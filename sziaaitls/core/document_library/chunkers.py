"""
Simple text chunker for raw Markdown (text) content.
In a later phase, more sophisticated chunking logic can be added.

TODO: semantic chunking, hierarchical chunking, summary and context
      generation.
"""

from typing import Any

import tiktoken


def chunk_text(
    text: str,
    *,
    method: str = "simple_intervals",
    max_chunk_size: int = 500,
    min_chunk_size: int = 0,
    min_leftover_chunk_size: int = 0,
    text_split_order: list[str] = None,
    encoding_name: str = "cl100k_base",
    **kwargs,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Split `text` into token-based chunks with metadata.

    Args:
        text: Full document text.
        method: Chunking strategy: "simple" or "simple_intervals".
        max_chunk_size: Maximum tokens per chunk.
        min_chunk_size: Minimum tokens per chunk for interval merging.
        min_leftover_chunk_size: If leftover chunk < this, merge it into the previous chunk (simple method).
        text_split_order: Preferred delimiters for interval splitting.
        encoding_name: Tokenizer encoding (e.g. "cl100k_base").
        **kwargs: Extra params for future methods.

    Returns:
        chunks: List of dicts with keys:
            - 'chunk_order': int
            - 'chunk_token_size': int
            - 'chunk_content': str
        metadata: Dict with 'document_token_size': total token count

    TODO: the current header could fit to code comments as well - that should be handled in a next version.
        - we can also create chunking functions for python codes, etc.
    """
    # Default split order
    if text_split_order is None:
        text_split_order = [
            "\n# ",
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            "?",
            "!",
            ".",
            ",",
            " ",
        ]
    text_split_order = [delim for delim in text_split_order if delim in text]

    # Initialize tokenizer and metadata
    enc = tiktoken.get_encoding(encoding_name)
    doc_tokens = enc.encode(text)
    total_tokens = len(doc_tokens)
    metadata = {"document_token_size": total_tokens}

    # SIMPLE: fixed windows + optional leftover merge
    if method == "simple":
        token_chunks: list[list[int]] = []
        for i in range(0, total_tokens, max_chunk_size):
            token_chunks.append(doc_tokens[i : i + max_chunk_size])
        # Merge a too-small final chunk
        if (
            min_leftover_chunk_size > 0
            and len(token_chunks) > 1
            and len(token_chunks[-1]) < min_leftover_chunk_size
        ):
            token_chunks[-2].extend(token_chunks[-1])
            token_chunks.pop()

        chunks: list[dict[str, Any]] = []
        for idx, tok_chunk in enumerate(token_chunks, start=1):
            chunks.append(
                {
                    "chunk_order": idx,
                    "chunk_token_size": len(tok_chunk),
                    "chunk_content": enc.decode(tok_chunk),
                }
            )
        return chunks, metadata

    # INTERVALS: hierarchical splitting + interval grouping
    elif method == "simple_intervals":
        # First, split oversized text into smaller segments
        segments = [text]
        for delim in text_split_order:
            new_segments: list[str] = []
            for seg in segments:
                seg_tokens = enc.encode(seg)
                if len(seg_tokens) > max_chunk_size:
                    parts = seg.split(delim)
                    for i, part in enumerate(parts):
                        suffix = delim if i < len(parts) - 1 else ""
                        new_segments.append(part + suffix)
                else:
                    new_segments.append(seg)
            segments = new_segments

        # Group segments into raw token lists
        raw_chunks: list[dict[str, Any]] = []
        curr_tokens: list[int] = []
        curr_segments: list[str] = []
        for seg in segments:
            seg_tokens = enc.encode(seg)
            if not curr_tokens:
                curr_tokens = seg_tokens.copy()
                curr_segments = [seg]
            elif len(curr_tokens) + len(seg_tokens) <= max_chunk_size:
                curr_tokens.extend(seg_tokens)
                curr_segments.append(seg)
            else:
                raw_chunks.append({"tokens": curr_tokens.copy(), "text": "".join(curr_segments)})
                curr_tokens = seg_tokens.copy()
                curr_segments = [seg]
        # Add final segment
        if curr_segments:
            raw_chunks.append({"tokens": curr_tokens.copy(), "text": "".join(curr_segments)})

        # Merge a too-small final raw chunk
        if (
            min_chunk_size > 0
            and len(raw_chunks) > 1
            and len(raw_chunks[-1]["tokens"]) < min_chunk_size
        ):
            prev = raw_chunks[-2]
            last = raw_chunks[-1]
            prev["tokens"].extend(last["tokens"])
            prev["text"] += last["text"]
            raw_chunks.pop()

        # Build structured chunks
        out_chunks: list[dict[str, Any]] = []
        for idx, obj in enumerate(raw_chunks, start=1):
            out_chunks.append(
                {
                    "chunk_order": idx,
                    "chunk_token_size": len(obj["tokens"]),
                    "chunk_content": obj["text"],
                }
            )

        # Merge any chunks smaller than min_chunk_size into previous
        if min_chunk_size > 0:
            merged: list[dict[str, Any]] = []
            for chunk in out_chunks:
                if merged and chunk["chunk_token_size"] < min_chunk_size:
                    merged[-1]["chunk_content"] += chunk["chunk_content"]
                    merged[-1]["chunk_token_size"] += chunk["chunk_token_size"]
                else:
                    merged.append(chunk)
            out_chunks = merged

        return out_chunks, metadata

    # Unknown method
    else:
        raise ValueError(f"Unknown chunking method: {method}")
