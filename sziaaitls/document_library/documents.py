"""
Defining Document classes for various content types that can be used under
the DocumentLibrary module.

The codes are written by o4-mini-high (based on a provided design).
"""

# ─── Import Base Libraries ──────────────────────────────────────────────────

from abc import ABC, abstractmethod
from typing import Any

from sziaaitls.document_library.chunkers import chunk_text

# ─── Import parsers and chunkers ────────────────────────────────────────────
from sziaaitls.document_library.parsers import (
    parse_html,
    parse_markdown,
    parse_pdf,
    parse_text,
)

# ─── DocumentMetadata ────────────────────────────────────────────────────────


class DocumentMetadata:
    """
    A container for document metadata.

    Stores arbitrary key→value pairs, exposes them as attributes
    (e.g. `.uri`, `.source`) and via dict-style lookup.
    """

    def __init__(self, metadata: dict[str, Any]):
        """
        Initialize DocumentMetadata.

        Args:
            metadata: A dict of initial metadata values. Each key becomes
                      both a dict entry and an attribute on this object.
        """
        # copy incoming dict so external changes don’t affect us
        self._data = metadata.copy()

        # expose each key as an attribute
        for key, value in self._data.items():
            setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """
        Produce a shallow copy of all metadata as a dict.

        Returns:
            A dict of all current metadata key→value pairs.
        """
        return self._data.copy()

    def add_items(self, items: dict[str, Any]) -> None:
        """
        Add or update multiple metadata entries.

        Args:
            items: A dict of key→value pairs to merge into existing metadata.
                   Existing keys will be overwritten; new ones will be added.
        """
        for key, value in items.items():
            self._data[key] = value
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """
        Enable dict-style access: meta['uri'].
        """
        return self._data[key]

    def __repr__(self) -> str:
        """
        Unambiguous representation, useful for debugging.
        """
        return f"DocumentMetadata({self._data})"


# ─── Base Document ───────────────────────────────────────────────────────────


class Document(ABC):
    """
    Abstract base class for all document types in Generative AI Pipelines.

    A Document holds:
      - the original raw content (page_content),
      - a parsed representation suitable for downstream processing (parsed_content),
      - and a list of text “chunks” (chunks) ready for embedding/retrieval.

    Subclasses must implement `get_parsed_content`, which turns `page_content`
    into raw Markdown in `parsed_content`.  The `get_chunks` logic is shared.
    """

    def __init__(
        self,
        page_content: Any,
        metadata: DocumentMetadata,
        encoding: str = "utf-8-sig",
        ignore_single_newline: bool = True,
        chunk_method: str = "simple",
        max_chunk_size: int = 500,
        min_chunk_size: int = 0,
        min_leftover_chunk_size: int = 0,
        text_split_order: list[str] = None,
    ):
        """
        Initialize a Document.

        Args:
            page_content: Raw content as loaded (bytes for PDFs, str for HTML/MD/Text).
            metadata:     A DocumentMetadata object carrying source info, etc.
        """
        self.page_content = page_content
        self.parsed_content: str = ""
        self.chunks: list[str] = []
        self.metadata = metadata

        # parsing settings
        self.encoding = encoding
        self.ignore_single_newline = ignore_single_newline

        # chunking settings
        self.chunk_method = chunk_method
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.min_leftover_chunk_size = min_leftover_chunk_size
        self.text_split_order = text_split_order

    @abstractmethod
    async def get_parsed_content(self) -> str:
        """
        Parse raw `page_content` into text.

        This method must:
          - read self.page_content,
          - produce a raw Markdown representation of the content,
          - store it in self.parsed_content, and
          - return that string.

        Returns:
            The parsed text of this document.
        """
        pass

    async def get_chunks(
        self,
        method: str = None,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        min_leftover_chunk_size: int = None,
        text_split_order: list[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Chunk `parsed_content` using either instance defaults or provided overrides.

        Args:
            method: Override chunking algorithm.
            max_chunk_size: Override max tokens per chunk.
            min_chunk_size: Override min tokens per chunk for interval merging.
            min_leftover_chunk_size: Override merge threshold for simple method.
            text_split_order: Override delimiters for interval splitting.

        If `parsed_content` is empty, calls `get_parsed_content` first.

        Returns:
            Tuple of:
            - List of chunk dicts ({'chunk_order', 'chunk_token_size', 'chunk_content'})
            - metadata dict ({'document_token_size': ...})
        """
        # ensure parsed_content
        if not self.parsed_content:
            text, parse_meta = await self.get_parsed_content()
            self.parsed_content = text
            self.metadata.add_items(parse_meta)

        # Use overrides or fall back to defaults
        cfg = {
            "method": method or self.chunk_method,
            "max_chunk_size": max_chunk_size or self.max_chunk_size,
            "min_chunk_size": min_chunk_size if min_chunk_size is not None else self.min_chunk_size,
            "min_leftover_chunk_size": min_leftover_chunk_size
            if min_leftover_chunk_size is not None
            else self.min_leftover_chunk_size,
            "text_split_order": text_split_order or self.text_split_order,
            "encoding_name": self.encoding,
        }

        # perform chunking
        chunks, chunk_meta = chunk_text(
            self.parsed_content,
            method=cfg["method"],
            max_chunk_size=cfg["max_chunk_size"],
            min_chunk_size=cfg["min_chunk_size"],
            min_leftover_chunk_size=cfg["min_leftover_chunk_size"],
            text_split_order=cfg["text_split_order"],
            encoding_name=cfg["encoding_name"],
        )
        self.chunks = chunks
        self.metadata.add_items(chunk_meta)
        return chunks


# ─── Concrete Document Types ─────────────────────────────────────────────────
# TODO: adding further content types, starting with docx


class PdfDocument(Document):
    """A Document wrapping raw PDF bytes and extracting raw Markdown from it."""

    async def get_parsed_content(self) -> str:
        # assume page_content is bytes
        self.parsed_content, metadata_addition = parse_pdf(
            self.page_content, self.encoding, self.ignore_single_newline
        )
        self.metadata.add_items(metadata_addition)
        return self.parsed_content


class HtmlDocument(Document):
    """A Document wrapping raw HTML content and extracting raw Markdown from it."""

    async def get_parsed_content(self) -> str:
        self.parsed_content, metadata_addition = parse_html(
            self.page_content, self.encoding, self.ignore_single_newline
        )
        self.metadata.add_items(metadata_addition)
        return self.parsed_content


class MarkdownDocument(Document):
    """A Document wrapping raw Markdown content."""

    async def get_parsed_content(self) -> str:
        self.parsed_content, metadata_addition = parse_markdown(
            self.page_content, self.encoding, self.ignore_single_newline
        )
        self.metadata.add_items(metadata_addition)
        return self.parsed_content


class TextDocument(Document):
    """A Document wrapping raw plain text content."""

    async def get_parsed_content(self) -> str:
        self.parsed_content, metadata_addition = parse_text(
            self.page_content, self.encoding, self.ignore_single_newline
        )
        self.metadata.add_items(metadata_addition)
        return self.parsed_content
