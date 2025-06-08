"""
DocumentLibrary Module:
This module provides functionality to load, parse and chunk various document types.

TODO:
 - add proper logging
 - add the stat remover to the DocumentLibrary (calculating the dict, and removing
   inside the document class)

The codes are written by o4-mini-high (based on a provided design).
"""

import asyncio
import io
import os
import pickle
from typing import Any

import pandas as pd

from .crawlers import Crawler
from .documents import (
    Document,
    DocumentMetadata,
    HtmlDocument,
    MarkdownDocument,
    PdfDocument,
    TextDocument,
)


async def file_loader(file_paths: list[str]) -> list[Document]:
    """
    Asynchronously load local files into Document objects.

    Args:
        file_paths: List of file system paths (pdf, html, md, txt).

    Returns:
        A list of Document instances. On error or unknown type,
        returns an empty MarkdownDocument with crawling_error metadata.
    """
    docs: list[Document] = []
    for path in file_paths:
        metadata = DocumentMetadata({"uri": path})
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == ".pdf":
                with open(path, "rb") as f:
                    content = f.read()
                doc = PdfDocument(content, metadata)
            elif ext in (".html", ".htm"):
                text = open(path, encoding="utf-8").read()
                doc = HtmlDocument(text, metadata)
            elif ext == ".md":
                text = open(path, encoding="utf-8").read()
                doc = MarkdownDocument(text, metadata)
            elif ext == ".txt":
                text = open(path, encoding="utf-8").read()
                doc = TextDocument(text, metadata)
            else:
                print(f"Unsupported file type: {ext}, {path} can't be loaded.")
        except Exception as e:
            metadata.add_items({"crawling_error": True, "crawling_error_message": str(e)})
            doc = MarkdownDocument("", metadata)
        docs.append(doc)
    return docs


class DocumentLibrary:
    """
    Collection of Documents with unified metadata and chunk access.
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
    ):
        """
        Initialize a DocumentLibrary with existing Document instances.

        Args:
            documents: Optional list of already-constructed Document objects.
        """
        self.documents: list[Document] = documents.copy() if documents else []
        # mark all as enabled by default
        for doc in self.documents:
            doc.metadata.add_items({"enabled": True})

    @classmethod
    async def create(
        cls,
        documents: list[Document] | None = None,
        file_paths: list[str] | None = None,
        urls: list[str] | None = None,
        crawler_kwargs: dict[str, Any] | None = None,
        encoding: str = "utf-8-sig",
        ignore_single_newline: bool = True,
        chunk_method: str = "simple",
        max_chunk_size: int = 500,
        min_chunk_size: int = 0,
        min_leftover_chunk_size: int = 0,
        text_split_order: list[str] = None,
        tokenizer: str = "cl100k_base",
    ) -> "DocumentLibrary":
        """
        Async factory to build a DocumentLibrary from Documents, local files, and/or URLs.

        Args:
            documents: Pre-built Document objects.
            file_paths: List of file system paths to load.
            urls:       List of URLs to crawl and load.

        Returns:
            A fully initialized DocumentLibrary.
        """

        docs: list[Document] = documents.copy() if documents else []
        if file_paths:
            docs.extend(await file_loader(file_paths))
        if urls:
            crawler = Crawler(**crawler_kwargs) if crawler_kwargs else Crawler()
            docs.extend(await crawler.get_documents(urls))

        # Apply global settings to each Document
        for doc in docs:
            doc.encoding = encoding
            doc.ignore_single_newline = ignore_single_newline
            doc.chunk_method = chunk_method
            doc.max_chunk_size = max_chunk_size
            doc.min_chunk_size = min_chunk_size
            doc.min_leftover_chunk_size = min_leftover_chunk_size
            doc.text_split_order = text_split_order
            doc.tokenizer = tokenizer

        # Mark all documents as enabled by default
        for doc in docs:
            doc.metadata.add_items({"enabled": True})

        # Construct and return
        return cls(documents=docs)

    def save(self, path: str) -> None:
        """
        Serialize this DocumentLibrary to disk for later loading.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "DocumentLibrary":
        """
        Load a previously saved DocumentLibrary.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    async def get_chunks_with_metadata(self) -> list[dict[str, Any]]:
        """
        Retrieve all chunks across enabled, successfully parsed documents in parallel.

        Returns a flat list of chunk dicts, each including:
          - chunk_order, chunk_token_size, chunk_content
          - chunk_metadata: originating document metadata dict
        """
        # filter docs
        valid_docs = []
        for doc in self.documents:
            meta = doc.metadata.to_dict()
            if not meta.get("enabled", True):
                continue
            if meta.get("parse_error") or meta.get("crawling_error"):
                continue
            valid_docs.append((doc, meta))

        # schedule all chunking tasks concurrently
        # chunk is loaded from Document.chunks if exists (see Document.get_chunks)
        tasks = [doc.get_chunks() for doc, _ in valid_docs]
        results = await asyncio.gather(*tasks)

        # collect combined chunks
        combined: list[dict[str, Any]] = []
        for idx, (_, meta) in enumerate(valid_docs):
            res = results[idx]
            enriched = [{**r, "chunk_metadata": meta} for r in res]
            combined.extend(enriched)
        return combined

    async def get_chunk_df(self) -> pd.DataFrame:
        """
        Produce a DataFrame of all chunks, sorted by URI and chunk_order.

        Columns: uri, chunk_order, chunk_token_size, chunk_content
        TODO: test the speed and decide how much we'd like to cache this.
        """
        records = await self.get_chunks_with_metadata()
        # expand metadata.uri into top-level column
        for r in records:
            r["uri"] = r["chunk_metadata"].get("uri")
        df = pd.DataFrame(records)
        return df.sort_values(["uri", "chunk_order"])[
            ["uri", "chunk_order", "chunk_token_size", "chunk_content"]
        ]

    def get_metadata_df(self) -> pd.DataFrame:
        """
        Produce a DataFrame of document-level metadata.
        """
        rows: list[dict[str, Any]] = []
        for doc in self.documents:
            rows.append(doc.metadata.to_dict())
        return pd.DataFrame(rows)

    def set_metadata_item(self, flags: dict[str, bool]) -> None:
        """
        Enable or disable documents by their URI.

        Args:
            flags: Mapping of uri -> enabled (True/False).
        """
        for doc in self.documents:
            uri = doc.metadata.to_dict().get("uri")
            if uri in flags:
                doc.metadata.add_items({"enabled": flags[uri]})
