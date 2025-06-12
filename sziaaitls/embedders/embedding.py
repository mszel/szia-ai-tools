"""
Embedding and EmbeddingSimilarity classes for vector stores.
"""

from typing import Any


class Embedding:
    def __init__(
        self,
        id: str,
        embedding_content: str,
        embedding_value: list[float],
        embedding_metadata: dict[str, Any] | None = None,
    ):
        self.embedding_content: str = embedding_content
        self.embedding_value: list[float] = embedding_value
        self.metadata = embedding_metadata
        self.id = id


class EmbeddingSimilarity:
    def __init__(self, embedding: Embedding, similarity: float):
        self.embedding = embedding
        self.similarity = similarity
