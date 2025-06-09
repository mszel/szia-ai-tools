"""
Vector Store covering class for supporting RAG-dependent applications.
This is a preliminary version with very limited functionality. TODO:
 - Add save
 - Add load
 - Add delete
 - Add multi-vector support
 - Add cloud-based vector stores (e.g. Pinecone)

The codes are written by o4-mini-high (based on a provided design).
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import faiss
import numpy as np

# VDBs
from usearch.index import Index

from sziaaitls.embedders import Embedding, EmbeddingSimilarity


class VectorStoreBase(ABC):
    """
    Abstract base for vector store implementations.
    """

    @abstractmethod
    async def add(self, embeddings: list[Embedding]) -> None:
        """
        Upsert a list of Embedding objects into the store.
        """
        pass

    @abstractmethod
    async def search(self, embedding: list[float], k: int) -> list[EmbeddingSimilarity]:
        """
        Find the top-k most similar embeddings to the input vector.
        """
        pass


class VectorStoreUSearch(VectorStoreBase):
    """
    Vector store backed by USearch library (https://github.com/unum-cloud/usearch).
    """

    def __init__(self, dimension: int, metric: str = "cos", **index_kwargs: Any):
        """
        Initialize a USearch Index for given dimensionality and metric.

        Args:
            dimension: Number of dimensions of input vectors.
            metric: Distance metric (e.g. 'cos', 'l2sq', 'ip').
            **index_kwargs: Other Index parameters (connectivity, expansion_add, etc.).
        """

        self.index = Index(ndim=dimension, metric=metric, **index_kwargs)
        # keep metadata mappings
        self.id_to_meta: dict[str, dict[str, Any]] = {}
        self.id_to_content: dict[str, str] = {}
        self.np = np

    async def add(self, embeddings: list[Embedding]) -> None:
        """
        Upsert embeddings into the USearch index.

        Args:
            embeddings: List of Embedding instances to add.
        """
        keys = []
        vecs = []
        for emb in embeddings:
            keys.append(emb.id)
            vecs.append(self.np.array(emb.embedding_value, dtype=self.np.float32))
            # store metadata & content
            self.id_to_meta[emb.id] = emb.metadata
            self.id_to_content[emb.id] = emb.embedding_content

        xb = self.np.stack(vecs)
        # index.add can take array or single vector; wrap in thread
        await asyncio.to_thread(self.index.add, keys, xb)

    async def search(self, embedding: list[float], k: int) -> list[EmbeddingSimilarity]:
        """
        Query the index for top-k nearest neighbors.

        Args:
            embedding: Query vector.
            k: Number of neighbors to retrieve.

        Returns:
            List of EmbeddingSimilarity results.
        """
        vec = self.np.array(embedding, dtype=self.np.float32)
        # perform search in thread
        matches = await asyncio.to_thread(self.index.search, vec, k)
        results: list[EmbeddingSimilarity] = []
        for hit in matches:
            key = hit.key
            # For cosine, hit.distance is 1 - cosine_similarity
            similarity = 1.0 - hit.distance if hasattr(hit, "distance") else 0.0

            # rebuild Embedding
            meta = self.id_to_meta.get(key, {})
            content = self.id_to_content.get(key, "")
            chunk = {
                "embedding_content": content,
                "embedding_metadata": meta,
            }
            emb = Embedding(id=key, **chunk, embedding_value=[])
            results.append(EmbeddingSimilarity(emb, similarity))
        return results


class VectorStoreFAISS(VectorStoreBase):
    """
    FAISS-backed vector store with ID mapping and cosine similarity.
    """

    def __init__(self, dimension: int):
        # Use flat inner-product index + ID mapping
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))
        self.id_to_meta: dict[int, dict[str, Any]] = {}
        self.str_to_int: dict[str, int] = {}
        self.next_int_id = 1
        self.faiss = faiss
        self.np = np

    async def add(self, embeddings: list[Embedding]) -> None:
        """
        Upsert embeddings into the FAISS index, normalizing for cosine similarity.
        """
        to_add_ids = []
        to_add_vecs = []
        for emb in embeddings:
            sid = emb.id
            if sid in self.str_to_int:
                iid = self.str_to_int[sid]
                # Remove old vector
                self.index.remove_ids(self.np.array([iid], dtype="int64"))
            else:
                iid = self.next_int_id
                self.next_int_id += 1
                self.str_to_int[sid] = iid

            vec = self.np.array(emb.embedding_value, dtype="float32")
            # Normalize for cosine
            self.faiss.normalize_L2(vec.reshape(1, -1))
            to_add_ids.append(iid)
            to_add_vecs.append(vec)
            # store metadata
            meta = emb.metadata.copy()
            meta.update(
                {
                    "embedding_content": emb.embedding_content,
                    "embedding_id": emb.id,
                }
            )
            self.id_to_meta[iid] = meta

        # Bulk add
        xb = self.np.stack(to_add_vecs)
        id_arr = self.np.array(to_add_ids, dtype="int64")
        self.index.add_with_ids(xb, id_arr)

    async def search(self, embedding: list[float], k: int) -> list[EmbeddingSimilarity]:
        """
        Search for top-k similar entries to the given embedding.
        """
        vec = self.np.array(embedding, dtype="float32")
        self.faiss.normalize_L2(vec.reshape(1, -1))
        D, I = self.index.search(vec.reshape(1, -1), k)
        results: list[EmbeddingSimilarity] = []
        for score, iid in zip(D[0], I[0], strict=False):
            meta = self.id_to_meta.get(iid, {})
            # Reconstruct chunk dict
            chunk = {
                "embedding_content": meta.get("embedding_content"),
                "id": meta.get("embedding_id"),
                "embedding_metadata": dict(
                    [
                        (_k, _v)
                        for _k, _v in meta.items()
                        if _k != "embedding_content" and _k != "embedding_id"
                    ]
                ),
            }
            emb = Embedding(**chunk, embedding_value=[])  # no vector
            results.append(EmbeddingSimilarity(emb, float(score)))
        return results


# Facade for both stores
class VectorStore(VectorStoreBase):
    """
    Facade for vector stores. Select 'usearch' or 'faiss' backend.
    """

    def __init__(self, name: str = "faiss", **kwargs: Any):
        provider = name.lower()
        if provider in ("faiss",):
            # For FAISS, expect `dimension` kwarg
            self.impl = VectorStoreFAISS(**kwargs)
        elif provider in ("usearch", "u-search"):
            # For USearch, expect `ndim` and optional index settings
            self.impl = VectorStoreUSearch(**kwargs)
        else:
            raise ValueError(f"Unknown vector store: {name}")

    async def add(self, embeddings: list[Embedding]) -> None:
        """
        Upsert embeddings into the selected vector store.
        """
        await self.impl.add(embeddings)

    async def search(self, embedding: list[float], k: int) -> list[EmbeddingSimilarity]:
        """
        Search for top-k similar embeddings using the selected backend.
        """
        return await self.impl.search(embedding, k)
