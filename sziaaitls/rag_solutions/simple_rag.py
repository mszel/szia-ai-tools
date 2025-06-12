"""
Simple RAG solution: using a vector store and an embedder provides a RAG instance
from a document library.

It's a simple solution, will be extended by (TODO):
 - delete option
 - making the data load even more parallel (not just the embedding part)
"""

from typing import Any

from sziaaitls.document_library import DocumentLibrary
from sziaaitls.embedders import Embedding, EmbeddingSimilarity, TextEmbedder
from sziaaitls.vector_stores import VectorStore


class SimpleRAG:
    """
    Simple Retrieval-Augmented Generation (RAG) wrapper:
      - builds a vector index from a DocumentLibrary
      - runs similarity searches for queries
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: TextEmbedder,
    ):
        """
        Initialize an empty SimpleRAG instance.

        Args:
            vector_store: a VectorStore instance (FAISS or USearch)
            embedder: a TextEmbedder instance (OpenAI or Ollama)
        """
        self.vector_store = vector_store
        self.embedder = embedder
        # list of dicts with keys: id, uri, chunk_order, chunk_token_size
        self.metadata_list: list[dict[str, Any]] = []
        self.next_id: int = 1

    @classmethod
    async def create(
        cls,
        doclib: DocumentLibrary,
        embedder: TextEmbedder,
        vector_store: VectorStore,
    ) -> "SimpleRAG":
        """
        Async factory: build and populate the vector store from a DocumentLibrary.
        """
        rag = cls(vector_store, embedder)
        await rag.add_document_library(doclib)
        return rag

    async def add_document_library(self, doclib: DocumentLibrary) -> None:
        """
        Embed all chunks from the DocumentLibrary and add them to the vector store.

        Populates self.metadata_list.
        """
        # get all chunks with metadata
        chunks = await doclib.get_chunks_with_metadata()
        # generate embeddings for all chunk contents
        embeddings = await self._embedding_list_from_chunks(chunks)
        # add to vector store
        await self.vector_store.add(embeddings)
        # update metadata_list
        for emb in embeddings:
            md = emb.metadata or {}
            self.metadata_list.append(
                {
                    "id": emb.id,
                    "uri": md.get("uri"),
                    "chunk_order": md.get("chunk_order"),
                    "chunk_token_size": md.get("chunk_token_size"),
                }
            )

    async def _embedding_list_from_chunks(self, chunks: list[dict[str, Any]]) -> list[Embedding]:
        """
        Internal: generate Embedding objects for all chunks.

        Uses the embedder to compute vectors in batch.
        """
        # extract texts and ids
        texts: list[str] = []
        ids: list[int] = []
        metas: list[dict[str, Any]] = []
        for chunk in chunks:
            # assign a new integer ID
            eid = self.next_id
            self.next_id += 1
            ids.append(eid)
            texts.append(chunk["chunk_content"])
            # include chunk_order and token_size explicitly
            metas.append(
                {
                    **chunk["chunk_metadata"],
                    "chunk_order": chunk.get("chunk_order"),
                    "chunk_token_size": chunk.get("chunk_token_size"),
                }
            )
        # compute embeddings
        vectors = await self.embedder.acreate(texts)
        # build Embedding instances
        embeddings: list[Embedding] = []
        for eid, text, vec, md in zip(ids, texts, vectors, metas, strict=False):
            embeddings.append(
                Embedding(
                    id=eid,
                    embedding_content=text,
                    embedding_value=vec,
                    embedding_metadata=md,
                )
            )
        return embeddings

    async def search_query(
        self,
        query: str,
        max_results: int = 10,
        token_limit: int = 0,
    ) -> list[EmbeddingSimilarity]:
        """
        Search the vector store for the query text.

        Args:
            query: input query string to embed and search.
            max_results: maximum number of hits to return.
            token_limit: if >0, cap the sum of chunk_token_size across results.

        Returns:
            List of EmbeddingSimilarity, sorted by descending similarity,
            respecting max_results and token_limit.
        """
        # embed the query
        vecs = await self.embedder.acreate([query])
        if not vecs:
            return []
        qvec = vecs[0]
        # search vector store
        sims = await self.vector_store.search(qvec, max_results)
        # apply token_limit
        if token_limit and token_limit > 0:
            filtered: list[EmbeddingSimilarity] = []
            total_tokens = 0
            for sim in sims:
                ts = sim.embedding.metadata.get("chunk_token_size", 0)
                if total_tokens + ts > token_limit:
                    break
                filtered.append(sim)
                total_tokens += ts
            return filtered
        return sims
