"""
TextEmbedder class that covers openai, ollama and (later) other text embedding models.

TODO: this is a simplified class, the followings should be added later:
- Adding cache (sqlite)
- Error handling
- Proper logging

The codes are written by o4-mini-high (based on a provided design).
"""

import asyncio
from abc import ABC, abstractmethod

import aiohttp
import openai
from aiolimiter import AsyncLimiter


class TextEmbedderBase(ABC):
    """
    Abstract base class defining the interface for text embedders.

    Subclasses must implement `acreate`, which asynchronously produces
    embeddings for a list of input texts.
    """

    @abstractmethod
    async def acreate(self, texts: list[str]) -> list[list[float]]:
        """
        Asynchronously generate embeddings for the given texts.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors (one list of floats per input text).
        """
        pass


class TextEmbedderOpenai(TextEmbedderBase):
    """
    OpenAI-based text embedder using the OpenAI Embeddings API.

    Uses an AsyncLimiter to respect rate limits (requests per time period),
    and batches inputs to stay within token limits.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        rate_limit: int = 3000,
        period: int = 60,
        max_batch_tokens: int = 8191,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            api_key: Your OpenAI API key.
            model: Name of the embedding model to use.
            rate_limit: Maximum number of requests per `period`.
            period: Time window in seconds for the rate limit.
            max_batch_tokens: Max total tokens per batch of inputs.
            encoding_name: Tokenizer name for batching (e.g. 'cl100k_base').
        """
        import tiktoken

        openai.api_key = api_key
        self.model = model
        self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=period)
        self.max_batch_tokens = max_batch_tokens
        self.encoding_name = encoding_name
        self._tiktoken = tiktoken.get_encoding(encoding_name)

    async def acreate(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for `texts` via OpenAI, using batching and async.

        Args:
            texts: A list of strings to embed.

        Returns:
            A list of embedding vectors corresponding to each input.
        """
        # Tokenize each text to count tokens
        token_lists = [self._tiktoken.encode(t) for t in texts]
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0
        # Create batches under max_batch_tokens
        for text, tokens in zip(texts, token_lists, strict=False):
            if current_batch and current_tokens + len(tokens) > self.max_batch_tokens:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(text)
            current_tokens += len(tokens)
        if current_batch:
            batches.append(current_batch)

        # Define helper to embed a single batch in a thread
        async def _embed_batch(batch_texts: list[str]):
            async with self.limiter:
                # run sync call in threadpool
                resp = await asyncio.to_thread(
                    openai.embeddings.create,
                    input=batch_texts,
                    model=self.model,
                )
            # extract embeddings, handling both object and dict
            embs: list[list[float]] = []
            for d in resp.data:
                if hasattr(d, "embedding"):
                    embs.append(d.embedding)
                else:
                    embs.append(d["embedding"])
            return embs

        # Embed all batches concurrently
        tasks = [_embed_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        # Flatten in original order
        embeddings: list[list[float]] = []
        for batch_emb in results:
            embeddings.extend(batch_emb)
        return embeddings


class TextEmbedderOllama(TextEmbedderBase):
    """
    Ollama-based text embedder using a local Ollama server.

    Wraps the `/embeddings` endpoint. Uses AsyncLimiter similarly.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text",
        api_key: str | None = None,
        rate_limit: int = 3000,
        period: int = 60,
    ):
        """
        Initialize the Ollama embedder.

        Args:
            base_url: URL of the Ollama embeddings endpoint.
            model: Ollama model identifier.
            api_key: Token for authorization if required.
            rate_limit: Max requests per `period`.
            period: Rate limit time window in seconds.
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=period)
        self._session: aiohttp.ClientSession | None = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def acreate(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings via Ollama's REST API.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        payload = {"model": self.model, "input": texts}
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with self.limiter:
            async with self.session.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
            ) as resp:
                data = await resp.json()

        return [item["embedding"] for item in data.get("data", [])]


class TextEmbedder:
    """
    Factory/Facade for text embedders. Selects a backend by name.

    Supported names: "openai", "ollama".
    """

    def __init__(self, name: str = "openai", **kwargs):
        """
        Instantiate the specified text embedder.

        Args:
            name: Provider name (e.g. "openai", "ollama").
            **kwargs: Passed to the specific embedder constructor.
        """
        provider = name.lower()
        if provider == "openai":
            self.impl = TextEmbedderOpenai(**kwargs)
        elif provider == "ollama":
            self.impl = TextEmbedderOllama(**kwargs)
        else:
            raise ValueError(f"Unknown embedder provider: {name}")

    @classmethod
    async def acreate(cls, name: str = "openai", **kwargs) -> "TextEmbedder":
        """
        Async constructor. Returns a ready-to-use TextEmbedder.
        """
        return cls(name, **kwargs)

    async def acreate(self, texts: list[str]) -> list[list[float]]:
        """
        Proxy to underlying embedder's `acreate`.
        """
        return await self.impl.acreate(texts)
