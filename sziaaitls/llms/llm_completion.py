import asyncio
from abc import ABC, abstractmethod
from typing import Any

import openai
from aiolimiter import AsyncLimiter

from .prompts import ChatCompletionPrompt, Message


class LLMPromptProcessorBase(ABC):
    """
    Abstract base for LLM chat completion providers.
    """

    @abstractmethod
    async def acreate(
        self,
        prompt: ChatCompletionPrompt,
        functions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Message:
        """
        Asynchronously run a chat completion and return the assistant Message.
        """
        pass


class LLMPromptProcessorOpenai(LLMPromptProcessorBase):
    """
    OpenAI chat completion using the OpenAI Python V2 client.

    Uses an AsyncLimiter to respect rate limits (requests per time period).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        rate_limit: int = 350,
        period: int = 60,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the OpenAI chat completion client.

        Args:
            api_key:      Your OpenAI API key.
            model:        Model name for chat completions.
            rate_limit:   Max requests per period.
            period:       Time window in seconds.
            temperature:  Sampling temperature.
            max_tokens:   Max tokens in the response.
            base_url:     Optional override for OpenAI API base (e.g. Ollama).
        """
        # Instantiate V2 client
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.limiter = AsyncLimiter(max_rate=rate_limit, time_period=period)

    async def acreate(
        self,
        prompt: ChatCompletionPrompt,
        functions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Message:
        """
        Send a chat completion request and return the assistant's Message.

        Supports function-calling if `functions` is provided.
        """
        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": prompt.to_dicts(),
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            api_kwargs["max_tokens"] = self.max_tokens
        if functions is not None:
            api_kwargs["functions"] = functions
        api_kwargs.update(kwargs)

        async with self.limiter:
            # run synchronous call in threadpool
            resp = await asyncio.to_thread(self.client.chat.completions.create, **api_kwargs)
        # Extract the first choice's message
        choice = resp.choices[0].message
        return Message.from_openai_message(choice)


# Facade
class LLMPromptProcessor:
    """
    High-level interface selecting a chat completion backend.
    Supported names: 'openai', 'ollama' (stub).
    """

    def __init__(self, name: str = "openai", **kwargs):
        provider = name.lower()
        if provider == "openai":
            self.impl = LLMPromptProcessorOpenai(**kwargs)
        elif provider == "ollama":
            # Placeholder for Ollama-based implementation
            raise NotImplementedError("Ollama chat not yet implemented")
        else:
            raise ValueError(f"Unknown LLM provider: {name}")

    @classmethod
    async def from_params(cls, name: str = "openai", **kwargs) -> "LLMPromptProcessor":
        """
        Async constructor for LLMCompletion.
        """
        return cls(name, **kwargs)

    async def acreate(
        self,
        prompt: ChatCompletionPrompt,
        functions: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Message:
        """
        Delegate to the selected provider's acreate.
        """
        return await self.impl.acreate(prompt, functions=functions, **kwargs)
