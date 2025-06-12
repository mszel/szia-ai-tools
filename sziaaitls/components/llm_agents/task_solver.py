"""
Solving tasks using the LLMPromptProcessor.

TODO:
 - cache
 - after openai resolves the message format, follow the new structure
"""

import asyncio
from typing import Any

from sziaaitls.core.llms import ChatCompletionPrompt, LLMPromptProcessor, Message


class TaskSolver:
    """
    A generic task executor that formats a prompt from base messages,
    fills placeholders, and invokes an async LLM prompt processor.

    Supports single and batch execution, including rich content (images, audio).
    """

    def __init__(
        self,
        base_messages: list[Message],
        llm: LLMPromptProcessor,
    ):
        """
        Initialize TaskSolver.

        Args:
            base_messages: A list of Message instances defining the task template.
            llm:           An LLMPromptProcessor instance for chat completions.
        """
        self.base_messages = base_messages
        self.llm = llm

    async def solve(
        self,
        input_object: dict[str, Any] | Message,
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Message:
        """
        Fill placeholders in base_messages with `params`, build a prompt,
        and call the LLM to get a completion Message.

        Args:
            input_object:    Can be a Message (preferably user role or a mapping of
                             placeholder names to values for formatting.
            functions:       Optional list of function definitions for the chat API.
            **kwargs:        Additional keyword args to pass to the LLM `.acreate()`.

        Returns:
            The assistant's response as a Message.
        """

        if isinstance(input_object, dict):
            # old behavior: format every template
            formatted = [m.format(**input_object) for m in self.base_messages]
        elif isinstance(input_object, Message):
            # split out system vs. everything‐else in your base template
            # as only one user message is allowed
            system_msgs = [m for m in self.base_messages if m.role == "system"]
            formatted = system_msgs + [input_object]
        else:
            raise TypeError("solve() accepts either a param dict or a Message")

        prompt = ChatCompletionPrompt(formatted)
        return await self.llm.acreate(prompt, functions=functions, **kwargs)

    async def solve_batch(
        self,
        input_list: list[dict[str, Any] | Message],
        functions: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[Message]:
        """
        Execute `solve` concurrently over a list of param dicts.

        Args:
            params_list: A list of parameter mappings for each call.
            functions:   Optional shared function definitions.
            **kwargs:    Additional args to pass to each `.solve()`.

        Returns:
            List of assistant Messages corresponding to each param set.
        """
        tasks = [self.solve(params, functions=functions, **kwargs) for params in input_list]
        return await asyncio.gather(*tasks)

    @classmethod
    async def create(
        cls, base_messages: Message | list[Message], llm_name: str = "openai", **llm_kwargs
    ) -> "TaskSolver":
        """
        Async factory: build a TaskSolver with system and user message templates.

        Args:
            base_messages:   A Message or list of Messages for system role.
            llm_name:        Name of LLM provider ('openai', 'ollama').
            **llm_kwargs:    Passed to LLMPromptProcessor.constructor.
        """

        # instantiate LLM client
        llm = await LLMPromptProcessor.acreate(name=llm_name, **llm_kwargs)
        return cls(base_messages, llm)
