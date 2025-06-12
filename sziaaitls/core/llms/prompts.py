"""
Message and ChatCompletionPrompt classes for handling chat messages
and formatting for LLM chat completions.

TODO: the structure of the messages are slightly messed up, but we won't
      fix it until the new openai method comes out (threads).
"""

from typing import Any


class Message:
    """
    A flexible container for chat messages (system/user/assistant),
    with support for formatting placeholders in content.
    """

    def __init__(
        self, role: str, content: str | None = None, name: str | None = None, **kwargs: Any
    ):
        self.role = role
        self.content = content
        self.name = name
        self.extra = kwargs

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to the raw dict for API calls.
        """
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.name is not None:
            msg["name"] = self.name
        msg.update(self.extra)
        return msg

    def format(self, **params: Any) -> "Message":
        """
        Return a new Message with placeholders in content replaced.

        Unknown placeholders are left intact.
        """
        if self.content is None:
            # nothing to format
            return Message(self.role, None, self.name, **self.extra)

        # a dict that returns "{key}" for missing keys
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        try:
            formatted = self.content.format_map(_SafeDict(params))
        except Exception:
            # fall back in the very unlikely case of other format errors
            formatted = self.content

        return Message(self.role, formatted, self.name, **self.extra)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """
        Create a Message instance from an API response dict.
        """
        role = data.get("role")
        content = data.get("content", None)
        name = data.get("name", None)
        extras = {k: v for k, v in data.items() if k not in {"role", "content", "name"}}
        return cls(role=role, content=content, name=name, **extras)

    @classmethod
    def from_openai_message(cls, msg: Any) -> "Message":
        """
        Create a Message from an OpenAI ChatCompletionMessage or similar object.
        This will use `to_dict()` if available, or fall back to attributes.
        """
        # Try to use msg.to_dict() if present
        if hasattr(msg, "to_dict") and callable(msg.to_dict):
            data = msg.to_dict()
        else:
            # Fallback: extract common attributes
            data = {
                "role": getattr(msg, "role", None),
                "content": getattr(msg, "content", None),
                "name": getattr(msg, "name", None),
            }
            # include any function_call or other extras
            for attr in ("function_call", "tool_calls", "annotations", "audio"):
                if hasattr(msg, attr):
                    val = getattr(msg, attr)
                    if val is not None:
                        data[attr] = val

        return cls.from_dict(data)


class ChatCompletionPrompt:
    """
    Bundles a list of Message instances for a chat completion.
    """

    def __init__(self, messages: list[Message]):
        self.messages = messages

    def to_dicts(self) -> list[dict[str, Any]]:
        """
        Convert each Message to the dict form expected by the API.
        """
        return [m.to_dict() for m in self.messages]
