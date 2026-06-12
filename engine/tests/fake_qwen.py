"""A scripted stand-in for QwenClient — pipeline tests never hit the network."""

from __future__ import annotations

from typing import Any

from engine.qwen import LLMError


class FakeQwen:
    """Duck-typed QwenClient: feed it canned chat/embedding responses."""

    def __init__(
        self,
        chat_responses: list[str | Exception] | None = None,
        embeddings: list[list[float]] | Exception | None = None,
    ) -> None:
        self.chat_responses = list(chat_responses or [])
        self.embeddings = embeddings if embeddings is not None else [[0.1, 0.2, 0.3]]
        self.chat_calls: list[list[dict[str, str]]] = []
        self.embed_calls: list[list[str]] = []

    async def chat(self, messages: list[dict[str, str]], **_: Any) -> str:
        self.chat_calls.append(messages)
        if not self.chat_responses:
            raise AssertionError("FakeQwen ran out of scripted chat responses")
        item = self.chat_responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def chat_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        import json

        from engine.qwen import strip_code_fences

        text = await self.chat(messages, **kwargs)
        try:
            parsed = json.loads(strip_code_fences(text))
        except json.JSONDecodeError:
            # Mirror the real client's single repair attempt.
            text = await self.chat(messages, **kwargs)
            parsed = json.loads(strip_code_fences(text))
        if not isinstance(parsed, dict):
            raise LLMError("not an object")
        return parsed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        if isinstance(self.embeddings, Exception):
            raise self.embeddings
        return [list(v) for v in self.embeddings][: len(texts)]
