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

    async def aclose(self) -> None:
        pass


class SmartFakeQwen:
    """Content-routing brain: replies based on which pipeline stage is asking,
    so tests are robust to concurrent background tasks (curiosity, consolidation)
    interleaving with later turns. No scripted call counts to get wrong."""

    def __init__(self) -> None:
        self.chat_calls: list[list[dict[str, str]]] = []
        self.embed_calls: list[list[str]] = []

    async def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        self.chat_calls.append(messages)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "understanding stage" in system:
            return self._decompose(user)
        if "curiosity stage" in system:
            return "A short contextual definition."
        if "consolidation stage" in system:
            return "A consolidated summary of the cluster."
        # A normal answer — tagged so tests can tell the two panes apart.
        if "long-term memory" in system:
            return f"MEM[{self._mem_signal(system)}]: answer to {user[:40]}"
        return f"RAW: answer to {user[:40]}"

    async def chat_json(self, messages: list[dict[str, str]], **_: object) -> dict:
        import json

        return json.loads(self._decompose(messages[-1]["content"]))

    def _decompose(self, user: str) -> str:
        import json

        from engine.decompose import lite_keywords

        # The decompose prompt embeds the raw message after "USER MESSAGE:".
        message = user
        if "USER MESSAGE:" in user:
            message = user.split("USER MESSAGE:", 1)[1].split("ASSISTANT REPLY:", 1)[0].strip()
        keywords = lite_keywords(message) or ["topic"]
        likes = ["coffee"] if "love coffee" in message.lower() else []
        dislikes = ["coffee"] if "hate coffee" in message.lower() else []
        return json.dumps(
            {
                "meaning": f"The user said: {message[:120]}",
                "intent": "converse",
                "keywords": keywords[:5],
                "entities": [],
                "preferences_delta": {"likes": likes, "dislikes": dislikes, "facts": []},
                "emotional_tone": "neutral",
                "importance": 0.7,
            }
        )

    def _mem_signal(self, system: str) -> str:
        return "has-memory" if "no memories yet" not in system else "empty"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        # Distinct-ish vectors by text length so cosine isn't always 1.0.
        return [[1.0, len(t) % 7 / 7.0, (len(t) // 7) % 5 / 5.0] for t in texts]

    async def aclose(self) -> None:
        pass
