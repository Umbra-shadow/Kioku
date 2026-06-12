"""Qwen Cloud client — chat + embeddings over the OpenAI-compatible endpoint.

All LLM calls in Kioku go through this module. Async httpx with timeouts,
tenacity retries on transient failures, structured logging with no message
bodies (and never a key) at INFO.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from engine.config import LLMConfig, settings

log = logging.getLogger("kioku.qwen")


class LLMError(RuntimeError):
    """The brain is unreachable or answered nonsense. Never a crash upstream."""


class _RetryableHTTP(LLMError):
    """429 or 5xx — worth another attempt."""


def _is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException, _RetryableHTTP))


def strip_code_fences(text: str) -> str:
    """Models sometimes wrap JSON in ```json fences despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class QwenClient:
    """One brain, shared. Chat and embeddings against Qwen Cloud / Model Studio."""

    def __init__(self, config: LLMConfig | None = None, client: httpx.AsyncClient | None = None) -> None:
        self.config = config or settings().llm
        self._client = client or httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=httpx.Timeout(self.config.timeout_s, connect=10.0),
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential_jitter(initial=0.5, max=8.0),
            retry=retry_if_exception(_is_retryable),
            reraise=True,
        ):
            with attempt:
                response = await self._client.post(path, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise _RetryableHTTP(f"{path}: HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise LLMError(f"{path}: HTTP {response.status_code}: {response.text[:500]}")
                return response.json()
        raise LLMError("unreachable")  # pragma: no cover — reraise above

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        log.info(
            "chat call provider=%s model=%s messages=%d json_mode=%s",
            self.config.provider, self.config.model, len(messages), json_mode,
        )
        data = await self._post("/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"malformed chat response: {e}") from e

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Strict-JSON chat: one round trip, one repair attempt on bad JSON."""
        text = await self.chat(
            messages, json_mode=True, temperature=temperature, max_tokens=max_tokens
        )
        try:
            parsed = json.loads(strip_code_fences(text))
            if isinstance(parsed, dict):
                return parsed
            raise ValueError("top level is not an object")
        except (json.JSONDecodeError, ValueError) as first_error:
            log.warning("chat_json got invalid JSON, asking for a repair: %s", first_error)
            repair = messages + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": "That was not valid JSON. Respond again with ONLY the "
                    f"corrected JSON object. Parser said: {first_error}",
                },
            ]
            text = await self.chat(
                repair, json_mode=True, temperature=0.0, max_tokens=max_tokens
            )
            try:
                parsed = json.loads(strip_code_fences(text))
            except json.JSONDecodeError as e:
                raise LLMError(f"model cannot produce valid JSON: {e}") from e
            if not isinstance(parsed, dict):
                raise LLMError("model cannot produce a JSON object")
            return parsed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        log.info(
            "embed call provider=%s model=%s texts=%d",
            self.config.provider, self.config.embed_model, len(texts),
        )
        data = await self._post(
            "/embeddings", {"model": self.config.embed_model, "input": texts}
        )
        try:
            items = sorted(data["data"], key=lambda d: d["index"])
            return [item["embedding"] for item in items]
        except (KeyError, TypeError) as e:
            raise LLMError(f"malformed embeddings response: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()
