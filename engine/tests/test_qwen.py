"""QwenClient against a mocked OpenAI-compatible endpoint (respx)."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from engine.config import LLMConfig
from engine.qwen import LLMError, QwenClient, strip_code_fences

BASE = "https://qwen.test/v1"

CONFIG = LLMConfig(
    base_url=BASE,
    api_key="sk-test-not-a-real-key",
    model="qwen-max",
    embed_model="text-embedding-v3",
    max_retries=3,
)


def chat_response(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def client() -> QwenClient:
    return QwenClient(CONFIG)


@respx.mock
async def test_chat_happy_path(client: QwenClient) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response("hello"))
    )
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "hello"
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen-max"
    assert route.calls[0].request.headers["authorization"] == f"Bearer {CONFIG.api_key}"


@respx.mock
async def test_retries_on_5xx_then_succeeds(client: QwenClient) -> None:
    route = respx.post(f"{BASE}/chat/completions")
    route.side_effect = [
        httpx.Response(500),
        httpx.Response(429),
        httpx.Response(200, json=chat_response("third time lucky")),
    ]
    out = await client.chat([{"role": "user", "content": "hi"}])
    assert out == "third time lucky"
    assert route.call_count == 3


@respx.mock
async def test_4xx_is_not_retried(client: QwenClient) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(401, text="bad key")
    )
    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert route.call_count == 1


@respx.mock
async def test_chat_json_repairs_once(client: QwenClient) -> None:
    route = respx.post(f"{BASE}/chat/completions")
    route.side_effect = [
        httpx.Response(200, json=chat_response("{broken")),
        httpx.Response(200, json=chat_response('{"a": 1}')),
    ]
    out = await client.chat_json([{"role": "user", "content": "decompose"}])
    assert out == {"a": 1}
    assert route.call_count == 2
    # The repair turn includes the parser error.
    repair = json.loads(route.calls[1].request.content)["messages"][-1]
    assert "not valid JSON" in repair["content"]


@respx.mock
async def test_chat_json_requests_json_mode(client: QwenClient) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_response('{"ok": true}'))
    )
    await client.chat_json([{"role": "user", "content": "x"}])
    sent = json.loads(route.calls[0].request.content)
    assert sent["response_format"] == {"type": "json_object"}


@respx.mock
async def test_embed_preserves_order(client: QwenClient) -> None:
    respx.post(f"{BASE}/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )
    )
    out = await client.embed(["first", "second"])
    assert out == [[1.0], [2.0]]


async def test_embed_empty_is_free(client: QwenClient) -> None:
    assert await client.embed([]) == []


def test_strip_code_fences() -> None:
    assert strip_code_fences('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_code_fences('{"a":1}') == '{"a":1}'
    assert strip_code_fences("```\n[]\n```") == "[]"
