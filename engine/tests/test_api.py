"""HTTP surface — routes, validation, isolation, gauges — over a fake brain."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from engine.config import settings
from engine.main import create_app
from engine.store import PyStore
from engine.tenants import KiokuEngine, TenantRegistry
from engine.tests.fake_qwen import SmartFakeQwen


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KIOKU_RATELIMIT", "off")  # no limiter bleed across tests
    store = PyStore(tmp_path / "pystore", ceiling_bytes=512 << 20)
    engine = KiokuEngine(TenantRegistry(store, SmartFakeQwen(), settings(), message_cap=50))
    engine._store = store
    with TestClient(create_app(engine)) as c:
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["backend"] == "pystore"


def test_chat_returns_dual_panes_and_address(client: TestClient) -> None:
    r = client.post("/api/chat", json={"message": "I love coffee"})
    assert r.status_code == 200
    body = r.json()
    assert body["kioku_reply"].startswith("MEM[")
    assert body["raw_reply"].startswith("RAW")
    assert "planet" in body["address"] and "cell 0x" in body["address"]
    assert body["token"] == "kioku"
    assert body["pack"]["budget"] == 1200


def test_chat_validates_empty_message(client: TestClient) -> None:
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_new_mind_is_empty_and_isolated(client: TestClient) -> None:
    client.post("/api/chat", json={"message": "I love coffee"})  # feeds shared
    token = client.post("/api/mind/new").json()["token"]
    assert token != "kioku"
    mem = client.get(f"/api/memory?token={token}").json()
    assert mem["total"] == 0  # newborn remembers nothing


def test_memory_browser_pagination(client: TestClient) -> None:
    for i in range(5):
        client.post("/api/chat", json={"message": f"fact number {i}", "send_to_both": False})
    page = client.get("/api/memory?limit=2&offset=0").json()
    assert page["total"] == 5
    assert len(page["engrams"]) == 2
    assert page["limit"] == 2


def test_memory_detail_roundtrips_from_disk(client: TestClient) -> None:
    chat = client.post("/api/chat", json={"message": "remember Kyoto", "send_to_both": False}).json()
    eid = chat["engram_id"]
    detail = client.get(f"/api/memory/kioku/{eid}").json()
    assert detail["engram_id"] == eid
    assert "Kyoto" in detail["message"]
    assert client.get("/api/memory/kioku/nonexistent").status_code == 404


def test_lexicon_and_forgetting_and_stats(client: TestClient) -> None:
    client.post("/api/chat", json={"message": "tell me about espresso", "send_to_both": False})

    lex = client.get("/api/lexicon").json()
    assert "definitions" in lex and lex["count"] >= 0

    forget = client.get("/api/forgetting").json()
    assert "retention" in forget and "last_consolidation" in forget

    stats = client.get("/api/stats").json()
    gauge = stats["gauge"]
    assert gauge["vram_virtual"] == 1 << 40
    assert gauge["disk_virtual"] == 1 << 42
    assert "small outside, huge inside" in gauge["headline"]
    assert gauge["backend"] == "pystore"


def test_consolidate_endpoint(client: TestClient) -> None:
    r = client.post("/api/consolidate")
    assert r.status_code == 200
    assert "did_anything" in r.json()


def test_stream_replays_recent_events(client: TestClient) -> None:
    client.post("/api/chat", json={"message": "I love coffee", "send_to_both": False})
    # replay_then_close gives a finite SSE stream — the live path is the same
    # generator without the early return.
    r = client.get("/api/stream/kioku?replay_then_close=true")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    data_lines = [ln for ln in r.text.splitlines() if ln.startswith("data:")]
    assert data_lines and any("stage" in ln for ln in data_lines)
