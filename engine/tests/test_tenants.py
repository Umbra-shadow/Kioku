"""The turn engine and tenancy: shared mind, newborns, isolation, the loop."""

from __future__ import annotations

import pytest

from engine.config import settings
from engine.store import PyStore
from engine.tenants import SHARED_TENANT, KiokuEngine, MindFull, TenantRegistry
from engine.tests.fake_qwen import SmartFakeQwen


@pytest.fixture
def env(tmp_path):
    store = PyStore(tmp_path / "pystore", ceiling_bytes=512 << 20)
    qwen = SmartFakeQwen()
    registry = TenantRegistry(store, qwen, settings(), message_cap=5)
    engine = KiokuEngine(registry)
    yield engine, registry, qwen
    store.close()


async def test_shared_mind_exists_at_birth(env) -> None:
    _, registry, _ = env
    mind = registry.resolve(None)
    assert mind.tenant_id == SHARED_TENANT
    assert registry.resolve("garbage-token").tenant_id == SHARED_TENANT


async def test_turn_returns_dual_answers_and_commits(env) -> None:
    engine, registry, _ = env
    mind = registry.resolve(None)
    result = await engine.turn(mind, "I love coffee", send_to_both=True)
    assert result.kioku_reply.startswith("MEM[")
    assert result.raw_reply.startswith("RAW")
    assert "planet" in result.receipt.address
    assert "coffee" in result.engram.meaning
    assert len(mind.index.live_engrams()) == 1
    await engine.drain_background()


async def test_send_to_both_false_skips_raw(env) -> None:
    engine, registry, _ = env
    mind = registry.resolve(None)
    result = await engine.turn(mind, "hello there", send_to_both=False)
    assert result.raw_reply is None
    await engine.drain_background()


async def test_cross_session_recall_within_a_mind(env) -> None:
    engine, registry, qwen = env
    mind = registry.resolve(None)
    await engine.turn(mind, "I love coffee", session_id="morning")
    await engine.drain_background()
    # New session, same mind: the coffee memory rides into the system prompt.
    await engine.turn(mind, "what do I like to drink?", session_id="evening")
    mem_systems = [c[0]["content"] for c in qwen.chat_calls if "long-term memory" in c[0]["content"]]
    assert any("coffee" in s for s in mem_systems[1:])
    await engine.drain_background()


async def test_newborn_is_isolated_from_shared(env) -> None:
    engine, registry, _ = env
    shared = registry.resolve(None)
    await engine.turn(shared, "I love coffee", session_id="s")
    await engine.drain_background()

    newborn = await registry.new_mind()
    assert newborn.tenant_id != SHARED_TENANT
    assert newborn.space != shared.space
    assert newborn.index.live_engrams() == []
    assert newborn.index.recall(["coffee"], [], None) == []


async def test_message_cap_is_enforced(env) -> None:
    engine, registry, _ = env
    mind = registry.resolve(None)
    for i in range(5):
        await engine.turn(mind, f"message number {i}", send_to_both=False)
    with pytest.raises(MindFull):
        await engine.turn(mind, "one too many", send_to_both=False)
    await engine.drain_background()


async def test_pipeline_events_are_emitted(env) -> None:
    engine, registry, _ = env
    mind = registry.resolve(None)
    queue = mind.subscribe()
    await engine.turn(mind, "I love coffee", send_to_both=False)
    await engine.drain_background()
    stages = []
    while not queue.empty():
        stages.append(queue.get_nowait()["stage"])
    assert {"captured", "decomposed", "committed"} <= set(stages)


async def test_curiosity_runs_in_background_and_fills_lexicon(env) -> None:
    engine, registry, _ = env
    mind = registry.resolve(None)
    await engine.turn(mind, "tell me about espresso machines", send_to_both=False)
    await engine.drain_background()
    assert len(mind.index.lexicon) > 0
