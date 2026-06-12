"""Forgetting — decay reporting, consolidation, contradiction, compaction."""

from __future__ import annotations

import time

import pytest

from engine.config import settings
from engine.engram import Engram, PreferencesDelta
from engine.forget import ForgetManager
from engine.retrieve import MemoryIndex
from engine.store import PyStore
from engine.tests.fake_qwen import FakeQwen

DAY = 86400.0


@pytest.fixture
def index(tmp_path):
    store = PyStore(tmp_path / "pystore", ceiling_bytes=256 << 20)
    space = store.open_space(8 << 20)
    yield MemoryIndex(store, space, settings())
    store.close()


def episodic(meaning: str, *, keywords, importance=0.25, age_days=30.0, **kw) -> Engram:
    e = Engram(
        tenant="kioku",
        user_id="u1",
        session_id="s1",
        message=meaning,
        reply="ok",
        meaning=meaning,
        keywords=keywords,
        entities=[],
        importance=importance,
        ts=time.time() - age_days * DAY,
        **kw,
    )
    return e


def test_retention_report_orders_weakest_first(index: MemoryIndex) -> None:
    strong = episodic("durable", keywords=["a"], importance=0.9, age_days=0.0)
    weak = episodic("fading", keywords=["b"], importance=0.2, age_days=60.0)
    index.commit(strong)
    index.commit(weak)
    fm = ForgetManager(index, FakeQwen())
    rows = fm.retention_report()
    assert rows[0].engram_id == weak.engram_id
    assert rows[0].retention < rows[-1].retention


async def test_consolidation_summarizes_tombstones_and_reclaims(index: MemoryIndex) -> None:
    # Three aging, low-retention memories about the same topic.
    for i in range(3):
        index.commit(episodic(f"Looked at gardens in Kyoto, day {i}.", keywords=["kyoto", "gardens"]))
    # One unrelated durable memory must survive untouched.
    keeper = Engram(
        tenant="kioku", user_id="u1", session_id="s1",
        message="My name is Aiko", reply="Nice to meet you",
        meaning="The user's name is Aiko.", keywords=["name"], entities=[],
        importance=0.95, preferences_delta=PreferencesDelta(facts=["name is Aiko"]),
    )
    index.commit(keeper)

    qwen = FakeQwen(chat_responses=["The user spent time visiting gardens in Kyoto."])
    fm = ForgetManager(index, qwen)
    diff = await fm.consolidate()

    assert diff.did_anything
    assert len(diff.tombstoned_ids) == 3
    assert len(diff.created_ids) == 1
    assert diff.summaries == ["The user spent time visiting gardens in Kyoto."]
    assert diff.reclaimed_bytes > 0

    # The summary is a live semantic memory; originals are gone from recall.
    survivors = index.live_engrams()
    classes = {e.memory_class for e in survivors}
    assert "semantic" in classes
    assert keeper.engram_id in {e.engram_id for e in survivors}
    hits = index.recall(["kyoto"], query_embedding=[], session_id=None)
    assert all(not h.engram.tombstoned for h in hits)
    assert any("gardens in Kyoto" in h.engram.meaning for h in hits)


async def test_consolidation_noop_when_everything_is_fresh(index: MemoryIndex) -> None:
    index.commit(episodic("recent thought", keywords=["x"], age_days=0.0, importance=0.5))
    fm = ForgetManager(index, FakeQwen())
    diff = await fm.consolidate()
    assert not diff.did_anything
    assert diff.reclaimed_bytes == 0


async def test_singletons_are_not_consolidated(index: MemoryIndex) -> None:
    # Two aging memories but on different topics → no cluster reaches size 2.
    index.commit(episodic("about cats", keywords=["cats"]))
    index.commit(episodic("about trains", keywords=["trains"]))
    fm = ForgetManager(index, FakeQwen(chat_responses=["unused"]))
    diff = await fm.consolidate()
    assert not diff.did_anything


def test_contradiction_supersedes_old_preference(index: MemoryIndex) -> None:
    old = Engram(
        tenant="kioku", user_id="u1", session_id="s1",
        message="I love coffee", reply="ok", meaning="The user likes coffee.",
        keywords=["coffee"], entities=[], importance=0.8,
        preferences_delta=PreferencesDelta(likes=["coffee"]),
    )
    index.commit(old)
    new = Engram(
        tenant="kioku", user_id="u1", session_id="s2",
        message="Actually I hate coffee now", reply="noted",
        meaning="The user dislikes coffee.", keywords=["coffee"], entities=[],
        importance=0.8, preferences_delta=PreferencesDelta(dislikes=["coffee"]),
    )
    index.commit(new)

    fm = ForgetManager(index, FakeQwen())
    diff = fm.supersede_contradictions(new)
    assert old.engram_id in diff.superseded_ids
    assert old.tombstoned and old.superseded_by == new.engram_id
    assert "coffee" in diff.reason


def test_fact_overwrite_supersedes(index: MemoryIndex) -> None:
    old = Engram(
        tenant="kioku", user_id="u1", session_id="s1",
        message="I live in Kyoto", reply="ok", meaning="Lives in Kyoto.",
        keywords=["kyoto"], entities=[], importance=0.8,
        preferences_delta=PreferencesDelta(facts=["lives in Kyoto"]),
    )
    index.commit(old)
    new = Engram(
        tenant="kioku", user_id="u1", session_id="s2",
        message="I moved to Osaka", reply="ok", meaning="Lives in Osaka.",
        keywords=["osaka"], entities=[], importance=0.8,
        preferences_delta=PreferencesDelta(facts=["lives in Osaka now"]),
    )
    index.commit(new)
    fm = ForgetManager(index, FakeQwen())
    diff = fm.supersede_contradictions(new)
    assert old.engram_id in diff.superseded_ids


def test_unrelated_preferences_do_not_supersede(index: MemoryIndex) -> None:
    old = Engram(
        tenant="kioku", user_id="u1", session_id="s1", message="m", reply="r",
        meaning="likes tea", keywords=["tea"], entities=[], importance=0.8,
        preferences_delta=PreferencesDelta(likes=["tea"]),
    )
    index.commit(old)
    new = Engram(
        tenant="kioku", user_id="u1", session_id="s2", message="m", reply="r",
        meaning="likes hiking", keywords=["hiking"], entities=[], importance=0.8,
        preferences_delta=PreferencesDelta(likes=["hiking"]),
    )
    index.commit(new)
    fm = ForgetManager(index, FakeQwen())
    assert fm.supersede_contradictions(new).superseded_ids == []
    assert not old.tombstoned


def test_compact_rewrites_live_and_reclaims(index: MemoryIndex) -> None:
    survivors = [episodic(f"keep {i}", keywords=[f"k{i}"], importance=0.6, age_days=0.0) for i in range(3)]
    doomed = [episodic(f"drop {i}", keywords=[f"d{i}"], importance=0.6, age_days=0.0) for i in range(3)]
    for e in survivors + doomed:
        index.commit(e)
    for e in doomed:
        e.tombstoned = True

    old_space = index.space
    freed = index.compact()
    assert freed > 0
    assert index.space != old_space  # moved to a fresh planet
    # Live ones still recall from the new planet; their blobs round-trip.
    hits = index.recall(["k0"], query_embedding=[], session_id=None)
    assert hits and hits[0].engram.meaning == "keep 0"
    assert index.get_blob_engram(survivors[0].engram_id).meaning == "keep 0"
    # Tombstoned ones are gone entirely.
    assert index.recall(["d0"], query_embedding=[], session_id=None) == []
    assert len(index.live_engrams()) == 3
