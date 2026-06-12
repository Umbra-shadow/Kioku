"""Retrieval, scoring, and the memory-pack budget — over a real PyStore."""

from __future__ import annotations

import time

import pytest

from engine.config import settings
from engine.engram import Engram, PreferencesDelta
from engine.retrieve import MemoryIndex, cosine, estimate_tokens
from engine.store import PyStore, keyword_cell


@pytest.fixture
def index(tmp_path):
    store = PyStore(tmp_path / "pystore", ceiling_bytes=256 << 20)
    space = store.open_space(8 << 20)
    yield MemoryIndex(store, space, settings())
    store.close()


def make_engram(**kw) -> Engram:
    base = dict(
        tenant="kioku",
        user_id="u1",
        session_id="s1",
        message="msg",
        reply="reply",
        meaning="a memory",
        keywords=["alpha", "beta"],
        entities=[],
        importance=0.5,
    )
    base.update(kw)
    return Engram(**base)


def test_commit_returns_physical_address(index: MemoryIndex) -> None:
    e = make_engram(keywords=["hanami"], importance=0.8)
    receipt = index.commit(e)
    assert receipt.engram_id == e.engram_id
    assert receipt.cell == keyword_cell("hanami")
    assert "planet" in receipt.address and "cell 0x" in receipt.address
    # The blob really landed on the virtual disk.
    assert index.get_blob_engram(e.engram_id).meaning == "a memory"


def test_is_known_is_a_shift_mask_lookup(index: MemoryIndex) -> None:
    assert not index.is_known("hanami")
    index.commit(make_engram(keywords=["hanami"]))
    assert index.is_known("hanami")
    assert not index.is_known("sakura")
    # Confirm it really reads the cell at hash64(term) & mask.
    assert index.store.get_cell(index.space, keyword_cell("hanami")) is not None


def test_keyword_recall_finds_committed_memory(index: MemoryIndex) -> None:
    e = make_engram(keywords=["kyoto", "hanami"], meaning="trip to Kyoto")
    index.commit(e)
    hits = index.recall(["kyoto"], query_embedding=[], session_id=None)
    assert [h.engram.engram_id for h in hits] == [e.engram_id]
    assert hits[0].hit_kind == "keyword"


def test_unknown_query_term_returns_nothing(index: MemoryIndex) -> None:
    index.commit(make_engram(keywords=["alpha"]))
    assert index.recall(["nonexistent"], query_embedding=[], session_id=None) == []


def test_vector_recall_and_ranking(index: MemoryIndex) -> None:
    near = make_engram(keywords=["x"], meaning="near", importance=0.4)
    near.embedding = [1.0, 0.0, 0.0]
    far = make_engram(keywords=["y"], meaning="far", importance=0.4)
    far.embedding = [0.0, 1.0, 0.0]
    index.commit(near)
    index.commit(far)
    hits = index.recall([], query_embedding=[0.9, 0.1, 0.0], session_id=None)
    assert hits[0].engram.engram_id == near.engram_id
    assert hits[0].similarity > hits[1].similarity


def test_importance_and_recency_factor_into_score(index: MemoryIndex) -> None:
    now = time.time()
    important = make_engram(keywords=["shared"], meaning="important", importance=0.9, ts=now)
    trivial = make_engram(keywords=["shared"], meaning="trivial", importance=0.1, ts=now - 30 * 86400)
    index.commit(important)
    index.commit(trivial)
    hits = index.recall(["shared"], query_embedding=[], session_id=None)
    assert hits[0].engram.engram_id == important.engram_id


def test_session_recency_walk(index: MemoryIndex) -> None:
    e = make_engram(keywords=["nomatch"], session_id="s9", meaning="recent turn")
    index.commit(e)
    # No keyword/vector hit, but the session walk surfaces it.
    hits = index.recall(["totally_other"], query_embedding=[], session_id="s9")
    assert any(h.engram.engram_id == e.engram_id and h.hit_kind == "recency" for h in hits)


def test_tombstoned_memories_are_not_recalled(index: MemoryIndex) -> None:
    e = make_engram(keywords=["ghost"])
    index.commit(e)
    e.tombstoned = True
    assert index.recall(["ghost"], query_embedding=[], session_id=None) == []


def test_reinforce_increments_access(index: MemoryIndex) -> None:
    e = make_engram()
    index.commit(e)
    index.reinforce([e])
    index.reinforce([e])
    assert e.access_count == 2


def test_pack_respects_token_budget(index: MemoryIndex) -> None:
    for i in range(40):
        e = make_engram(
            keywords=["bulk"],
            meaning=f"Memory number {i} with a fairly long descriptive sentence to spend tokens.",
            importance=0.5,
        )
        index.commit(e)
    hits = index.recall(["bulk"], query_embedding=[], session_id=None, top_k=40)
    pack = index.build_pack(hits, token_budget=120)
    assert pack.tokens <= 120
    assert 0 < len(pack.hits) < 40  # budget forced a cut
    assert pack.budget == 120


def test_pack_surfaces_preferences_and_definitions(index: MemoryIndex) -> None:
    e = make_engram(
        keywords=["hanami"],
        meaning="The user loves hanami.",
        preferences_delta=PreferencesDelta(likes=["hanami"], facts=["lives in Kyoto"]),
        importance=0.9,
    )
    e.definitions["hanami"] = "Japanese flower viewing."
    index.commit(e)
    hits = index.recall(["hanami"], query_embedding=[], session_id=None)
    pack = index.build_pack(hits, token_budget=400)
    assert "Known about the user" in pack.text
    assert "lives in Kyoto" in pack.text
    assert "hanami" in pack.definitions
    assert "Japanese flower viewing" in pack.text


def test_pack_dedupes_identical_meanings(index: MemoryIndex) -> None:
    for _ in range(3):
        index.commit(make_engram(keywords=["dup"], meaning="exactly the same thought"))
    hits = index.recall(["dup"], query_embedding=[], session_id=None, top_k=10)
    pack = index.build_pack(hits, token_budget=400)
    assert pack.text.count("exactly the same thought") == 1


def test_cosine_and_token_helpers() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([], [1]) == 0.0
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 40) == 10
