"""Engram schema, normalization, serialization, and retention math."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from engine.engram import Engram, PreferencesDelta, classify, new_ulid, normalize_term


def make_engram(**overrides) -> Engram:
    base = dict(
        tenant="kioku",
        user_id="u1",
        session_id="s1",
        message="I love hanami season in Kyoto",
        reply="Cherry blossom viewing is wonderful there.",
        meaning="The user loves hanami season in Kyoto.",
        intent="share a preference",
        keywords=["Hanami", "kyoto", "hanami", "x"],
        entities=["Kyoto"],
        importance=0.9,
    )
    base.update(overrides)
    return Engram(**base)


def test_ulids_are_unique_and_sortable() -> None:
    a, b = new_ulid(), new_ulid()
    assert a != b and len(a) == 26
    assert a < b  # monotonic in time


def test_terms_are_normalized_and_deduped() -> None:
    e = make_engram()
    # "Hanami" and "hanami" collapse; "x" is below MIN_TERM_LEN.
    assert e.keywords == ["hanami", "kyoto"]
    assert e.index_terms() == ["hanami", "kyoto"]


def test_normalize_term_unicode() -> None:
    assert normalize_term("  Café  Au   Lait ") == "café au lait"
    assert normalize_term("記憶") == "記憶"


def test_importance_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        make_engram(importance=1.5)
    with pytest.raises(ValidationError):
        make_engram(importance=-0.1)


def test_bytes_roundtrip() -> None:
    e = make_engram()
    e.definitions["hanami"] = "Japanese flower viewing."
    e.embedding = [0.1, -0.2, 0.3]
    again = Engram.from_bytes(e.to_bytes())
    assert again == e


def test_classify() -> None:
    pref = make_engram(preferences_delta=PreferencesDelta(likes=["hanami"]))
    assert classify(pref) == "preference"
    chatter = make_engram(importance=0.1)
    assert classify(chatter) == "smalltalk"
    assert classify(make_engram(importance=0.5)) == "episodic"


def test_retention_decays_with_age_and_class() -> None:
    lambdas = {"preference": 0.005, "episodic": 0.08, "smalltalk": 0.5}
    now = 1_750_000_000.0
    week_ago = now - 7 * 86400

    pref = make_engram(ts=week_ago, importance=0.9, memory_class="preference")
    chat = make_engram(ts=week_ago, importance=0.9, memory_class="smalltalk")
    assert pref.retention(lambdas, now) > chat.retention(lambdas, now)

    fresh = make_engram(ts=now, importance=0.9, memory_class="episodic")
    assert math.isclose(fresh.retention(lambdas, now), 0.9, rel_tol=1e-6)

    # Reinforcement amplifies but never gates.
    touched = make_engram(ts=week_ago, importance=0.5, access_count=10)
    untouched = make_engram(ts=week_ago, importance=0.5, access_count=0)
    assert touched.retention(lambdas, now) > untouched.retention(lambdas, now) > 0.0
