"""The understanding pipeline, with a scripted brain."""

from __future__ import annotations

import json

from engine.decompose import Capture, PipelineEvent, decompose_exchange
from engine.qwen import LLMError
from engine.tests.fake_qwen import FakeQwen

GOOD_DECOMPOSITION = json.dumps(
    {
        "meaning": "The user is planning a flower-viewing trip to Kyoto in June.",
        "intent": "plan a trip",
        "keywords": ["hanami", "kyoto", "trip planning"],
        "entities": ["Kyoto"],
        "preferences_delta": {"likes": ["hanami"], "dislikes": [], "facts": ["planning a June trip to Kyoto"]},
        "emotional_tone": "excited",
        "importance": 0.8,
    }
)

CAPTURE = Capture(
    tenant="kioku",
    user_id="u1",
    session_id="s1",
    message="I'm planning a hanami trip to Kyoto this June!",
    reply="Wonderful — early June is past peak bloom, but the gardens are lovely.",
    session_prev="01PREV",
)


async def test_decompose_builds_full_engram() -> None:
    qwen = FakeQwen(chat_responses=[GOOD_DECOMPOSITION], embeddings=[[0.5, 0.5]])
    engram = await decompose_exchange(qwen, CAPTURE)

    assert engram.meaning.startswith("The user is planning")
    assert engram.intent == "plan a trip"
    assert engram.keywords == ["hanami", "kyoto", "trip planning"]
    assert engram.entities == ["kyoto"]
    assert engram.preferences_delta.likes == ["hanami"]
    assert engram.importance == 0.8
    assert engram.memory_class == "preference"  # prefs delta → preference class
    assert engram.embedding == [0.5, 0.5]
    assert engram.links.session_prev == "01PREV"
    assert engram.links.topics == ["hanami", "kyoto", "trip planning"]
    assert engram.tenant == "kioku" and engram.session_id == "s1"
    # Embedding input is meaning + keywords.
    assert "hanami" in qwen.embed_calls[0][0]


async def test_events_fire_in_pipeline_order() -> None:
    events: list[PipelineEvent] = []

    async def sink(e: PipelineEvent) -> None:
        events.append(e)

    qwen = FakeQwen(chat_responses=[GOOD_DECOMPOSITION])
    engram = await decompose_exchange(qwen, CAPTURE, emit=sink)
    assert [e.stage for e in events] == ["captured", "decomposed", "embedded"]
    assert events[1].detail["importance"] == 0.8
    assert events[1].engram_id == engram.engram_id
    assert events[2].detail["dims"] == 3


async def test_fenced_and_then_repaired_json_is_tolerated() -> None:
    fenced = "```json\n" + GOOD_DECOMPOSITION + "\n```"
    qwen = FakeQwen(chat_responses=[fenced])
    engram = await decompose_exchange(qwen, CAPTURE)
    assert engram.importance == 0.8

    qwen = FakeQwen(chat_responses=["{not json", GOOD_DECOMPOSITION])
    engram = await decompose_exchange(qwen, CAPTURE)
    assert engram.keywords == ["hanami", "kyoto", "trip planning"]
    assert len(qwen.chat_calls) == 2  # one repair round-trip


async def test_partial_decomposition_still_becomes_a_memory() -> None:
    sloppy = json.dumps({"meaning": "Something was said.", "importance": "not-a-number"})
    qwen = FakeQwen(chat_responses=[sloppy])
    engram = await decompose_exchange(qwen, CAPTURE)
    assert engram.meaning == "Something was said."
    assert engram.importance == 0.0
    assert engram.keywords == []
    assert engram.memory_class == "smalltalk"


async def test_embedding_failure_keeps_the_memory() -> None:
    qwen = FakeQwen(chat_responses=[GOOD_DECOMPOSITION], embeddings=LLMError("embeddings down"))
    engram = await decompose_exchange(qwen, CAPTURE)
    assert engram.embedding == []
    assert engram.meaning  # the memory survived


async def test_event_sink_failure_never_breaks_the_pipeline() -> None:
    async def broken_sink(_: PipelineEvent) -> None:
        raise RuntimeError("inspector exploded")

    qwen = FakeQwen(chat_responses=[GOOD_DECOMPOSITION])
    engram = await decompose_exchange(qwen, CAPTURE, emit=broken_sink)
    assert engram.meaning
