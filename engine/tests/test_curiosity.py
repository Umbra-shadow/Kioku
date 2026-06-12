"""The curiosity loop: look up the unknown, respect the budget, never block."""

from __future__ import annotations

from engine.curiosity import curiosity_pass
from engine.decompose import PipelineEvent
from engine.qwen import LLMError
from engine.tests.fake_qwen import FakeQwen
from engine.tests.test_engram import make_engram


async def test_unknown_terms_are_researched_and_locked_in() -> None:
    engram = make_engram(keywords=["hanami", "kyoto"], entities=[])
    qwen = FakeQwen(chat_responses=["Flower viewing, a Japanese spring tradition.", "A city in Japan the user plans to visit."])
    learned = await curiosity_pass(qwen, engram, is_known=lambda t: False)
    assert set(learned) == {"hanami", "kyoto"}
    assert engram.definitions["hanami"].startswith("Flower viewing")
    # Context travels with the lookup.
    assert "hanami" in qwen.chat_calls[0][1]["content"]


async def test_known_terms_are_skipped() -> None:
    engram = make_engram(keywords=["hanami", "kyoto"], entities=[])
    qwen = FakeQwen(chat_responses=["def"])
    learned = await curiosity_pass(qwen, engram, is_known=lambda t: t == "hanami")
    assert set(learned) == {"kyoto"}
    assert len(qwen.chat_calls) == 1


async def test_budget_cap_is_respected() -> None:
    engram = make_engram(keywords=["aaa", "bbb", "ccc", "ddd", "eee"], entities=[])
    qwen = FakeQwen(chat_responses=["1", "2"])
    learned = await curiosity_pass(qwen, engram, is_known=lambda t: False, max_lookups=2)
    assert len(learned) == 2
    assert len(qwen.chat_calls) == 2


async def test_lookup_failure_is_skipped_not_raised() -> None:
    engram = make_engram(keywords=["alpha", "beta"], entities=[])
    qwen = FakeQwen(chat_responses=[LLMError("brain offline"), "Beta is the second letter."])
    learned = await curiosity_pass(qwen, engram, is_known=lambda t: False)
    assert learned == {"beta": "Beta is the second letter."}


async def test_curious_events_emitted_per_term() -> None:
    events: list[PipelineEvent] = []

    async def sink(e: PipelineEvent) -> None:
        events.append(e)

    engram = make_engram(keywords=["hanami"], entities=[])
    qwen = FakeQwen(chat_responses=["Flower viewing."])
    await curiosity_pass(qwen, engram, is_known=lambda t: False, emit=sink)
    assert [e.stage for e in events] == ["curious"]
    assert events[0].detail == {"term": "hanami"}
