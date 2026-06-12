"""The understanding pipeline — Kioku's heart.

capture → decompose (one structured Qwen call) → embed → [curiosity, async]
→ commit. Every stage emits a pipeline event so the inspector can show a
memory forming live.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from engine.engram import Engram, PreferencesDelta, classify
from engine.qwen import LLMError, QwenClient

log = logging.getLogger("kioku.decompose")

DECOMPOSE_SYSTEM = """You are the understanding stage of Kioku, a memory engine.
You receive one exchange between a user and an assistant. Decompose it into
structured memory. Respond with ONLY a JSON object, no prose, exactly these keys:

{
  "meaning": "one sentence: what was actually said in this exchange",
  "intent": "what the user is trying to achieve",
  "keywords": ["3-8 lowercase topical words or short phrases"],
  "entities": ["proper nouns and specific things mentioned: people, places, products, terms"],
  "preferences_delta": {
    "likes": ["new durable likes the user revealed, else empty"],
    "dislikes": ["new durable dislikes, else empty"],
    "facts": ["new durable personal facts (name, job, city, plans), else empty"]
  },
  "emotional_tone": "one or two words",
  "importance": 0.0
}

Importance rubric (a number from 0 to 1):
- 0.9-1.0  durable personal facts and preferences (name, allergies, loves/hates)
- 0.6-0.8  plans, goals, decisions, commitments
- 0.3-0.5  contextual information that may matter later
- 0.0-0.2  small talk, acknowledgements, filler

Write keywords/entities in the language the user used. Be terse and factual."""


@dataclass(frozen=True, slots=True)
class Capture:
    """Stage 3.1 — the raw exchange plus its identifiers (ULIDs upstream)."""

    tenant: str
    user_id: str
    session_id: str
    message: str
    reply: str
    session_prev: str | None = None
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """One inspector chip: captured → decomposed → embedded → curious(term)
    → committed @planet/segment/cell."""

    stage: str
    engram_id: str
    detail: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


EventSink = Callable[[PipelineEvent], Awaitable[None]]


async def _emit(emit: EventSink | None, event: PipelineEvent) -> None:
    if emit is None:
        return
    try:
        await emit(event)
    except Exception:  # noqa: BLE001 — the inspector must never break the pipeline
        log.exception("event sink failed for stage %s", event.stage)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _engram_from_raw(capture: Capture, raw: dict[str, Any]) -> Engram:
    """Validate the model's decomposition into an Engram; tolerate partial
    output — a sloppy field must not lose the memory."""
    prefs_raw = raw.get("preferences_delta") or {}
    if not isinstance(prefs_raw, dict):
        prefs_raw = {}
    try:
        importance = min(1.0, max(0.0, float(raw.get("importance", 0.0))))
    except (TypeError, ValueError):
        importance = 0.0
    engram = Engram(
        tenant=capture.tenant,
        user_id=capture.user_id,
        session_id=capture.session_id,
        ts=capture.ts,
        message=capture.message,
        reply=capture.reply,
        meaning=str(raw.get("meaning", "")).strip(),
        intent=str(raw.get("intent", "")).strip(),
        keywords=_as_str_list(raw.get("keywords")),
        entities=_as_str_list(raw.get("entities")),
        preferences_delta=PreferencesDelta(
            likes=_as_str_list(prefs_raw.get("likes")),
            dislikes=_as_str_list(prefs_raw.get("dislikes")),
            facts=_as_str_list(prefs_raw.get("facts")),
        ),
        emotional_tone=str(raw.get("emotional_tone", "")).strip(),
        importance=importance,
    )
    engram.links.session_prev = capture.session_prev
    engram.links.topics = engram.keywords[:4]
    engram.memory_class = classify(engram)
    return engram


async def decompose_exchange(
    qwen: QwenClient,
    capture: Capture,
    emit: EventSink | None = None,
) -> Engram:
    """Stages 3.1-3.3: capture → decompose → embed. One structured Qwen
    call plus one embedding call. Curiosity (3.4) and commit (3.5) are the
    caller's next moves — curiosity runs async and must not block."""
    placeholder_id = "pending"
    await _emit(emit, PipelineEvent("captured", placeholder_id, {"chars": len(capture.message)}))

    raw = await qwen.chat_json(
        [
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {
                "role": "user",
                "content": f"USER MESSAGE:\n{capture.message}\n\nASSISTANT REPLY:\n{capture.reply}",
            },
        ]
    )
    engram = _engram_from_raw(capture, raw)
    await _emit(
        emit,
        PipelineEvent(
            "decomposed",
            engram.engram_id,
            {
                "meaning": engram.meaning,
                "intent": engram.intent,
                "importance": engram.importance,
                "memory_class": engram.memory_class,
                "keywords": engram.keywords,
                "entities": engram.entities,
            },
        ),
    )

    embed_text = engram.meaning + " | " + " ".join(engram.keywords)
    try:
        vectors = await qwen.embed([embed_text])
        engram.embedding = vectors[0] if vectors else []
    except LLMError as e:
        # A memory without a vector still has its keyword index — keep it.
        log.warning("embedding failed for %s: %s", engram.engram_id, e)
        engram.embedding = []
    await _emit(
        emit,
        PipelineEvent("embedded", engram.engram_id, {"dims": len(engram.embedding)}),
    )
    return engram
