"""The engram — Kioku's unit of memory.

Every (user_message, assistant_reply) exchange is decomposed into one
engram: not the transcript, the *understanding* of it. The full engram is
persisted as a blob on the virtual disk; its keywords/entities/topics are
indexed as shift+mask cells in the vRAM planet (layout in
docs/MEMORY_MODEL.md).
"""

from __future__ import annotations

import json
import math
import time
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from ulid import ULID

MemoryClass = Literal["preference", "semantic", "episodic", "smalltalk"]

# Terms shorter than this are noise, not index keys.
MIN_TERM_LEN = 2
MAX_TERMS_PER_ENGRAM = 24


def new_ulid() -> str:
    return str(ULID())


def normalize_term(term: str) -> str:
    """One canonical spelling per concept: NFKC, casefold, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", term).casefold().split())


class PreferencesDelta(BaseModel):
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.likes or self.dislikes or self.facts)


class EngramLinks(BaseModel):
    session_prev: str | None = None
    topics: list[str] = Field(default_factory=list)


class Engram(BaseModel):
    """Schema from the build spec (§3), plus the bookkeeping forgetting needs."""

    engram_id: str = Field(default_factory=new_ulid)
    tenant: str
    user_id: str
    session_id: str
    ts: float = Field(default_factory=time.time)

    message: str
    reply: str

    meaning: str = ""
    intent: str = ""
    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    preferences_delta: PreferencesDelta = Field(default_factory=PreferencesDelta)
    emotional_tone: str = ""
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    definitions: dict[str, str] = Field(default_factory=dict)
    embedding: list[float] = Field(default_factory=list)
    links: EngramLinks = Field(default_factory=EngramLinks)

    # Bookkeeping for retrieval reinforcement and timely forgetting (§4, §5).
    memory_class: MemoryClass = "episodic"
    access_count: int = 0
    tombstoned: bool = False
    superseded_by: str | None = None

    @field_validator("keywords", "entities", mode="after")
    @classmethod
    def _normalize_terms(cls, terms: list[str]) -> list[str]:
        seen: list[str] = []
        for t in terms:
            n = normalize_term(t)
            if len(n) >= MIN_TERM_LEN and n not in seen:
                seen.append(n)
        return seen[:MAX_TERMS_PER_ENGRAM]

    def index_terms(self) -> list[str]:
        """The terms this engram is findable by — keywords, entities, topics."""
        out: list[str] = []
        for t in (*self.keywords, *self.entities, *self.links.topics):
            n = normalize_term(t)
            if len(n) >= MIN_TERM_LEN and n not in out:
                out.append(n)
        return out[:MAX_TERMS_PER_ENGRAM]

    # -- (de)serialization: the blob format on the virtual disk -----------

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "Engram":
        return cls.model_validate(json.loads(raw.decode("utf-8")))

    # -- forgetting math (§5) — used by forget.py, defined with the schema --

    def age_days(self, now: float | None = None) -> float:
        return max(0.0, ((now or time.time()) - self.ts) / 86400.0)

    def retention(self, lambda_per_class: dict[str, float], now: float | None = None) -> float:
        """retention = importance · e^(−λ·age_days) · log(1 + access_count).

        The log term is floored at 1 (log(1+0)=0 would erase untouched but
        important memories instantly — reinforcement amplifies, never gates).
        """
        lam = lambda_per_class.get(self.memory_class, 0.08)
        reinforcement = max(1.0, math.log(1 + self.access_count + 1))
        return self.importance * math.exp(-lam * self.age_days(now)) * reinforcement


def classify(engram: Engram) -> MemoryClass:
    """Memory class drives the decay rate λ. Preferences outlive everything."""
    if not engram.preferences_delta.is_empty():
        return "preference"
    if engram.importance < 0.2:
        return "smalltalk"
    return "episodic"
