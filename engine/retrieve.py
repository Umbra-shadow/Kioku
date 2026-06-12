"""Recall — fast, small, right (spec §4).

`MemoryIndex` is both sides of the same structure: the commit path that
writes an engram into the Cadran substrate (§3.5), and the recall path that
finds it again under a strict token budget (§4).

Addressing discipline (docs/MEMORY_MODEL.md):
- The full engram is a CRC-verified blob on the 4 TiB virtual disk.
- Each index term writes one vRAM cell at ``hash64(term) & PLANET_CELL_MASK``
  — so "have I seen this word?" and "which memories mention it?" are one
  shift+mask jump, never a search. The cell records (importance, postings
  count, latest blob block); the postings expansion (term → engram ids) and
  the vector pages are kept as an in-process accelerator, rebuildable from
  the disk blobs.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from engine.config import Settings, settings
from engine.engram import Engram, normalize_term
from engine.metrics import METRICS
from engine.store import (
    PLANET_CELL_MASK,
    SEGMENT_CELL_BITS,
    BlobHandle,
    Cell,
    MemoryStore,
    keyword_cell,
)

log = logging.getLogger("kioku.retrieve")

# Recency half-life for the retrieval score's γ term (distinct from the
# forgetting λ, which governs deletion). ~10-day half-life.
RECENCY_LAMBDA = 0.069
# access_frequency saturates here: enough reinforcement is enough.
ACCESS_SATURATION = 8.0
# Rough token estimate: ~4 chars/token. Good enough to hold a budget.
CHARS_PER_TOKEN = 4


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class CommitReceipt:
    """What the inspector shows when a memory lands: its physical address."""

    engram_id: str
    block: int
    planet: int
    segment: int
    cell: int

    @property
    def address(self) -> str:
        return f"planet {self.planet} · segment {self.segment} · cell 0x{self.cell:05X}"


@dataclass(frozen=True, slots=True)
class ScoredEngram:
    engram: Engram
    score: float
    similarity: float
    components: dict[str, float]
    hit_kind: str  # "keyword" | "vector" | "recency"


@dataclass(frozen=True, slots=True)
class MemoryPack:
    """Recalled context, assembled to fit a strict token budget (§4)."""

    text: str
    tokens: int
    budget: int
    hits: list[ScoredEngram] = field(default_factory=list)
    definitions: dict[str, str] = field(default_factory=dict)

    def hit_list(self) -> list[dict[str, object]]:
        return [
            {
                "engram_id": h.engram.engram_id,
                "meaning": h.engram.meaning,
                "score": round(h.score, 4),
                "similarity": round(h.similarity, 4),
                "hit_kind": h.hit_kind,
            }
            for h in self.hits
        ]


class MemoryIndex:
    """One tenant's living memory over one substrate space."""

    def __init__(self, store: MemoryStore, space: int, config: Settings | None = None) -> None:
        self.store = store
        self.space = space
        self.config = config or settings()
        # Derived accelerators (rebuildable from disk blobs).
        self._engrams: dict[str, Engram] = {}
        self._handles: dict[str, BlobHandle] = {}
        self._postings: dict[int, list[str]] = {}  # term cell -> engram ids
        self._session_order: dict[str, list[str]] = {}  # session -> engram ids, oldest first
        self._lexicon: dict[str, str] = {}  # term -> definition (curiosity, §3.4)

    # -- commit (§3.5) ----------------------------------------------------

    def commit(self, engram: Engram) -> CommitReceipt:
        """Blob → virtual disk; index cells → vRAM planet; postings updated."""
        with METRICS.timer("commit_ms"):
            handle = self.store.put_blob(self.space, engram.to_bytes())
            terms = engram.index_terms()
            cells = [
                Cell(
                    cell=keyword_cell(term),
                    act=float(engram.importance),
                    expert=min(len(self._postings.get(keyword_cell(term), [])) + 1, 0xFFFF_FFFF),
                    weight=handle.block,
                )
                for term in terms
            ]
            if cells:
                self.store.put_cells(self.space, cells)

            self._engrams[engram.engram_id] = engram
            self._handles[engram.engram_id] = handle
            for term in terms:
                self._postings.setdefault(keyword_cell(term), []).append(engram.engram_id)
            self._session_order.setdefault(engram.session_id, []).append(engram.engram_id)
            for term, definition in engram.definitions.items():
                self._lexicon.setdefault(normalize_term(term), definition)

        METRICS.incr("engrams_committed")
        anchor = keyword_cell(terms[0]) if terms else (handle.block & PLANET_CELL_MASK)
        return CommitReceipt(
            engram_id=engram.engram_id,
            block=handle.block,
            planet=self.space,
            segment=(anchor >> SEGMENT_CELL_BITS),
            cell=anchor,
        )

    def learn_terms(self, definitions: dict[str, str]) -> None:
        """Curiosity results join the global lexicon."""
        for term, definition in definitions.items():
            self._lexicon.setdefault(normalize_term(term), definition)

    # -- the genuine shift+mask lookup curiosity asks for (§3.4) ----------

    def is_known(self, term: str) -> bool:
        """One shift+mask jump: is this term's index cell populated, or is it
        already in the lexicon?"""
        if normalize_term(term) in self._lexicon:
            return True
        return self.store.get_cell(self.space, keyword_cell(term)) is not None

    # -- recall (§4) ------------------------------------------------------

    def _live(self, engram_id: str) -> Engram | None:
        e = self._engrams.get(engram_id)
        if e is None or e.tombstoned:
            return None
        return e

    def recall(
        self,
        query_terms: list[str],
        query_embedding: list[float],
        session_id: str | None = None,
        top_k: int = 12,
    ) -> list[ScoredEngram]:
        """Hybrid retrieval (a) keyword/entity exact cells, (b) cosine top-k
        over vector pages, (c) session-recency walk — then scored and ranked."""
        with METRICS.timer("retrieve_ms"):
            candidates: dict[str, str] = {}  # engram_id -> first hit_kind

            # (a) keyword/entity exact hits via shift+mask cells.
            for term in query_terms:
                cell = keyword_cell(term)
                if self.store.get_cell(self.space, cell) is None:
                    continue  # one shift+mask jump says "never seen"
                for eid in self._postings.get(cell, []):
                    candidates.setdefault(eid, "keyword")

            # (b) cosine over vector pages.
            if query_embedding:
                for eid, engram in self._engrams.items():
                    if engram.embedding:
                        candidates.setdefault(eid, "vector")

            # (c) session-recency walk: the last few turns of this session.
            if session_id:
                for eid in self._session_order.get(session_id, [])[-6:]:
                    candidates.setdefault(eid, "recency")

            scored: list[ScoredEngram] = []
            now = time.time()
            for eid, kind in candidates.items():
                engram = self._live(eid)
                if engram is None:
                    continue
                scored.append(self._score(engram, query_embedding, kind, now))

            scored.sort(key=lambda s: s.score, reverse=True)
            result = scored[:top_k]

        METRICS.incr("recalls")
        return result

    def _score(
        self, engram: Engram, query_embedding: list[float], kind: str, now: float
    ) -> ScoredEngram:
        c = self.config
        similarity = cosine(query_embedding, engram.embedding) if query_embedding else 0.0
        similarity = max(0.0, similarity)
        recency = math.exp(-RECENCY_LAMBDA * engram.age_days(now))
        access = min(1.0, math.log1p(engram.access_count) / math.log1p(ACCESS_SATURATION))
        components = {
            "similarity": c.score_alpha * similarity,
            "importance": c.score_beta * engram.importance,
            "recency": c.score_gamma * recency,
            "access": c.score_delta * access,
        }
        return ScoredEngram(
            engram=engram,
            score=sum(components.values()),
            similarity=similarity,
            components=components,
            hit_kind=kind,
        )

    def reinforce(self, engrams: list[Engram]) -> None:
        """Touched memories get access_count += 1 (spec §4, reinforcement)."""
        for e in engrams:
            e.access_count += 1

    # -- memory pack builder (§4) -----------------------------------------

    def build_pack(
        self,
        scored: list[ScoredEngram],
        token_budget: int | None = None,
        max_definitions: int = 6,
    ) -> MemoryPack:
        """Assemble retrieved engrams into a strict token budget: dedupe,
        collapse near-duplicates, order by score, render compact structured
        context. This is "recall critical memories within limited context
        windows," measured."""
        budget = token_budget or self.config.pack_token_budget

        # Dedupe exact/near-duplicate meanings, highest score wins.
        kept: list[ScoredEngram] = []
        seen_meanings: list[str] = []
        for s in scored:
            meaning_key = normalize_term(s.engram.meaning)
            if any(meaning_key == m for m in seen_meanings):
                continue
            if any(self._near_duplicate(s.engram, k.engram) for k in kept):
                continue
            kept.append(s)
            seen_meanings.append(meaning_key)

        # Preferences block — highest-value, always considered first.
        prefs = self._collect_preferences(kept)
        lines: list[str] = []
        used = 0

        def add_block(header: str, body_lines: list[str]) -> None:
            nonlocal used
            if not body_lines:
                return
            block = "\n".join([header, *body_lines])
            cost = estimate_tokens(block)
            if used + cost > budget:
                return
            lines.append(block)
            used += cost

        if prefs:
            add_block("[Known about the user]", prefs)

        # Relevant memories, by score, until the budget is spent.
        mem_lines: list[str] = []
        included: list[ScoredEngram] = []
        for s in kept:
            line = f"- ({s.engram.importance:.1f}) {s.engram.meaning or s.engram.message[:160]}"
            if used + estimate_tokens("\n".join(mem_lines + [line])) > budget and mem_lines:
                break
            mem_lines.append(line)
            included.append(s)
        if mem_lines:
            add_block("[Relevant memories]", mem_lines)

        # Lexicon definitions for terms in the surviving memories.
        defs = self._relevant_definitions(included, max_definitions)
        if defs:
            add_block("[Definitions]", [f"- {t}: {d}" for t, d in defs.items()])

        text = "\n\n".join(lines)
        tokens = estimate_tokens(text) if text else 0
        METRICS.record("pack_tokens", float(tokens))
        return MemoryPack(
            text=text,
            tokens=tokens,
            budget=budget,
            hits=included,
            definitions=defs,
        )

    def _near_duplicate(self, a: Engram, b: Engram) -> bool:
        if a.embedding and b.embedding:
            return cosine(a.embedding, b.embedding) > 0.97
        return False

    def _collect_preferences(self, scored: list[ScoredEngram]) -> list[str]:
        likes: list[str] = []
        dislikes: list[str] = []
        facts: list[str] = []
        for s in scored:
            d = s.engram.preferences_delta
            for src, dst in ((d.likes, likes), (d.dislikes, dislikes), (d.facts, facts)):
                for item in src:
                    if item not in dst:
                        dst.append(item)
        out: list[str] = []
        if likes:
            out.append(f"- likes: {', '.join(likes[:8])}")
        if dislikes:
            out.append(f"- dislikes: {', '.join(dislikes[:8])}")
        if facts:
            out.extend(f"- {f}" for f in facts[:8])
        return out

    def _relevant_definitions(
        self, scored: list[ScoredEngram], limit: int
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for s in scored:
            for term in s.engram.index_terms():
                t = normalize_term(term)
                if t in self._lexicon and t not in out:
                    out[t] = self._lexicon[t]
                    if len(out) >= limit:
                        return out
        return out

    # -- inspector views --------------------------------------------------

    @property
    def lexicon(self) -> dict[str, str]:
        return dict(self._lexicon)

    def all_engrams(self) -> list[Engram]:
        return list(self._engrams.values())

    def get(self, engram_id: str) -> Engram | None:
        return self._engrams.get(engram_id)

    def get_blob_engram(self, engram_id: str) -> Engram | None:
        """Re-read an engram straight from the virtual disk (proof the blob
        round-trips), bypassing the in-memory cache."""
        handle = self._handles.get(engram_id)
        if handle is None:
            return None
        return Engram.from_bytes(self.store.get_blob(self.space, handle))

    def live_engrams(self) -> list[Engram]:
        return [e for e in self._engrams.values() if not e.tombstoned]

    def session_last(self, session_id: str) -> str | None:
        """The most recent engram id in a session — the link for the next turn."""
        order = self._session_order.get(session_id)
        return order[-1] if order else None

    # -- physical reclaim (§5) --------------------------------------------

    def compact(self) -> int:
        """Reclaim space the only way the substrate does: rewrite the live
        engrams into a fresh planet and release the old one. The append-only
        virtual disk has no per-blob free, so tombstoned blobs are reclaimed
        wholesale here. Returns bytes freed by the release."""
        live = self.live_engrams()
        budget = next(
            (s.budget for s in self.store.stats().spaces if s.space == self.space),
            8 << 20,
        )
        old_space = self.space
        new_space = self.store.open_space(budget)

        # Reset the derived caches and re-commit live engrams into the new
        # planet (fresh blob + index cells). The lexicon is rebuilt from the
        # engrams' own definitions during re-commit.
        self.space = new_space
        self._engrams = {}
        self._handles = {}
        self._postings = {}
        self._session_order = {}
        self._lexicon = {}
        for engram in live:
            self.commit(engram)

        freed = self.store.release_space(old_space)
        METRICS.incr("compactions")
        METRICS.record("reclaimed_bytes", float(freed))
        return freed
