"""Forgetting — the part everyone else skips (spec §5).

Three mechanisms, all observable in the inspector's Forgetting tab:

- **Decay**: every engram carries a retention score (defined on the engram,
  §5 math). Low-retention episodic/small-talk memories are consolidation
  candidates.
- **Consolidation**: clusters of old low-retention episodic engrams are
  summarized by Qwen into one semantic engram ("user spent June planning a
  flower-viewing trip"); the originals are tombstoned. A compaction then
  reclaims their disk blocks via space release.
- **Contradiction**: a new preference that conflicts with an old one
  supersedes it — the old engram is tombstoned with ``superseded_by``.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from engine.engram import Engram, classify, normalize_term
from engine.qwen import LLMError, QwenClient
from engine.retrieve import MemoryIndex

log = logging.getLogger("kioku.forget")

# Episodic/small-talk memories below this retention are consolidation fodder.
DEFAULT_RETENTION_FLOOR = 0.15
# Don't consolidate anything younger than this — give it a chance to be used.
MIN_AGE_DAYS = 1.0
# A cluster needs at least this many memories to be worth summarizing.
MIN_CLUSTER_SIZE = 2

CONSOLIDATE_SYSTEM = (
    "You are the consolidation stage of Kioku, a memory engine. You are given "
    "several old, low-importance memories that share a topic. Summarize them "
    "into ONE durable semantic memory — a single sentence capturing what "
    "lastingly matters, in the user's own frame. Reply with ONLY that sentence."
)


@dataclass(frozen=True, slots=True)
class RetentionRow:
    engram_id: str
    meaning: str
    memory_class: str
    importance: float
    access_count: int
    age_days: float
    retention: float
    tombstoned: bool


@dataclass(slots=True)
class ConsolidationDiff:
    """The last consolidation, for the Forgetting tab."""

    summaries: list[str] = field(default_factory=list)
    tombstoned_ids: list[str] = field(default_factory=list)
    created_ids: list[str] = field(default_factory=list)
    reclaimed_bytes: int = 0

    @property
    def did_anything(self) -> bool:
        return bool(self.tombstoned_ids or self.created_ids)


@dataclass(slots=True)
class SupersedeDiff:
    superseded_ids: list[str] = field(default_factory=list)
    reason: str = ""


class ForgetManager:
    """Decay reporting, consolidation, and contradiction handling for one mind."""

    def __init__(
        self,
        index: MemoryIndex,
        qwen: QwenClient,
        retention_floor: float = DEFAULT_RETENTION_FLOOR,
    ) -> None:
        self.index = index
        self.qwen = qwen
        self.retention_floor = retention_floor
        self.last_consolidation = ConsolidationDiff()

    # -- decay reporting --------------------------------------------------

    def retention_report(self, now: float | None = None) -> list[RetentionRow]:
        """Every engram with its current retention, weakest first."""
        now = now or time.time()
        lambdas = self.index.config.lambda_per_class
        rows = [
            RetentionRow(
                engram_id=e.engram_id,
                meaning=e.meaning,
                memory_class=e.memory_class,
                importance=e.importance,
                access_count=e.access_count,
                age_days=round(e.age_days(now), 3),
                retention=round(e.retention(lambdas, now), 4),
                tombstoned=e.tombstoned,
            )
            for e in self.index.all_engrams()
        ]
        rows.sort(key=lambda r: r.retention)
        return rows

    def _candidates(self, now: float) -> list[Engram]:
        lambdas = self.index.config.lambda_per_class
        out = []
        for e in self.index.live_engrams():
            if e.memory_class in ("episodic", "smalltalk") and e.age_days(now) >= MIN_AGE_DAYS:
                if e.retention(lambdas, now) < self.retention_floor:
                    out.append(e)
        return out

    # -- consolidation ----------------------------------------------------

    async def consolidate(self, now: float | None = None, compact: bool = True) -> ConsolidationDiff:
        """Cluster aging low-retention memories by topic, summarize each
        cluster into one semantic engram, tombstone the originals, and
        optionally reclaim their disk blocks via compaction."""
        now = now or time.time()
        candidates = self._candidates(now)
        diff = ConsolidationDiff()
        if not candidates:
            self.last_consolidation = diff
            return diff

        clusters: dict[str, list[Engram]] = defaultdict(list)
        for e in candidates:
            topic = (e.links.topics or e.keywords or ["misc"])[0]
            clusters[normalize_term(topic)].append(e)

        for topic, cluster in clusters.items():
            if len(cluster) < MIN_CLUSTER_SIZE:
                continue
            summary_text = await self._summarize(topic, cluster)
            if not summary_text:
                continue
            summary = self._semantic_engram(topic, cluster, summary_text)
            self.index.commit(summary)
            diff.created_ids.append(summary.engram_id)
            diff.summaries.append(summary_text)
            for e in cluster:
                e.tombstoned = True
                e.superseded_by = summary.engram_id
                diff.tombstoned_ids.append(e.engram_id)

        if compact and diff.did_anything:
            diff.reclaimed_bytes = self.index.compact()

        self.last_consolidation = diff
        log.info(
            "consolidation: %d tombstoned, %d created, %d bytes reclaimed",
            len(diff.tombstoned_ids), len(diff.created_ids), diff.reclaimed_bytes,
        )
        return diff

    async def _summarize(self, topic: str, cluster: list[Engram]) -> str:
        bullets = "\n".join(f"- {e.meaning or e.message[:160]}" for e in cluster)
        try:
            text = await self.qwen.chat(
                [
                    {"role": "system", "content": CONSOLIDATE_SYSTEM},
                    {"role": "user", "content": f"TOPIC: {topic}\nMEMORIES:\n{bullets}"},
                ],
                temperature=0.3,
                max_tokens=120,
            )
        except LLMError as e:
            log.warning("consolidation summary failed for topic %r: %s", topic, e)
            return ""
        return text.strip()

    def _semantic_engram(self, topic: str, cluster: list[Engram], summary_text: str) -> Engram:
        first = cluster[0]
        # A consolidated memory is worth keeping: at least its strongest source.
        importance = max(0.5, max(e.importance for e in cluster))
        merged_keywords: list[str] = []
        for e in cluster:
            for kw in e.keywords:
                if kw not in merged_keywords:
                    merged_keywords.append(kw)
        engram = Engram(
            tenant=first.tenant,
            user_id=first.user_id,
            session_id=first.session_id,
            ts=time.time(),
            message=f"(consolidated {len(cluster)} memories about {topic})",
            reply="",
            meaning=summary_text,
            intent="retain the lasting gist",
            keywords=merged_keywords[:8],
            entities=[],
            importance=min(1.0, importance),
        )
        engram.memory_class = "semantic"
        engram.links.topics = [topic]
        return engram

    # -- contradiction handling -------------------------------------------

    def supersede_contradictions(self, new_engram: Engram) -> SupersedeDiff:
        """A new like that was an old dislike (or vice versa), or a new fact
        that overwrites an old one about the same subject, tombstones the
        stale memory. Preferences keep up with a changing person."""
        diff = SupersedeDiff()
        delta = new_engram.preferences_delta
        if delta.is_empty():
            return diff

        new_likes = {normalize_term(x) for x in delta.likes}
        new_dislikes = {normalize_term(x) for x in delta.dislikes}
        new_fact_subjects = {_fact_subject(f) for f in delta.facts}

        for old in self.index.live_engrams():
            if old.engram_id == new_engram.engram_id or old.preferences_delta.is_empty():
                continue
            old_likes = {normalize_term(x) for x in old.preferences_delta.likes}
            old_dislikes = {normalize_term(x) for x in old.preferences_delta.dislikes}
            old_subjects = {_fact_subject(f) for f in old.preferences_delta.facts}

            flipped = (new_likes & old_dislikes) | (new_dislikes & old_likes)
            fact_overwrite = new_fact_subjects & old_subjects - {""}
            if flipped or fact_overwrite:
                old.tombstoned = True
                old.superseded_by = new_engram.engram_id
                diff.superseded_ids.append(old.engram_id)
                reason = ", ".join(sorted(flipped | fact_overwrite))
                diff.reason = f"superseded by newer preference ({reason})"

        if diff.superseded_ids:
            log.info("superseded %d stale preference(s): %s", len(diff.superseded_ids), diff.reason)
        return diff


def _fact_subject(fact: str) -> str:
    """Crude subject key for a personal fact, so 'lives in Kyoto' and 'lives
    in Osaka now' collide on 'lives in' and the newer one wins. The first two
    normalized words catch the common 'verb preposition object' facts (lives
    in, name is, works at); a documented v1 heuristic, not full coreference."""
    words = normalize_term(fact).split()
    return " ".join(words[:2])
