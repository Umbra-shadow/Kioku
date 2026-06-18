# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1
"""Tenancy and the turn engine — the shared mind and the newborns (spec §6).

Default: one **shared mind** (``tenant = "kioku"``, one Cadran space) — everyone
who talks to the demo feeds the same memory. "New mind" mints a fresh tenant,
opens a new space, and a newborn with empty memory wakes up, isolated by Cadran
space isolation. No accounts: a tenant token (the tenant id) lives in the
browser's localStorage; per-tenant message caps and per-IP rate limits keep it
safe.

``KiokuEngine.turn`` is the full loop: recall → build pack → answer (with and
without memory) → understand the new exchange → commit → curiosity (async) →
supersede contradictions → consolidate on cadence.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from engine.config import LLMConfig, Settings, settings
from engine.curiosity import curiosity_pass
from engine.decompose import Capture, PipelineEvent, decompose_exchange, lite_keywords
from engine.engram import Engram, new_ulid
from engine.forget import ConsolidationDiff, ForgetManager, SupersedeDiff
from engine.metrics import METRICS
from engine.qwen import LLMError, QwenClient
from engine.retrieve import CommitReceipt, MemoryIndex, MemoryPack
from engine.store import MemoryStore

log = logging.getLogger("kioku.tenants")

SHARED_TENANT = "kioku"

KIOKU_SYSTEM = """You are a warm, concise assistant with a genuine long-term memory \
of this user, recalled below from Kioku. Use what you remember naturally, as a \
friend would — do not announce that you have a memory system unless asked. If the \
recalled memory is empty, simply answer normally.

{pack}"""

RAW_SYSTEM = "You are a helpful, concise assistant."

DEFAULT_SPACE_BUDGET = 64 << 20  # 64 MiB per mind
DEFAULT_MESSAGE_CAP = 500
CONSOLIDATE_EVERY = 12  # turns


@dataclass(frozen=True, slots=True)
class TurnResult:
    kioku_reply: str
    raw_reply: str | None
    pack: MemoryPack
    receipt: CommitReceipt
    engram: Engram
    supersede: SupersedeDiff
    session_id: str


class MindFull(RuntimeError):
    """A tenant hit its per-mind message cap."""


class Mind:
    """One tenant's memory and the live pipeline event bus the inspector reads."""

    def __init__(self, tenant_id: str, store: MemoryStore, qwen: QwenClient, config: Settings) -> None:
        self.tenant_id = tenant_id
        self.config = config
        self.space = store.open_space(DEFAULT_SPACE_BUDGET)
        self.index = MemoryIndex(store, self.space, config)
        self.forget = ForgetManager(self.index, qwen)
        self.created_ts = time.time()
        self.message_count = 0
        self.turn_count = 0
        self._subscribers: set[asyncio.Queue[dict]] = set()
        self._recent_events: list[dict] = []  # replay buffer for late subscribers

    # -- SSE event bus ----------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=256)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        self._subscribers.discard(q)

    async def emit(self, event: PipelineEvent) -> None:
        payload = {
            "stage": event.stage,
            "engram_id": event.engram_id,
            "detail": event.detail,
            "ts": event.ts,
            "tenant": self.tenant_id,
        }
        self._recent_events.append(payload)
        del self._recent_events[:-64]
        for q in list(self._subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass  # a slow inspector must never stall the pipeline

    @property
    def recent_events(self) -> list[dict]:
        return list(self._recent_events)


class TenantRegistry:
    """The box of minds. Hands out shared and newborn tenants over one store."""

    def __init__(
        self,
        store: MemoryStore,
        qwen: QwenClient,
        config: Settings | None = None,
        message_cap: int = DEFAULT_MESSAGE_CAP,
    ) -> None:
        self.store = store
        self.qwen = qwen
        self.config = config or settings()
        self.message_cap = message_cap
        self._minds: dict[str, Mind] = {}
        self._lock = asyncio.Lock()
        # The shared mind exists from birth — 25,000 people, one self.
        self._minds[SHARED_TENANT] = Mind(SHARED_TENANT, store, qwen, self.config)

    def resolve(self, token: str | None) -> Mind:
        """A token is a tenant id. Unknown/blank tokens fall back to the
        shared mind rather than erroring."""
        if not token or token not in self._minds:
            return self._minds[SHARED_TENANT]
        return self._minds[token]

    async def new_mind(self) -> Mind:
        async with self._lock:
            tenant_id = new_ulid()
            mind = Mind(tenant_id, self.store, self.qwen, self.config)
            self._minds[tenant_id] = mind
            log.info("newborn mind %s on space %d", tenant_id, mind.space)
            return mind

    def named_mind(self, tenant_id: str) -> Mind:
        """Get — or create — a mind under a STABLE id. Unlike ``new_mind`` (a
        random newborn), this is how a persistent per-user memory is pinned: the
        Researcher's whole history lives under one named tenant, rehydrated from
        the database on startup, so memory is per-user and survives restarts."""
        mind = self._minds.get(tenant_id)
        if mind is None:
            mind = Mind(tenant_id, self.store, self.qwen, self.config)
            self._minds[tenant_id] = mind
            log.info("named mind %s on space %d", tenant_id, mind.space)
        return mind

    def all_minds(self) -> list[Mind]:
        return list(self._minds.values())


_BRAIN_CACHE_SIZE = 32   # max per-key QwenClient instances held in RAM
_KEY_MAX = 128           # trim oversized X-Qwen-Key headers before using as dict keys


class KiokuEngine:
    """The turn loop over the registry — comparable, observable, honest."""

    def __init__(self, registry: TenantRegistry) -> None:
        self.registry = registry
        self.qwen = registry.qwen
        self.config = registry.config
        self._tasks: set[asyncio.Task] = set()
        # Durability hooks: list of fn(tenant_id, engram) -> None. Append-only so
        # multiple managers (e.g. in tests) can each register without overwriting.
        self.persistor: list[Callable[[str, Engram], None]] = []
        # Per-window brains keyed by raw API key, bounded LRU (never persisted).
        # The key is trimmed to _KEY_MAX bytes here so an oversized header can't
        # grow the cache unboundedly or land in logs.
        self._brains: OrderedDict[str, QwenClient] = OrderedDict()

    def qwen_for(self, api_key: str | None) -> QwenClient:
        """The brain to use for this request: the caller's keyed client if they
        brought a key, else the server's default brain."""
        if not api_key:
            return self.qwen
        api_key = api_key[:_KEY_MAX]
        client = self._brains.get(api_key)
        if client is None:
            base = self.config.llm
            keyed = LLMConfig(
                base_url=base.base_url, api_key=api_key, model=base.model,
                embed_model=base.embed_model, provider=base.provider,
                timeout_s=base.timeout_s, max_retries=base.max_retries,
            )
            client = QwenClient(keyed)
            if len(self._brains) >= _BRAIN_CACHE_SIZE:
                self._brains.popitem(last=False)  # evict the least-recently used
            self._brains[api_key] = client
        else:
            self._brains.move_to_end(api_key)
        return client

    async def aclose_brains(self) -> None:
        for client in list(self._brains.values()):
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass
        self._brains.clear()

    def _persist(self, mind: Mind, engram: Engram) -> None:
        for fn in self.persistor:
            try:
                fn(mind.tenant_id, engram)
            except Exception:  # noqa: BLE001 — durability must never break a reply
                log.exception("persistor failed for %s", engram.engram_id)

    # -- background work that must never block the reply ------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain_background(self) -> None:
        """Await all in-flight curiosity/consolidation tasks (tests, shutdown).

        Only awaits tasks that are not yet done, then yields so their
        ``discard`` done-callbacks can run. Awaiting an already-finished task
        does not suspend, so gathering only done tasks would busy-spin forever
        (the set never drains because the callbacks never get a turn)."""
        while True:
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                self._tasks.clear()
                return
            await asyncio.gather(*pending, return_exceptions=True)
            await asyncio.sleep(0)  # let the done-callbacks discard finished tasks

    # -- the turn ---------------------------------------------------------

    async def turn(
        self,
        mind: Mind,
        message: str,
        session_id: str | None = None,
        send_to_both: bool = True,
        qwen: QwenClient | None = None,
        extra_context: str = "",
    ) -> TurnResult:
        if mind.message_count >= self.registry.message_cap:
            raise MindFull(f"mind {mind.tenant_id} reached its {self.registry.message_cap}-message cap")
        qwen = qwen or self.qwen  # the caller's per-window brain, or the server default
        session_id = session_id or new_ulid()
        mind.message_count += 1
        mind.turn_count += 1

        # 1. Recall: decompose-lite (local keywords + one cheap embedding).
        terms = lite_keywords(message)
        try:
            vectors = await qwen.embed([message])
            query_embedding = vectors[0] if vectors else []
        except LLMError as e:
            log.warning("query embed failed, keyword-only recall: %s", e)
            query_embedding = []
        scored = mind.index.recall(terms, query_embedding, session_id)
        pack = mind.index.build_pack(scored)
        mind.index.reinforce([s.engram for s in scored])

        # 2. Answer — with memory, and (for the comparison) without.
        system = KIOKU_SYSTEM.format(pack=pack.text or "(no memories yet)")
        if extra_context:
            system += "\n\n" + extra_context
        kioku_reply = await qwen.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": message},
            ]
        )
        raw_reply = None
        if send_to_both:
            raw_reply = await qwen.chat(
                [{"role": "system", "content": RAW_SYSTEM}, {"role": "user", "content": message}]
            )

        # 3. Understand and commit the new exchange.
        prev = mind.index.session_last(session_id)
        capture = Capture(
            tenant=mind.tenant_id,
            user_id=mind.tenant_id,
            session_id=session_id,
            message=message,
            reply=kioku_reply,
            session_prev=prev,
        )
        engram = await decompose_exchange(qwen, capture, emit=mind.emit)
        supersede = mind.forget.supersede_contradictions(engram)
        # Snapshot which terms are novel BEFORE commit writes this engram's own
        # cells — otherwise curiosity would see every term as already "known".
        novel_terms = [t for t in engram.index_terms() if not mind.index.is_known(t)]
        receipt = mind.index.commit(engram)
        self._persist(mind, engram)
        await mind.emit(
            PipelineEvent("committed", engram.engram_id, {"address": receipt.address, "block": receipt.block})
        )

        # 4. Curiosity and consolidation run in the background (never block).
        self._spawn(self._curiosity(mind, engram, novel_terms, qwen))
        if mind.turn_count % CONSOLIDATE_EVERY == 0:
            self._spawn(self._consolidate(mind))

        return TurnResult(
            kioku_reply=kioku_reply,
            raw_reply=raw_reply,
            pack=pack,
            receipt=receipt,
            engram=engram,
            supersede=supersede,
            session_id=session_id,
        )

    async def remember(
        self,
        mind: Mind,
        message: str,
        reply: str,
        session_id: str | None = None,
        *,
        importance_floor: float = 0.0,
        qwen: QwenClient | None = None,
    ) -> Engram:
        """Commit an exchange into memory WITHOUT generating an answer.

        ``turn`` is the chat loop (it asks the brain twice for the dual-pane
        comparison). The researcher already *has* the content — a sub-question
        and the finding it researched — and only needs Kioku to *understand and
        remember* it. This is that write half of the turn: decompose → embed →
        supersede contradictions → commit, plus background curiosity. So a whole
        research run (every question, every finding, the final report) becomes
        recallable memory the model can be asked about afterwards.
        """
        qwen = qwen or self.qwen
        session_id = session_id or new_ulid()
        mind.message_count += 1
        prev = mind.index.session_last(session_id)
        capture = Capture(
            tenant=mind.tenant_id,
            user_id=mind.tenant_id,
            session_id=session_id,
            message=message,
            reply=reply,
            session_prev=prev,
        )
        engram = await decompose_exchange(qwen, capture, emit=mind.emit)
        if engram.importance < importance_floor:
            engram.importance = importance_floor
            engram.memory_class = "semantic"
        mind.forget.supersede_contradictions(engram)
        novel_terms = [t for t in engram.index_terms() if not mind.index.is_known(t)]
        mind.index.commit(engram)
        self._persist(mind, engram)
        self._spawn(self._curiosity(mind, engram, novel_terms, qwen))
        return engram

    async def _curiosity(
        self, mind: Mind, engram: Engram, novel_terms: list[str], qwen: QwenClient | None = None
    ) -> None:
        try:
            novel = set(novel_terms)
            learned = await curiosity_pass(
                qwen or self.qwen,
                engram,
                is_known=lambda t: t not in novel,
                max_lookups=self.config.curiosity_max_lookups,
                emit=mind.emit,
            )
            mind.index.learn_terms(learned)
        except Exception:  # noqa: BLE001 — background task, log and move on
            log.exception("curiosity task failed for %s", engram.engram_id)

    async def _consolidate(self, mind: Mind) -> ConsolidationDiff:
        try:
            return await mind.forget.consolidate()
        except Exception:  # noqa: BLE001
            log.exception("consolidation task failed for %s", mind.tenant_id)
            return ConsolidationDiff()
