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
from dataclasses import dataclass, field

from engine.config import Settings, settings
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

    def all_minds(self) -> list[Mind]:
        return list(self._minds.values())


class KiokuEngine:
    """The turn loop over the registry — comparable, observable, honest."""

    def __init__(self, registry: TenantRegistry) -> None:
        self.registry = registry
        self.qwen = registry.qwen
        self.config = registry.config
        self._tasks: set[asyncio.Task] = set()

    # -- background work that must never block the reply ------------------

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain_background(self) -> None:
        """Await all in-flight curiosity/consolidation tasks (tests, shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    # -- the turn ---------------------------------------------------------

    async def turn(
        self,
        mind: Mind,
        message: str,
        session_id: str | None = None,
        send_to_both: bool = True,
    ) -> TurnResult:
        if mind.message_count >= self.registry.message_cap:
            raise MindFull(f"mind {mind.tenant_id} reached its {self.registry.message_cap}-message cap")
        session_id = session_id or new_ulid()
        mind.message_count += 1
        mind.turn_count += 1

        # 1. Recall: decompose-lite (local keywords + one cheap embedding).
        terms = lite_keywords(message)
        try:
            vectors = await self.qwen.embed([message])
            query_embedding = vectors[0] if vectors else []
        except LLMError as e:
            log.warning("query embed failed, keyword-only recall: %s", e)
            query_embedding = []
        scored = mind.index.recall(terms, query_embedding, session_id)
        pack = mind.index.build_pack(scored)
        mind.index.reinforce([s.engram for s in scored])

        # 2. Answer — with memory, and (for the comparison) without.
        kioku_reply = await self.qwen.chat(
            [
                {"role": "system", "content": KIOKU_SYSTEM.format(pack=pack.text or "(no memories yet)")},
                {"role": "user", "content": message},
            ]
        )
        raw_reply = None
        if send_to_both:
            raw_reply = await self.qwen.chat(
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
        engram = await decompose_exchange(self.qwen, capture, emit=mind.emit)
        supersede = mind.forget.supersede_contradictions(engram)
        # Snapshot which terms are novel BEFORE commit writes this engram's own
        # cells — otherwise curiosity would see every term as already "known".
        novel_terms = [t for t in engram.index_terms() if not mind.index.is_known(t)]
        receipt = mind.index.commit(engram)
        await mind.emit(
            PipelineEvent("committed", engram.engram_id, {"address": receipt.address, "block": receipt.block})
        )

        # 4. Curiosity and consolidation run in the background (never block).
        self._spawn(self._curiosity(mind, engram, novel_terms))
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

    async def _curiosity(self, mind: Mind, engram: Engram, novel_terms: list[str]) -> None:
        try:
            novel = set(novel_terms)
            learned = await curiosity_pass(
                self.qwen,
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
