# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""Run manager — owns the live research runs, their shared memory, and history.

Two design choices carry the operator's intent:

1. **Memory is per user, not per session.** Every run and every chat writes into
   ONE persistent mind (the ``researcher`` tenant), not a fresh newborn per run.
   So a brand-new session — even years later — recalls a run from long ago. The
   memory accumulates; nothing is siloed by session.

2. **Nothing is lost.** A ``ResearchDB`` (SQLite) persists every run, every chat
   turn, and every committed engram. On startup the manager rehydrates the mind
   from those engrams and reloads the run history — so a restart loses nothing.

A ``ResearchRun`` is one investigation: a topic, its status, findings, the final
report, and a small pub/sub bus so the browser can watch the work live (SSE).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from engine.engram import new_ulid
from engine.research.persistence import ResearchDB
from engine.research.researcher import DEFAULT_NUM_QUESTIONS, Researcher
from engine.research.websearch import WebSearch
from engine.tenants import KiokuEngine, Mind

log = logging.getLogger("kioku.research.runs")

# The single persistent per-user memory the whole Researcher shares.
# DESIGN: this is intentionally a single-user tool. All runs share one tenant
# and one mind; run history is readable by anyone who can reach the server.
# If this ever becomes multi-tenant, replace with a session cookie / auth token
# that maps each browser to its own tenant id.
RESEARCH_TENANT = "researcher"


@dataclass
class ResearchRun:
    run_id: str
    topic: str
    mind: Mind
    num_questions: int = DEFAULT_NUM_QUESTIONS
    status: str = "starting"  # starting|expanding|researching|synthesizing|done|error
    error: str | None = None
    questions: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    report: str = ""
    provider: str = ""
    grounded_count: int = 0
    session_id: str = ""
    created_ts: float = field(default_factory=time.time)
    done_ts: float | None = None

    _events: list[dict] = field(default_factory=list, repr=False)
    _subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def token(self) -> str:
        return self.mind.tenant_id

    @property
    def is_terminal(self) -> bool:
        return self.status in ("done", "error")

    # -- pub/sub ----------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    async def emit(self, stage: str, detail: dict) -> None:
        event = {"stage": stage, "detail": detail, "ts": time.time(), "run_id": self.run_id}
        self._events.append(event)
        del self._events[:-256]
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def recent_events(self) -> list[dict]:
        return list(self._events)

    def public(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "token": self.token,
            "tenant": self.token,
            "status": self.status,
            "error": self.error,
            "num_questions": self.num_questions,
            "questions": self.questions,
            "findings": self.findings,
            "report": self.report,
            "provider": self.provider,
            "grounded_count": self.grounded_count,
            "session_id": self.session_id,
            "created_ts": self.created_ts,
            "done_ts": self.done_ts,
        }


_STATUS_BY_STAGE = {
    "expanding": "expanding",
    "expanded": "researching",
    "synthesizing": "synthesizing",
    "done": "done",
}


class RunManager:
    """All research runs for this engine, over one shared persistent memory."""

    def __init__(self, engine: KiokuEngine, db: ResearchDB | None = None,
                 tenant: str = RESEARCH_TENANT) -> None:
        self.engine = engine
        self.db = db
        self.tenant = tenant
        self._runs: dict[str, ResearchRun] = {}
        self._lock = asyncio.Lock()
        self.user_mind: Mind = engine.registry.named_mind(tenant)
        self._bootstrapped = False

    def bootstrap(self) -> int:
        """Wire durability and rebuild memory from disk. Idempotent. Returns the
        number of engrams rehydrated into the per-user mind."""
        if self._bootstrapped:
            return 0
        self._bootstrapped = True
        if self.db is None:
            return 0
        # Append rather than overwrite so that multiple RunManagers on the same
        # engine (e.g. two test fixtures in one process) each get their callbacks
        # registered without silently dropping the other's.
        self.engine.persistor.append(self.db.save_engram)
        # rebuild the in-RAM index from the durable engrams — memory survives restart
        n = 0
        for engram in self.db.load_engrams(self.tenant):
            self.user_mind.index.commit(engram)
            n += 1
        if n:
            log.info("rehydrated %d engrams into the '%s' memory", n, self.tenant)
        return n

    # -- lookup -----------------------------------------------------------

    def get(self, run_id: str) -> ResearchRun | None:
        run = self._runs.get(run_id)
        if run is not None:
            return run
        # not live in memory — try the durable history (e.g. after a restart)
        if self.db is not None:
            row = self.db.load_run(run_id)
            if row:
                run = self._from_row(row)
                self._runs[run_id] = run
                return run
        return None

    def list(self) -> list[dict[str, Any]]:
        """Run history — live runs merged with the durable record."""
        seen: dict[str, dict] = {}
        if self.db is not None:
            for row in self.db.load_runs(self.tenant):
                seen[row["run_id"]] = row
        for r in self._runs.values():
            seen[r.run_id] = r.public()
        return sorted(seen.values(), key=lambda r: r.get("created_ts") or 0, reverse=True)

    def _from_row(self, row: dict[str, Any]) -> ResearchRun:
        run = ResearchRun(
            run_id=row["run_id"], topic=row["topic"], mind=self.user_mind,
            num_questions=row.get("num_questions") or DEFAULT_NUM_QUESTIONS,
            status=row["status"], error=row.get("error"),
            questions=row.get("questions") or [], findings=row.get("findings") or [],
            report=row.get("report") or "", provider=row.get("provider") or "",
            grounded_count=row.get("grounded_count") or 0,
            session_id=row.get("session_id") or "",
            created_ts=row.get("created_ts") or time.time(), done_ts=row.get("done_ts"),
        )
        return run

    def _persist_run(self, run: ResearchRun) -> None:
        if self.db is not None:
            try:
                self.db.save_run({**run.public(), "tenant": run.token})
            except Exception:  # noqa: BLE001
                log.exception("failed to persist run %s", run.run_id)

    # -- start + drive ----------------------------------------------------

    async def start(self, topic: str, num_questions: int = DEFAULT_NUM_QUESTIONS,
                    qwen=None) -> ResearchRun:
        if not self._bootstrapped:
            self.bootstrap()
        async with self._lock:
            run = ResearchRun(
                run_id=new_ulid(), topic=topic.strip(), mind=self.user_mind,
                num_questions=num_questions,
            )
            self._runs[run.run_id] = run
        self._persist_run(run)
        run._task = asyncio.create_task(self._drive(run, qwen))
        return run

    async def _drive(self, run: ResearchRun, qwen=None) -> None:
        web = WebSearch()
        run.provider = web.provider

        async def progress(stage: str, detail: dict) -> None:
            if stage in _STATUS_BY_STAGE:
                run.status = _STATUS_BY_STAGE[stage]
            if stage == "expanded":
                run.questions = detail.get("questions", [])
                run.findings = [
                    {"id": i + 1, "question": q, "answer": "", "grounded": False, "sources": []}
                    for i, q in enumerate(run.questions)
                ]
                self._persist_run(run)
            elif stage == "studied":
                fid = detail.get("id")
                for f in run.findings:
                    if f["id"] == fid:
                        f["answer"] = detail.get("answer", "")
                        f["grounded"] = detail.get("grounded", False)
                        f["sources"] = detail.get("sources", f.get("sources", []))
                        break
            await run.emit(stage, detail)

        researcher = Researcher(self.engine, run.mind, web=web, progress=progress, qwen=qwen)
        try:
            run.status = "expanding"
            result = await researcher.run(run.topic, run.num_questions)
            run.report = result["report"]
            run.findings = result["findings"]
            run.questions = result["questions"]
            run.grounded_count = result["grounded_count"]
            run.session_id = result["session_id"]
            run.status = "done"
        except Exception as e:  # noqa: BLE001 — surface the failure, don't crash the server
            log.exception("research run %s failed", run.run_id)
            run.status = "error"
            run.error = f"{type(e).__name__}: {e}"
            await run.emit("error", {"error": run.error})
        finally:
            run.done_ts = time.time()
            self._persist_run(run)
            await researcher.aclose()
