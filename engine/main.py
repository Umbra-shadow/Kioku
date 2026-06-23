# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1
"""Kioku v1 — FastAPI app: the arena's backend, the inspector's source.

Routes (all JSON unless noted):
  GET  /api/health                      liveness + which substrate backend
  POST /api/chat                        the turn: dual answers + memory pack
  POST /api/mind/new                    mint a newborn tenant (empty memory)
  GET  /api/stream/{token}              SSE: live pipeline stage chips
  GET  /api/memory?token&limit&offset   paginated engram browser
  GET  /api/memory/{token}/{engram_id}  one engram, full JSON (re-read from disk)
  GET  /api/lexicon?token               curiosity definitions
  GET  /api/forgetting?token            retention rows + last consolidation
  POST /api/consolidate                 force a consolidation pass (demo button)
  GET  /api/stats?token                 substrate gauge + latency percentiles
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from engine.config import REPO_ROOT
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from engine.config import settings
from engine.metrics import METRICS
from engine.qwen import LLMError, QwenClient
from engine.store import StoreStats, open_store
from engine.tenants import KiokuEngine, Mind, MindFull, TenantRegistry

log = logging.getLogger("kioku.api")
logging.basicConfig(level=os.environ.get("KIOKU_LOG_LEVEL", "INFO"))

limiter = Limiter(key_func=get_remote_address, default_limits=["240/minute"])


# --- request/response models -------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    token: str | None = None
    session_id: str | None = None
    send_to_both: bool = True


class PackView(BaseModel):
    text: str
    tokens: int
    budget: int
    hits: list[dict[str, Any]]
    definitions: dict[str, str]


class ChatResponse(BaseModel):
    token: str
    session_id: str
    kioku_reply: str
    raw_reply: str | None
    pack: PackView
    address: str
    block: int
    engram_id: str
    superseded: list[str]


class NewMindResponse(BaseModel):
    token: str
    space: int


# --- gauge assembly ----------------------------------------------------------


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def _gauge(stats: StoreStats, mind: Mind) -> dict[str, Any]:
    return {
        "backend": stats.backend,
        "vram_committed": stats.vram_committed,
        "vram_virtual": stats.vram_virtual,
        "disk_committed": stats.disk_committed,
        "disk_virtual": stats.disk_virtual,
        "headline": f"{_human(stats.vram_committed)} of {_human(stats.vram_virtual)} "
        f"— small outside, huge inside",
        "open_minds": stats.open_spaces,
        "live_engrams": len(mind.index.live_engrams()),
        "retrieve_ms": _summary("retrieve_ms"),
        "commit_ms": _summary("commit_ms"),
        "pack_tokens": _summary("pack_tokens", unit="tok"),
        "reclaimed_bytes": METRICS.counter("compactions"),
    }


def _summary(name: str, unit: str = "ms") -> dict[str, Any]:
    s = METRICS.summary(name, unit=unit)
    return {"count": s.count, "p50": s.p50, "p95": s.p95, "p99": s.p99, "unit": s.unit}


# --- app factory -------------------------------------------------------------


def build_engine() -> KiokuEngine:
    cfg = settings()
    store = open_store(cfg.data_dir)
    qwen = QwenClient(cfg.llm)
    # This is a single-user research tool, not a public arena — the per-message
    # cap is high so a long research life (many runs × ~20 findings) never trips it.
    message_cap = int(os.environ.get("KIOKU_MESSAGE_CAP", "1000000"))
    registry = TenantRegistry(store, qwen, cfg, message_cap=message_cap)
    engine = KiokuEngine(registry)
    engine._store = store  # type: ignore[attr-defined]  # held for shutdown
    return engine


def create_app(engine: KiokuEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or build_engine()
        # The Researcher's durable history + per-user memory. The DB path is
        # configurable; ":memory:" gives an ephemeral store (tests).
        from engine.research.persistence import ResearchDB
        from engine.research.runs import RunManager

        # Prefer a real cloud database (DATABASE_URL, e.g. Neon Postgres) so the
        # history + per-user memory survive redeploys; fall back to a local file.
        db_dsn = (
            os.environ.get("DATABASE_URL")
            or os.environ.get("KIOKU_RESEARCH_DB")
            or str(REPO_ROOT / "kioku_data" / "research.db")
        )
        app.state.db = ResearchDB(db_dsn)
        app.state.runs = RunManager(app.state.engine, app.state.db)
        app.state.runs.bootstrap()  # rehydrate memory + wire durability
        yield
        eng = app.state.engine
        await eng.drain_background()
        store = getattr(eng, "_store", None)
        if store is not None:
            store.close()
        await eng.qwen.aclose()
        await eng.aclose_brains()
        db = getattr(app.state, "db", None)
        if db is not None:
            db.close()

    app = FastAPI(title="Kioku v1", version="0.1.0", lifespan=lifespan)
    # Rate limiting is on by default; tests and load benchmarks turn it off.
    limiter.enabled = os.environ.get("KIOKU_RATELIMIT", "on").lower() not in ("off", "0", "false")
    app.state.limiter = limiter

    @app.exception_handler(RateLimitExceeded)
    async def _ratelimit(request: Request, exc: RateLimitExceeded):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})

    origins = os.environ.get("KIOKU_WEB_ORIGIN", "http://localhost:8080").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def get_engine(request: Request) -> KiokuEngine:
        return request.app.state.engine

    def get_mind(request: Request, token: str | None) -> Mind:
        return request.app.state.engine.registry.resolve(token)

    # -- routes -----------------------------------------------------------

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        eng = get_engine(request)
        store = getattr(eng, "_store", None)
        backend = store.stats().backend if store is not None else "unknown"
        return {"ok": True, "service": "kioku", "version": "0.1.0", "backend": backend}

    @app.post("/api/chat", response_model=ChatResponse)
    @limiter.limit("30/minute")
    async def chat(request: Request, body: ChatRequest) -> ChatResponse:
        eng = get_engine(request)
        mind = eng.registry.resolve(body.token)
        try:
            result = await eng.turn(
                mind, body.message, session_id=body.session_id, send_to_both=body.send_to_both,
                qwen=eng.qwen_for(request.headers.get("X-Qwen-Key")),
            )
        except MindFull as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        except LLMError as e:
            raise HTTPException(status_code=502, detail=f"Qwen Cloud unavailable: {e}") from e
        return ChatResponse(
            token=mind.tenant_id,
            session_id=result.session_id,
            kioku_reply=result.kioku_reply,
            raw_reply=result.raw_reply,
            pack=PackView(
                text=result.pack.text,
                tokens=result.pack.tokens,
                budget=result.pack.budget,
                hits=result.pack.hit_list(),
                definitions=result.pack.definitions,
            ),
            address=result.receipt.address,
            block=result.receipt.block,
            engram_id=result.engram.engram_id,
            superseded=result.supersede.superseded_ids,
        )

    @app.post("/api/mind/new", response_model=NewMindResponse)
    @limiter.limit("10/minute")
    async def new_mind(request: Request) -> NewMindResponse:
        mind = await get_engine(request).registry.new_mind()
        return NewMindResponse(token=mind.tenant_id, space=mind.space)

    @app.get("/api/stream/{token}")
    async def stream(
        request: Request, token: str, replay_then_close: bool = False
    ) -> StreamingResponse:
        mind = get_mind(request, token)

        async def event_source():
            # Replay the recent buffer so a late inspector still sees the chips.
            for event in mind.recent_events[-12:]:
                yield f"data: {json.dumps(event)}\n\n"
            if replay_then_close:
                return  # finite stream for tests / one-shot polls
            queue = mind.subscribe()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                mind.unsubscribe(queue)

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/memory")
    async def memory(
        request: Request,
        token: str | None = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        include_tombstoned: bool = False,
    ) -> dict[str, Any]:
        mind = get_mind(request, token)
        engrams = mind.index.all_engrams() if include_tombstoned else mind.index.live_engrams()
        engrams.sort(key=lambda e: e.ts, reverse=True)
        window = engrams[offset : offset + limit]
        return {
            "total": len(engrams),
            "limit": limit,
            "offset": offset,
            "engrams": [
                {
                    "engram_id": e.engram_id,
                    "ts": e.ts,
                    "message": e.message,
                    "meaning": e.meaning,
                    "intent": e.intent,
                    "keywords": e.keywords,
                    "entities": e.entities,
                    "importance": e.importance,
                    "memory_class": e.memory_class,
                    "access_count": e.access_count,
                    "tombstoned": e.tombstoned,
                }
                for e in window
            ],
        }

    @app.get("/api/memory/{token}/{engram_id}")
    async def memory_detail(request: Request, token: str, engram_id: str) -> dict[str, Any]:
        mind = get_mind(request, token)
        # Re-read from the virtual disk to prove the blob round-trips.
        engram = mind.index.get_blob_engram(engram_id) or mind.index.get(engram_id)
        if engram is None:
            raise HTTPException(status_code=404, detail="no such engram")
        return json.loads(engram.model_dump_json())

    @app.get("/api/lexicon")
    async def lexicon(request: Request, token: str | None = None) -> dict[str, Any]:
        mind = get_mind(request, token)
        lex = mind.index.lexicon
        return {"count": len(lex), "definitions": lex}

    @app.get("/api/forgetting")
    async def forgetting(request: Request, token: str | None = None) -> dict[str, Any]:
        mind = get_mind(request, token)
        rows = mind.forget.retention_report()
        last = mind.forget.last_consolidation
        return {
            "retention": [
                {
                    "engram_id": r.engram_id,
                    "meaning": r.meaning,
                    "memory_class": r.memory_class,
                    "importance": r.importance,
                    "access_count": r.access_count,
                    "age_days": r.age_days,
                    "retention": r.retention,
                    "tombstoned": r.tombstoned,
                }
                for r in rows[:100]
            ],
            "last_consolidation": {
                "summaries": last.summaries,
                "tombstoned": last.tombstoned_ids,
                "created": last.created_ids,
                "reclaimed_bytes": last.reclaimed_bytes,
            },
        }

    @app.post("/api/consolidate")
    @limiter.limit("6/minute")
    async def consolidate(request: Request, token: str | None = None) -> dict[str, Any]:
        mind = get_mind(request, token)
        diff = await mind.forget.consolidate()
        return {
            "did_anything": diff.did_anything,
            "summaries": diff.summaries,
            "tombstoned": diff.tombstoned_ids,
            "created": diff.created_ids,
            "reclaimed_bytes": diff.reclaimed_bytes,
        }

    @app.get("/api/stats")
    async def stats(request: Request, token: str | None = None) -> dict[str, Any]:
        eng = get_engine(request)
        mind = eng.registry.resolve(token)
        store = getattr(eng, "_store", None)
        store_stats = store.stats() if store is not None else None
        if store_stats is None:
            raise HTTPException(status_code=503, detail="store unavailable")
        return {
            "gauge": _gauge(store_stats, mind),
            "counters": METRICS.snapshot()["counters"],
            "tenant": mind.tenant_id,
        }

    _AUTOCONV_SYSTEM = (
        "You are scripting a realistic multi-turn demo for a memory AI. "
        "Generate a sequence of short user messages (1–3 sentences each) for the given domain. "
        "The sequence must:\n"
        "1. First message: introduce yourself with a specific name and role in that domain.\n"
        "2. Middle messages: ask real, interesting questions about the domain; occasionally "
        "   reference who you are or what you said earlier.\n"
        "3. One late message: explicitly test memory (e.g. 'do you remember what I told you about myself?').\n"
        "4. Last message: a recall probe — 'what do you know about me and my work in this field?'\n"
        "Messages must feel natural. Make personal details specific and memorable.\n"
        "Respond ONLY with a JSON object: {\"messages\": [\"...\", \"...\"]}"
    )

    @app.post("/api/autoconv/plan")
    async def autoconv_plan(request: Request) -> dict:
        data = await request.json()
        domain = str(data.get("domain", "")).strip()
        turns = int(data.get("turns", 7))
        if len(domain) < 2:
            raise HTTPException(status_code=422, detail="domain must be at least 2 characters")
        turns = max(4, min(12, turns))
        engine: KiokuEngine = get_engine(request)
        qwen = engine.qwen_for(request.headers.get("X-Qwen-Key"))
        try:
            result = await qwen.chat_json(
                [
                    {"role": "system", "content": _AUTOCONV_SYSTEM},
                    {"role": "user", "content": f"Domain: {domain}\nTurns: {turns}"},
                ],
                temperature=0.75,
                max_tokens=8192,
            )
            messages = result.get("messages") if isinstance(result, dict) else None
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError("bad response shape")
            return {"messages": [str(m).strip() for m in messages[:turns] if str(m).strip()]}
        except LLMError as e:
            raise HTTPException(status_code=502, detail=f"LLM unavailable: {e}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not plan conversation: {e}") from e

    # The Researcher: one question → ~20 deep questions → live web research →
    # one complete report, all remembered in a Kioku mind. Registered before the
    # static mount so its /api routes win.
    from engine.research_api import add_research_routes

    add_research_routes(app)

    # Serve the web arena at the root, same-origin with the API, so the browser
    # never needs a separate host/port. Mounted last: API routes take priority.
    web_dir = REPO_ROOT / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


app = None  # built lazily by uvicorn via the factory below


def get_app() -> FastAPI:
    return create_app()
