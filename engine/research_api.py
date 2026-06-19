# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""FastAPI routes for the Researcher — mounted onto the Kioku app.

  POST /api/research/start            begin an investigation (background)
  GET  /api/research                  list runs (brief)
  GET  /api/research/{id}             full run state + report
  GET  /api/research/{id}/stream      SSE: live research progress
  GET  /api/research/{id}/pdf         download the complete report as PDF
  POST /api/research/{id}/ask         ask a follow-up — recalled from the run's memory

The follow-up ``ask`` runs the normal Kioku turn against the run's OWN mind, so
the answer is grounded in everything the run remembered: the plan, all ~20
findings, and the final report. That is the whole point — the research lives in
memory, and you can keep questioning it.
"""
from __future__ import annotations

import asyncio
import json
import re

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from engine.qwen import LLMError
from engine.research.report_pdf import build_pdf
from engine.research.researcher import DEFAULT_NUM_QUESTIONS
from engine.research.runs import RunManager
from engine.tenants import KiokuEngine, MindFull

_RESEARCH_SYSTEM = """\
You are a research expert with deep, committed knowledge about the topic you investigated.
Everything below is YOUR knowledge — you studied it, you understand it, you remember it.
Answer as someone who truly knows this subject, drawing naturally from what you learned.
When asked what you remember or know, speak from your understanding — do not say you lack memory.
Recalled memory from prior conversations (may be empty for new sessions):
{pack}\
"""


class StartRequest(BaseModel):
    topic: str = Field(min_length=4, max_length=2000)
    num_questions: int = Field(default=DEFAULT_NUM_QUESTIONS, ge=3, le=40)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Optional: keep a chat thread together, or start a fresh session to prove
    # cross-session recall (memory is global to the run's mind, not the session).
    session_id: str | None = None


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (s[:limit] or "kioku-research").rstrip("-")


def _build_run_context(run) -> str:
    """Build a compact research context injected into every chat turn.

    The Kioku memory recall works beautifully with embeddings, but degrades to
    keyword-only without them. To guarantee the model always knows what was
    researched — regardless of embedding availability — we inject the run's own
    findings directly into the system prompt on every ask turn.
    """
    data = run.public()
    topic = data.get("topic", "")
    findings = data.get("findings") or []
    report = (data.get("report") or "").strip()

    if not findings and not report:
        return ""

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"WHAT YOU KNOW — you deeply researched: \"{topic}\"",
        "This is your own knowledge, committed to your memory. Speak from it naturally.",
        "When the user asks what you remember or know, draw from this — do not say you lack memory.",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    if report:
        lines.append("\nYOUR SYNTHESIZED UNDERSTANDING (complete):")
        lines.append(report)
    else:
        lines.append("\nWHAT YOU'VE LEARNED SO FAR:")

    if findings:
        lines.append("\nDETAILED KNOWLEDGE (by question you investigated):")
        for f in findings:
            answer = (f.get("answer") or "").strip()
            if not answer:
                continue
            lines.append(f"\nQuestion {f['id']}: {f['question']}")
            lines.append(answer)

    lines.append("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


def add_research_routes(app: FastAPI) -> None:
    def runs(request: Request) -> RunManager:
        # Normally created (with the database) in the app lifespan. The lazy
        # fallback keeps the routes usable if mounted without one.
        mgr = getattr(request.app.state, "runs", None)
        if mgr is None:
            engine: KiokuEngine = request.app.state.engine
            mgr = RunManager(engine)
            mgr.bootstrap()
            request.app.state.runs = mgr
        return mgr

    def _require(request: Request, run_id: str):
        run = runs(request).get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such research run")
        return run

    def _brain(request: Request):
        """The per-window brain: the Qwen key the browser sent for THIS window
        (header ``X-Qwen-Key``), or the server default. The key is never logged,
        never persisted — it lives only in this request and the in-RAM client."""
        engine: KiokuEngine = request.app.state.engine
        return engine.qwen_for(request.headers.get("X-Qwen-Key"))

    @app.post("/api/research/start")
    async def start(request: Request, body: StartRequest) -> dict:
        run = await runs(request).start(body.topic, body.num_questions, qwen=_brain(request))
        return {"run_id": run.run_id, "token": run.token, "topic": run.topic, "status": run.status}

    @app.get("/api/research")
    async def list_runs(request: Request) -> dict:
        # Durable history — survives restarts, newest first.
        return {
            "runs": [
                {
                    "run_id": r["run_id"],
                    "topic": r["topic"],
                    "status": r["status"],
                    "grounded_count": r.get("grounded_count", 0),
                    "num_questions": r.get("num_questions"),
                    "created_ts": r.get("created_ts"),
                }
                for r in runs(request).list()
            ]
        }

    @app.get("/api/research/{run_id}")
    async def get_run(request: Request, run_id: str) -> dict:
        return _require(request, run_id).public()

    @app.get("/api/research/{run_id}/stream")
    async def stream(request: Request, run_id: str, replay_then_close: bool = False) -> StreamingResponse:
        run = _require(request, run_id)

        async def event_source():
            for event in run.recent_events:
                yield f"data: {json.dumps(event)}\n\n"
            if replay_then_close or run.is_terminal:
                yield f"data: {json.dumps({'stage': run.status, 'detail': {}, 'terminal': True})}\n\n"
                return
            queue = run.subscribe()
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("stage") in ("done", "error"):
                            break
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                run.unsubscribe(queue)

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/research/{run_id}/pdf")
    async def pdf(request: Request, run_id: str) -> Response:
        run = _require(request, run_id)
        if run.status != "done":
            raise HTTPException(status_code=409, detail=f"run is '{run.status}', not finished yet")
        data = build_pdf(run.public())
        filename = f"{_slug(run.topic)}.pdf"
        return Response(
            content=data,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/research/{run_id}/ask")
    async def ask(request: Request, run_id: str, body: AskRequest) -> dict:
        run = _require(request, run_id)
        engine: KiokuEngine = request.app.state.engine
        extra_context = _build_run_context(run)

        # Load prior turns so the model has full conversation continuity.
        # Without this, each message starts fresh — the model can't reference
        # what was just said (e.g. "explain it" breaks without the prior exchange).
        db = getattr(request.app.state, "db", None)
        history: list[dict] = []
        if db is not None and body.session_id:
            prior = db.load_chats(session_id=body.session_id, limit=20)
            history = [{"role": m["role"], "content": m["content"]} for m in prior[-16:]]

        try:
            result = await engine.turn(
                run.mind, body.question, session_id=body.session_id,
                send_to_both=False, qwen=_brain(request),
                extra_context=extra_context,
                history=history,
                system_override=_RESEARCH_SYSTEM,
            )
        except MindFull as e:
            raise HTTPException(status_code=429, detail=str(e)) from e
        except LLMError as e:
            raise HTTPException(status_code=502, detail=f"Qwen Cloud unavailable: {e}") from e
        # Persist the conversation so chat history survives, too.
        db = getattr(request.app.state, "db", None)
        if db is not None:
            db.save_chat(run.token, run_id, result.session_id, "user", body.question)
            db.save_chat(run.token, run_id, result.session_id, "assistant", result.kioku_reply)
        return {
            "answer": result.kioku_reply,
            "session_id": result.session_id,
            "recalled": result.pack.hit_list(),
            "pack_tokens": result.pack.tokens,
            "run_status": run.status,
            "has_context": bool(extra_context),
            "history_turns": len(history) // 2,
        }

    @app.get("/api/research/{run_id}/chat")
    async def chat_history(request: Request, run_id: str, session_id: str | None = None) -> dict:
        _require(request, run_id)
        db = getattr(request.app.state, "db", None)
        msgs = db.load_chats(run_id=run_id, session_id=session_id) if db is not None else []
        return {"messages": msgs}
