# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""The researcher — Kioku turned from a memory into a mind that investigates.

Give it one question — "How can we cure blindness?" — and it:

  1. EXPANDS it into ~20 genuinely deep sub-questions (one Qwen call).
  2. STUDIES each sub-question: pulls live sources from the Internet
     (``WebSearch``), reads them, and synthesizes a grounded finding that
     separates established fact from hypothesis and cites what it rests on.
  3. REMEMBERS every finding into a Kioku ``Mind`` — so nothing is lost and the
     whole run can be recalled and questioned afterwards, no matter how large.
  4. SYNTHESIZES everything into one complete, *uncut* report — procedures,
     methodologies, directions — drawing on the memory it just built.

The honesty law is inherited from Foreman: never fabricate a source, a number,
or a result; label every speculation a hypothesis. When the network is down the
finding is still written, clearly tagged ``[model-knowledge, unsourced]``.

This module is pure orchestration over the existing engine — it owns no storage
of its own. ``progress`` is an async callback the run manager uses to stream the
work live to the browser.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from engine.engram import new_ulid
from engine.qwen import LLMError, QwenClient
from engine.research.websearch import Source, WebSearch
from engine.tenants import KiokuEngine, Mind

log = logging.getLogger("kioku.research")

Progress = Callable[[str, dict], Awaitable[None]]

DEFAULT_NUM_QUESTIONS = 20
MAX_SOURCES_PER_Q = 4
_STUDY_CONCURRENCY = 4  # how many sub-questions are researched at once


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Finding:
    id: int
    question: str
    answer: str = ""
    sources: list[Source] = field(default_factory=list)
    grounded: bool = False  # True if at least one live source backed it

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "grounded": self.grounded,
            "sources": [{"url": s.url, "title": s.title, "domain": s.domain} for s in self.sources],
        }


# ── prompts ──────────────────────────────────────────────────────────────────

_EXPAND_SYSTEM = """You are the research-planning stage of Kioku, an autonomous \
research engine. You receive ONE top-level question. Break it into deep, distinct \
sub-questions that together would let a careful analyst answer the original \
completely.

Rules:
- Each sub-question must be SPECIFIC and INDEPENDENTLY RESEARCHABLE (a real
  search could find sources for it).
- Cover the whole problem: mechanisms/causes, what has been tried, the current
  state of the art, the gaps and open problems, methods and procedures, risks
  and constraints, and the most promising directions.
- No yes/no questions. No duplicates. No vague "what about X?".
- Order them from foundational to frontier.

Respond with ONLY a JSON object: {"subquestions": ["...", "...", ...]} with \
exactly %(n)d items."""

_STUDY_SYSTEM = """You are the research stage of Kioku. You are answering ONE \
sub-question using the SOURCES provided (pulled live from the web). Write a \
thorough, self-contained finding.

Discipline (this is the law):
- Ground every specific claim in the sources. After a sourced claim, cite it
  inline like [S1], [S2] matching the numbered sources.
- Separate ESTABLISHED FACT from HYPOTHESIS. Label speculation clearly.
- NEVER invent a citation, a number, a study, or a result. If the sources do not
  cover something important, say so plainly and mark your own reasoning
  [inference] — do not dress a guess as fact.
- Be concrete: mechanisms, methods, procedures, quantities (with their basis).
- Length follows substance — do not pad, do not truncate. Plain prose, no preamble.

If NO sources were provided, answer from your own knowledge but begin the finding \
with "[model-knowledge, unsourced]" and stay especially careful to flag uncertainty."""

_SYNTH_SYSTEM = """You are the synthesis stage of Kioku. You have the original \
question and the full set of researched findings for its sub-questions (each \
already grounded and cited). Write the COMPLETE final report — the "everything \
put together" answer.

Requirements:
- Do NOT summarize away the substance. This is the full report, uncut: include
  the procedures, methodologies, and concrete directions in detail.
- Structure it clearly with these sections, each a markdown H2 (## ):
  ## Overview
  ## What is established
  ## What has been tried
  ## The core gaps
  ## Procedures & methodologies
  ## Promising directions
  ## Open questions & risks
  ## Bottom line
- Keep the inline [S#]/[hypothesis]/[inference] discipline from the findings.
- Honest to the end: established fact and hypothesis stay clearly separated.
Plain markdown prose. No preamble, start at "## Overview"."""


def _norm_questions(raw: object, n: int) -> list[str]:
    items: list[str] = []
    if isinstance(raw, dict):
        raw = raw.get("subquestions") or raw.get("questions") or []
    if isinstance(raw, list):
        for q in raw:
            q = str(q).strip().lstrip("0123456789.)- ").strip()
            if q and q not in items:
                items.append(q)
    return items[:n]


class Researcher:
    """One research run, against one Kioku mind."""

    def __init__(
        self,
        engine: KiokuEngine,
        mind: Mind,
        web: WebSearch | None = None,
        *,
        progress: Progress | None = None,
        qwen: QwenClient | None = None,
    ) -> None:
        self.engine = engine
        # The brain for this run: the caller's per-window key, or the server default.
        self.qwen: QwenClient = qwen or engine.qwen
        self.mind = mind
        self.web = web or WebSearch()
        self._progress = progress
        self.session_id = new_ulid()  # one session ties the whole run together in memory

    async def _emit(self, stage: str, detail: dict) -> None:
        if self._progress:
            try:
                await self._progress(stage, detail)
            except Exception:  # noqa: BLE001 — telemetry must never break the run
                log.exception("progress sink failed at %s", stage)

    # -- 1. expand --------------------------------------------------------

    async def expand(self, topic: str, n: int = DEFAULT_NUM_QUESTIONS) -> list[str]:
        await self._emit("expanding", {"topic": topic, "n": n})
        raw = await self.qwen.chat_json(
            [
                {"role": "system", "content": _EXPAND_SYSTEM % {"n": n}},
                {"role": "user", "content": f"TOP-LEVEL QUESTION:\n{topic}"},
            ],
            temperature=0.4,
            max_tokens=8192,
        )
        questions = _norm_questions(raw, n)
        if not questions:
            raise LLMError("question expansion produced no sub-questions")
        # Remember the plan with numbered questions so "what was question N?" is recallable.
        await self.engine.remember(
            self.mind,
            f"Research plan for '{topic}': {len(questions)} sub-questions",
            "Research questions:\n" + "\n".join(f"Question {i+1}: {q}" for i, q in enumerate(questions)),
            session_id=self.session_id,
            importance_floor=0.7,
            qwen=self.qwen,
        )
        await self._emit("expanded", {"questions": questions})
        return questions

    # -- 2. study one sub-question ---------------------------------------

    async def study(self, finding: Finding) -> Finding:
        await self._emit("studying", {"id": finding.id, "question": finding.question})
        sources = await self.web.gather(finding.question, k=MAX_SOURCES_PER_Q)
        finding.sources = sources
        finding.grounded = bool(sources)

        if sources:
            src_block = "\n\n".join(
                f"[S{i+1}] {s.title} ({s.domain})\n{s.text}" for i, s in enumerate(sources)
            )
            user = f"SUB-QUESTION:\n{finding.question}\n\nSOURCES:\n{src_block}"
        else:
            user = f"SUB-QUESTION:\n{finding.question}\n\nSOURCES: (none — the web pull returned nothing)"

        try:
            finding.answer = (
                await self.qwen.chat(
                    [
                        {"role": "system", "content": _STUDY_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.3,
                    max_tokens=8192,
                )
            ).strip()
        except LLMError as e:
            log.warning("study failed for q%d: %s", finding.id, e)
            finding.answer = f"[research-error] could not synthesize this finding: {e}"

        # Commit the finding into memory. Wrapped so a storage failure never
        # prevents the studied event from firing — the UI spinner must always resolve.
        cite = ""
        if sources:
            cite = "\nSources: " + " | ".join(s.cite() for s in sources)
        try:
            await self.engine.remember(
                self.mind,
                f"Research question {finding.id}: {finding.question}",
                finding.answer + cite,
                session_id=self.session_id,
                importance_floor=0.6,
                qwen=self.qwen,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("memory commit failed for q%d (finding still used): %s", finding.id, e)
        await self._emit(
            "studied",
            {"id": finding.id, "question": finding.question, "grounded": finding.grounded,
             "sources": len(sources), "answer": finding.answer},
        )
        return finding

    # -- 3. synthesize ----------------------------------------------------

    async def synthesize(self, topic: str, findings: list[Finding]) -> str:
        await self._emit("synthesizing", {"count": len(findings)})
        blocks = []
        for f in findings:
            tag = "" if f.grounded else " [unsourced]"
            blocks.append(f"### Q{f.id}{tag}: {f.question}\n{f.answer}")
        corpus = "\n\n".join(blocks)
        # Qwen context is large; if the corpus is huge, the model still handles it,
        # but keep a generous cap to stay within one call.
        _synth_max = 8192
        report = (
            await self.qwen.chat(
                [
                    {"role": "system", "content": _SYNTH_SYSTEM},
                    {"role": "user", "content": f"ORIGINAL QUESTION:\n{topic}\n\nFINDINGS:\n{corpus}"},
                ],
                temperature=0.35,
                max_tokens=_synth_max,
            )
        ).strip()
        if len(report) >= _synth_max * 3:
            log.warning("synthesis may be truncated: %d chars at max_tokens=%d", len(report), _synth_max)
        await self.engine.remember(
            self.mind,
            f"Complete research report on '{topic}' — synthesized from {len(findings)} findings",
            report,
            session_id=self.session_id,
            importance_floor=0.9,
            qwen=self.qwen,
        )
        await self._emit("synthesized", {"chars": len(report)})
        return report

    # -- the whole run ----------------------------------------------------

    async def run(self, topic: str, n: int = DEFAULT_NUM_QUESTIONS) -> dict:
        questions = await self.expand(topic, n)
        findings = [Finding(id=i + 1, question=q) for i, q in enumerate(questions)]

        # Research the sub-questions with bounded concurrency — fast, but kind to
        # the network and the rate limiter.
        sem = asyncio.Semaphore(_STUDY_CONCURRENCY)

        async def _guarded(f: Finding) -> Finding:
            async with sem:
                return await self.study(f)

        # return_exceptions=True: one slow/failing question never cancels the others.
        raw = await asyncio.gather(*(_guarded(f) for f in findings), return_exceptions=True)
        findings = [r if isinstance(r, Finding) else f for r, f in zip(raw, findings)]
        report = await self.synthesize(topic, findings)
        grounded = sum(1 for f in findings if f.grounded)
        await self._emit("done", {"grounded": grounded, "total": len(findings)})
        return {
            "topic": topic,
            "report": report,
            "questions": questions,
            "findings": [f.to_dict() for f in findings],
            "grounded_count": grounded,
            "provider": self.web.provider,
            "session_id": self.session_id,
        }

    async def aclose(self) -> None:
        await self.web.aclose()
