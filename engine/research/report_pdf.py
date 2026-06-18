# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Guardianity — Kioku v1 · Researcher
"""Render a finished research run to a downloadable PDF — the whole thing, uncut.

Pure Python (fpdf2), no system dependencies, so "press download and it downloads
the complete file" is a single in-process call: ``build_pdf(run) -> bytes``.

The PDF carries the full final report (rendered from its markdown), then an
appendix with every one of the ~20 sub-questions, its researched finding, and the
sources it was grounded in — nothing summarized away.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from fpdf import FPDF

# Core PDF fonts are latin-1; map the unicode the model commonly emits, drop the rest.
_REP = {
    "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "•": "-", "→": "->",
    "×": "x", "≈": "~", "…": "...", "≥": ">=", "≤": "<=", "·": "-", "±": "+/-",
    "°": " deg", "←": "<-", "™": "(TM)", "©": "(c)", " ": " ",
}


def _s(txt: Any) -> str:
    if txt is None:
        return ""
    t = str(txt)
    for k, v in _REP.items():
        t = t.replace(k, v)
    t = "".join(ch for ch in t if ch == "\n" or ch >= " ")
    return t.encode("latin-1", "replace").decode("latin-1")


class _Report(FPDF):
    title_text = "Kioku research report"

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, _s(f"Kioku Researcher  ·  Guardianity  ·  page {self.page_no()}"), align="C")
        self.set_text_color(0, 0, 0)


# brand-ish palette (amber / rust on paper)
_AMBER = (217, 142, 43)
_RUST = (155, 70, 40)
_INK = (30, 28, 26)
_GREY = (120, 120, 120)
_GREEN = (70, 110, 70)


def build_pdf(run: dict) -> bytes:
    topic = run.get("topic") or "Research report"
    report_md = run.get("report") or ""
    findings = run.get("findings") or []
    provider = run.get("provider") or "web"
    grounded = run.get("grounded_count", 0)

    pdf = _Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    pdf.add_page()
    W = pdf.epw

    def block(text: str, h: float = 6, size: float = 11, style: str = "", color=_INK) -> None:
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        txt = _s(text) or " "
        try:
            pdf.multi_cell(W, h, txt)
        except Exception:  # never let one bad line kill the report
            try:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(W, h, txt[:1500] or " ")
            except Exception:
                pass
        pdf.set_text_color(*_INK)

    def rule() -> None:
        pdf.set_draw_color(*_AMBER)
        pdf.set_line_width(0.6)
        y = pdf.get_y() + 1
        pdf.line(pdf.l_margin, y, pdf.l_margin + W, y)
        pdf.ln(4)

    # -- cover block --
    block(topic, 10, 22, "B", _RUST)
    block(
        "Autonomous research by Kioku  ·  Qwen brain + Cadran memory  ·  "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        5, 9, "I", _GREY,
    )
    block(f"{len(findings)} sub-questions researched  ·  {grounded} grounded in live sources  ·  via {provider}",
          5, 9, "", _GREY)
    pdf.ln(2)
    rule()

    # -- the report, rendered from markdown --
    _render_markdown(pdf, report_md, block)

    # -- appendix: every sub-question, finding, sources --
    pdf.add_page()
    block("Appendix — the full research trail", 9, 16, "B", _RUST)
    block("Every sub-question, its researched finding, and the sources it rests on.", 5, 9, "I", _GREY)
    pdf.ln(2)
    rule()
    for f in findings:
        tag = "" if f.get("grounded") else "  [unsourced]"
        block(f"Q{f.get('id')}.  {f.get('question', '')}{tag}", 6, 12, "B", _AMBER)
        pdf.ln(0.5)
        block(f.get("answer") or "(no finding)", 5.4, 10)
        srcs = f.get("sources") or []
        if srcs:
            pdf.ln(0.5)
            for s in srcs:
                block(f"  - {s.get('title') or s.get('domain')}  <{s.get('url')}>", 4.6, 8.5, "I", _GREEN)
        pdf.ln(4)

    pdf.ln(4)
    block("Made by Guardianity   ·   contact: balingenensiidan@gmail.com", 5, 9, "", _GREY)
    block("We build what shouldn't exist. Because we dream of it.", 5, 9, "I", _GREY)
    return bytes(pdf.output())


_H2 = re.compile(r"^\s{0,3}##\s+(.*)$")
_H3 = re.compile(r"^\s{0,3}###\s+(.*)$")
_H1 = re.compile(r"^\s{0,3}#\s+(.*)$")
_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
_NUM = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")


def _strip_inline(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


def _render_markdown(pdf: FPDF, md: str, block) -> None:
    """A small, forgiving markdown renderer — headings, bullets, numbered lists,
    paragraphs. Good enough for the model's report; never raises."""
    if not md.strip():
        block("(the report came back empty)", 6, 11, "I", _GREY)
        return
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            pdf.ln(2)
            continue
        m = _H2.match(line)
        if m:
            pdf.ln(2)
            block(_strip_inline(m.group(1)), 8, 14, "B", _RUST)
            continue
        m = _H3.match(line)
        if m:
            block(_strip_inline(m.group(1)), 6.5, 12, "B", _AMBER)
            continue
        m = _H1.match(line)
        if m:
            pdf.ln(2)
            block(_strip_inline(m.group(1)), 9, 16, "B", _RUST)
            continue
        m = _BULLET.match(line)
        if m:
            block("  -  " + _strip_inline(m.group(1)), 5.6, 11)
            continue
        m = _NUM.match(line)
        if m:
            block(f"  {m.group(1)}.  " + _strip_inline(m.group(2)), 5.6, 11)
            continue
        block(_strip_inline(line), 6, 11)
