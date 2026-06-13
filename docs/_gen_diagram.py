#!/usr/bin/env python3
"""Render docs/architecture.png with PIL — no mermaid toolchain needed.

A hand-laid version of architecture.mmd in the Kioku hardware-dusk palette.
Run: python docs/_gen_diagram.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 900
BG = (11, 13, 16)
INK = (230, 237, 243)
DIM = (147, 161, 176)
LINE = (35, 42, 51)
AMBER = (245, 182, 89)
TEAL = (79, 209, 197)
VIOLET = (176, 140, 255)
GREEN = (95, 211, 139)


def font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
        "/usr/share/fonts/truetype/liberation/LiberationSans%s.ttf" % ("-Bold" if bold else ""),
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


img = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(img)
def cjk_font(size: int):
    for p in ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return None


F_TITLE = font(30, True)
F_CJK = cjk_font(34)
F_H = font(18, True)
F = font(14)
F_S = font(12)


def text_center(box, s, fnt, fill):
    x0, y0, x1, y1 = box
    tw = d.textlength(s, font=fnt)
    asc, desc = fnt.getmetrics()
    d.text(((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - (asc + desc) / 2), s, font=fnt, fill=fill)


def box(xy, title, lines, accent, fill=(22, 27, 34)):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=14, fill=fill, outline=accent, width=2)
    d.text((x0 + 16, y0 + 12), title, font=F_H, fill=accent)
    yy = y0 + 40
    for ln in lines:
        d.text((x0 + 16, yy), ln, font=F_S, fill=DIM)
        yy += 18
    return xy


def arrow(p1, p2, color=AMBER, label=None, dashed=False, width=2):
    d.line([p1, p2], fill=color, width=width)
    import math

    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    L = 10
    for da in (2.6, -2.6):
        d.line([p2, (p2[0] - L * math.cos(ang + da), p2[1] - L * math.sin(ang + da))], fill=color, width=width)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        tw = d.textlength(label, font=F_S)
        d.rectangle([mx - tw / 2 - 4, my - 9, mx + tw / 2 + 4, my + 9], fill=BG)
        d.text((mx - tw / 2, my - 8), label, font=F_S, fill=color)


# Title
tx = 40
if F_CJK is not None:
    d.text((tx, 24), "記憶", font=F_CJK, fill=AMBER)
    tx += int(d.textlength("記憶", font=F_CJK)) + 16
d.text((tx, 28), "Kioku v1", font=F_TITLE, fill=AMBER)
d.text((tx + int(d.textlength("Kioku v1", font=F_TITLE)) + 16, 36),
       "a living memory for any LLM, on Cadran virtual hardware", font=F, fill=DIM)

# Browser layer
box((40, 90, 470, 200), "Web Arena  (web/)", [
    "Dual chat panes — Qwen+Kioku vs Qwen raw",
    "Memory Inspector — pipeline · memory · lexicon",
    "          forgetting · substrate gauge",
], TEAL)

# Engine layer
box((40, 250, 720, 470), "FastAPI Engine  (engine/)", [
    "main.py     — routes + SSE pipeline stream",
    "tenants.py  — shared mind + newborns, the turn loop",
    "decompose.py — understand each exchange into an engram",
    "curiosity.py — self-research unknown terms (async)",
    "retrieve.py  — hybrid recall + memory pack (token-budgeted)",
    "forget.py    — decay · consolidation · supersession",
    "metrics.py   — retrieval p50/p95 · pack tokens",
], AMBER)

# Qwen layer
box((790, 250, 1240, 470), "Qwen Cloud — the brain  (qwen.py)", [
    "chat: qwen-max / qwen-plus",
    "   decompose · answer · consolidate · define",
    "",
    "embeddings: text-embedding-v3",
    "   meaning + keywords  →  vector pages",
    "",
    "OpenAI-compatible · async httpx · retries",
], GREEN)

# Substrate layer
box((40, 530, 720, 830), "Cadran virtual hardware  (substrate/)", [
    "store.py seam  →  kiokud.rs daemon",
    "        (Unix socket, newline-JSON line protocol)",
    "",
    "vRAM  1 TiB sparse  ·  keyword cells at hash64(term) & MASK",
    "virtual disk  4 TiB  ·  CRC-verified engram blobs",
    "Spaces — one mind, one planet: isolated + ceiling-bounded",
    "",
    "PyStore — pure-Python fallback, identical arithmetic",
], VIOLET)

# Numbers callout
box((790, 530, 1240, 830), "Measured  (eval/METRICS.md)", [
    "cross-session recall      100%  (5/5 probes)",
    "retrieval p95             33 ms  over 10,000 engrams",
    "                          budget <= 50 ms   (PASS)",
    "memory pack median        105 tokens  (budget 1200)",
    "",
    "Track 1 requirements:",
    "  persistent multi-turn + cross-session memory",
    "  efficient storage & retrieval (shift+mask)",
    "  timely forgetting (decay + consolidation)",
    "  recall within a limited context window",
], AMBER)

# Arrows
arrow((255, 200), (255, 250), TEAL, "HTTP / SSE")
arrow((720, 330), (790, 330), GREEN, "chat")
arrow((720, 400), (790, 400), GREEN, "embed")
arrow((380, 470), (380, 530), VIOLET, "store / recall")
arrow((720, 680), (790, 680), AMBER, "honest numbers", width=1)

# Footer
d.text((40, 858), "Kioku v1 — built on Cadran virtual hardware · Guardianity", font=F_S, fill=DIM)

out = Path(__file__).resolve().parent / "architecture.png"
img.save(out)
print("wrote", out)
