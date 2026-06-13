# Kioku v1 — 3-minute demo video script

One browser tab (the arena), one terminal (optional, for the substrate gauge).
Beats with exact timestamps. The whole pitch: **same Qwen model, same key on
both sides — only the memory differs.**

> Setup before recording: `cp .env.example .env`, set `QWEN_API_KEY`, run
> `make demo`, open `http://localhost:8080`. Make sure the "⇉ send to both"
> toggle is ON. Use the **shared mind** (default).

---

### 0:00 — The hook (do they remember you?)
- Type into the shared input: **"Do you remember me?"** Hit send.
- **Qwen raw** (right): "I don't have memory of past conversations…"
- **Qwen + Kioku** (left): also nothing *yet* — "We haven't talked before."
- Line: *"Same model. Same key. Right now they're equal. Watch what twenty
  seconds of memory does."*

### 0:30 — Teach it, across a session
Send each (to both):
- **"My name is Aiko, I live in Kyoto, and I'm allergic to peanuts."**
- **"I love hanami and I'm planning a flower-viewing trip this June."**
- **"I prefer quiet temples over crowded tourist spots."**
- As each lands, glance at the **Inspector → Pipeline**: the chips light
  `captured → decomposed → embedded → curious → committed`, ending in a real
  address like `planet 1 · segment 12 · cell 0x9F3A`.
- Line: *"It isn't storing my messages — it's understanding them. Watch it look
  up a word it didn't know."* Point at the **curious** chip / Lexicon tab.

### 1:15 — A new session, cross-session recall
- Click nothing that resets the mind. (A real demo can reload the page — the
  tenant token persists in localStorage; the session id rolls.)
- Type the **🔎 recall probe** button, or: **"What do you remember about me?"**
- **Qwen raw**: generic "I don't know anything about you."
- **Qwen + Kioku**: *"You're Aiko from Kyoto, you're allergic to peanuts, you
  love hanami and you're planning a quiet-temple trip this June."*
- Point at the dashed **memory pack** line above Kioku's answer:
  *"~110 tokens, 8 memories recalled — that's the whole context it needed."*

### 1:45 — The inspector: a memory forming + curiosity
- Open **Inspector → Memory**: the engram table — meaning, intent, importance,
  retention, address. Click one → full JSON, re-read straight from the virtual
  disk.
- Open **Lexicon**: the definitions curiosity locked, e.g. *hanami* — *"Japanese
  cherry-blossom viewing, which Aiko enjoys."*

### 2:20 — Forgetting and consolidation
- Open **Inspector → Forgetting**: retention bars, weakest first (the small-talk
  decays fastest).
- Click **⟳ Consolidate now**. Toast: *"Consolidated 3 → 1, reclaimed 768 KiB."*
- Line: *"Old, low-value memories get summarized into one durable semantic
  memory, the originals tombstoned, and the disk blocks physically reclaimed —
  forgetting, on purpose."*

### 2:45 — Substrate gauge + close
- Open **Inspector → Substrate**: *"3.1 MiB committed of 1 TiB virtual — small
  outside, huge inside,"* retrieval p50/p95, pack token size, backend `kiokud`.
- Close on the footer: **"Kioku v1 — built on Cadran virtual hardware ·
  Guardianity."**
- Final line: *"Persistent across sessions, fast to recall, honest about
  forgetting — a living memory for any LLM. This is v1 of a much bigger system."*

---

**If Qwen is unreachable mid-demo:** the UI shows a clear error, never a crash;
switch to the committed `eval/METRICS.md` numbers and the inspector, which work
from stored memory.
