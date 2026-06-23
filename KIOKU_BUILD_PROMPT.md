# BUILD PROMPT — KIOKU v1

You are Claude Code. Build **Kioku v1** from top to bottom in this folder.
Kioku is a memory engine that gives any LLM API key a living memory: it
doesn't store messages, it **understands** them — decomposing every
exchange into meaning, intent, keywords, preferences, and self-researched
definitions, then committing the result into a Cadran virtual-hardware
substrate (1 TiB sparse vRAM + 4 TiB virtual disk) where retrieval is
shift+mask fast. Target: **first place, Track 1 (MemoryAgent), Global AI
Hackathon Series with Qwen Cloud**.

Track 1 verbatim requirements you must visibly nail: persistent memory
across multi-turn AND cross-session interactions; efficient memory
storage and retrieval; **timely forgetting** of outdated information;
recalling critical memories **within limited context windows**.

Judging weights: Technical Depth & Engineering 30% · Innovation & AI
Creativity 30% · Problem Value & Impact 25% · Presentation & Docs 15%.
Every architectural choice below maps to one of these. Do not cut any.

---

## 0. Hard constraints (hackathon rules — violating these disqualifies)

1. **Qwen Cloud is the brain.** All LLM calls go to Qwen models via the
   Qwen Cloud / Model Studio OpenAI-compatible endpoint
   (`QWEN_BASE_URL`, `QWEN_API_KEY`, `QWEN_MODEL` default `qwen-max` or
   `qwen-plus`, `QWEN_EMBED_MODEL` default `text-embedding-v3`). A
   secondary `GENERIC_BASE_URL/GENERIC_API_KEY` mode may exist (any
   OpenAI-compatible key gets a Kioku memory) but Qwen is default and
   what the demo shows.
2. **Public repo, open-source license.** `LICENSE` = Apache-2.0 at repo
   root, referenced in README first screen and in the GitHub About.
3. **Alibaba Cloud deployment proof.** Provide `deploy/alibaba/` with a
   working `deploy.sh` (ECS, Docker) + `PROOF.md` that links to the exact
   code files calling Qwen Cloud APIs and explains the recording to make.
4. **Architecture diagram** committed as `docs/architecture.png` (also
   keep the Mermaid source `docs/architecture.mmd`).
5. **3-minute demo video script** in `docs/DEMO_SCRIPT.md` with exact
   beats and timestamps.
6. All secrets via `.env` only (`.env.example` committed, `.env`
   gitignored). No key ever in code or logs.

---

## 1. Repo layout (create exactly this)

```
kioku/
├── LICENSE                      # Apache-2.0
├── README.md                    # see §8 — short, judge-first
├── .env.example
├── .gitignore
├── docker-compose.yml           # api + web + substrate volume
├── Makefile                     # make dev / test / eval / demo / deploy
├── substrate/                   # PROVIDED — do not rewrite, extend only
│   ├── cadran_vram.rs           # 1 TiB sparse vRAM (provided)
│   ├── cadran_vgpu.rs           # lane engine + kernels (provided)
│   ├── cadran_storage.rs        # 4 TiB virtual disk (provided)
│   ├── space.rs                 # Space/TheBox + ceiling (provided)
│   ├── kiokud.rs                # NEW — daemon: line-protocol server over
│   │                            #   the substrate (see §2)
│   └── README.md                # provided
├── engine/                      # Python 3.11+, FastAPI
│   ├── main.py                  # app factory, routes
│   ├── qwen.py                  # Qwen Cloud client (chat + embeddings)
│   ├── decompose.py             # the understanding pipeline (§3)
│   ├── curiosity.py             # the curiosity loop (§3.4)
│   ├── engram.py                # engram schema + (de)serialization
│   ├── store.py                 # talks to kiokud; fallback PyStore (§2)
│   ├── retrieve.py              # hybrid recall + memory pack (§4)
│   ├── forget.py                # decay, consolidation, tombstones (§5)
│   ├── tenants.py               # shared mind + newborn spaces (§6)
│   ├── metrics.py               # latency/recall instrumentation
│   └── tests/                   # pytest — every module
├── web/                         # single-page, vanilla HTML/CSS/JS or Next.js
│   └── …                        # the dual-pane arena + inspector (§7)
├── eval/
│   ├── fixtures/                # scripted conversations + recall probes
│   ├── run_eval.py              # produces METRICS.md
│   └── METRICS.md               # committed sample run
├── deploy/alibaba/
│   ├── deploy.sh, Dockerfile(s), PROOF.md
└── docs/
    ├── ARCHITECTURE.md, architecture.mmd, architecture.png
    ├── MEMORY_MODEL.md          # the engram, addressing, forgetting math
    └── DEMO_SCRIPT.md
```

The five provided substrate files are already in `substrate/`. They are
stable Rust, build with
`rustc --edition=2021 -C opt-level=3 cadran_vgpu.rs` and pass 23 tests.
Read their README before touching anything.

## 2. The substrate seam — `kiokud`

Write `substrate/kiokud.rs`: a long-lived daemon that owns one `TheBox`
(vRAM universe + virtual disk + spaces) and serves a tiny line protocol
over a Unix socket (newline-delimited JSON):

- `{"op":"open_space","budget":..} → {"space":N}`
- `{"op":"put","space":N,"key":u64,"cells":[...]}` — write engram index
  cells into the space's planet (keyword/postings/vector-page slots)
- `{"op":"get","space":N,"key":u64}` / `{"op":"scan","space":N,...}`
- `{"op":"put_blob","space":N,"b64":...} → {"block":B}` — full engram
  JSON persisted to the virtual disk (CRC-verified objects)
- `{"op":"get_blob","space":N,"block":B}`
- `{"op":"stats"}` — committed vs virtual bytes, per-space residency
- `{"op":"release_space","space":N}`

Addressing discipline: keyword index lives at deterministic cell
addresses — `cell = hash64(keyword) & PLANET_CELL_MASK` — so lookup is
one shift+mask jump, no search. Vector pages are appended segments.
Document the layout in `docs/MEMORY_MODEL.md`.

`engine/store.py` speaks this protocol. **Fallback:** if `rustc` is
unavailable at runtime, a `PyStore` implements the same interface with
the same planet/segment/masking arithmetic in pure Python (mmap'd file),
clearly logged as fallback. The demo must run either way; the Rust path
is the headline.

## 3. The understanding pipeline (`decompose.py`) — Kioku's heart

Every `(user_message, assistant_reply)` becomes an **engram** via one
structured Qwen call (JSON mode, one round-trip, strict schema):

```json
{
  "engram_id": "...", "tenant": "...", "user_id": "...",
  "session_id": "...", "ts": 0,
  "message": "...", "reply": "...",
  "meaning": "one-sentence what was actually said",
  "intent": "what the user is trying to achieve",
  "keywords": ["..."], "entities": ["..."],
  "preferences_delta": {"likes": [], "dislikes": [], "facts": []},
  "emotional_tone": "...",
  "importance": 0.0,
  "definitions": {},
  "embedding": [],
  "links": {"session_prev": "...", "topics": []}
}
```

3.1 **Capture** raw message + reply + ids (session_id, user_id,
message_id are ULIDs).
3.2 **Decompose** via Qwen → meaning, intent, keywords, entities,
preferences_delta, tone, importance (0–1 rubric in the prompt).
3.3 **Embed** `meaning + keywords` with `text-embedding-v3`.
3.4 **Curiosity loop** (`curiosity.py`) — the differentiator. For every
entity/keyword Kioku has never seen (one shift+mask lookup), it asks
Qwen for a 1–2 sentence definition *in this user's context*, and locks
it into the engram and into a global `lexicon` space. Like a human who
hears a new word and quietly looks it up. Budget-capped (max N
lookups/turn, configurable) and async — never blocks the reply.
3.5 **Commit**: blob → virtual disk; index cells (keywords, entities,
topic links, vector page ref) → vRAM planet; postings updated.

The pipeline emits an SSE event per stage (`captured → decomposed →
embedded → curious(term) → committed @planet/segment/cell`) so the
inspector can show memory forming live (§7).

## 4. Recall (`retrieve.py`) — fast, small, right

On every incoming message:
1. Decompose-lite (keywords + embedding of the query, one cheap call).
2. **Hybrid retrieval**: (a) keyword/entity exact hits via shift+mask
   cells; (b) cosine top-k over vector pages; (c) session-recency walk.
3. **Score** = α·similarity + β·importance + γ·recency_decay +
   δ·access_frequency (constants in config, documented).
4. **Memory pack builder**: assemble retrieved engrams into a strict
   token budget (default 1,200 tokens): dedupe, collapse near-duplicates,
   order by score, render as compact structured context (preferences
   block + relevant engrams + lexicon definitions). This is the "recall
   critical memories within limited context windows" requirement —
   measured and shown in the UI (pack size in tokens, hit list).
5. Inject pack into the Qwen system prompt; answer; then §3 runs on the
   new exchange. Touched engrams get `access_count += 1` (reinforcement).

Latency budget: retrieval ≤ 50 ms p95 on 10k engrams (measured in eval).

## 5. Forgetting (`forget.py`) — the part everyone else skips

- Each engram carries `retention = importance × e^(−λ·age) ×
  log(1+access_count)`; λ per memory class (preferences decay slowest,
  small-talk fastest).
- A background **consolidation** job (every M turns): clusters of old
  low-retention episodic engrams are summarized by Qwen into one
  semantic engram ("user spent June planning a flower-viewing trip"),
  originals tombstoned; disk blocks reclaimed via space release —
  show reclaimed bytes in the inspector.
- Contradiction handling: a new preference that conflicts with an old
  one supersedes it (old engram tombstoned with `superseded_by`).
- Everything observable: the inspector has a "Forgetting" tab showing
  retention scores and the last consolidation diff.

## 6. Tenancy — the shared mind and the newborns

- Default: **one shared mind** (`tenant = "kioku"`, one Space). Everyone
  who talks to the demo feeds the same memory — 25,000 people, one self.
- "**New mind**" button: creates a fresh tenant → `open_space` → a
  newborn with empty memory, isolated by Cadran space isolation
  (budgeted, ceiling-enforced). No accounts, no auth — a tenant token in
  localStorage. Rate-limit per IP (slowapi) and per-tenant message cap.

## 7. The web arena (`web/`) — what the judges see

One focused page, mobile-first (breakpoints 640/768/1024/1280, no
horizontal scroll at 360px, inputs ≥44px, aria-labels):

- **Dual chat panes**: left "Qwen + Kioku", right "Qwen raw" (no
  memory). One input bar with a `⇉ send to both` toggle (default on) —
  the comparison IS the pitch. Same model, same key, only Kioku differs.
- **Memory Inspector drawer** (right side / bottom sheet on mobile):
  - *Live pipeline*: the SSE stage chips lighting up as an engram forms,
    ending with its physical address `planet 1 · segment 12 · cell 0x9F3A`.
  - *Memory browser*: searchable table of engrams (message, meaning,
    intent, keywords, importance, retention, address), click → full JSON.
  - *Lexicon*: every curiosity definition Kioku has locked.
  - *Forgetting tab*: retention curves, last consolidation, reclaimed bytes.
  - *Substrate gauge*: committed vs virtual (e.g. "3.1 MiB of 1 TiB —
    small outside, huge inside"), retrieval p50/p95, pack token size.
- A "recall probe" quick-button: asks "what do you remember about me?"
  to both panes simultaneously. This is the money shot for the video.

## 8. Docs — judge-first, minimum words

- **README.md** (≤120 lines): name + one-line thesis; 30-second GIF
  placeholder; Track 1 mapping table (requirement → where in code);
  quickstart (`cp .env.example .env && make dev`, ≤6 commands);
  architecture diagram embed; license badge; link to METRICS.md.
- **ARCHITECTURE.md**: the diagram + one paragraph per box; the Qwen
  Cloud touchpoints highlighted (judging asks for sophisticated Qwen use).
- **MEMORY_MODEL.md**: engram schema, addressing math, scoring formula,
  forgetting math — short, precise, with the constants.
- **DEMO_SCRIPT.md**: 0:00 hook (ask both panes "do you remember me?" —
  raw fails, Kioku answers); 0:30 teach it things across a session;
  1:15 new session, cross-session recall; 1:45 inspector: watch an
  engram form live + curiosity lookup; 2:20 forgetting/consolidation;
  2:45 substrate gauge + close ("v1 of a bigger system").
- **PROOF.md**: exact files using Qwen Cloud APIs + ECS recording steps.

## 9. Engineering bar (non-negotiable)

- Python: type hints everywhere, pydantic models, async httpx with
  timeouts + retries (tenacity), structured logging (no message bodies
  at INFO), input validation at boundaries, proper status codes,
  pagination on list endpoints, CORS locked to the web origin.
- Rust: stable only, zero warnings, all unsafe interior, tests pass.
- Tests: pytest for decompose (mock Qwen), retrieve scoring, forget
  math, store protocol, tenant isolation; `make test` green.
- Eval: `make eval` runs fixtures → cross-session recall accuracy %,
  retrieval p50/p95 ms, pack token sizes, with/without-memory answer
  comparison — written to `eval/METRICS.md`. Commit one honest run.
- Determinism where possible (seeded), graceful degradation when Qwen
  is unreachable (clear error in UI, never a crash).
- No fabricated numbers anywhere. Every claim in docs is measured or
  marked TODO.

## 10. Build order (do it in this order, commit per step)

1. Substrate daemon `kiokud.rs` + `store.py` + PyStore fallback + tests
2. Engram + decompose + curiosity (mocked Qwen in tests) + qwen.py
3. Retrieve + memory pack + metrics
4. Forget + consolidation
5. FastAPI routes + SSE inspector stream + tenants + rate limiting
6. Web arena (dual pane + inspector)
7. Eval harness + METRICS.md
8. Docs + diagram + deploy/alibaba + DEMO_SCRIPT
9. Final pass: README polish, license headers, `make demo` one-command.

Name everything Kioku v1. The footer of the web page reads:
**Kioku v1 — built on Cadran virtual hardware · Guardianity**.
