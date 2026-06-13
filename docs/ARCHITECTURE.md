# Kioku v1 — Architecture

![Kioku architecture](architecture.png)

_Source: [`architecture.mmd`](architecture.mmd) (Mermaid) · the PNG is generated
by [`_gen_diagram.py`](_gen_diagram.py), no toolchain required._

Kioku gives any LLM a **living memory**. It does not store messages — it
*understands* every exchange, decomposing it into meaning, intent, keywords,
preferences and self-researched definitions, then committing the result into a
**Cadran virtual-hardware** substrate where retrieval is a shift+mask jump.

One box per component.

## Web Arena — `web/`
A dependency-free single-page app. Two chat panes share one input: **Qwen +
Kioku** (memory on) and **Qwen raw** (memory off) — same model, same key, only
the memory differs, so the comparison *is* the pitch. A Memory Inspector drawer
shows a memory forming live (SSE stage chips ending in its physical address),
browses engrams, lists curiosity's lexicon, charts retention/forgetting, and
reads the substrate gauge.

## FastAPI Engine — `engine/`
- **`main.py`** — routes and the SSE pipeline stream; per-IP rate limits
  (slowapi); CORS locked to the web origin.
- **`tenants.py`** — the shared mind and the newborns (§6), and `KiokuEngine.turn`:
  recall → build pack → answer (with memory and raw) → understand → commit →
  curiosity + consolidation in the background.
- **`decompose.py`** — one structured Qwen call turns an exchange into an engram.
- **`curiosity.py`** — Kioku looks up words it has never seen, in *your* context.
- **`retrieve.py`** — hybrid recall (shift+mask keyword cells, bounded vector
  window, session recency) and the token-budgeted memory pack.
- **`forget.py`** — retention decay, consolidation, contradiction supersession.
- **`qwen.py`** — the Qwen Cloud client. **`metrics.py`** — latency/percentiles.

## Qwen Cloud — the brain (`engine/qwen.py`)
**Every** LLM call goes to Qwen models on the Model Studio OpenAI-compatible
endpoint. Qwen is used four sophisticated ways, not just to chat:
1. **Structured decomposition** — JSON-mode extraction of meaning/intent/
   keywords/entities/preferences/importance from each exchange (one round trip,
   strict schema, one repair retry).
2. **Curiosity** — on-demand, context-grounded definitions of novel terms.
3. **Consolidation** — summarizing clusters of fading memories into one durable
   semantic memory.
4. **Embeddings** — `text-embedding-v3` over `meaning + keywords` for the vector
   pages, and over each query for recall.

The answer itself is a fifth call, with the recalled memory pack injected into
the system prompt.

## Cadran virtual hardware — `substrate/`
The five provided files (`cadran_vram.rs`, `cadran_vgpu.rs`, `cadran_storage.rs`,
`space.rs`, plus README) are the stable substrate: a 1 TiB sparse vRAM universe,
a 4 TiB virtual disk, and **Spaces** (one mind = one planet, isolated and
ceiling-bounded). Kioku adds:
- **`kiokud.rs`** — a long-lived daemon that owns one `TheBox` and serves a tiny
  newline-JSON line protocol over a Unix socket (`open_space`, `put`/`get`/`scan`
  cells, `put_blob`/`get_blob`, `check_budget`, `stats`, `release_space`).
- **`engine/store.py`** — the client, plus **`PyStore`**, a pure-Python fallback
  with identical planet/segment/mask arithmetic and a byte-compatible virtual
  disk format, so the demo runs even without `rustc`. The Rust path is the
  headline; the fallback keeps it alive.

Addressing discipline lives in [`MEMORY_MODEL.md`](MEMORY_MODEL.md): a keyword's
index cell is `hash64(keyword) & PLANET_CELL_MASK` — lookup is one shift+mask
jump, never a search.

## A turn, end to end
1. **Recall** — extract query keywords locally, embed the query (one cheap call),
   hybrid-recall candidate engrams, score them, and pack the best into ≤1200
   tokens.
2. **Answer** — inject the pack into Qwen's system prompt; (optionally) also
   answer with no memory for the side-by-side.
3. **Understand** — decompose the new exchange into an engram; supersede any
   contradicted preference.
4. **Commit** — blob to the virtual disk, index cells to the vRAM planet.
5. **Background** — curiosity defines novel terms; every N turns, consolidation
   compresses fading memories and reclaims disk via space release.

Every step emits an SSE event, so the inspector shows the memory forming in real
time. Numbers in [`../eval/METRICS.md`](../eval/METRICS.md) are measured, never
fabricated.
