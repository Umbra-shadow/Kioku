# Kioku Researcher

> One question in. A complete, sourced report out. And it remembers everything it
> went through — forever — so you can keep asking.

The Researcher turns Kioku from a *memory* into a *mind that investigates*. It is
built directly on the existing engine (the Qwen brain + the Cadran memory), and
adds four things: a live web pull, a research orchestrator, a downloadable PDF,
and a durable per-user memory.

## The loop

1. **Expand** — one top-level question (e.g. *"How can we cure blindness?"*) is
   broken into ~20 deep, independently-researchable sub-questions.
   ([`researcher.py`](../engine/research/researcher.py) `expand`)
2. **Study** — each sub-question is researched **live from the web**
   ([`websearch.py`](../engine/research/websearch.py): free DuckDuckGo, or Tavily
   if `TAVILY_API_KEY` is set), the pages are read, and Qwen synthesizes a
   grounded finding — established fact separated from hypothesis, every claim
   cited `[S1]`. If the web returns nothing, the finding is still written, clearly
   tagged `[model-knowledge, unsourced]` — it never fabricates a source.
3. **Remember** — every finding is committed into a Kioku mind as it completes, so
   nothing is lost and the whole run is recallable.
4. **Synthesize** — all findings are combined into one complete, *uncut* report
   (overview · established · tried · gaps · procedures & methodologies · directions
   · risks · bottom line), which you can **download as a PDF**
   ([`report_pdf.py`](../engine/research/report_pdf.py), pure-python).

## Talk to it while it works

The Researcher window is a split screen: research on the left, a **live chat on the
right**. Because each finding lands in memory the moment it's researched, you can
ask *"what did you find on question 2?"* **while the run is still going**, and it
answers from memory. Start a new chat session and it still recalls the whole
investigation — memory is **per user, not per session**.

## Memory is durable and per-user

All runs and chats write into one persistent mind (the `researcher` tenant), not a
fresh one per run. That memory — plus the run and chat history — is written to a
database ([`persistence.py`](../engine/research/persistence.py)):

- **`DATABASE_URL`** set (a `postgres://` DSN, e.g. **Neon**) → durable cloud
  Postgres, surviving redeploys and machines.
- otherwise → a local **SQLite** file (`KIOKU_RESEARCH_DB`, default
  `kioku_data/research.db`).

On startup the engine **rehydrates** the in-RAM index from the stored engrams, so a
restart loses nothing and a brand-new session — even years later — recalls a run
from long ago.

## Bring your own key (per window)

The Researcher window has a field to enter **your own Qwen API key**. It is stored
in `sessionStorage` — it lives only in that browser window and is wiped the moment
you close it — and is sent per request as the `X-Qwen-Key` header. The backend uses
that key as the brain for that window (a per-key client kept only in RAM), and
never logs or persists it. `QWEN_API_KEY` on the server is just the default
fallback.

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/research/start` | begin an investigation (background); `{topic, num_questions}` |
| GET  | `/api/research` | run history (durable, newest first) |
| GET  | `/api/research/{id}` | full run state + report |
| GET  | `/api/research/{id}/stream` | SSE: live research progress |
| GET  | `/api/research/{id}/pdf` | download the complete report as PDF |
| POST | `/api/research/{id}/ask` | follow-up, recalled from the run's memory |
| GET  | `/api/research/{id}/chat` | persisted chat history |

All LLM-touching routes accept an optional `X-Qwen-Key` header (the per-window key).

## Try it

```bash
make demo                 # serves the arena + API at :8000
# open http://localhost:8000/research.html
# enter your Qwen key, ask one big question, watch ~20 get researched,
# chat with it live, download the PDF, then close the tab — the key is gone,
# but the research is remembered.
```

Tests are fully offline (a fake brain + fake web + SQLite `:memory:`):
`pytest engine/tests/test_research.py`.
