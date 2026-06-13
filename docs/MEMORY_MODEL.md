# Kioku v1 — The Memory Model

The engram, the addressing, the scoring, the forgetting math — short and exact.
Constants live in [`engine/config.py`](../engine/config.py) and
[`engine/retrieve.py`](../engine/retrieve.py).

## 1. The engram

One `(user_message, assistant_reply)` exchange becomes one **engram** — not the
transcript, the understanding of it. Schema (`engine/engram.py`):

```jsonc
{
  "engram_id": "ULID", "tenant": "...", "user_id": "...", "session_id": "...",
  "ts": 0.0,
  "message": "...", "reply": "...",
  "meaning": "one sentence: what was actually said",
  "intent": "what the user is trying to achieve",
  "keywords": ["..."], "entities": ["..."],
  "preferences_delta": { "likes": [], "dislikes": [], "facts": [] },
  "emotional_tone": "...", "importance": 0.0,           // 0..1
  "definitions": { "term": "curiosity's definition" },  // §3.4
  "embedding": [/* text-embedding-v3 */],
  "links": { "session_prev": "ULID", "topics": ["..."] },
  "memory_class": "preference|semantic|episodic|smalltalk",
  "access_count": 0, "tombstoned": false, "superseded_by": null
}
```

Keywords and entities are normalized (NFKC + casefold + single spaces), deduped,
and capped, so one concept has one spelling and therefore one cell.

## 2. Addressing — shift+mask, never search

The substrate geometry (mirrors `cadran_vram.rs`):

```
PLANET_BITS = 14          NUM_PLANETS        = 16,384   (one per mind)
PLANET_CELL_BITS = 22     CELLS_PER_PLANET   = 4,194,304
SEGMENT_CELL_BITS = 14    CELLS_PER_SEGMENT  = 16,384    (fault granularity)
CELL_BYTES = 16           PLANET_CELL_MASK   = CELLS_PER_PLANET - 1
```

**Keyword index cell** — the one rule both the daemon and the engine agree on:

```
cell  =  hash64(term) & PLANET_CELL_MASK          # FNV-1a 64-bit
addr  = (planet << PLANET_CELL_BITS) | cell        # absolute universe address
```

So "have I seen this term?" and "which memories mention it?" are a single
shift+mask jump into the planet — no scan. The cell stores `(importance,
postings_count, latest_blob_block)`; the postings expansion (term → engram ids)
and the vector pages are an in-process accelerator, rebuildable from the disk
blobs.

**Engram blob** — the full engram JSON is a CRC-verified object appended to the
mind's 256 MiB disk room (`cadran_storage.rs`), returning a `(block, len)`
handle. **A mind is one planet**: isolation and the host-safe ceiling are
enforced by the substrate's `Space`, not by Kioku.

## 3. Retrieval score

For each candidate engram (config constants α, β, γ, δ):

```
score = α·similarity + β·importance + γ·recency_decay + δ·access_frequency
        α=0.55         β=0.20         γ=0.15            δ=0.10

similarity        = max(0, cosine(query_vec, engram_vec))
recency_decay     = e^(−λ_r · age_days),                λ_r = 0.069  (~10-day half-life)
access_frequency  = min(1, ln(1+access_count) / ln(1+8))
```

**Hybrid recall** unions three sources: (a) exact keyword/entity hits via
shift+mask cells (unbounded reach, O(1) per term); (b) cosine over a **bounded
recent vector window** (default 1024) so latency does not grow with the corpus;
(c) a session-recency walk. Touched engrams get `access_count += 1`
(reinforcement).

**Memory pack** (the "recall within a limited context window" requirement):
dedupe and collapse near-duplicates, order by score, and render
`preferences block + ranked memories + lexicon definitions` into a strict token
budget (default **1200**, ~4 chars/token). Measured: median ~105 tokens over the
eval probes.

## 4. Forgetting

Each engram's **retention**:

```
retention = importance · e^(−λ_class · age_days) · max(1, ln(2 + access_count))
```

λ per memory class — preferences fade slowest, small talk fastest:

| class       | λ      | half-life |
|-------------|--------|-----------|
| preference  | 0.005  | ~139 days |
| semantic    | 0.02   | ~35 days  |
| episodic    | 0.08   | ~8.7 days |
| smalltalk   | 0.5    | ~1.4 days |

- **Consolidation** (background, every N turns): aging episodic/small-talk
  engrams below a retention floor are clustered by topic and summarized by Qwen
  into one durable **semantic** engram; the originals are tombstoned. A
  **compaction** then reclaims disk the only way the append-only substrate can —
  rewrite the live engrams into a fresh planet and **release** the old one,
  reporting bytes freed.
- **Contradiction**: a new like that was an old dislike (or vice versa), or a new
  fact overwriting an old one about the same subject, tombstones the stale
  memory with `superseded_by`.

Everything is observable in the inspector's Forgetting tab: retention bars, the
last consolidation diff, and reclaimed bytes.
