# Kioku v1 — Alibaba Cloud + Qwen Cloud proof

This file points to the exact code that calls **Qwen Cloud** (Alibaba Cloud Model
Studio) and explains how to record the **Alibaba Cloud ECS** deployment for the
submission.

## 1. Qwen Cloud is the brain — exact call sites

Every LLM call in Kioku goes through one client, against the Qwen / Model Studio
**OpenAI-compatible** endpoint.

| What | File · symbol | Qwen API |
|---|---|---|
| HTTP client (base URL, key, retries) | [`engine/qwen.py`](../../engine/qwen.py) · `QwenClient` | `POST /chat/completions`, `POST /embeddings` |
| Endpoint + model config | [`engine/config.py`](../../engine/config.py) · `_llm_from_env` | `QWEN_BASE_URL`, `QWEN_MODEL` (`qwen-max`), `QWEN_EMBED_MODEL` (`text-embedding-v3`) |
| Structured decomposition (JSON mode) | [`engine/decompose.py`](../../engine/decompose.py) · `decompose_exchange` → `chat_json` | chat |
| Curiosity definitions | [`engine/curiosity.py`](../../engine/curiosity.py) · `curiosity_pass` → `chat` | chat |
| Consolidation summaries | [`engine/forget.py`](../../engine/forget.py) · `ForgetManager._summarize` → `chat` | chat |
| Query + memory embeddings | `decompose_exchange`, [`engine/tenants.py`](../../engine/tenants.py) `KiokuEngine.turn` → `embed` | embeddings |
| The answer (memory pack injected) | [`engine/tenants.py`](../../engine/tenants.py) · `KiokuEngine.turn` → `chat` | chat |

Default endpoint (international): `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
(mainland: `https://dashscope.aliyuncs.com/compatible-mode/v1`). The key is read
only from the environment (`.env`, gitignored) and is never logged or committed —
see `QwenClient.__init__` and the structured logging in `engine/qwen.py`.

## 2. Deploy to Alibaba Cloud ECS

1. **Create an ECS instance** (console or `aliyun ecs CreateInstance`):
   Ubuntu 22.04, ≥ 2 GiB RAM (e.g. `ecs.t6-c1m2.large`), assign a public IP.
2. **Security group**: allow inbound TCP **8000** (app) and **22** (SSH).
3. **Install Docker** on the instance: `curl -fsSL https://get.docker.com | sh`.
4. **Get a Qwen key** from Model Studio, put it in `.env` (`QWEN_API_KEY=…`).
5. **Deploy** from your machine:
   ```bash
   ./deploy/alibaba/deploy.sh ecs ubuntu@<instance-public-ip>
   ```
   This rsyncs the repo and runs `docker compose up --build -d`. The image builds
   the Rust `kiokud` daemon and serves the arena + API on `:8000`.
6. **Verify**: `curl http://<public-ip>:8000/api/health` →
   `{"ok":true,"backend":"kiokud", …}`.

## 3. What to record (the submission video / screenshots)

1. The Alibaba Cloud **ECS console** showing the running instance + its public IP.
2. A terminal: `curl http://<public-ip>:8000/api/health` returning
   `"backend":"kiokud"` — proof the Rust substrate is live on Alibaba Cloud.
3. The arena at `http://<public-ip>:8000`: teach it facts, then the recall probe —
   Qwen+Kioku remembers, Qwen raw does not.
4. The **Inspector → Substrate** gauge: committed vs 1 TiB/4 TiB virtual, and the
   backend reading `kiokud`.
5. Optional: Model Studio's usage dashboard showing the API calls this demo made,
   tying the traffic back to Qwen Cloud.

Numbers shown anywhere come from [`eval/METRICS.md`](../../eval/METRICS.md), a
real `make eval` run — nothing hand-written.
