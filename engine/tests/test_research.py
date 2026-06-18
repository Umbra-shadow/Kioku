"""The Researcher — expansion, grounded study, memory, report, PDF, and the API
(live chat during a run, cross-session recall). All offline: a research-aware
fake brain and a fake web pull, no network, no real Qwen."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from engine.config import settings
from engine.main import create_app
from engine.research.report_pdf import build_pdf
from engine.research.researcher import Researcher
from engine.research.websearch import Source, html_to_text
from engine.store import PyStore
from engine.tenants import KiokuEngine, TenantRegistry
from engine.tests.fake_qwen import SmartFakeQwen


# ── fakes ────────────────────────────────────────────────────────────────────

class ResearchFakeQwen(SmartFakeQwen):
    """Adds the three research stages on top of the memory-pipeline fake."""

    def __init__(self, n: int = 20) -> None:
        super().__init__()
        self.n = n

    async def chat(self, messages, **kw):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "research stage" in system:
            self.chat_calls.append(messages)
            tag = "[model-knowledge, unsourced]\n" if "(none" in user else ""
            return f"{tag}Finding: a grounded answer with a claim [S1] and a [hypothesis]."
        if "synthesis stage" in system:
            self.chat_calls.append(messages)
            return (
                "## Overview\nThe complete answer.\n\n## What is established\nFacts [S1].\n\n"
                "## What has been tried\nPrior work.\n\n## The core gaps\nGaps.\n\n"
                "## Procedures & methodologies\n1. Step one.\n2. Step two.\n\n"
                "## Promising directions\n- Direction A [hypothesis]\n\n"
                "## Open questions & risks\nRisks.\n\n## Bottom line\nDo this."
            )
        return await super().chat(messages, **kw)

    async def chat_json(self, messages, **kw):
        if "research-planning stage" in messages[0]["content"]:
            self.chat_calls.append(messages)
            return {"subquestions": [f"Deep sub-question number {i+1}?" for i in range(self.n)]}
        return await super().chat_json(messages, **kw)


class FakeWeb:
    """Stands in for WebSearch. Returns canned sources (or none) — no network."""

    def __init__(self, sources_per_q: int = 2, provider: str = "duckduckgo") -> None:
        self.provider = provider
        self.sources_per_q = sources_per_q
        self.queries: list[str] = []

    async def gather(self, query: str, k: int = 4):
        self.queries.append(query)
        return [
            Source(url=f"https://example.com/{i}", title=f"Source {i} on {query[:20]}",
                   text="Evidence text about the topic.", snippet="snippet")
            for i in range(min(self.sources_per_q, k))
        ]

    async def aclose(self):
        pass


def _engine(tmp_path):
    store = PyStore(tmp_path / "store", ceiling_bytes=512 << 20)
    engine = KiokuEngine(TenantRegistry(store, ResearchFakeQwen(), settings(), message_cap=500))
    engine._store = store
    return engine


# ── unit: the researcher ─────────────────────────────────────────────────────

async def test_expand_makes_n_questions_and_remembers_the_plan(tmp_path):
    engine = _engine(tmp_path)
    mind = await engine.registry.new_mind()
    r = Researcher(engine, mind, web=FakeWeb(), progress=None)
    qs = await r.expand("How can we cure blindness?", n=20)
    assert len(qs) == 20
    await engine.drain_background()
    # the plan is now a memory
    assert any("Research plan" in e.message for e in mind.index.live_engrams())


async def test_full_run_grounds_studies_and_writes_report(tmp_path):
    engine = _engine(tmp_path)
    mind = await engine.registry.new_mind()
    web = FakeWeb(sources_per_q=2)
    r = Researcher(engine, mind, web=web, progress=None)
    out = await r.run("How can we cure blindness?", n=6)
    await engine.drain_background()

    assert len(out["questions"]) == 6
    assert len(out["findings"]) == 6
    assert out["grounded_count"] == 6  # every question got sources
    assert all(f["sources"] for f in out["findings"])
    assert out["report"].startswith("## Overview")
    assert len(web.queries) == 6  # one web pull per sub-question

    # everything is in memory: plan + 6 findings + final report
    live = mind.index.live_engrams()
    assert any("Final research report" in e.message for e in live)
    assert len([e for e in live if e.message.startswith("Deep sub-question")]) == 6


async def test_run_degrades_honestly_when_web_is_empty(tmp_path):
    engine = _engine(tmp_path)
    mind = await engine.registry.new_mind()
    web = FakeWeb(sources_per_q=0)  # network returned nothing
    r = Researcher(engine, mind, web=web, progress=None)
    out = await r.run("An offline question?", n=3)
    await engine.drain_background()
    assert out["grounded_count"] == 0
    assert all(not f["grounded"] for f in out["findings"])
    assert all("unsourced" in f["answer"] for f in out["findings"])


async def test_memory_recall_after_run(tmp_path):
    engine = _engine(tmp_path)
    mind = await engine.registry.new_mind()
    r = Researcher(engine, mind, web=FakeWeb(), progress=None)
    await r.run("How can we cure blindness?", n=4)
    await engine.drain_background()
    # ask the mind a follow-up — recall should surface the run's engrams
    result = await engine.turn(mind, "what did you research?", send_to_both=False)
    assert result.pack.hits, "follow-up recalled nothing from the research memory"


# ── unit: PDF + html extraction ──────────────────────────────────────────────

def test_pdf_builds_complete_bytes():
    run = {
        "topic": "How can we cure blindness?",
        "report": "## Overview\nThe answer — with a dash and a bullet •.\n\n## Bottom line\nDo this.",
        "provider": "duckduckgo",
        "grounded_count": 2,
        "findings": [
            {"id": 1, "question": "Q one?", "answer": "Finding one [S1].", "grounded": True,
             "sources": [{"url": "https://example.com/1", "title": "Src", "domain": "example.com"}]},
            {"id": 2, "question": "Q two?", "answer": "Finding two.", "grounded": False, "sources": []},
        ],
    }
    data = build_pdf(run)
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_html_to_text_strips_scripts():
    text = html_to_text("<html><body><script>evil()</script><p>Hello world</p><style>x{}</style></body></html>")
    assert "Hello world" in text and "evil" not in text and "x{}" not in text


# ── API: start → stream → done → pdf → ask (during + cross-session) ──────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KIOKU_RATELIMIT", "off")
    monkeypatch.setenv("KIOKU_RESEARCH_DB", ":memory:")  # isolated durable store per test
    # No network: every WebSearch instance the run manager builds returns canned sources.
    import engine.research.runs as runs_mod

    monkeypatch.setattr(runs_mod, "WebSearch", lambda *a, **k: FakeWeb())
    store = PyStore(tmp_path / "store", ceiling_bytes=512 << 20)
    engine = KiokuEngine(TenantRegistry(store, ResearchFakeQwen(), settings(), message_cap=500))
    engine._store = store
    with TestClient(create_app(engine)) as c:
        yield c


def _wait_done(client, run_id, tries=80):
    for _ in range(tries):
        run = client.get(f"/api/research/{run_id}").json()
        if run["status"] in ("done", "error"):
            return run
        time.sleep(0.05)
    raise AssertionError("research run did not finish in time")


def test_research_api_end_to_end(client: TestClient):
    start = client.post("/api/research/start", json={"topic": "How can we cure blindness?", "num_questions": 5})
    assert start.status_code == 200
    run_id = start.json()["run_id"]

    run = _wait_done(client, run_id)
    assert run["status"] == "done", run.get("error")
    assert len(run["findings"]) == 5
    assert run["report"].startswith("## Overview")

    # PDF downloads as a complete attachment
    pdf = client.get(f"/api/research/{run_id}/pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert "attachment" in pdf.headers["content-disposition"]
    assert pdf.content[:5] == b"%PDF-"

    # ask a follow-up — recalled from the run's memory
    ask = client.post(f"/api/research/{run_id}/ask", json={"question": "what did you find?"})
    assert ask.status_code == 200
    body = ask.json()
    assert body["answer"] and body["recalled"]  # it recalled something
    sess = body["session_id"]

    # a NEW session against the same run still recalls the research (cross-session)
    ask2 = client.post(f"/api/research/{run_id}/ask",
                       json={"question": "remind me what you researched", "session_id": "s_brand_new"})
    assert ask2.json()["session_id"] == "s_brand_new" != sess
    assert ask2.json()["recalled"]


def test_pdf_before_done_is_conflict(client: TestClient):
    run_id = client.post("/api/research/start", json={"topic": "Slow topic here"}).json()["run_id"]
    # immediately (very likely still running) — but tolerate a fast finish
    r = client.get(f"/api/research/{run_id}/pdf")
    assert r.status_code in (200, 409)
    _wait_done(client, run_id)


def test_chat_history_persists(client: TestClient):
    run_id = client.post("/api/research/start", json={"topic": "History topic?", "num_questions": 3}).json()["run_id"]
    _wait_done(client, run_id)
    client.post(f"/api/research/{run_id}/ask", json={"question": "q1", "session_id": "s1"})
    client.post(f"/api/research/{run_id}/ask", json={"question": "q2", "session_id": "s1"})
    hist = client.get(f"/api/research/{run_id}/chat?session_id=s1").json()["messages"]
    roles = [m["role"] for m in hist]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert hist[0]["content"] == "q1"


# ── persistence + per-user shared memory ─────────────────────────────────────

from engine.research.persistence import ResearchDB  # noqa: E402
from engine.research.runs import RunManager  # noqa: E402


def test_db_roundtrips_runs_and_chats(tmp_path):
    db = ResearchDB(tmp_path / "r.db")
    db.save_run({"run_id": "R1", "tenant": "researcher", "topic": "T", "status": "starting",
                 "num_questions": 5, "questions": ["a"], "findings": [], "report": "",
                 "created_ts": 1.0})
    db.save_run({"run_id": "R1", "tenant": "researcher", "topic": "T", "status": "done",
                 "report": "## Overview\nx", "grounded_count": 3, "created_ts": 1.0, "done_ts": 2.0})
    loaded = db.load_run("R1")
    assert loaded["status"] == "done" and loaded["report"].startswith("## Overview")
    assert len(db.load_runs("researcher")) == 1
    db.save_chat("researcher", "R1", "s1", "user", "hi")
    assert db.load_chats(session_id="s1")[0]["content"] == "hi"
    db.close()


async def test_two_runs_share_one_per_user_memory(tmp_path, monkeypatch):
    import engine.research.runs as runs_mod
    monkeypatch.setattr(runs_mod, "WebSearch", lambda *a, **k: FakeWeb())
    engine = _engine(tmp_path)
    mgr = RunManager(engine, ResearchDB(":memory:"))
    mgr.bootstrap()
    r1 = await mgr.start("Topic Alpha question?", num_questions=2); await r1._task
    r2 = await mgr.start("Topic Beta question?", num_questions=2); await r2._task
    await engine.drain_background()
    assert r1.mind is r2.mind  # one memory per user, not per run/session
    res = await engine.turn(mgr.user_mind, "alpha and beta findings", send_to_both=False)
    assert res.pack.hits  # recall reaches across both runs


async def test_memory_and_history_survive_restart(tmp_path, monkeypatch):
    import engine.research.runs as runs_mod
    monkeypatch.setattr(runs_mod, "WebSearch", lambda *a, **k: FakeWeb())
    db = ResearchDB(tmp_path / "durable.db")

    # session one: research a topic with engine A
    engine_a = _engine(tmp_path / "a")
    mgr_a = RunManager(engine_a, db)
    mgr_a.bootstrap()
    run = await mgr_a.start("How can we cure blindness?", num_questions=3)
    await run._task
    await engine_a.drain_background()
    saved = db.count_engrams("researcher")
    assert saved > 0

    # "years later" — a brand-new process: fresh engine + manager over the same DB
    engine_b = _engine(tmp_path / "b")
    mgr_b = RunManager(engine_b, db)
    rehydrated = mgr_b.bootstrap()
    assert rehydrated == saved  # memory rebuilt from disk, nothing lost

    # a new session still recalls the old research
    res = await engine_b.turn(mgr_b.user_mind, "what did you research about blindness?", send_to_both=False)
    assert res.pack.hits
    # and the run history is there too
    assert any(r["topic"] == "How can we cure blindness?" for r in mgr_b.list())
    db.close()


# ── engine: qwen_for key cache ────────────────────────────────────────────────

def test_qwen_for_same_key_returns_same_client(tmp_path):
    engine = _engine(tmp_path)
    c1 = engine.qwen_for("key-abc")
    c2 = engine.qwen_for("key-abc")
    assert c1 is c2  # LRU hit: same object


def test_qwen_for_different_keys_return_different_clients(tmp_path):
    engine = _engine(tmp_path)
    assert engine.qwen_for("key-aaa") is not engine.qwen_for("key-bbb")


def test_qwen_for_none_returns_server_default(tmp_path):
    engine = _engine(tmp_path)
    assert engine.qwen_for(None) is engine.qwen
    assert engine.qwen_for("") is engine.qwen


def test_qwen_for_evicts_oldest_at_capacity(tmp_path):
    from engine.tenants import _BRAIN_CACHE_SIZE
    engine = _engine(tmp_path)
    for i in range(_BRAIN_CACHE_SIZE + 1):
        engine.qwen_for(f"key-{i:04d}")
    assert len(engine._brains) == _BRAIN_CACHE_SIZE


def test_qwen_for_trims_oversized_key(tmp_path):
    from engine.tenants import _KEY_MAX
    engine = _engine(tmp_path)
    long_key = "x" * (_KEY_MAX + 100)
    c = engine.qwen_for(long_key)
    # the stored key must be at most _KEY_MAX chars
    assert all(len(k) <= _KEY_MAX for k in engine._brains)
    assert c is engine.qwen_for(long_key)  # same trimmed key → same client


# ── SSE stream endpoint ───────────────────────────────────────────────────────

def test_stream_replay_on_finished_run(client: TestClient):
    run_id = client.post("/api/research/start",
                         json={"topic": "stream test topic?", "num_questions": 3}).json()["run_id"]
    _wait_done(client, run_id)
    # replay_then_close=true on a terminal run should emit all buffered events
    # then a terminal sentinel and close immediately.
    with client.stream("GET", f"/api/research/{run_id}/stream?replay_then_close=true") as resp:
        assert resp.status_code == 200
        raw = b"".join(resp.iter_bytes()).decode()
    data_lines = [ln[6:] for ln in raw.splitlines() if ln.startswith("data:")]
    stages = [json.loads(d).get("stage") for d in data_lines if d]
    assert any(s in ("done", "error") for s in stages), f"no terminal event in: {stages}"
