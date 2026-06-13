/* Kioku v1 arena — vanilla JS, no build step. Talks to the FastAPI engine.
   API base: ?api=… query param, else window.KIOKU_API, else same origin :8000. */
"use strict";

// API base: ?api=… wins; then window.KIOKU_API; else same origin (the engine
// serves this page and /api together). For a separately-hosted page, set
// window.KIOKU_API before this script loads.
const API = (new URLSearchParams(location.search).get("api")
  || window.KIOKU_API
  || (location.protocol === "file:" ? "http://localhost:8000" : location.origin)
).replace(/\/$/, "");

const $ = (id) => document.getElementById(id);
const state = {
  token: localStorage.getItem("kioku_token") || "kioku",
  session: localStorage.getItem("kioku_session") || null,
  stream: null,
};

// ---- helpers ----------------------------------------------------------------
function toast(msg) {
  const t = $("toast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 2400);
}
function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }
function humanBytes(n) {
  const u = ["B", "KiB", "MiB", "GiB", "TiB"]; let i = 0; n = Number(n);
  while (Math.abs(n) >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 ? n : n.toFixed(1)) + " " + u[i];
}
async function api(path, opts) {
  const r = await fetch(API + path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}
function setMindChip() {
  $("mindChip").textContent = "mind: " + (state.token === "kioku" ? "shared" : state.token.slice(0, 10) + "…");
}

// ---- chat -------------------------------------------------------------------
function addMsg(paneId, who, text, cls = "") {
  const log = $(paneId);
  const hint = log.querySelector(".empty-hint"); if (hint) hint.remove();
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.innerHTML = `<div class="who">${esc(who)}</div>${esc(text)}`;
  log.appendChild(div); log.scrollTop = log.scrollHeight;
  return div;
}
function addTyping(paneId) {
  const log = $(paneId);
  const d = document.createElement("div"); d.className = "typing"; d.textContent = "thinking…";
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}

async function send(message, sendToBoth) {
  if (!message.trim()) return;
  $("sendBtn").disabled = true; $("status").textContent = "";
  addMsg("logKioku", "you", message, "user");
  if (sendToBoth) addMsg("logRaw", "you", message, "user");
  const tk = addTyping("logKioku");
  const tr = sendToBoth ? addTyping("logRaw") : null;
  resetChips();
  try {
    const res = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, token: state.token, session_id: state.session, send_to_both: sendToBoth }),
    });
    state.token = res.token; state.session = res.session_id;
    localStorage.setItem("kioku_token", res.token);
    localStorage.setItem("kioku_session", res.session_id);
    setMindChip();
    tk.remove();
    if (res.pack && res.pack.tokens > 0) {
      addMsg("logKioku", `memory pack · ${res.pack.tokens}/${res.pack.budget} tok · ${res.pack.hits.length} recalled`,
        res.pack.text, "pack");
    }
    addMsg("logKioku", "qwen + kioku", res.kioku_reply, "bot");
    if (tr) { tr.remove(); addMsg("logRaw", "qwen raw", res.raw_reply || "(no answer)", "bot"); }
    $("address").innerHTML = `<span class="lbl">committed at</span>${esc(res.address)}`;
    if (res.superseded && res.superseded.length) toast(`Superseded ${res.superseded.length} stale memory(ies)`);
    refreshActiveTab();
  } catch (e) {
    tk.remove(); if (tr) tr.remove();
    $("status").textContent = "⚠ " + e.message;
    toast("Error: " + e.message);
  } finally {
    $("sendBtn").disabled = false;
  }
}

// ---- SSE pipeline -----------------------------------------------------------
const CHIP_ORDER = ["captured", "decomposed", "embedded", "curious", "committed"];
function resetChips() {
  document.querySelectorAll(".chip").forEach((c) => c.classList.remove("lit"));
  $("eventFeed").innerHTML = "";
}
function litChip(stage) { const c = document.querySelector(`.chip[data-stage="${stage}"]`); if (c) c.classList.add("lit"); }
function feedEvent(ev) {
  const feed = $("eventFeed");
  let txt = ev.stage;
  if (ev.stage === "curious" && ev.detail.term) txt = `curious → "${ev.detail.term}"`;
  else if (ev.stage === "decomposed") txt = `decomposed · ${(ev.detail.keywords || []).join(", ")}`;
  else if (ev.stage === "committed" && ev.detail.address) txt = `committed @ ${ev.detail.address}`;
  const line = document.createElement("div");
  line.className = "ev"; line.innerHTML = `<b>▸</b> ${esc(txt)}`;
  feed.prepend(line);
  while (feed.children.length > 12) feed.lastChild.remove();
}
function connectStream() {
  if (state.stream) state.stream.close();
  const es = new EventSource(`${API}/api/stream/${encodeURIComponent(state.token)}`);
  es.onmessage = (m) => {
    try {
      const ev = JSON.parse(m.data);
      litChip(ev.stage); feedEvent(ev);
    } catch (_) {}
  };
  es.onerror = () => {};
  state.stream = es;
}

// ---- inspector tabs ---------------------------------------------------------
let activeTab = "pipeline";
document.querySelectorAll(".insp-tabs button").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".insp-tabs button").forEach((x) => x.setAttribute("aria-selected", "false"));
    b.setAttribute("aria-selected", "true");
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    $("panel-" + b.dataset.tab).classList.add("active");
    activeTab = b.dataset.tab; refreshActiveTab();
  });
});
function refreshActiveTab() {
  if (activeTab === "memory") loadMemory();
  else if (activeTab === "lexicon") loadLexicon();
  else if (activeTab === "forgetting") loadForgetting();
  else if (activeTab === "substrate") loadSubstrate();
}

async function loadMemory() {
  try {
    const d = await api(`/api/memory?token=${encodeURIComponent(state.token)}&limit=100`);
    $("memCount").textContent = `(${d.total})`;
    const list = $("memList");
    if (!d.engrams.length) { list.innerHTML = '<p class="muted">No memories yet.</p>'; return; }
    list.innerHTML = d.engrams.map((e) => `
      <div class="card click ${e.tombstoned ? "tomb" : ""}" data-id="${esc(e.engram_id)}">
        <div class="meaning">${esc(e.meaning || e.message)}</div>
        <div class="meta">
          <span class="tag ${e.memory_class}">${esc(e.memory_class)}</span>
          <span>imp ${e.importance.toFixed(2)}</span>
          <span>×${e.access_count}</span>
          <span>${(e.keywords || []).slice(0, 4).join(" · ")}</span>
        </div>
      </div>`).join("");
    list.querySelectorAll(".card").forEach((c) => c.addEventListener("click", () => showEngram(c.dataset.id)));
  } catch (e) { $("memList").innerHTML = `<p class="muted">⚠ ${esc(e.message)}</p>`; }
}
async function showEngram(id) {
  try {
    const d = await api(`/api/memory/${encodeURIComponent(state.token)}/${encodeURIComponent(id)}`);
    $("modalTitle").textContent = d.meaning || "Engram";
    $("modalBody").textContent = JSON.stringify(d, null, 2);
    $("modalBack").classList.add("open");
  } catch (e) { toast("Error: " + e.message); }
}
async function loadLexicon() {
  try {
    const d = await api(`/api/lexicon?token=${encodeURIComponent(state.token)}`);
    const list = $("lexList");
    const terms = Object.entries(d.definitions);
    if (!terms.length) { list.innerHTML = '<p class="muted">Kioku hasn\'t been curious yet.</p>'; return; }
    list.innerHTML = terms.map(([t, def]) =>
      `<div class="card"><span class="lex-term">${esc(t)}</span><div style="margin-top:5px">${esc(def)}</div></div>`).join("");
  } catch (e) { $("lexList").innerHTML = `<p class="muted">⚠ ${esc(e.message)}</p>`; }
}
async function loadForgetting() {
  try {
    const d = await api(`/api/forgetting?token=${encodeURIComponent(state.token)}`);
    const last = d.last_consolidation;
    const diff = $("consoDiff");
    if (last && (last.created.length || last.tombstoned.length)) {
      diff.style.display = "block";
      diff.innerHTML = `<b>Last consolidation:</b> ${esc(last.summaries.join("; "))}<br>
        <span class="muted">${last.tombstoned.length} tombstoned → ${last.created.length} semantic · reclaimed ${humanBytes(last.reclaimed_bytes)}</span>`;
    } else diff.style.display = "none";
    const list = $("retList");
    if (!d.retention.length) { list.innerHTML = '<p class="muted">Nothing to forget yet.</p>'; return; }
    const max = Math.max(...d.retention.map((r) => r.retention), 1);
    list.innerHTML = d.retention.map((r) => `
      <div class="card ${r.tombstoned ? "tomb" : ""}">
        <div class="meaning">${esc(r.meaning || "(memory)")}</div>
        <div class="meta"><span class="tag ${r.memory_class}">${esc(r.memory_class)}</span>
          <span>ret ${r.retention.toFixed(3)}</span><span>${r.age_days.toFixed(1)}d</span><span>×${r.access_count}</span></div>
        <div class="ret-bar"><span style="width:${Math.min(100, (r.retention / max) * 100)}%"></span></div>
      </div>`).join("");
  } catch (e) { $("retList").innerHTML = `<p class="muted">⚠ ${esc(e.message)}</p>`; }
}
async function loadSubstrate() {
  try {
    const d = await api(`/api/stats?token=${encodeURIComponent(state.token)}`);
    const g = d.gauge;
    const vramPct = (g.vram_committed / g.vram_virtual) * 100;
    const diskPct = (g.disk_committed / g.disk_virtual) * 100;
    $("gauge").innerHTML = `
      <div class="big">${esc(g.headline)}</div>
      <div class="bar-wrap">
        <div class="label"><span>vRAM committed</span><span>${humanBytes(g.vram_committed)} / ${humanBytes(g.vram_virtual)}</span></div>
        <div class="bar"><span style="width:${Math.max(0.5, vramPct).toFixed(4)}%"></span></div>
      </div>
      <div class="bar-wrap">
        <div class="label"><span>virtual disk committed</span><span>${humanBytes(g.disk_committed)} / ${humanBytes(g.disk_virtual)}</span></div>
        <div class="bar disk"><span style="width:${Math.max(0.5, diskPct).toFixed(4)}%"></span></div>
      </div>
      <div class="stat-grid">
        <div class="stat"><div class="v">${(g.retrieve_ms.p50 ?? 0).toFixed(1)}</div><div class="k">retrieve p50 ms</div></div>
        <div class="stat"><div class="v">${(g.retrieve_ms.p95 ?? 0).toFixed(1)}</div><div class="k">retrieve p95 ms</div></div>
        <div class="stat"><div class="v">${g.pack_tokens.p50 ?? 0}</div><div class="k">pack tokens p50</div></div>
        <div class="stat"><div class="v">${g.live_engrams}</div><div class="k">live engrams</div></div>
        <div class="stat"><div class="v">${g.open_minds}</div><div class="k">open minds</div></div>
        <div class="stat"><div class="v" style="font-size:13px">${esc(g.backend)}</div><div class="k">backend</div></div>
      </div>`;
  } catch (e) { $("gauge").innerHTML = `<p class="muted">⚠ ${esc(e.message)}</p>`; }
}

// ---- actions ----------------------------------------------------------------
$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("input");
  send(input.value, $("bothToggle").checked);
  input.value = ""; input.style.height = "auto";
});
$("input").addEventListener("input", (e) => { e.target.style.height = "auto"; e.target.style.height = e.target.scrollHeight + "px"; });
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("composer").requestSubmit(); }
});
$("probeBtn").addEventListener("click", () => send("What do you remember about me?", $("bothToggle").checked));
$("newMindBtn").addEventListener("click", async () => {
  try {
    const d = await api("/api/mind/new", { method: "POST" });
    state.token = d.token; state.session = null;
    localStorage.setItem("kioku_token", d.token); localStorage.removeItem("kioku_session");
    setMindChip(); connectStream();
    $("logKioku").innerHTML = '<p class="empty-hint">A newborn mind. Empty memory, its own Cadran space. Teach it from scratch.</p>';
    $("logRaw").innerHTML = '<p class="empty-hint">The raw pane never had a memory to begin with.</p>';
    resetChips(); $("address").innerHTML = '<span class="lbl">physical address</span>—';
    refreshActiveTab(); toast("New mind: " + d.token.slice(0, 10) + "…");
  } catch (e) { toast("Error: " + e.message); }
});
$("consolidateBtn").addEventListener("click", async () => {
  try {
    const d = await api(`/api/consolidate?token=${encodeURIComponent(state.token)}`, { method: "POST" });
    toast(d.did_anything ? `Consolidated ${d.tombstoned.length} → ${d.created.length}, reclaimed ${humanBytes(d.reclaimed_bytes)}` : "Nothing old enough to consolidate yet");
    loadForgetting();
  } catch (e) { toast("Error: " + e.message); }
});
$("modalClose").addEventListener("click", () => $("modalBack").classList.remove("open"));
$("modalBack").addEventListener("click", (e) => { if (e.target === $("modalBack")) $("modalBack").classList.remove("open"); });

// ---- boot -------------------------------------------------------------------
setMindChip();
connectStream();
api("/api/health").then((h) => { if (!h.backend) return; $("status").textContent = `substrate: ${h.backend}`; })
  .catch(() => { $("status").textContent = "⚠ engine offline — run make dev"; });
