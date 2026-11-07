/* jepa — the local install's web page. Plain JS, no build step. */

const $ = (id) => document.getElementById(id);
let runtime = { status: "loading" };

/* ---------- tabs ---------- */
document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    document.querySelectorAll("main section").forEach((s) => (s.hidden = true));
    $("tab-" + b.dataset.tab).hidden = false;
    if (b.dataset.tab === "bench") loadBench();
    if (b.dataset.tab === "train") loadTrain();
    if (b.dataset.tab === "about") loadAbout();
  })
);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: String(r.status) }));
    throw err;
  }
  return r.json();
}

/* ---------- status pills ---------- */
function pill(ok, text) {
  return `<span class="pill ${ok ? "ok" : "warn"}">${ok ? "●" : "○"} ${text}</span>`;
}

async function refreshHealth() {
  try {
    runtime = await api("/api/health");
    const r = runtime;
    $("pills").innerHTML = [
      pill(r.runtime_status === "ready" || r.runtime_status === "model", `model ${r.step ? `step ${r.step.toLocaleString()}` : "not loaded"}`),
      pill(r.gallery_images > 0, `photo gallery ${r.gallery_images ? r.gallery_images.toLocaleString() : "— missing"}`),
      pill(r.rotation_ready, `rotation ${r.rotation_ready ? "fitted" : "— not yet"}`),
      pill(r.caption_bank > 0, `caption bank ${r.caption_bank ? r.caption_bank.toLocaleString() : "—"}`),
      pill(true, "device: CPU"),
    ].join(" ");
  } catch (e) {
    $("pills").innerHTML = pill(false, "server error");
  }
}

/* ---------- searching ---------- */
async function search(text, mode, grid) {
  grid.innerHTML = `<div class="dim">embedding “${text.replace(/</g, "&lt;")}”…</div>`;
  try {
    const res = await api("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, k: 12, mode }),
    });
    if (mode === "images") {
      grid.innerHTML = res.results
        .map(
          (x) =>
            `<figure class="card"><img loading="lazy" src="${x.url}" alt="">
             <figcaption>${x.score.toFixed(3)}</figcaption></figure>`
        )
        .join("");
    } else {
      grid.innerHTML = res.results
        .map((x) => `<div class="row"><span class="score">${x.score.toFixed(3)}</span><span>${x.text}</span></div>`)
        .join("");
    }
  } catch (e) {
    const d = e.detail || e;
    grid.innerHTML = `<div class="dim">⚠ ${typeof d.error === "string" ? d.error : JSON.stringify(d)} —
      ${d.hint || ""}</div>`;
  }
}

$("home-go").onclick = () => { const t = $("home-query").value.trim(); if (t) { search(t, "images", $("home-results")); select("photos", t); } };
$("home-go-caps").onclick = () => { const t = $("home-query").value.trim(); if (t) { search(t, "captions", $("home-results")); } };
$("photo-go").onclick = () => { const t = $("photo-query").value.trim(); if (t) search(t, "images", $("photo-grid")); };
$("photo-query").addEventListener("keydown", (e) => e.key === "Enter" && $("photo-go").click());
$("cap-go").onclick = async () => {
  const t = $("cap-query").value.trim();
  if (!t) return;
  $("cap-list").innerHTML = `<div class="dim">embedding…</div>`;
  let vec = "";
  try {
    const v = await api("/api/embed", { method: "POST", headers: { "Content-Type": "application/json" },
                                        body: JSON.stringify({ texts: [t] }) });
    vec = `step ${v.step} · ${v.dim}-d · first dims: [${v.embeddings[0].slice(0, 6).join(", ")}, …}`;
  } catch (e) { /* shown by the list error below */ }
  $("cap-vec").textContent = vec;
  await search(t, "captions", $("cap-list"));
};
$("cap-query").addEventListener("keydown", (e) => e.key === "Enter" && $("cap-go").click());

function select(tab, text) {
  const input = $(tab === "photos" ? "photo-query" : "cap-query");
  const button = $(tab === "photos" ? "photo-go" : "cap-go");
  input.value = text; button.click();
}

/* ---------- benchmarks ---------- */
const fmt = (x) => (x == null ? "—" : typeof x === "number" ? x.toFixed(3) : x);

async function loadBench() {
  $("bench-wrap").innerHTML = `<div class="dim">loading metrics…</div>`;
  try {
    const m = await api("/api/metrics");
    const ref = m.reference || {};
    let html = `<h3>Flickr · retrieval through one rotation (fit: val, score: test, chance 0.001)</h3>
      <table class="tbl"><tr><th>metric</th><th>this install</th><th>paper quick-start ref.</th></tr>
      ${["mknn", "R@1", "R@5", "wR@1", "wR@5", "cos"]
        .map(
          (k) =>
            `<tr><td>${k}</td><td>${fmt((m.flickr_alignment || {})[k]) || "—"}</td><td>${fmt(ref.flickr_quickstart[k]) || "—"}</td></tr>`
        )
        .join("")}</table>
      <h3>WIT-1024 · mutual 10-NN of the PRH paper vs DINOv2-B (per-layer max, chance ~0.01)</h3>
      <table class="tbl"><tr><th>model</th><th>score</th></tr>
      ${Object.entries(ref.wit1024_dinov2_b || {})
        .map(([k, v]) => (k === "note" ? "" : `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`))
        .join("")}
      ${Object.entries(m.wit1024 || {})
        .map(([k, v]) => `<tr class="local"><td>${k}</td><td>${fmt(v)}</td></tr>`)
        .join("")}</table>
      <h3>Text-only quality</h3>
      <table class="tbl"><tr><th>measure</th><th>this install</th></tr>
        <tr><td>STS-B Spearman (test)</td><td>${fmt(m.sts_b_spearman)}</td></tr>
        <tr><td>Flickr caption-to-caption R@1</td><td>${fmt(m.flickr_caption_r1)}</td></tr></table>
      <p class="dim">${ref.wit1024_dinov2_b?.note || ""} <br>${m.local_compute_note || ""}<br>
      ${m.wit1024_error ? "WIT protocol pending: " + m.wit1024_error : ""}</p>`;
    $("bench-wrap").innerHTML = html;
  } catch (e) {
    const d = e.detail || e;
    $("bench-wrap").innerHTML = `<div class="dim">No metrics yet: ${typeof d.error === "string" ? d.error : ""} —
      run <code>jepa evaluate</code> after the first checkpoints and photos arrive.</div>`;
  }
}

/* ---------- training ---------- */
async function loadTrain() {
  await refreshHealth();
  const r = runtime;
  $("tr-step").textContent = (r.step || 0).toLocaleString();
  $("tr-epoch").textContent = `${Math.max(1, Math.ceil((r.step || 0) * r.config.batch_size / r.config.epoch_size))} / ${r.config.epochs}`;
  $("tr-ckpt").textContent = r.checkpoint || "no checkpoint yet";
  $("tr-pills").innerHTML = [
    pill(r.runtime_status === "ready" || r.runtime_status === "model", `serving: ${r.runtime_status}`),
    pill(true, `${r.config.arch} · ${r.config.params.toLocaleString()} params · batch ${r.config.batch_size}`),
    pill(r.gallery_images > 0, `gallery ${r.gallery_images}`),
    pill(r.rotation_ready, `rotation ${r.rotation_ready ? "✓" : "…"}`),
  ].join(" ");
  let t = [];
  try {
    const info = await api("/api/training");
    t = info.log_tail || [];
  } catch (e) {}
  $("tr-log").textContent = t.slice(-40).join("\n") || "no training log yet";
}

$("tr-start").onclick = async () => {
  await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
  setTimeout(loadTrain, 800);
};
$("tr-stop").onclick = async () => {
  await api("/api/stop-train", { method: "POST", body: "{}" });
  setTimeout(loadTrain, 800);
};
$("tr-refresh").onclick = loadTrain;

/* model card */
async function loadAbout() {
  const md = await api("/api/model");
  $("about-content").innerHTML = `
  <h2>What is installed here</h2>
  <table class="tbl">
    <tr><td>Paper</td><td><a href="https://arxiv.org/abs/2610.39302" target="_blank">${md.paper.title}</a> — ${md.paper.authors}</td></tr>
    <tr><td>Code</td><td><a href="https://github.com/galilai-group/prh-proof" target="_blank">galilai-group/prh-proof</a> (MIT), trained and served from C:\\Users\\user\\JEPA\\repo</td></tr>
    <tr><td>Architecture</td><td>${md.architecture.arch} BERT-width encoder: ${md.architecture.width}-d, ${md.architecture.layers} layers, ${md.architecture.heads} heads, ${md.architecture.params.toLocaleString()} parameters, tokenizer bert-base-uncased, pooled as [CLS]</td></tr>
    <tr><td>Objective</td><td>${md.objective.name}: ${md.objective.views}, LeJEPA invariance + SIGReg (λ=${md.objective.sigreg_lamb})</td></tr>
    <tr><td>Data</td><td>COCO Karpathy captions (113k images × 5) + Flickr30k captions (30k × 5) — good-view pairs only (two captions describing one image)</td></tr>
    <tr><td>Training</td><td>this install trained the model on this CPU; official 109M checkpoints are restricted, the paper ran 16–32 GPUs for the headline result</td></tr>
    <tr><td>Serving</td><td>App: app/jepa_server.py (FastAPI). CLI: <code>jepa serve | stop | status | train | chat | evaluate</code>. Artifacts under app/state/</td></tr>
    <tr><td>Why a rotation?</td><td>${md.headline}</td></tr>
  </table>`;
}

/* ---------- boot ---------- */
refreshHealth();
setInterval(refreshHealth, 10000);
