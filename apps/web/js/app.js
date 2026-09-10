// Scamless web app: on-device inference via transformers.js (ONNX Runtime Web).
// Flow: load thresholds -> load model (progress shown) -> scan -> sigmoid probs
// -> per-label verdicts + calibrated risk score -> rule-based why panel.

import { AutoTokenizer, AutoModelForSequenceClassification, env } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1";
import { CONFIG } from "./config.js";
import { CATEGORIES, riskBand } from "./categories.js";
import { heuristic_scan } from "./heuristic.js";

const $ = (id) => document.getElementById(id);

const state = {
  phase: "idle", // idle | loading | ready | demo
  tokenizer: null,
  model: null,
  thresholds: null,
};

// transformers.js fetches model files itself; route its progress into our UI
function wireProgress(callback) {
  const perFile = new Map();
  return (data) => {
    if (data.status === "progress" && data.total) {
      perFile.set(data.file, { loaded: data.loaded, total: data.total });
      let loaded = 0, total = 0;
      for (const f of perFile.values()) { loaded += f.loaded; total += f.total; }
      callback(loaded, total);
    }
  };
}

function setStatus(text, cls) {
  $("status-text").textContent = text;
  const dot = $("status-dot");
  dot.className = "dot" + (cls ? " " + cls : "");
}

async function loadThresholds() {
  const url = `https://huggingface.co/${CONFIG.MODEL_REPO}/resolve/main/thresholds.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`thresholds fetch failed: ${res.status}`);
  return res.json();
}

async function loadModel() {
  setStatus("Downloading detection model (one time, ~118 MB, cached after this)");
  const progress = wireProgress((loaded, total) => {
    const mb = (loaded / 1e6).toFixed(0);
    const totalMb = (total / 1e6).toFixed(0);
    const pct = total ? Math.round((loaded / total) * 100) : 0;
    $("load-progress").hidden = false;
    $("load-bar").style.width = pct + "%";
    $("load-detail").textContent = `${mb} / ${totalMb} MB - runs locally after this, never downloads again`;
  });

  const tokenizer = await AutoTokenizer.from_pretrained(CONFIG.MODEL_REPO, { progress_callback: progress });
  const model = await AutoModelForSequenceClassification.from_pretrained(CONFIG.MODEL_REPO, {
    progress_callback: progress,
    model_file_name: "model_int8",
  });
  state.tokenizer = tokenizer;
  state.model = model;
  state.thresholds = await loadThresholds();
  state.phase = "ready";

  setStatus("Model ready - running locally on this device", "ready");
  $("load-progress").hidden = true;
  $("load-detail").textContent = "Inference happens inside this browser tab. Scan away.";
  $("scan-btn").disabled = false;
}

function enterDemoMode(reason) {
  state.phase = "demo";
  setStatus("Demo mode: keyword rules only (model unavailable)", "error");
  $("load-detail").textContent =
    `Full model could not load (${reason}). Scan results below use simple keyword rules, ` +
    "not the trained detector. Reload later for full protection.";
  $("load-progress").hidden = true;
  $("scan-btn").disabled = false;
}

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function labelThreshold(label) {
  const t = state.thresholds && state.thresholds[label];
  return typeof t === "number" ? t : 0.5;
}

async function scanWithModel(text) {
  const enc = state.tokenizer(text, { truncation: true, max_length: CONFIG.MAX_LEN });
  const output = await state.model({
    input_ids: [enc.input_ids],
    attention_mask: [enc.attention_mask],
  });
  const logits = output.logits.tolist()[0];
  return logits.map(sigmoid);
}

function renderRisk(score) {
  const band = riskBand(score);
  const arc = $("meter-arc");
  const circumference = 2 * Math.PI * 52;
  arc.style.strokeDashoffset = String(circumference * (1 - score / 100));
  arc.style.stroke = band.color;
  $("meter-score").textContent = String(score);
  $("verdict-band").textContent = band.label;
  $("verdict-band").style.color = band.color;
}

function renderLabels(hits) {
  const list = $("verdict-labels");
  list.textContent = "";
  if (!hits.length) {
    list.textContent = "No scam pattern matched above its calibrated threshold.";
  } else {
    list.textContent = hits.map((h) => CATEGORIES[h.label].name).join(" + ");
  }
}

function renderWhy(hits, weak) {
  const list = $("why-list");
  list.textContent = "";
  const rows = [...hits, ...weak];
  if (!rows.length) {
    const li = document.createElement("li");
    li.textContent = "No scam indicators found in this text.";
    list.appendChild(li);
    return;
  }
  for (const row of rows) {
    const li = document.createElement("li");
    if (row.weak) li.className = "weak";
    const head = document.createElement("div");
    head.className = "why-head";
    const name = document.createElement("span");
    name.textContent = (row.weak ? "Possible " : "") + CATEGORIES[row.label].name;
    const prob = document.createElement("span");
    prob.className = "why-prob";
    prob.textContent = Math.round(row.prob * 100) + "%";
    head.append(name, prob);
    const body = document.createElement("div");
    body.className = "why-body";
    body.textContent = CATEGORIES[row.label].blurb;
    li.append(head, body);
    list.appendChild(li);
  }
}

function renderPlaybook(hits) {
  const panel = $("playbook");
  const text = $("playbook-text");
  if (!hits.length) {
    panel.hidden = true;
    return;
  }
  const top = hits[0];
  text.textContent = CATEGORIES[top.label].action;
  panel.hidden = false;
}

function computeHits(probs, labelNames) {
  const hits = [];
  const weak = [];
  labelNames.forEach((label, i) => {
    const prob = probs[i];
    const row = { label, prob, weak: prob < labelThreshold(label) };
    if (!row.weak) hits.push(row);
    else if (prob >= CONFIG.WEAK_SIGNAL) weak.push(row);
  });
  hits.sort((a, b) => b.prob - a.prob);
  weak.sort((a, b) => b.prob - a.prob);
  return { hits, weak };
}

function renderScan(hits, weak, engineTag) {
  const top = hits[0];
  const score = top ? Math.min(100, Math.round(top.prob * 100)) : 0;
  renderRisk(score);
  renderLabels(hits);
  renderWhy(hits, weak);
  renderPlaybook(hits);
  $("result").hidden = false;
  const tag = $("engine-tag");
  tag.hidden = false;
  tag.textContent = engineTag;
  saveHistory({ score, hits: hits.map((h) => h.label), engine: engineTag, at: new Date().toISOString() });
}

function saveHistory(entry) {
  try {
    const key = "scamless_history";
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    history.unshift(entry);
    localStorage.setItem(key, JSON.stringify(history.slice(0, 50)));
  } catch { /* private mode: history is a nice-to-have, never block the scan */ }
}

function onScan() {
  const text = $("input").value.trim();
  if (!text) return;
  if (state.phase === "ready") {
    const labelNames = Object.keys(state.thresholds || {});
    const run = async () => {
      let probs;
      try {
        probs = await scanWithModel(text);
      } catch (err) {
        enterDemoMode("inference error: " + err.message);
        return onScan();
      }
      const names = labelNames.length ? labelNames : probs.map((_, i) => `label_${i}`);
      const { hits, weak } = computeHits(probs, names);
      renderScan(hits, weak, "full model - ran locally");
    };
    run();
  } else if (state.phase === "demo") {
    const hits = heuristic_scan(text).sort((a, b) => b.prob - a.prob);
    const weak = [];
    renderScan(hits, weak, "demo keyword rules - not the trained model");
  }
}

async function boot() {
  $("scan-btn").addEventListener("click", onScan);
  $("clear-btn").addEventListener("click", () => {
    $("input").value = "";
    $("result").hidden = true;
  });
  $("input").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onScan();
  });

  // transformers.js runs wasm/webgpu fully in-browser; no local server needed
  env.allowLocalModels = false;
  try {
    await loadModel();
  } catch (err) {
    enterDemoMode(err.message || "model repo unreachable");
  }
}

boot();
