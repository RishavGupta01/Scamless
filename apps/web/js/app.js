// Scamless web app - bulletproof boot sequence.
//
// Architecture:
// 1. Rule engine is available IMMEDIATELY (zero dependencies)
// 2. transformers.js loads dynamically (non-blocking) for model inference
// 3. If transformers.js or the model fails, rule engine continues working
//
// No static imports of remote modules. No blocking initialization.
// The site is never useless.

import { CONFIG } from "./config.js";
import { CATEGORIES, riskBand } from "./categories.js";
import { rule_scan } from "./heuristic.js";
import { fuse } from "./fusion.js";

const $ = (id) => document.getElementById(id);

const state = {
  phase: "demo", // starts as demo: rule engine works without any model
  tokenizer: null,
  model: null,
  thresholds: null,
  accelerated: "WASM",
};

// ---------------------------------------------------------------------------
// model loading (background, non-blocking)
// ---------------------------------------------------------------------------

let transformersModule = null;

async function loadTransformers() {
  if (!transformersModule) {
    transformersModule = await import(
      "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1"
    );
  }
  return transformersModule;
}

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

async function loadThresholds() {
  const url = `https://huggingface.co/${CONFIG.MODEL_REPO}/resolve/main/thresholds.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`thresholds fetch failed: ${res.status}`);
  return res.json();
}

async function loadModel() {
  const { AutoTokenizer, AutoModelForSequenceClassification } = await loadTransformers();

  setStatus("Downloading detection model (one time, cached after this)");
  const progress = wireProgress((loaded, total) => {
    const mb = (loaded / 1e6).toFixed(0);
    const totalMb = (total / 1e6).toFixed(0);
    const pct = total ? Math.round((loaded / total) * 100) : 0;
    $("load-progress").hidden = false;
    $("load-bar").style.width = pct + "%";
    $("load-detail").textContent = `${mb} / ${totalMb} MB - runs locally after this`;
  });

  // if the download takes > 8 minutes, abort
  const timeout = new Promise((_, reject) =>
    setTimeout(() => reject(new Error("model download timed out after 8 minutes")), 8 * 60 * 1000)
  );

  const load = async () => {
    state.tokenizer = await AutoTokenizer.from_pretrained(CONFIG.MODEL_REPO, { progress_callback: progress });
    state.model = await AutoModelForSequenceClassification.from_pretrained(CONFIG.MODEL_REPO, {
      progress_callback: progress,
      model_file_name: "model_int8",
      dtype: "fp32",
    });
    state.thresholds = await loadThresholds();
  };

  await Promise.race([load(), timeout]);
  state.phase = "ready";

  if (navigator.gpu) state.accelerated = "WebGPU";
  setStatus(`Model ready - running locally (${state.accelerated})`, "ready");
  $("load-progress").hidden = true;
  $("load-detail").textContent = "Inference happens inside this browser tab. Scan away.";
}

// ---------------------------------------------------------------------------
// inference
// ---------------------------------------------------------------------------

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

async function scanWithModel(text) {
  const inputs = state.tokenizer(text, {
    truncation: true,
    max_length: CONFIG.MAX_LEN,
    return_tensor: true,
  });
  const output = await state.model(inputs);
  const logits = output.logits.tolist()[0];
  return logits.map(sigmoid);
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------

function setStatus(text, cls) {
  $("status-text").textContent = text;
  $("status-dot").className = "dot" + (cls ? " " + cls : "");
}

function renderRisk(score) {
  const band = riskBand(score);
  const arc = $("meter-arc");
  const circumference = 2 * Math.PI * 52;
  arc.style.strokeDashoffset = String(circumference * (1 - score / 100));
  arc.style.stroke = band.color;
  animateScore($("meter-score"), score);
  $("verdict-band").textContent = band.label;
  $("verdict-band").style.color = band.color;
}

function animateScore(el, target) {
  const t0 = performance.now();
  const dur = 900;
  function step(now) {
    const t = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = String(Math.round(target * eased));
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const TACTIC_PATTERNS = [
  { cls: "link", re: /\b(?:https?:\/\/|www\.)[^\s<>"')\]]+/gi },
  { cls: "otp", re: /\b\d{6,8}\b/g },
  { cls: "money", re: /(?:₹|Rs\.?\s?|INR|\$|€|£)\s?\d[\d,]*(?:\.\d+)?/g },
  { cls: "urgency", re: /\b(?:urgent(?:ly)?|final warning|last chance|act now|immediately|within \d+ hours|digital arrest|virtual custody|legal action|arrest(?:ed)?)\b/gi },
];

function annotate(text, hits) {
  const panel = $("annotated-panel");
  if (!hits.length) { panel.hidden = true; return; }

  const ranges = [];
  for (const { cls, re } of TACTIC_PATTERNS) {
    re.lastIndex = 0;
    for (const m of text.matchAll(re)) {
      ranges.push({ start: m.index, end: m.index + m[0].length, cls, text: m[0] });
    }
  }
  ranges.sort((a, b) => a.start - b.start || b.end - a.start);
  const picked = [];
  let lastEnd = -1;
  for (const r of ranges) {
    if (r.start >= lastEnd) { picked.push(r); lastEnd = r.end; }
  }

  const host = $("annotated-text");
  host.textContent = "";
  let cursor = 0;
  for (const r of picked) {
    if (r.start > cursor) host.appendChild(document.createTextNode(text.slice(cursor, r.start)));
    const mark = document.createElement("mark");
    mark.className = r.cls;
    mark.textContent = r.text;
    host.appendChild(mark);
    cursor = r.end;
  }
  if (cursor < text.length) host.appendChild(document.createTextNode(text.slice(cursor)));
  panel.hidden = false;
}

function renderLabels(hits, weak) {
  const list = $("verdict-labels");
  list.textContent = "";
  if (hits.length) {
    list.textContent = hits.map((h) =>
      CATEGORIES[h.label].name + (h.engine === "rules" ? " (link analysis)" : "")
    ).join(" + ");
  } else if (weak.length) {
    list.textContent = "Weak signals detected - below alert threshold";
  } else {
    list.textContent = "No scam indicators detected.";
  }
}

function whyRow(head, prob, body, cls) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  const headEl = document.createElement("div");
  headEl.className = "why-head";
  const name = document.createElement("span");
  name.textContent = head;
  const probEl = document.createElement("span");
  probEl.className = "why-prob";
  if (prob !== null) probEl.textContent = Math.round(prob * 100) + "%";
  headEl.append(name, probEl);
  const bodyEl = document.createElement("div");
  bodyEl.className = "why-body";
  bodyEl.textContent = body;
  li.append(headEl, bodyEl);
  return li;
}

let _signals = [];

function renderWhy(all) {
  const list = $("why-list");
  list.textContent = "";

  const relevant = all
    .filter((r) => (r.fused ?? r.prob) >= 0.05)
    .sort((a, b) => (b.fused ?? b.prob) - (a.fused ?? a.prob))
    .slice(0, 8);

  if (!relevant.length && !_signals.length) {
    const li = document.createElement("li");
    li.textContent = "No scam indicators found in this text.";
    list.appendChild(li);
    return;
  }

  for (const r of relevant) {
    const isHit = r.fused !== undefined && r.fused >= (r.threshold || 0.5);
    const head = (isHit ? "" : "Possible ") + CATEGORIES[r.label].name;
    const prob = r.fused ?? r.prob;
    list.appendChild(whyRow(head, prob, CATEGORIES[r.label].blurb, isHit ? "" : "weak"));
  }

  for (const s of _signals) {
    list.appendChild(whyRow("Signal: " + s.id.replace(/_/g, " "), null, s.detail, "weak"));
  }
}

function renderPlaybook(hits) {
  const panel = $("playbook");
  const text = $("playbook-text");
  if (!hits.length) { panel.hidden = true; return; }
  text.textContent = CATEGORIES[hits[0].label].action;
  panel.hidden = false;
}

function renderScan(all, score, engineTag, text) {
  renderRisk(score);
  const hits = all.filter((r) => r.fused !== undefined && r.fused >= (r.threshold || 0.5));
  renderLabels(hits, all.filter((r) => !hits.includes(r)));
  renderWhy(all);
  renderPlaybook(hits);
  annotate(text, hits);
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
  } catch { /* private mode */ }
}

// ---------------------------------------------------------------------------
// scan dispatch
// ---------------------------------------------------------------------------

const SAMPLES = [
  { label: "Safe (real bank OTP)", text: "Your SBI OTP for net banking login is 448210. Valid for 10 minutes. Never share this OTP with anyone including bank staff. -SBI" },
  { label: "Phishing (KYC)", text: "Dear customer your KYC has expired!! update it immediately on secure-sbi-kyc.com otherwise your account will be blocked within 24hrs - SBI" },
  { label: "OTP theft attempt", text: "Sir aapke card se 62,000 ki shopping ho rahi hai abhi. Block karne ke liye jo OTP aaya hai wo batao jaldi" },
  { label: "Task scam", text: "Ghar baithe 4,000 rozana kamao. Simple copy paste ka kaam. Sirf 1,200 ka activation charge hai. Aaj se pehla payment kal milega" },
  { label: "Digital arrest", text: "Sir I am officer Rajesh from CBI. There is a complaint against your Aadhaar for money laundering. This is virtual police notice. Transfer the verification amount immediately" },
  { label: "Safe (delivery)", text: "Delivered: Your Amazon order #402-8817734-9912 (1 item) was handed directly to resident at 2:42 PM. No signature required. Thank you for shopping with us" },
];

async function onScan() {
  const text = $("input").value.trim();
  if (!text) return;
  $("scan-btn").disabled = true;

  try {
    if (state.phase === "ready") {
      let probs;
      try {
        probs = await scanWithModel(text);
      } catch {
        // model inference failed: fall back to rule engine for this scan
        probs = null;
      }
      if (probs) {
        const labelNames = Object.keys(state.thresholds || {});
        const names = labelNames.length ? labelNames : probs.map((_, i) => `label_${i}`);
        const fused = fuse(probs, names, text, state.thresholds);
        _signals = fused.signals;
        renderScan(fused.hits.concat(fused.weak), fused.score, `full model - ran locally (${state.accelerated})`, text);
      } else {
        const result = rule_scan(text);
        _signals = result.signals;
        renderScan(
          [...result.hits, ...result.weak].map((r) => ({ ...r, threshold: 0.5 })),
          result.score,
          "rule engine - full model not loaded",
          text
        );
      }
    } else {
      // demo mode (model not loaded yet)
      const result = rule_scan(text);
      _signals = result.signals;
      renderScan(
        [...result.hits, ...result.weak].map((r) => ({ ...r, threshold: 0.5 })),
        result.score,
        "rule engine - full model not loaded",
        text
      );
    }
  } finally {
    $("scan-btn").disabled = false;
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------

const SAMPLE_LABELS = SAMPLES;

function renderSamples() {
  const box = $("samples");
  box.textContent = "";
  const label = document.createElement("span");
  label.className = "samples-label";
  label.textContent = "Try an example:";
  box.appendChild(label);
  for (const sample of SAMPLE_LABELS) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = sample.label;
    chip.addEventListener("click", () => {
      $("input").value = sample.text;
      onScan();
    });
    box.appendChild(chip);
  }
}

async function boot() {
  // 1. wire up UI - the site is NEVER useless
  $("scan-btn").addEventListener("click", onScan);
  $("clear-btn").addEventListener("click", () => {
    $("input").value = "";
    $("result").hidden = true;
  });
  $("input").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onScan();
  });

  // 2. rule engine works IMMEDIATELY - zero dependencies
  state.phase = "demo";
  setStatus("Rule engine ready - model loading in background", "ready");
  $("load-detail").textContent = "Scanning with pattern analysis. Trained model loading in background...";
  $("scan-btn").disabled = false;
  renderSamples();

  // 3. model downloads in background (non-blocking)
  //    when it arrives, future scans use the trained model + fusion
  loadModel().catch((err) => {
    setStatus("Rule engine active (model unavailable: " + err.message + ")", "error");
    $("load-detail").textContent = "The rule engine is still fully functional.";
  });
}

boot();
