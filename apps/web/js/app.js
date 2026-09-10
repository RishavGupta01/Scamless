// Scamless web app: on-device inference via transformers.js (ONNX Runtime Web).
// Flow: load thresholds -> load model (progress shown) -> scan -> sigmoid probs
// -> fusion (model + rule signals) -> risk score -> verdict + why panel.

import { AutoTokenizer, AutoModelForSequenceClassification, env } from
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1";
import { CONFIG } from "./config.js";
import { CATEGORIES, riskBand } from "./categories.js";
import { rule_scan } from "./heuristic.js";
import { fuse } from "./fusion.js";

const $ = (id) => document.getElementById(id);

const state = {
  phase: "idle", // idle | loading | ready | demo
  tokenizer: null,
  model: null,
  thresholds: null,
  accelerated: "WASM",
};

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
  $("status-dot").className = "dot" + (cls ? " " + cls : "");
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

  state.tokenizer = await AutoTokenizer.from_pretrained(CONFIG.MODEL_REPO, { progress_callback: progress });
  state.model = await AutoModelForSequenceClassification.from_pretrained(CONFIG.MODEL_REPO, {
    progress_callback: progress,
    model_file_name: "model_int8",
    dtype: "fp32", // artifact is fp32 content under the model_int8 name; don't let transformers.js append _quantized
  });
  state.thresholds = await loadThresholds();
  state.phase = "ready";

  if (navigator.gpu) state.accelerated = "WebGPU";
  setStatus(`Model ready - running locally (${state.accelerated})`, "ready");
  $("load-progress").hidden = true;
  $("load-detail").textContent = "Inference happens inside this browser tab. Scan away.";
  $("scan-btn").disabled = false;
  renderSamples();
}

function enterDemoMode(reason) {
  state.phase = "demo";
  setStatus("Rule engine active - full model unavailable", "error");
  $("load-detail").textContent =
    `Full model could not load (${reason}). Scans use the built-in rule engine: ` +
    "weighted pattern analysis + link/homoglyph/OTP signals. Reload later for the trained model.";
  $("load-progress").hidden = true;
  $("scan-btn").disabled = false;
  renderSamples();
}

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

// tactic span highlighting: mark urgency language, links, codes and amounts
// directly inside the scanned message (rule-based, deterministic)
const TACTIC_PATTERNS = [
  { cls: "link", re: /\b(?:https?:\/\/|www\.)[^\s<>"')\]]+/gi },
  { cls: "otp", re: /\b\d{4,8}\b/g },
  { cls: "money", re: /(?:rs\.?\s?|inr\s?|\$|€|£|₹)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|lakh|crore|million))?|\b\d[\d,]+\s?(?:rupees|dollars|euros)\b/gi },
  { cls: "urgency", re: /\b(?:urgent(?:ly)?|immediately|final warning|last chance|act now|suspended|deactivated|legal action|arrest(?:ed)?|digital arrest|virtual custody|expire[sd]?|perm(?:anently)?antly?)\b/gi },
];

function annotate(text) {
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
}

function renderLabels(hits) {
  const list = $("verdict-labels");
  list.textContent = "";
  if (!hits.length) {
    list.textContent = "No scam pattern matched above its calibrated threshold.";
  } else {
    list.textContent = hits.map((h) => CATEGORIES[h.label].name + (h.engine === "rules" ? " (link analysis)" : "")).join(" + ");
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

function renderWhy(hits, weak, signals) {
  const list = $("why-list");
  list.textContent = "";
  if (!hits.length && !weak.length && !signals.length) {
    const li = document.createElement("li");
    li.textContent = "No scam indicators found in this text.";
    list.appendChild(li);
    return;
  }
  for (const h of hits) {
    list.appendChild(whyRow(
      (h.engine === "rules" ? "Link analysis: " : "") + CATEGORIES[h.label].name,
      h.fused ?? h.prob,
      CATEGORIES[h.label].blurb
    ));
  }
  for (const w of weak) {
    list.appendChild(whyRow(
      "Possible " + CATEGORIES[w.label].name,
      w.fused ?? w.prob,
      CATEGORIES[w.label].blurb,
      "weak"
    ));
  }
  for (const s of signals) {
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

function computeHits(modelProbs, labelNames) {
  const thresholdFor = (label) => {
    const t = state.thresholds && state.thresholds[label];
    return typeof t === "number" ? t : 0.5;
  };
  const hits = [];
  const weak = [];
  labelNames.forEach((label, i) => {
    const prob = modelProbs[i];
    const row = { label, prob, weak: prob < thresholdFor(label), engine: "model" };
    (row.weak ? weak : hits).push(row);
  });
  hits.sort((a, b) => b.prob - a.prob);
  weak.sort((a, b) => b.prob - a.prob);
  return { hits, weak };
}

function renderScan(hits, weak, signals, score, engineTag, text) {
  renderRisk(score);
  renderLabels(hits);
  renderWhy(hits, weak, signals);
  renderPlaybook(hits);
  if (text) {
    annotate(text);
    $("annotated-panel").hidden = false;
  } else {
    $("annotated-panel").hidden = true;
  }
  $("result").hidden = false;
  const tag = $("engine-tag");
  tag.hidden = false;
  tag.textContent = engineTag;
  saveHistory({ score, hits: hits.map((h) => h.label), engine: engineTag, at: new Date().toISOString() });
}

function onScan() {
  const text = $("input").value.trim();
  if (!text) return;
  if (state.phase === "ready") {
    (async () => {
      let probs;
      try {
        probs = await scanWithModel(text);
      } catch (err) {
        enterDemoMode("inference error: " + err.message);
        return onScan();
      }
      const labelNames = Object.keys(state.thresholds || {});
      const names = labelNames.length ? labelNames : probs.map((_, i) => `label_${i}`);
      const modelHits = computeHits(probs, names);
      const fused = fuse(probs, names, text, state.thresholds);
      renderScan(fused.hits, fused.weak, fused.signals, fused.score, `full model - ran locally (${state.accelerated})`, text);
    })();
  } else if (state.phase === "demo") {
    const result = rule_scan(text);
    renderScan(result.hits, result.weak, result.signals, result.score, "rule engine - full model not loaded", text);
  }
}

function saveHistory(entry) {
  try {
    const key = "scamless_history";
    const history = JSON.parse(localStorage.getItem(key) || "[]");
    history.unshift(entry);
    localStorage.setItem(key, JSON.stringify(history.slice(0, 50)));
  } catch { /* private mode: history is a nice-to-have, never block the scan */ }
}

const SAMPLES = [
  { label: "Safe (real bank OTP)", text: "Your SBI OTP for net banking login is 448210. Valid for 10 minutes. Never share this OTP with anyone including bank staff. -SBI" },
  { label: "Phishing (KYC)", text: "Dear customer your KYC has expired!! update it immediately on secure-sbi-kyc.com otherwise your account will be blocked within 24hrs - SBI" },
  { label: "OTP theft attempt", text: "Sir aapke card se 62,000 ki shopping ho rahi hai abhi. Block karne ke liye jo OTP aaya hai wo batao jaldi" },
  { label: "Task scam", text: "Ghar baithe 4,000 rozana kamao. Simple copy paste ka kaam. Sirf 1,200 ka activation charge hai. Aaj se pehla payment kal milega" },
];

function renderSamples() {
  const box = $("samples");
  box.textContent = "";
  const label = document.createElement("span");
  label.className = "samples-label";
  label.textContent = "Try an example:";
  box.appendChild(label);
  for (const sample of SAMPLES) {
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
  $("scan-btn").addEventListener("click", onScan);
  $("clear-btn").addEventListener("click", () => {
    $("input").value = "";
    $("result").hidden = true;
  });
  $("input").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") onScan();
  });

  env.allowLocalModels = false;
  try {
    await loadModel();
  } catch (err) {
    enterDemoMode(err.message || "model repo unreachable");
  }
}

boot();
