// Scamless popup: paste-scan, verdict display, sensitivity slider, and the
// local-only correction loop (spec section 9 - "this was wrong" feeds future
// training data; nothing is ever uploaded).

const $ = (id) => document.getElementById(id);

function bandFor(score) {
  if (score >= 70) return { text: "DANGEROUS", cls: "danger" };
  if (score >= 40) return { text: "SUSPICIOUS", cls: "susp" };
  return { text: "SAFE", cls: "safe" };
}

let lastScan = null;

function render(verdict) {
  const result = $("result");
  const band = $("band");
  const hits = $("hits");
  hits.textContent = "";

  if (!verdict.ok) {
    band.className = "band danger";
    band.textContent = "Scan failed";
    const err = document.createElement("div");
    err.className = "hint";
    err.textContent = verdict.error || "unknown error";
    hits.appendChild(err);
    result.hidden = false;
    return;
  }

  lastScan = verdict;
  const top = verdict.hits[0];
  const score = top ? Math.min(100, Math.round(top.prob * 100)) : 0;
  const b = bandFor(score);
  band.className = "band " + b.cls;
  band.textContent = top ? `${b.text} - risk ${score}/100` : "SAFE - no scam pattern matched";

  const rows = (verdict.hits.length ? verdict.hits : verdict.weak).slice(0, 5);
  for (const hit of rows) {
    const row = document.createElement("div");
    row.className = "hit";
    const name = document.createElement("span");
    name.textContent = (hit.weak ? "possible " : "") + hit.label.replace(/_/g, " ");
    name.className = hit.weak ? "weak-label" : "";
    const prob = document.createElement("span");
    prob.className = "prob";
    prob.textContent = Math.round(hit.prob * 100) + "%";
    row.append(name, prob);
    hits.appendChild(row);
  }
  result.hidden = false;
}

async function scan() {
  const text = $("input").value.trim();
  const status = $("status");
  if (!text) return;
  $("scan").disabled = true;
  status.textContent = "scanning...";
  try {
    const verdict = await chrome.runtime.sendMessage({ type: "scan-request", text });
    render(verdict);
    status.textContent = "";
  } catch (err) {
    status.textContent = "error: " + (err && err.message);
  } finally {
    $("scan").disabled = false;
  }
}

async function markWrong() {
  if (!lastScan) return;
  const { corrections = [] } = await chrome.storage.local.get({ corrections: [] });
  corrections.push({
    at: new Date().toISOString(),
    verdict: lastScan,
    marked: "wrong",
  });
  await chrome.storage.local.set({ corrections });
  $("status").textContent = "correction saved locally";
}

async function exportCorrections() {
  const { corrections = [] } = await chrome.storage.local.get({ corrections: [] });
  const blob = new Blob([JSON.stringify(corrections, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "scamless-corrections.json";
  link.click();
  URL.revokeObjectURL(url);
}

async function initSensitivity() {
  const { sensitivity = 0 } = await chrome.storage.sync.get({ sensitivity: 0 });
  $("sensitivity").value = sensitivity;
  updateSensitivityLabel(sensitivity);
  $("sensitivity").addEventListener("input", async (e) => {
    const value = Number(e.target.value);
    updateSensitivityLabel(value);
    await chrome.storage.sync.set({ sensitivity: value });
  });
}

function updateSensitivityLabel(value) {
  $("sens-value").textContent =
    value < 25 ? "lenient" : value < 60 ? "balanced" : value < 85 ? "strict" : "maximum";
}

document.addEventListener("DOMContentLoaded", () => {
  $("scan").addEventListener("click", scan);
  $("wrong").addEventListener("click", markWrong);
  $("export").addEventListener("click", (e) => {
    e.preventDefault();
    exportCorrections();
  });
  initSensitivity();
});
