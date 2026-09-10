// Scamless offscreen document: hosts the ONNX model via transformers.js.
// Loads once per browser session (weights cached in this document's
// IndexedDB by transformers.js), then answers scan requests forever.
// Zero network calls during scanning; the model files arrive once on setup.

import { AutoTokenizer, AutoModelForSequenceClassification } from "../vendor/transformers.min.js";
import { fuse } from "../fusion.js";
import { rule_scan } from "../heuristic.js";

const MODEL_REPO = "RishavGupta01/scamless-model-v1";
const MAX_LEN = 256;

let tokenizer = null;
let model = null;
let thresholds = null;

let modelFailed = false;

async function ensureModel() {
  if (model || modelFailed) return;
  try {
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_REPO);
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_REPO, {
    model_file_name: "model_int8",
    dtype: "fp32",
  });
    const res = await fetch(`https://huggingface.co/${MODEL_REPO}/resolve/main/thresholds.json`);
    if (!res.ok) throw new Error(`thresholds fetch failed: ${res.status}`);
    thresholds = await res.json();
  } catch (err) {
    // model unavailable (offline first run, HF outage): degrade to the rule
    // engine instead of dead-ending - verdicts are tagged so the popup can
    // show which engine produced them
    modelFailed = true;
    throw err;
  }
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (!msg || msg.target !== "offscreen" || msg.type !== "scan") return;

  (async () => {
    try {
      await ensureModel();
    } catch (err) {
      // rule-engine fallback: same response shape, tagged engine
      const result = rule_scan(msg.text);
      sendResponse({
        ok: true,
        hits: result.hits,
        weak: result.weak,
        signals: result.signals,
        score: result.score,
        engine: "rules",
      });
      return;
    }

    try {
      const enc = tokenizer(msg.text, {
        truncation: true,
        max_length: MAX_LEN,
        return_tensor: true,
      });
      const output = await model(enc);
      const probs = output.logits.tolist()[0].map(sigmoid);

      const { sensitivity = 0 } = await chrome.storage.sync.get({ sensitivity: 0 });
      const margin = Number(sensitivity) * 0.0015; // slider 0..100 -> +0..0.15

      const labelNames = Object.keys(thresholds);
      const fused = fuse(probs, labelNames, msg.text, thresholds);
      const hits = [];
      const weak = [];
      labelNames.forEach((label, i) => {
        const threshold = Math.min(0.99, thresholds[label] + margin);
        const row = { label, prob: probs[i], weak: probs[i] < threshold };
        (row.weak ? weak : hits).push(row);
      });
      // fold rule-derived hits (e.g. malicious_link from link analysis) in
      for (const ruleHit of fused.hits.filter((h) => h.engine === "rules")) {
        if (!hits.some((h) => h.label === ruleHit.label)) hits.push(ruleHit);
      }
      hits.sort((a, b) => (b.fused ?? b.prob) - (a.fused ?? a.prob));

      const score = fused.score;
      sendResponse({ ok: true, hits, weak, signals: fused.signals, score, engine: "model" });
    } catch (err) {
      sendResponse({ ok: false, error: String((err && err.message) || err) });
    }
  })();

  return true; // async sendResponse
});
