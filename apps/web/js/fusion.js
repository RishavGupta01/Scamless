// Scamless fusion engine v1 - the spec's section-5 intelligence layer.
// Combines the trained model's probabilities with deterministic rule
// signals the classifier cannot see directly (URL structure, homoglyph
// attacks, OTP patterns, urgency language). Explainable by design: every
// point of the risk score traces to a named signal.
//
// This file is shared verbatim between the web app (js/) and the Chrome
// extension (root) - fetch-deps.py copies it at packaging time.

const SUSPICIOUS_TLDS = new Set([
  "xyz", "top", "info", "club", "online", "site", "icu", "buzz", "monster", "cfd",
]);
const URL_KEYWORDS = [
  "login", "verify", "secure", "account", "update", "confirm", "recover",
  "wallet", "refund", "billing", "kyc", "support", "unlock", "signin",
];
const BRANDS = [
  "paypal", "paytm", "google", "amazon", "apple", "facebook", "instagram",
  "whatsapp", "netflix", "sbi", "hdfc", "icici", "axis", "chase", "wellsfargo",
  "binance", "coinbase", "phonepe",
];
const OFFICIAL_HOSTS = new Set([
  "paypal.com", "paytm.com", "google.com", "amazon.com", "apple.com",
  "facebook.com", "instagram.com", "whatsapp.com", "netflix.com", "sbi.co.in",
  "onlinesbi.sbi", "hdfcbank.com", "icicibank.com", "axisbank.com", "chase.com",
  "wellsfargo.com", "binance.com", "coinbase.com", "phonepe.com",
]);

// Cyrillic + fullwidth lookalikes for common Latin scam targets
const HOMOGLYPHS = new Set("аеорсухіѕӏᴬ".concat("аеорсухі"));
const HOMOGLYPH_SET = new Set([
  ..."аеорсухі", // cyrillic а е о р с у х і
  ..."\u0430\u0435\u043E\u0440\u0441\u0443\u0445\u0456".split(""),
]);

const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>"')\]]+/gi;
const BARE_DOMAIN_RE =
  /\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|net|org|info|xyz|top|online|site|icu|club|in|co|io|ru|cn|buzz|cfd|monster)\b/gi;
const OTP_RE = /\b(?:otp|one[\s-]?time (?:password|code)|verification code|pin)\b[^.]{0,40}?\b(\d{4,8})\b|\b(\d{4,8})\b[^.]{0,40}\b(?:otp|verification code)\b/i;
const URGENCY_RE = /\b(?:urgent|immediately|final warning|last chance|act now|within \d+ (?:hours|hours|minutes)|suspended|deactivated|legal action|arrest)\b/gi;

const SIGNAL_WEIGHTS = {
  homoglyph: 25,
  url_punycode: 20,
  url_ip_host: 15,
  url_keywords: 12,
  url_brand_lookalike: 18,
  url_suspicious_tld: 10,
  url_deep_subdomain: 8,
  url_insecure: 6,
  otp_pattern: 15,
  urgency_language: 8,
  pii_request: 5,
};

function extractUrls(text) {
  const urls = new Set();
  for (const m of text.matchAll(URL_RE)) urls.add(m[0]);
  for (const m of text.matchAll(BARE_DOMAIN_RE)) urls.add("http://" + m[0]);
  return [...urls];
}

function urlRisk(url) {
  const signals = [];
  let host;
  try {
    host = new URL(url.startsWith("http") ? url : "http://" + url).hostname;
  } catch {
    return signals;
  }
  const h = host.toLowerCase();

  if (h.startsWith("xn--") || url.includes("xn--")) {
    signals.push({ id: "url_punycode", detail: `Punycode domain (can fake any brand): ${h}` });
  }
  if (/^\d{1,3}(\.\d{1,3}){3}$/.test(h)) {
    signals.push({ id: "url_ip_host", detail: `Raw IP address instead of a domain: ${h}` });
  }
  const parts = h.split(".");
  if (parts.length >= 4 && !OFFICIAL_HOSTS.has(h)) {
    signals.push({ id: "url_deep_subdomain", detail: `Unusually deep subdomain chain: ${h}` });
  }
  const tld = parts[parts.length - 1];
  if (SUSPICIOUS_TLDS.has(tld)) {
    signals.push({ id: "url_suspicious_tld", detail: `High-abuse domain extension ".${tld}"` });
  }
  const kw = URL_KEYWORDS.filter((k) => h.includes(k));
  if (kw.length && !OFFICIAL_HOSTS.has(h)) {
    signals.push({ id: "url_keywords", detail: `Phishing keywords in domain (${kw.join(", ")}): ${h}` });
  }
  const bare = h.replace(/^www\./, "");
  for (const brand of BRANDS) {
    if (bare.includes(brand) && !OFFICIAL_HOSTS.has(bare)) {
      signals.push({
        id: "url_brand_lookalike",
        detail: `Impersonates "${brand}" on a non-official domain: ${h}`,
      });
      break;
    }
  }
  if (url.startsWith("http://")) {
    signals.push({ id: "url_insecure", detail: "Insecure http:// link (no encryption)" });
  }
  return signals;
}

function dedupeSignals(signals) {
  const seen = new Set();
  return signals.filter((s) => (seen.has(s.id + s.detail) ? false : (seen.add(s.id + s.detail), true)));
}

export function analyzeText(text) {
  const signals = [];
  const boosts = {}; // label -> total weight

  const addBoost = (signal, ...labels) => {
    signals.push(signal);
    for (const label of labels) {
      boosts[label] = (boosts[label] || 0) + SIGNAL_WEIGHTS[signal.id] || 0;
    }
  };

  // 1. homoglyph attacks (lookalike unicode in latin words)
  const words = text.split(/\s+/);
  let homoglyphHit = null;
  for (const w of words) {
    if (w.length < 4) continue;
    const hasLatin = /[a-z]/i.test(w);
    const hasLookalike = [...w].some((c) => HOMOGLYPH_SET.has(c));
    if (hasLatin && hasLookalike) { homoglyphHit = w; break; }
  }
  if (homoglyphHit) {
    addBoost(
      { id: "homoglyph", detail: `Lookalike characters detected in "${homoglyphHit}" - a common brand-impersonation trick` },
      "phishing", "malicious_link"
    );
  }

  // 2. URL risk analysis
  const urls = extractUrls(text);
  const urlSignals = [];
  for (const url of urls.slice(0, 5)) {
    urlSignals.push(...urlRisk(url));
  }
  for (const s of dedupeSignals(urlSignals)) {
    addBoost(s, "malicious_link", "phishing");
  }
  if (urls.length >= 3) {
    signals.push({ id: "url_count", detail: `${urls.length} separate links in one message`, weight: 0 });
  }

  // 3. OTP extraction pattern
  if (OTP_RE.test(text)) {
    addBoost(
      { id: "otp_pattern", detail: "Message pairs an OTP / verification code with a request - classic account-takeover pattern" },
      "otp_request"
    );
  }

  // 4. urgency language
  const urgencyHits = text.match(URGENCY_RE) || [];
  if (urgencyHits.length >= 1) {
    addBoost(
      { id: "urgency_language", detail: `Pressure language detected (${urgencyHits.slice(0, 3).map((w) => `"${w}"`).join(", ")})` },
      "payment_pressure", "account_suspension", "phishing"
    );
  }

  // 5. PII request (informational, shown but never boosts alone)
  if (/\b(aadhaar|ssn|passport|card number|cvv|cvv2|date of birth)\b/i.test(text)) {
    signals.push({ id: "pii_request", detail: "Message asks about identity or card details", weight: 0 });
  }

  // rule-derived malicious_link hit: strong url evidence fills this label
  const urlStrong = urlSignals.length >= 2 || urlSignals.some((s) =>
    ["url_punycode", "url_brand_lookalike", "url_ip_host"].includes(s.id)
  );
  let ruleHit = null;
  if (urlStrong) {
    const boostTotal = boosts["malicious_link"] || 0;
    ruleHit = {
      label: "malicious_link",
      prob: Math.min(0.92, 0.5 + boostTotal * 0.02),
      engine: "rules",
    };
  }

  const totalBoost = Math.min(35, Object.values(boosts).reduce((a, b) => a + b, 0) * 0.35);
  return { signals, boosts, ruleHit, totalBoost, urls: urls.length };
}

export function fuse(modelProbs, labelNames, text, thresholds) {
  const analysis = analyzeText(text);
  const thresholdFor = (label) => {
    const t = thresholds && thresholds[label];
    return typeof t === "number" ? t : 0.5;
  };

  const hits = [];
  const weak = [];
  let topProb = 0;
  labelNames.forEach((label, i) => {
    const prob = modelProbs[i];
    const boost = analysis.boosts[label] || 0;
    // fusion: the model probability is nudged upward by matching rule signals
    const fused = Math.min(0.99, prob + boost / 250);
    topProb = Math.max(topProb, prob);
    const threshold = thresholdFor(label);
    const row = { label, prob, fused, weak: fused < threshold, engine: "model" };
    (row.weak ? weak : hits).push(row);
  });

  if (analysis.ruleHit) {
    const row = { ...analysis.ruleHit, weak: false };
    hits.push(row);
  }
  hits.sort((a, b) => (b.fused ?? b.prob) - (a.fused ?? a.prob));

  const driver = hits[0];
  const base = driver ? (driver.fused ?? driver.prob) * 100 : 0;
  const score = Math.min(100, Math.round(base + (driver && driver.engine === "model" ? analysis.totalBoost : 0)));

  return { hits, weak, signals: analysis.signals, score, urls: analysis.urls };
}
