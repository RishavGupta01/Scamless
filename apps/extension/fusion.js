// Scamless fusion engine v1.1 - the spec's section-5 intelligence layer.
// Combines the trained model's probabilities with deterministic rule
// signals (URL analysis, homoglyph attacks, OTP patterns, urgency language).
// Every point of the risk score traces to a named signal.
//
// Cross-label agreement: when multiple scam labels fire at moderate
// probability simultaneously, the top label gets boosted - multiple weak
// indicators pointing at the same conclusion are stronger than any alone.
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

const HOMOGLYPH_CHARS = new Set([
  ..."\u0430\u0435\u043E\u0440\u0441\u0443\u0445\u0456", // cyrillic а е о р с у х і
]);

const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>"')\]]+/gi;
const BARE_DOMAIN_RE =
  /\b(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+(?:com|net|org|info|xyz|top|online|site|icu|in|co|io|ru|cn|buzz|cfd|monster)\b/gi;
const OTP_RE = /\b(?:otp|one[\s-]?time (?:password|code)|verification code|pin)\b[^.]{0,40}?\b(\d{4,8})\b|\b(\d{4,8})\b[^.]{0,40}\b(?:otp|verification code)\b/i;
const URGENCY_RE = /\b(?:urgent|immediately|final warning|last chance|act now|within \d+ (?:hours|minutes)|suspended|deactivated|legal action|arrest)\b/gi;
const DIGITAL_ARREST_RE = /\b(?:digital arrest|virtual custody|video call (?:with )?(?:police|officer|constable)|skype (?:hearing|investigation)|cbi (?:officer|case)|cyber (?:cell|police) (?:case|notice))\b/i;
const UPI_REVERSAL_RE = /\b(?:enter|put|dial)\b[^.]{0,30}\b(?:upi )?pin\b[^.]{0,40}\b(?:receive|refund|get)\b|\b(?:receive|refund)\b[^.]{0,30}\b(?:upi )?pin\b/i;
const PRIZE_FEE_RE = /\b(?:won|winner|prize|lottery|lucky draw)\b/i;
const FEE_RE = /\b(?:fee|charge|gst|customs|processing|registration|deposit)\b/i;

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
  digital_arrest: 20,
  upi_pin_reversal: 18,
  prize_fee_combo: 15,
  kyc_link: 12,
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
  const boosts = {};

  const addBoost = (signal, ...labels) => {
    signals.push(signal);
    for (const label of labels) {
      boosts[label] = (boosts[label] || 0) + (SIGNAL_WEIGHTS[signal.id] || 0);
    }
  };

  // 1. homoglyph attacks
  const words = text.split(/\s+/);
  let homoglyphHit = null;
  for (const w of words) {
    if (w.length < 4) continue;
    const hasLatin = /[a-z]/i.test(w);
    const hasLookalike = [...w].some((c) => HOMOGLYPH_CHARS.has(c));
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

  // 4. digital arrest
  if (DIGITAL_ARREST_RE.test(text)) {
    addBoost(
      { id: "digital_arrest", detail: "Claims of virtual police custody or remote arrest - real law enforcement never operates this way" },
      "gov_bank_impersonation", "payment_pressure"
    );
  }

  // 5. UPI PIN reversal
  if (UPI_REVERSAL_RE.test(text)) {
    addBoost(
      { id: "upi_pin_reversal", detail: "Asks for your UPI PIN in a money-RECEIVING context - entering a PIN always SENDS money" },
      "otp_request", "payment_pressure"
    );
  }

  // 6. prize + fee combination
  const isPrize = PRIZE_FEE_RE.test(text);
  const isFee = FEE_RE.test(text);
  if (isPrize && isFee) {
    addBoost(
      { id: "prize_fee_combo", detail: "A prize or win combined with a fee or charge - real prizes never cost money to claim" },
      "lottery_prize", "advance_fee"
    );
  }

  // 7. KYC urgency paired with a link
  if (/\bkyc\b/i.test(text) && urls.length > 0) {
    addBoost(
      { id: "kyc_link", detail: "KYC update pushed through a non-official link" },
      "gov_bank_impersonation", "phishing"
    );
  }

  // 8. urgency language
  const urgencyHits = text.match(URGENCY_RE) || [];
  if (urgencyHits.length >= 1) {
    addBoost(
      { id: "urgency_language", detail: `Pressure language detected (${urgencyHits.slice(0, 3).map((w) => `"${w}"`).join(", ")})` },
      "payment_pressure", "account_suspension", "phishing"
    );
  }

  // 9. PII request (informational)
  if (/\b(aadhaar|ssn|passport|card number|cvv|cvv2|date of birth)\b/i.test(text)) {
    signals.push({ id: "pii_request", detail: "Message asks about identity or card details", weight: 0 });
  }

  // 10. e-commerce legitimacy dampener: known safe patterns that counter
  // the false-positive problem on delivery confirmations and order updates.
  // These signals SUBTRACT from scam scores when they match strongly.
  let dampener = 0;
  const dampDetails = [];
  if (/\b(?:order|tracking)\s*#?\s*\d{4,}[-\d]*\b/i.test(text)) {
    dampener += 0.12;
    dampDetails.push("valid order/tracking number format");
  }
  if (/\b(?:handed|delivered|shipped|dispatched|out for delivery)\b/i.test(text) &&
      /\b(?:resident|recipient|doorstep|mailbox|reception|guard)\b/i.test(text)) {
    dampener += 0.10;
    dampDetails.push("standard delivery confirmation language");
  }
  if (/\bthank you for (?:shopping|your order|choosing)\b/i.test(text)) {
    dampener += 0.08;
    dampDetails.push("standard merchant closing");
  }
  if (/\b(?:no signature required|no action (?:needed|required)|no payment)\b/i.test(text)) {
    dampener += 0.10;
    dampDetails.push("explicitly states no action or payment needed");
  }
  if (/\b(?:your|the)\s+(?:bill|invoice|statement)\s+(?:is attached|was sent|is ready)\b/i.test(text)) {
    dampener += 0.08;
    dampDetails.push("standard billing notification");
  }
  if (dampener > 0) {
    signals.push({
      id: "ecommerce_dampener",
      detail: `Legitimate e-commerce patterns detected (${dampDetails.join("; ")})`,
      dampen: Math.min(0.35, dampener),
    });
  }

  // rule-derived malicious_link: strong URL evidence fills this label
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

  // per-label fused probabilities
  // e-commerce dampener: suppresses scam scores when legitimate patterns match
  let dampen = 0;
  for (const s of analysis.signals) {
    if (s.dampen) dampen = Math.max(dampen, s.dampen);
  }

  const all = [];
  labelNames.forEach((label, i) => {
    const prob = modelProbs[i];
    const boost = analysis.boosts[label] || 0;
    let fused = Math.min(0.99, prob + boost / 250);
    // dampener suppresses scam probability when e-commerce legitimacy signals fire
    if (dampen > 0) fused = Math.max(0, fused - dampen * prob);
    all.push({ label, prob: fused, fused, threshold: thresholdFor(label) });
  });

  // cross-label agreement: when 2+ labels fire above 0.40, boost the top
  // label's fused score - combined evidence is stronger than any alone
  const strongSignals = all.filter((r) => r.fused >= 0.40 && r.label !== "generic_spam");
  if (strongSignals.length >= 2) {
    const top = all.reduce((a, b) => (b.fused > a.fused ? b : a));
    const bonus = Math.min(0.15, strongSignals.length * 0.05);
    top.fused = Math.min(0.99, top.fused + bonus);
  }

  // split into hits (crossed calibrated threshold) and weak (meaningful but below)
  const hits = all.filter((r) => r.fused >= r.threshold);
  const weak = all.filter((r) => r.fused < r.threshold && r.fused >= 0.25);
  hits.sort((a, b) => b.fused - a.fused);
  weak.sort((a, b) => b.fused - a.fused);

  // rule-derived malicious_link
  if (analysis.ruleHit && !hits.some((h) => h.label === "malicious_link")) {
    const row = { ...analysis.ruleHit, fused: analysis.ruleHit.prob, threshold: 0.5 };
    if (row.fused >= 0.5) hits.push(row);
    else if (row.fused >= 0.25) weak.push(row);
  }
  hits.sort((a, b) => b.fused - a.fused);
  weak.sort((a, b) => b.fused - a.fused);

  // risk score: based on the strongest fused probability across ALL labels,
  // not just hits - the model seeing 72% gov impersonation IS a risk signal
  // even if it doesn't cross the calibrated threshold
  const topFused = all.length ? Math.max(...all.map((r) => r.fused)) : 0;
  let score = Math.min(100, Math.round(topFused * 100));
  if (analysis.ruleHit && hits.some((h) => h.label === "malicious_link")) {
    score = Math.max(score, Math.round(analysis.ruleHit.prob * 100));
  }

  return { hits, weak, signals: analysis.signals, score, urls: analysis.urls };
}
