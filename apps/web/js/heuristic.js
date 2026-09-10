// Scamless rule engine - the no-model fallback detector.
//
// Weighted regex rules across all 15 scam categories, structural signals
// from the fusion engine (URL analysis, homoglyphs, OTP patterns), and
// benign dampeners so legitimate bank alerts and receipts score low.
// Per-label scores map to probability-like values through a saturating
// curve, producing the same {hits, weak, signals, score} shape as the
// trained model - the UI is identical in both modes.

import { analyzeText } from "./fusion.js";

// weight: how strongly this pattern indicates the label (2+ = strong)
// damp: benign patterns that subtract from the scam score
const RULES = [
  // ---- phishing ----
  { label: "phishing", w: 3.0, re: /(?:verify|confirm|validate|update|restore|reactivate|unlock|re[- ]?activate)\s+(?:your\s+|the\s+)?(?:account|identity|billing|payment|details|information|card|credentials)/i },
  { label: "phishing", w: 2.5, re: /(?:click|tap|press)\s+(?:here|the link|below|this link)/i },
  { label: "phishing", w: 2.0, re: /log\s?in\s+(?:to|into|at|on)\s+(?:your|the)\s+(?:account|portal|dashboard)/i },
  { label: "phishing", w: 2.5, re: /(?:password|credentials|netbanking|net banking|internet banking)\s+(?:expir\w+|update|renew|confirm|invalid|required)/i },
  { label: "phishing", w: 2.0, re: /(?:enter|provide|submit)\s+(?:your\s+)?(?:card|credit card|debit card|banking)\s+(?:number|details|information|pin)/i },
  { label: "phishing", w: 1.5, re: /(?:secure|official|verified)\s+(?:link|portal|page|website|site)/i },

  // ---- otp_request ----
  { label: "otp_request", w: 3.0, re: /(?:share|send|give|tell|provide|batao|bata|do)\b[^.]{0,30}\b(?:otp|one[\s-]?time password|verification code|pin|cvv|cvv2)/i },
  { label: "otp_request", w: 2.5, re: /\b(?:otp|verification code|pin)\b[^.]{0,40}\b(?:share|send|tell|provide|confirm|enter|batao)\b/i },
  { label: "otp_request", w: 2.0, re: /(?:never share|do not share|kisi se share)\b.{0,40}\b(?:otp|code|pin)\b/i },

  // ---- payment_pressure ----
  { label: "payment_pressure", w: 2.5, re: /(?:pay|send|transfer|chukwana)\b[^.]{0,40}\b(?:immediately|now|today|urgently|turant|abhi)\b/i },
  { label: "payment_pressure", w: 2.0, re: /(?:overdue|outstanding|pending payment|unpaid|bakaya|due amount)\b/i },
  { label: "payment_pressure", w: 2.0, re: /(?:legal action|court case|lawsuit|police case|notice will be sent|arrester?| warrant)/i },
  { label: "payment_pressure", w: 2.0, re: /(?:disconnec\w+|cut off|seal|reconnection)\b.{0,50}\b(?:today|tonight|24 hours|48 hours|tomorrow)/i },
  { label: "payment_pressure", w: 2.5, re: /(?:hospital|surgery|medical emergency|accident|icu)\b.{0,80}\b(?:money|amount|deposit|lakh|rupees|cash)/i },
  { label: "payment_pressure", w: 2.0, re: /(?:stuck|stranded|lost (?:my )?(?:wallet|baggage|passport|phone))\b.{0,80}\b(?:send|need|money|help)/i },

  // ---- investment_crypto ----
  { label: "investment_crypto", w: 3.0, re: /(?:guaranteed|assured|fixed)\s+(?:returns?|profits?|income|payout)/i },
  { label: "investment_crypto", w: 2.5, re: /(?:double|triple|10x|2x|3x|5x|multiply)\s+(?:your\s+)?(?:money|investment|capital|profits?|amount)/i },
  { label: "investment_crypto", w: 2.0, re: /(?:crypto|bitcoin|btc|forex|trading|trading bot|stock market)\b.{0,40}\b(?:profit|returns?|signal|earn)/i },
  { label: "investment_crypto", w: 2.0, re: /(?:daily|weekly|monthly)\s+(?:returns?|profit|income|payout|withdrawal)/i },
  { label: "investment_crypto", w: 1.5, re: /(?:limited slots|few slots|closing tonight|last chance to invest)/i },

  // ---- lottery_prize ----
  { label: "lottery_prize", w: 2.5, re: /(?:won|winner|winning)\b.{0,60}\b(?:lottery|prize|draw|jackpot|sweepstake|cash)/i },
  { label: "lottery_prize", w: 2.0, re: /(?:lucky draw|lucky dip|mega draw|bumper (?:draw|prize)|raffle)/i },
  { label: "lottery_prize", w: 2.0, re: /(?:claim|redeem)\s+(?:your|the|a)\s+(?:prize|reward|winnings?|amount)/i },
  { label: "lottery_prize", w: 1.5, re: /(?:free iphone|free samsung|free gift|prize card|winner selected)/i },

  // ---- job_task ----
  { label: "job_task", w: 2.5, re: /(?:earn|making|kamaye|kamao)\b[^.]{0,40}\b(?:\d[\d,]*\s*(?:per day|daily|daily payment|rupees|dollars)|daily|per day|per task|per assignment)/i },
  { label: "job_task", w: 2.5, re: /(?:registration|joining|activation|security|admin)\s+(?:fee|charge|cost)\b.{0,40}\b(?:start|begin|receive tasks|earn)/i },
  { label: "job_task", w: 1.5, re: /(?:simple tasks|copy paste|captcha|data entry|liking videos|youtube videos|reviews? posting)/i },
  { label: "job_task", w: 1.5, re: /(?:work from home|work from home|ghar baithe|wfh)\b.{0,60}\b(?:earn|income|job|daily)/i },
  { label: "job_task", w: 2.0, re: /(?:pre[- ]?approved|selected)\b.{0,60}\b(?:work[\s-]?from[\s-]?home|remote (?:job|position)|task)/i },

  // ---- advance_fee ----
  { label: "advance_fee", w: 2.5, re: /(?:inheritance|inherited|inherit|legacy|consignment box|diplomat|diplomatic)/i },
  { label: "advance_fee", w: 2.0, re: /(?:transfer|release|courier|shipping)\s+(?:fee|charges?|cost)\b.{0,60}\b(?:receive|claim|release|fund|box|prize)/i },
  { label: "advance_fee", w: 1.5, re: /(?:late|deceased)\s+(?:father|husband|wife|minister|client|general|relative)/i },
  { label: "advance_fee", w: 1.5, re: /(?:donation|charity|charitable)\b.{0,60}\b(?:million|approved|fund)/i },

  // ---- gov_bank_impersonation ----
  { label: "gov_bank_impersonation", w: 2.5, re: /\b(?:rbi|reserve bank|income tax (?:department|dept|notice)|gst (?:dept|department)|cbi|fbi|cyber (?:cell|crime|police)|crime branch|doT|traI)\b/i },
  { label: "gov_bank_impersonation", w: 2.5, re: /\bkyc\b.{0,60}\b(?:expired|suspend\w*|update|renew|invalid|pending|link)\b/i },
  { label: "gov_bank_impersonation", w: 2.0, re: /(?:digital arrest|virtual custody|video call (?:with |the )?(?:police|officer|constable)|skype (?:hearing|investigation))/i },
  { label: "gov_bank_impersonation", w: 2.0, re: /(?:bank|branch)\s+(?:manager|officer|head)\b.{0,60}\b(?:call|speak|confirm|verify|blocked)/i },
  { label: "gov_bank_impersonation", w: 2.0, re: /(?:aadhaar|aadhar|pan card|ssn|social security)\b.{0,50}\b(?:link|linking|update|verify|expired|suspend\w*|mismatch)/i },

  // ---- delivery_scam ----
  { label: "delivery_scam", w: 2.0, re: /(?:parcel|package|shipment|courier)\b.{0,60}\b(?:held|held up|stuck|detained|seized|blocked|customs)/i },
  { label: "delivery_scam", w: 2.0, re: /(?:customs|clearance|redelivery|delivery failed|address incomplete|import duty)\b/i },
  { label: "delivery_scam", w: 1.5, re: /(?:delivery|courier) (?:charge|fee|cost)\b/i },

  // ---- account_suspension ----
  { label: "account_suspension", w: 2.5, re: /(?:account|wallet|sim|subscription|number)\s+(?:will be|has been|is)\s+(?:permanently\s+)?(?:suspend\w+|blocked|closed|deactivated|terminated|disabled|barred|disconnected)/i },
  { label: "account_suspension", w: 2.0, re: /(?:account|page|profile)\b.{0,50}\b(?:delete\w*|remov\w*|permanently (?:removed|banned))/i },
  { label: "account_suspension", w: 1.5, re: /(?:avoid|stop|prevent)\b.{0,40}\b(?:suspension|closure|deactivation|disconnection)/i },

  // ---- sextortion ----
  { label: "sextortion", w: 3.0, re: /(?:recorded|captured|hacked|installed (?:a )?(?:virus|malware|spyware))\b.{0,80}\b(?:video|footage|camera|recording|adult)/i },
  { label: "sextortion", w: 2.5, re: /\b(?:bitcoin|btc|crypto wallet)\b.{0,80}\b(?:pay|send|wallet address|within \d+ hours)/i },
  { label: "sextortion", w: 2.0, re: /(?:send|forward|viral|share)\b.{0,60}\b(?:contacts|family|friends|relatives)\b.{0,40}\b(?:unless|if you (?:do not|dont)|before)/i },

  // ---- romance ----
  { label: "romance", w: 2.0, re: /(?:widow|widower|late husband|late wife|deployed|serving in|oil rig|army|military)\b.{0,80}\b(?:love|trust|friend|marry|single)/i },
  { label: "romance", w: 1.5, re: /(?:dear (?:friend|beloved|one)|my (?:love|dear|soulmate)|beautiful soul|found your profile)/i },
  { label: "romance", w: 1.5, re: /(?:feel|feeling)\s+(?:we can|a strong|a deep)\b.{0,40}\b(?:connection|friendship|bond)/i },

  // ---- tech_support ----
  { label: "tech_support", w: 2.5, re: /(?:virus|malware|trojan|spyware|trojans)\b.{0,60}\b(?:detected|found|infected|removed)/i },
  { label: "tech_support", w: 2.0, re: /(?:remote access|remote connection|remote support|allow access)\b/i },
  { label: "tech_support", w: 1.5, re: /(?:microsoft|windows|apple|quick heal|norton)\b.{0,50}\b(?:support|technician|engineer|license|warranty)/i },

  // ---- generic_spam ----
  { label: "generic_spam", w: 1.5, re: /(?:congratulations|congrats)\b.{0,60}\b(?:selected|won|winner|chosen)/i },
  { label: "generic_spam", w: 1.5, re: /(?:free|100% free)\b.{0,30}\b(?:recharge|gift|data|plan|trial)/i },
  { label: "generic_spam", w: 1.0, re: /(?:limited time|today only|hurry|expires tonight|offer closes|last day)/i },
  { label: "generic_spam", w: 1.0, re: /(?:forward this|share this|send to (?:\d+|ten|your))\b.{0,40}\b(?:friends|groups|contacts)/i },
  { label: "generic_spam", w: 1.0, re: /(?:dear customer|dear user|dear member|valued customer)\b/i },
];

// benign dampeners: strong legitimate patterns that subtract from every
// scam label - this is what keeps real bank alerts and receipts safe
const DAMPENERS = [
  { w: -2.5, re: /(?:your|aapka) (?:otp|code) (?:is|hai) \d{4,8}\b.{0,80}(?:valid|expires|do not share|never share|kisi se share)/i },
  { w: -2.0, re: /(?:payment|bill|transaction) (?:of )?(?:rs\.? ?|inr|₹|\$)?[\d,]+ (?:was |has been )?(?:successful|paid|debited|credited|processed)/i },
  { w: -1.5, re: /(?:delivered|dispatched|out for delivery|shipped)\b.{0,60}\b(?:order|parcel|package|shipment)/i },
  { w: -1.5, re: /(?:attached|please review|follow up|minutes of the|agenda|regarding the)/i },
  { w: -1.0, re: /(?:meeting|appointment|class|lecture|webinar)\b.{0,40}\b(?:at|on) \d{1,2}[:.]\d{2}/i },
  { w: -1.0, re: /(?:salary|invoice|receipt|refund) (?:of |for )?[\d,]+ (?:credited|processed|issued|paid)/i },
  { w: -1.0, re: /(?:temperature|weather|traffic|news|headlines) (?:is |are |update|report)/i },
];

// labels that indicate urgency-style manipulation get cross-label boosts
const URGENCY_BOOSTS = ["payment_pressure", "account_suspension", "phishing"];

function capsRatio(text) {
  const letters = text.replace(/[^a-zA-Z]/g, "");
  if (letters.length < 10) return 0;
  const caps = letters.replace(/[^A-Z]/g, "").length;
  return caps / letters.length;
}

function punctuationSpam(text) {
  return (text.match(/!{2,}|\?{3,}|\${2,}/g) || []).length;
}

export function rule_scan(text) {
  const normalized = text.replace(/\u200b/g, "");

  // structural signals from the fusion engine
  const analysis = analyzeText(text);

  // per-label weighted scores from regex rules + dampeners
  const scores = {};
  for (const rule of RULES) {
    if (rule.re.test(normalized)) {
      scores[rule.label] = (scores[rule.label] || 0) + rule.w;
    }
  }
  for (const d of DAMPENERS) {
    if (d.re.test(normalized)) {
      for (const label in scores) {
        scores[label] += d.w;
      }
    }
  }

  // structural signals boost their labels (same targets as model fusion)
  for (const [label, boost] of Object.entries(analysis.boosts)) {
    scores[label] = (scores[label] || 0) + boost * 0.6;
  }
  // caps / punctuation spam: mild generic boost when other signals exist
  const caps = capsRatio(text);
  const punct = punctuationSpam(text);
  if (caps > 0.5 && Object.keys(scores).length) {
    scores.generic_spam = (scores.generic_spam || 0) + 0.5;
  }
  if (punct >= 2 && Object.keys(scores).length) {
    scores.generic_spam = (scores.generic_spam || 0) + 0.5;
  }

  // saturating mapping to probability-like values: score 2 -> ~0.49,
  // score 4 -> ~0.74, score 6+ -> 0.86+ (never claims false certainty)
  const labelNames = Object.keys(scores).filter((l) => scores[l] > 0);
  const hits = [];
  const weak = [];
  let topScore = 0;
  for (const label of labelNames) {
    const s = scores[label];
    const prob = 1 - Math.exp(-Math.max(0, s) / 3);
    topScore = Math.max(topScore, s);
    const row = { label, prob, weak: prob < 0.5, engine: "rules" };
    (row.weak ? weak : hits).push(row);
  }
  // fusion-derived rule hit: strong URL evidence fills malicious_link
  if (analysis.ruleHit && !hits.some((h) => h.label === "malicious_link")) {
    hits.push({ ...analysis.ruleHit, engine: "rules" });
  }
  hits.sort((a, b) => b.prob - a.prob);
  weak.sort((a, b) => b.prob - a.prob);

  // risk score: saturating curve on the top rule score + fusion total boost
  const fusedScore = 1 - Math.exp(-Math.max(0, topScore + analysis.totalBoost * 0.35) / 3);
  const score = Math.min(100, Math.round(fusedScore * 100));

  return { hits, weak, signals: analysis.signals, score };
}
