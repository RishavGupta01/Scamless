// Scamless rule engine - the no-model fallback detector.
//
// Design: many simple, independent patterns per scam category. Each pattern
// fires independently and contributes weight. A convergence bonus is added
// when multiple patterns from the same category fire - because 5 weak
// indicators pointing at the same scam together are conclusive even if no
// single one is. Benign dampeners subtract for legitimate message formats.
//
// Every pattern is a simple substring or short regex - no compound
// co-occurrence requirements that break on real-world sentence variation.

import { analyzeText } from "./fusion.js";

// Each rule: [label, weight, pattern]
// Patterns are lowercase substrings checked against the lowercased text
// after zero-width stripping. Simple and fast.
const RULES = [
  // === job_task ===
  ["job_task", 2.5, "rozana kamao"],
  ["job_task", 2.5, "rozana kamaye"],
  ["job_task", 2.0, "ghar baithe"],
  ["job_task", 2.0, "ghar baiti"],
  ["job_task", 2.0, "ghar par kaam"],
  ["job_task", 2.0, "work from home"],
  ["job_task", 2.0, "work-from-home"],
  ["job_task", 2.0, "wfh job"],
  ["job_task", 1.5, "copy paste"],
  ["job_task", 1.5, "copy-paste"],
  ["job_task", 1.5, "simple task"],
  ["job_task", 1.5, "captcha"],
  ["job_task", 1.5, "data entry"],
  ["job_task", 1.5, "liking videos"],
  ["job_task", 1.5, "youtube videos"],
  ["job_task", 2.5, "activation charge"],
  ["job_task", 2.5, "activation fee"],
  ["job_task", 2.5, "registration fee"],
  ["job_task", 2.5, "registration charge"],
  ["job_task", 2.5, "joining fee"],
  ["job_task", 2.5, "joining charge"],
  ["job_task", 2.5, "security fee"],
  ["job_task", 2.0, "pehla payment"],
  ["job_task", 2.0, "first payment"],
  ["job_task", 2.0, "daily payment"],
  ["job_task", 2.0, "daily income"],
  ["job_task", 2.0, "daily payout"],
  ["job_task", 2.0, "earn daily"],
  ["job_task", 2.0, "daily earning"],
  ["job_task", 2.0, "rozana kamai"],
  ["job_task", 2.0, "per day income"],
  ["job_task", 2.0, "per day earning"],
  ["job_task", 1.5, "pre-approved for a"],
  ["job_task", 1.5, "you are selected"],
  ["job_task", 1.5, "aap select"],

  // === otp_request ===
  ["otp_request", 3.0, "share the otp"],
  ["otp_request", 3.0, "share otp"],
  ["otp_request", 3.0, "batao otp"],
  ["otp_request", 3.0, "otp batao"],
  ["otp_request", 3.0, "otp bata"],
  ["otp_request", 2.5, "tell me the otp"],
  ["otp_request", 2.5, "give me the otp"],
  ["otp_request", 2.5, "otp for verification"],
  ["otp_request", 2.5, "otp to verify"],
  ["otp_request", 2.5, "otp for payment"],
  ["otp_request", 2.0, "verification code"],
  ["otp_request", 2.0, "one time password"],
  ["otp_request", 1.5, "enter the otp"],
  ["otp_request", 1.5, "enter otp"],
  ["otp_request", 1.5, "otp aaya"],
  ["otp_request", 1.5, "otp aaya hai"],

  // === payment_pressure ===
  ["payment_pressure", 2.5, "immediately or"],
  ["payment_pressure", 2.5, "pay immediately"],
  ["payment_pressure", 2.5, "pay now"],
  ["payment_pressure", 2.5, "send money now"],
  ["payment_pressure", 2.0, "final warning"],
  ["payment_pressure", 2.0, "last warning"],
  ["payment_pressure", 2.0, "legal action"],
  ["payment_pressure", 2.0, "court case"],
  ["payment_pressure", 2.0, "police case"],
  ["payment_pressure", 2.0, "arresterd"],
  ["payment_pressure", 2.0, "pay within"],
  ["payment_pressure", 2.0, "pay today"],
  ["payment_pressure", 2.0, "pay now or"],
  ["payment_pressure", 1.5, "overdue"],
  ["payment_pressure", 1.5, "outstanding amount"],
  ["payment_pressure", 1.5, "pending payment"],
  ["payment_pressure", 1.5, "disconnection"],
  ["payment_pressure", 1.5, "electricity will be"],
  ["payment_pressure", 1.5, "internet will be"],
  ["payment_pressure", 1.5, "medical emergency"],
  ["payment_pressure", 1.5, "hospital deposit"],
  ["payment_pressure", 1.5, "surgery advance"],

  // === investment_crypto ===
  ["investment_crypto", 3.0, "guaranteed return"],
  ["investment_crypto", 3.0, "guaranteed profit"],
  ["investment_crypto", 3.0, "double your money"],
  ["investment_crypto", 3.0, "double your investment"],
  ["investment_crypto", 2.5, "10x your"],
  ["investment_crypto", 2.5, "daily returns"],
  ["investment_crypto", 2.5, "daily profit"],
  ["investment_crypto", 2.5, "daily withdrawal"],
  ["investment_crypto", 2.0, "trading bot"],
  ["investment_crypto", 2.0, "trading platform"],
  ["investment_crypto", 2.0, "forex trading"],
  ["investment_crypto", 2.0, "crypto trading"],
  ["investment_crypto", 2.0, "bitcoin profit"],
  ["investment_crypto", 2.0, "crypto profit"],
  ["investment_crypto", 1.5, "investment opportunity"],
  ["investment_crypto", 1.5, "risk free"],
  ["investment_crypto", 1.5, "risk-free"],
  ["investment_crypto", 1.5, "limited slots"],
  ["investment_crypto", 1.5, "few slots"],

  // === lottery_prize ===
  ["lottery_prize", 2.5, "lucky draw"],
  ["lottery_prize", 2.5, "lucky dip"],
  ["lottery_prize", 2.5, "mega draw"],
  ["lottery_prize", 2.5, "bumper draw"],
  ["lottery_prize", 2.5, "you have won"],
  ["lottery_prize", 2.5, "you won"],
  ["lottery_prize", 2.0, "congratulations you"],
  ["lottery_prize", 2.0, "congrats you"],
  ["lottery_prize", 2.0, "prize claim"],
  ["lottery_prize", 2.0, "claim your prize"],
  ["lottery_prize", 2.0, "claim your reward"],
  ["lottery_prize", 2.0, "winner selected"],
  ["lottery_prize", 1.5, "free iphone"],
  ["lottery_prize", 1.5, "free samsung"],
  ["lottery_prize", 1.5, "prize card"],

  // === gov_bank_impersonation ===
  ["gov_bank_impersonation", 3.0, "digital arrest"],
  ["gov_bank_impersonation", 3.0, "virtual custody"],
  ["gov_bank_impersonation", 2.5, "video call with the police"],
  ["gov_bank_impersonation", 2.5, "video call police"],
  ["gov_bank_impersonation", 2.5, "cyber cell"],
  ["gov_bank_impersonation", 2.5, "cyber police"],
  ["gov_bank_impersonation", 2.5, "cbi officer"],
  ["gov_bank_impersonation", 2.0, "income tax department"],
  ["gov_bank_impersonation", 2.0, "rbi notice"],
  ["gov_bank_impersonation", 2.0, "rbi guidelines"],
  ["gov_bank_impersonation", 2.0, "rbi new rule"],
  ["gov_bank_impersonation", 2.5, "kyc expired"],
  ["gov_bank_impersonation", 2.5, "kyc update"],
  ["gov_bank_impersonation", 2.5, "kyc pending"],
  ["gov_bank_impersonation", 2.5, "kyc re-verification"],
  ["gov_bank_impersonation", 2.0, "aadhaar linking"],
  ["gov_bank_impersonation", 2.0, "aadhaar link"],
  ["gov_bank_impersonation", 2.0, "pan card link"],
  ["gov_bank_impersonation", 2.0, "account frozen"],
  ["gov_bank_impersonation", 2.0, "account will be frozen"],

  // === phishing ===
  ["phishing", 2.5, "verify your account"],
  ["phishing", 2.5, "verify your identity"],
  ["phishing", 2.5, "confirm your identity"],
  ["phishing", 2.0, "click the link"],
  ["phishing", 2.0, "click here to"],
  ["phishing", 2.0, "login now"],
  ["phishing", 2.0, "login to your"],
  ["phishing", 2.0, "sign in to confirm"],
  ["phishing", 2.0, "unusual activity"],
  ["phishing", 2.0, "unusual login"],
  ["phishing", 1.5, "suspicious login"],
  ["phishing", 1.5, "secure your account"],
  ["phishing", 1.5, "password expires"],
  ["phishing", 1.5, "password expiring"],

  // === delivery_scam ===
  ["delivery_scam", 2.5, "customs fee"],
  ["delivery_scam", 2.5, "customs charge"],
  ["delivery_scam", 2.5, "clearance charge"],
  ["delivery_scam", 2.5, "clearance fee"],
  ["delivery_scam", 2.0, "redelivery charge"],
  ["delivery_scam", 2.0, "redelivery fee"],
  ["delivery_scam", 2.0, "delivery failed"],
  ["delivery_scam", 2.0, "address incomplete"],
  ["delivery_scam", 2.0, "parcel held"],
  ["delivery_scam", 2.0, "parcel detained"],
  ["delivery_scam", 2.0, "package held"],
  ["delivery_scam", 2.0, "shipment held"],
  ["delivery_scam", 1.5, "pay to release"],
  ["delivery_scam", 1.5, "pay to receive"],

  // === advance_fee ===
  ["advance_fee", 2.5, "consignment box"],
  ["advance_fee", 2.5, "transfer fee to receive"],
  ["advance_fee", 2.5, "processing fee to"],
  ["advance_fee", 2.0, "inheritance"],
  ["advance_fee", 2.0, "inherited"],
  ["advance_fee", 2.0, "late husband"],
  ["advance_fee", 2.0, "late wife"],
  ["advance_fee", 2.0, "late father"],
  ["advance_fee", 2.0, "donation approved"],
  ["advance_fee", 2.0, "charity approved"],
  ["advance_fee", 1.5, "diplomat"],
  ["advance_fee", 1.5, "diplomatic"],

  // === sextortion ===
  ["sextortion", 3.0, "recorded you"],
  ["sextortion", 3.0, "recorded your"],
  ["sextortion", 3.0, "hacked your camera"],
  ["sextortion", 3.0, "hacked your phone"],
  ["sextortion", 3.0, "installed a virus"],
  ["sextortion", 2.5, "bitcoin wallet"],
  ["sextortion", 2.5, "btc wallet"],
  ["sextortion", 2.0, "send the video to"],
  ["sextortion", 2.0, "viral to your"],
  ["sextortion", 2.0, "do not inform police"],
  ["sextortion", 2.0, "do not contact police"],

  // === tech_support ===
  ["tech_support", 2.5, "virus detected"],
  ["tech_support", 2.5, "viruses found"],
  ["tech_support", 2.5, "malware detected"],
  ["tech_support", 2.0, "remote access"],
  ["tech_support", 2.0, "remote connection"],
  ["tech_support", 2.0, "remote support"],
  ["tech_support", 1.5, "certified technician"],
  ["tech_support", 1.5, "certified engineer"],
  ["tech_support", 1.5, "microsoft support"],
  ["tech_support", 1.5, "windows support"],

  // === account_suspension ===
  ["account_suspension", 2.5, "permanently closed"],
  ["account_suspension", 2.5, "permanently deleted"],
  ["account_suspension", 2.5, "permanently removed"],
  ["account_suspension", 2.0, "will be suspended"],
  ["account_suspension", 2.0, "will be blocked"],
  ["account_suspension", 2.0, "will be deactivated"],
  ["account_suspension", 2.0, "will be disconnected"],
  ["account_suspension", 2.0, "will be barred"],
  ["account_suspension", 2.0, "sim card will be"],
  ["account_suspension", 2.0, "account will be"],
  ["account_suspension", 1.5, "temporarily disabled"],
  ["account_suspension", 1.5, "limited access"],

  // === romance ===
  ["romance", 2.0, "widow of late"],
  ["romance", 2.0, "widower of late"],
  ["romance", 2.0, "deployed in"],
  ["romance", 2.0, "serving in syria"],
  ["romance", 2.0, "serving in the army"],
  ["romance", 2.0, "oil rig"],
  ["romance", 1.5, "dear beloved"],
  ["romance", 1.5, "my dear"],
  ["romance", 1.5, "beautiful soul"],
  ["romance", 1.5, "found your profile"],

  // === generic_spam ===
  ["generic_spam", 2.0, "congratulations!!"],
  ["generic_spam", 2.0, "congratulations!!"],
  ["generic_spam", 1.5, "free recharge"],
  ["generic_spam", 1.5, "free data"],
  ["generic_spam", 1.5, "100% free"],
  ["generic_spam", 1.5, "limited time"],
  ["generic_spam", 1.5, "today only"],
  ["generic_spam", 1.5, "offer closes"],
  ["generic_spam", 1.0, "dear customer"],
  ["generic_spam", 1.0, "dear user"],
  ["generic_spam", 1.0, "dear member"],
  ["generic_spam", 1.0, "valued customer"],

  // === Hinglish patterns ===
  ["job_task", 2.5, "rozana kamao"],
  ["job_task", 2.5, "paise kamao"],
  ["job_task", 2.0, "ghar baithe kaam"],
  ["job_task", 2.0, "kaam karo"],
  ["job_task", 2.0, "sirf activation"],
  ["job_task", 2.0, "sirf charges"],
  ["job_task", 1.5, "aaj se pehla"],
  ["job_task", 1.5, "payment kal"],
  ["job_task", 1.5, "paisa kal"],
  ["payment_pressure", 2.5, "paise bhejo"],
  ["payment_pressure", 2.5, "paisa bhej"],
  ["payment_pressure", 2.5, "turant bhej"],
  ["payment_pressure", 2.0, "paise chahiye"],
  ["payment_pressure", 2.0, "urgent paise"],
  ["payment_pressure", 2.0, "hospital me admission"],
  ["gov_bank_impersonation", 2.0, "account freeze"],
  ["gov_bank_impersonation", 2.0, "kyc karo"],
  ["gov_bank_impersonation", 2.0, "kyc pending"],
  ["otp_request", 2.5, "otp batao"],
  ["otp_request", 2.5, "otp bhejo"],
  ["otp_request", 2.0, "otp dena"],
  ["otp_request", 2.0, "otp do mujhe"],
];

// benign dampeners: strong legitimate patterns that subtract from every
// scam label - this is what keeps real bank alerts and receipts safe
const DAMPENERS = [
  { w: -3.0, re: /(?:your |aapka |aapki )?(?:otp|code|pin) (?:is|hai) \d{4,8}\b.{0,80}(?:valid|expires|do not share|never share|kisi se share|10 min)/i },
  { w: -2.0, re: /(?:payment|bill|transaction) (?:of )?(?:rs\.? ?|inr|₹|\$)?[\d,]+ (?:was |has been )?(?:successful|paid|debited|credited|processed|auto-debited|auto-paid)/i },
  { w: -2.0, re: /(?:delivered|dispatched|out for delivery|shipped)\b.{0,60}\b(?:order|parcel|package|shipment|thank you)/i },
  { w: -1.5, re: /(?:attached|please review|follow up|minutes of the|agenda|regarding the)/i },
  { w: -1.5, re: /(?:meeting|appointment|class|lecture|webinar)\b.{0,40}\b(?:at|on) \d{1,2}[:.]\d{2}/i },
  { w: -1.0, re: /(?:salary|invoice|receipt|refund) (?:of |for )?[\d,]+ (?:credited|processed|issued|paid)/i },
  { w: -1.0, re: /(?:temperature|weather|traffic|news|headlines) (?:is |are |update|report)/i },
  { w: -1.0, re: /(?:thank you for shopping|your order.*has been|tracking number)/i },
];

// convergence bonus: when N rules from the same label fire, add a bonus
// that makes the combined evidence much stronger than any single rule
function convergenceBonus(count) {
  if (count >= 5) return 5.0;
  if (count >= 4) return 3.5;
  if (count >= 3) return 2.5;
  if (count >= 2) return 1.5;
  return 0;
}

function capsRatio(text) {
  const letters = text.replace(/[^a-zA-Z]/g, "");
  if (letters.length < 10) return 0;
  return text.replace(/[^A-Z]/g, "").length / letters.length;
}

export function rule_scan(text) {
  const normalized = text.replace(/\u200b/g, "");
  const lower = normalized.toLowerCase();

  // structural signals from the fusion engine
  const analysis = analyzeText(text);

  // per-label weighted scores from substring rules
  const scores = {};
  const labelHitCount = {};

  for (const [label, w, pattern] of RULES) {
    if (lower.includes(pattern)) {
      scores[label] = (scores[label] || 0) + w;
      labelHitCount[label] = (labelHitCount[label] || 0) + 1;
    }
  }

  // convergence bonus: multiple rules from the same category = strong signal
  for (const label in labelHitCount) {
    if (labelHitCount[label] >= 2) {
      scores[label] += convergenceBonus(labelHitCount[label]);
    }
  }

  // fusion signal boosts (URL analysis, homoglyphs, OTP patterns)
  for (const [label, boost] of Object.entries(analysis.boosts)) {
    scores[label] = (scores[label] || 0) + boost * 0.5;
  }

  // benign dampeners subtract from all labels
  for (const d of DAMPENERS) {
    if (d.re.test(normalized)) {
      for (const label in scores) {
        scores[label] += d.w;
      }
    }
  }

  // caps / punctuation spam: mild generic boost
  const caps = capsRatio(text);
  const punct = (text.match(/!{2,}|\?{3,}/g) || []).length;
  if (caps > 0.5) scores.generic_spam = (scores.generic_spam || 0) + 0.5;
  if (punct >= 2) scores.generic_spam = (scores.generic_spam || 0) + 0.5;

  // saturating mapping to probability-like values
  // score 2 -> 0.63, score 4 -> 0.86, score 6 -> 0.95, score 8 -> 0.98
  const labelNames = Object.keys(scores).filter((l) => scores[l] > 0);
  const hits = [];
  const weak = [];
  let topScore = 0;
  for (const label of labelNames) {
    const s = scores[label];
    const prob = 1 - Math.exp(-Math.max(0, s) / 2.2);
    topScore = Math.max(topScore, s);
    const row = { label, prob, weak: prob < 0.5, engine: "rules" };
    (row.weak ? weak : hits).push(row);
  }
  if (analysis.ruleHit && !hits.some((h) => h.label === "malicious_link")) {
    hits.push({ ...analysis.ruleHit, engine: "rules" });
  }
  hits.sort((a, b) => b.prob - a.prob);
  weak.sort((a, b) => b.prob - a.prob);

  const fusedScore = 1 - Math.exp(-Math.max(0, topScore + analysis.totalBoost * 0.35) / 2.2);
  const score = Math.min(100, Math.round(fusedScore * 100));

  return { hits, weak, signals: analysis.signals, score };
}
