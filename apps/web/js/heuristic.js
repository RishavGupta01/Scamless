// Demo-mode heuristic: a small JS port of the training baseline rules.
// Only used when the full model is unavailable (offline first visit, load
// failure). Clearly labelled in the UI so users never confuse it with the
// trained model verdicts.

const RULES = {
  phishing: ["verify your account", "confirm your identity", "click the link", "login here", "update your kyc"],
  otp_request: ["otp", "one-time", "one time password", "verification code", "share the code"],
  payment_pressure: ["send $", "send money", "immediately or", "pay now", "gift card", "rs ", "upi"],
  investment_crypto: ["double your", "investment opportunity", "guaranteed returns", "crypto profit", "daily returns"],
  lottery_prize: ["congratulations", "you won", "prize", "lucky draw", "winner"],
  job_task: ["work from home", "earn daily", "registration fee", "part time job", "task"],
  advance_fee: ["inheritance", "consignment", "diplomat", "donation approved", "transfer fee"],
  sextortion: ["recorded", "bitcoin", "viral to your contacts", "do not inform police"],
  delivery_scam: ["parcel held", "customs fee", "redelivery charge", "clearance charge"],
  account_suspension: ["account will be suspended", "account will be blocked", "permanently closed", "deactivated"],
  gov_bank_impersonation: ["rbi", "income tax department", "bank manager", "cyber cell", "kyc expired"],
  tech_support: ["virus detected", "remote access", "certified technician", "windows license"],
  generic_spam: ["free", "winner", "claim", "lottery", "cash prize", "selected"],
};

export function heuristic_scan(text) {
  const normalized = text.replace(/\u200b/g, "").toLowerCase();
  const hits = [];
  for (const [label, keywords] of Object.entries(RULES)) {
    if (keywords.some((kw) => normalized.includes(kw))) {
      hits.push({ label, prob: 0.62 });
    }
  }
  return hits;
}
