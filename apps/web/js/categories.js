// Scamless - category knowledge base (single source of truth for the app
// and the learn mode). Explanations are rule-based, deterministic text:
// no generative model in the explanation path, so nothing can hallucinate.

export const CATEGORIES = {
  phishing: {
    name: "Phishing (credential theft)",
    blurb: "Pretends to be a trusted service and pushes you to a lookalike login page to steal your username, password or card details.",
    redFlags: ["Verify your account", "Login link that is not the official domain", "Urgency: account closes soon", "Asks for password, OTP or card details"],
    action: "Do not click. Open the official website by typing its address yourself and check for alerts there.",
  },
  investment_crypto: {
    name: "Investment / crypto scam",
    blurb: "Promises guaranteed or outsized returns, doubling schemes, insider signals, or fake trading platforms.",
    redFlags: ["Guaranteed returns", "Doubling your money", "Urgency: last slots", "Unregistered platforms"],
    action: "No investment is guaranteed. Never pay to withdraw profits. Verify registration with the regulator.",
  },
  romance: {
    name: "Romance scam",
    blurb: "Builds a fake relationship, then asks for money, gift cards, or moves you to a private chat.",
    redFlags: ["Fast emotional attachment", "Never able to meet or video call", "Eventually asks for money"],
    action: "Never send money or financial details to someone you have not met in person.",
  },
  tech_support: {
    name: "Tech-support scam",
    blurb: "Claims your device is infected or your license expired, then asks for remote access or payment.",
    redFlags: ["Viruses detected out of nowhere", "Call this number now", "Allow remote access"],
    action: "Real companies do not cold-warn you about viruses. Never grant remote access to unsolicited callers.",
  },
  lottery_prize: {
    name: "Lottery / prize scam",
    blurb: "You won a draw you never entered - but need to pay a fee first, or share details to claim.",
    redFlags: ["You won without entering", "Processing / GST / delivery fee", "Claim within hours"],
    action: "Real prizes never require upfront payment. Do not pay anything to claim.",
  },
  job_task: {
    name: "Job / task scam",
    blurb: "Easy work-from-home income that requires a registration or security fee, or unpaid tasks that grow.",
    redFlags: ["Pay to start earning", "Daily high income, no skills", "Deposit for tasks"],
    action: "Real employers never charge you to work. Stop all contact and block.",
  },
  gov_bank_impersonation: {
    name: "Government / bank impersonation",
    blurb: "Pretends to be your bank, RBI, income tax, police or a government scheme to create fear and force action.",
    redFlags: ["KYC expired / account frozen", "Digital arrest", "Refund or subsidy needing a fee", "Calls claiming to be officers"],
    action: "Banks and government offices never ask for OTP, PIN or fees over call or message. Hang up and contact the official channel.",
  },
  otp_request: {
    name: "OTP request",
    blurb: "Asks you to share an OTP, PIN or verification code. This is how accounts get emptied.",
    redFlags: ["Any request to share OTP or PIN", "Someone calling and asking to read codes"],
    action: "No legitimate service ever asks for your OTP. Never share it with anyone.",
  },
  payment_pressure: {
    name: "Payment pressure",
    blurb: "Manufactures an emergency - overdue bills, family accidents, customs fees - to force an instant transfer.",
    redFlags: ["Pay within hours", "Family emergency from a new number", "Court / police threats"],
    action: "Verify through a known channel before sending anything. Emergencies can wait 10 minutes for a callback.",
  },
  advance_fee: {
    name: "Advance-fee (419) scam",
    blurb: "Offers a fortune - inheritance, diplomat box, donation - that only needs a small fee to release.",
    redFlags: ["Huge sum promised", "Small fee to release funds", "Foreign officials or widows"],
    action: "Any money that requires a fee to receive is a scam. Walk away.",
  },
  sextortion: {
    name: "Sextortion",
    blurb: "Claims to have compromising footage or data and threatens to send it to your contacts unless you pay.",
    redFlags: ["Claims hacked your device", "Bitcoin payment demand", "Send-to-contacts threat"],
    action: "Do not pay. Do not reply. Report to the cybercrime portal and block the sender.",
  },
  delivery_scam: {
    name: "Delivery / customs scam",
    blurb: "A parcel is held, failed, or contains valuables - but needs a fee to release or redeliver.",
    redFlags: ["Customs or clearance fee", "Address incomplete", "Valuable parcel you did not expect"],
    action: "Track parcels only on the courier's official app or website. Never pay via links in messages.",
  },
  account_suspension: {
    name: "Account suspension threat",
    blurb: "Your account, SIM, wallet or subscription will be closed unless you verify or update now.",
    redFlags: ["Suspended in hours", "Mandatory re-verification", "Links to update details"],
    action: "Check the official app or website directly. Real deadlines never arrive only by message.",
  },
  malicious_link: {
    name: "Malicious link",
    blurb: "Contains a link to a fake or dangerous site - lookalike domains, shortened links, unexpected attachments.",
    redFlags: ["Links you did not expect", "Misspelled or strange domains", "Shortened URLs"],
    action: "Do not open. If curious, inspect the domain carefully - scammers use lookalike spellings.",
  },
  generic_spam: {
    name: "Generic spam",
    blurb: "Bulk marketing, junk promotions or mass-sent garbage that wastes your attention.",
    redFlags: ["Mass promotions", "Too-good offers", "Random sender"],
    action: "Report as spam and block. Do not engage or reply.",
  },
};

export const RISK_BANDS = {
  safe: { label: "SAFE", color: "var(--ok)" },
  suspicious: { label: "SUSPICIOUS", color: "var(--warn)" },
  dangerous: { label: "DANGEROUS", color: "var(--bad)" },
};

export function riskBand(score) {
  if (score >= 70) return RISK_BANDS.dangerous;
  if (score >= 40) return RISK_BANDS.suspicious;
  return RISK_BANDS.safe;
}
