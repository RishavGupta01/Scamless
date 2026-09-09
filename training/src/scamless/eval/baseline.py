"""Keyword heuristic baseline. Purposes:
1. Validates the metrics harness end to end before any training happens.
2. Never the shipped detector; the trained model replaces it.
"""

from scamless.aug.adversarial import ZERO_WIDTH

RULES = {
    "generic_spam": [
        "free",
        "prize",
        "winner",
        "win ",
        "claim",
        "cash",
        "lottery",
        "selected",
    ],
    "otp_request": ["otp", "one-time", "one time password", "verification code", "share the code"],
    "phishing": ["verify your account", "confirm your identity", "click the link", "login here"],
    "payment_pressure": ["send $", "send money", "immediately or", "pay now", "gift card"],
    "investment_crypto": [
        "double your",
        "investment opportunity",
        "guaranteed returns",
        "crypto profit",
    ],
}


def heuristic_predict(text: str) -> list[str]:
    normalized = text.replace(ZERO_WIDTH, "").lower()
    hits = []
    for label, keywords in RULES.items():
        if any(kw in normalized for kw in keywords):
            hits.append(label)
    return hits
