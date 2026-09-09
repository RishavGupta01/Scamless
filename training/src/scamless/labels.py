"""Canonical label schema. Order is a public contract: never reorder, only append."""

SCAM_LABELS = [
    "phishing",  # 0
    "investment_crypto",  # 1
    "romance",  # 2
    "tech_support",  # 3
    "lottery_prize",  # 4
    "job_task",  # 5
    "gov_bank_impersonation",  # 6
    "otp_request",  # 7
    "payment_pressure",  # 8
    "advance_fee",  # 9
    "sextortion",  # 10
    "delivery_scam",  # 11
    "account_suspension",  # 12
    "malicious_link",  # 13
    "generic_spam",  # 14
]

SAFE = "safe"
LABELS = SCAM_LABELS
NUM_SCAM_LABELS = len(SCAM_LABELS)

# Tactic tags for token-level span tagging (Phase 4: manipulation highlighting).
# Append-only, like the scam labels: adding a tag never breaks old checkpoints.
TACTIC_TAGS = [
    "urgency",  # 0
    "authority",  # 1
    "fear",  # 2
    "payment_pressure",  # 3
]
NUM_TACTIC_TAGS = len(TACTIC_TAGS)

_INDEX = {name: i for i, name in enumerate(SCAM_LABELS)}


def label_index(name: str) -> int:
    return _INDEX[name]


def labels_to_vector(names: list[str]) -> list[float]:
    vec = [0.0] * NUM_SCAM_LABELS
    for name in names:
        if name != SAFE:
            vec[label_index(name)] = 1.0
    return vec


def vector_to_labels(vec: list[float]) -> list[str]:
    return [SCAM_LABELS[i] for i, v in enumerate(vec) if v > 0.5]
