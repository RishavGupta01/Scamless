from scamless.labels import (
    LABELS,
    NUM_SCAM_LABELS,
    SAFE,
    SCAM_LABELS,
    label_index,
    labels_to_vector,
    vector_to_labels,
)


def test_fifteen_scam_labels_plus_safe():
    assert len(SCAM_LABELS) == 15
    assert SAFE not in SCAM_LABELS
    assert LABELS == SCAM_LABELS  # model output space is the 15 scam labels


def test_label_index_is_stable():
    assert label_index("phishing") == 0
    assert label_index("generic_spam") == NUM_SCAM_LABELS - 1


def test_labels_to_vector_roundtrip():
    vec = labels_to_vector(["phishing", "payment_pressure"])
    assert len(vec) == NUM_SCAM_LABELS
    assert vec[label_index("phishing")] == 1.0
    assert vec[label_index("payment_pressure")] == 1.0
    assert vec[label_index("romance")] == 0.0
    assert set(vector_to_labels(vec)) == {"phishing", "payment_pressure"}


def test_vector_to_labels_all_zero_is_safe():
    assert vector_to_labels([0.0] * NUM_SCAM_LABELS) == []
