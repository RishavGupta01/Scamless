import numpy as np
import pytest

from scamless.eval.harness import tune_thresholds
from scamless.labels import labels_to_vector

# every matrix is full schema width; the interesting signal sits on "phishing" (index 0)


def _full_width(scam_probs, safe_probs):
    scam_rows = [[p] + [0.0] * 14 for p in scam_probs]
    safe_rows = [[p] + [0.0] * 14 for p in safe_probs]
    probs = np.array([r for row in scam_rows + safe_rows for r in [row]])
    truth = np.array([[1.0] + [0.0] * 14] * len(scam_rows) + [[0.0] + [0.0] * 14] * len(safe_rows))
    return probs, truth


def _probs_and_truth():
    scam_probs = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.58, 0.56, 0.55]
    safe_probs = [0.45, 0.4, 0.38, 0.35, 0.32, 0.3, 0.28, 0.25, 0.22, 0.7]
    return _full_width(scam_probs, safe_probs)


def test_precision_floor_forces_higher_threshold_than_pure_f1():
    probs, truth = _probs_and_truth()
    # at 0.55 recall=1.0 but the 0.7 safe row is flagged: precision 10/11
    # with floor 0.95 the tuner must move to a higher threshold
    tuned = tune_thresholds(probs, truth, precision_floor=0.95)
    assert tuned["phishing"] > 0.55


def test_pure_f1_mode_picks_lowest_threshold_region():
    probs, truth = _probs_and_truth()
    tuned = tune_thresholds(probs, truth, precision_floor=0.0)
    assert tuned["phishing"] <= 0.6


def test_zero_support_label_defaults_to_half():
    probs = np.array([[0.9] + [0.0] * 14, [0.1] + [0.0] * 14])
    truth = np.array([labels_to_vector([]), labels_to_vector([])])
    tuned = tune_thresholds(probs, truth)
    assert tuned["phishing"] == 0.5


def test_fallback_and_floor_pick_best_available():
    probs, truth = _full_width([0.8, 0.75, 0.7], [0.3, 0.65, 0.6])
    # threshold 0.65 reaches precision 1.0 AND f1 1.0 (above the 0.95 floor):
    # the tuner must find it rather than stopping at a lower, sloppier one
    tuned = tune_thresholds(probs, truth, precision_floor=0.95)
    assert tuned["phishing"] == pytest.approx(0.65)


def test_other_labels_remain_in_output():
    probs, truth = _probs_and_truth()
    tuned = tune_thresholds(probs, truth, precision_floor=0.9)
    from scamless.labels import SCAM_LABELS

    assert set(tuned.keys()) == set(SCAM_LABELS)
