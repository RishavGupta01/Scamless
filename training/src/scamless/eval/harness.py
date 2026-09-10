"""Metrics harness: per-label P/R/F1, macro-F1, false-positive rate.

fp_rate = share of records with NO true labels (safe) that get >= 1 predicted label.
This is the trust metric; it is optimized as hard as recall (spec section 7).
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from scamless.labels import SCAM_LABELS, labels_to_vector


def compute_metrics(df, preds: list[list[str]], min_support: int = 30) -> dict:
    y_true = np.array([labels_to_vector(list(r)) for r in df["labels"]])
    y_pred = np.array([labels_to_vector(list(p)) for p in preds])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(y_true[:, j].sum()),
        }
        for j, (name, p, r, f) in enumerate(zip(SCAM_LABELS, prec, rec, f1))
    }

    safe_mask = y_true.sum(axis=1) == 0
    flagged = y_pred.sum(axis=1) > 0
    fp_rate = (
        float((safe_mask & flagged).sum() / safe_mask.sum()) if safe_mask.any() else 0.0
    )

    # macro-F1 over evaluable labels only: categories with at least
    # min_support positive rows in the eval split. Zero-support categories
    # are data gaps to fill, not model failures to punish.
    support = y_true.sum(axis=0)
    evaluable = support >= min_support
    macro_f1_evaluable = (
        float(np.mean(f1[evaluable])) if evaluable.any() else 0.0
    )

    # product-level metric: "did we flag this message as a scam at all"
    # (union of all scam labels). A phishing email caught as generic_spam
    # still protects the user - the union measures that honestly.
    true_scam = y_true.sum(axis=1) > 0
    pred_scam = y_pred.sum(axis=1) > 0
    if true_scam.any():
        u_prec, u_rec, u_f1, _ = precision_recall_fscore_support(
            true_scam.astype(int), pred_scam.astype(int), average="binary", zero_division=0
        )
        scam_union = {
            "f1": float(u_f1),
            "precision": float(u_prec),
            "recall": float(u_rec),
        }
    else:
        scam_union = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "macro_f1": macro_f1_evaluable,
        "macro_f1_evaluable_labels": int(evaluable.sum()),
        "macro_f1_all_labels": float(np.mean(f1)) if len(f1) else 0.0,
        "scam_union": scam_union,
        "false_positive_rate": fp_rate,
        "per_label": per_label,
        "per_label_support": {name: int(v) for name, v in zip(SCAM_LABELS, support)},
        "n": len(df),
    }


def save_metrics(metrics: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def truth_matrix(df) -> np.ndarray:
    """(n, NUM_SCAM_LABELS) float matrix from a DataFrame with a 'labels' column."""
    return np.array([labels_to_vector(list(r)) for r in df["labels"]])


def tune_thresholds(
    probs, truth, sweep=None, precision_floor: float = 0.90
) -> dict:
    """Per-label thresholds: best F1 subject to a precision floor.

    Pure F1-maximization allowed the phishing head to fire at precision 0.31
    (the 54% FP disaster). This tuner first looks for the F1-best threshold
    among candidates whose precision meets the floor; if none qualify, it
    falls back to the highest-precision threshold with nonzero F1.
    Returns {label: threshold}.
    """
    if sweep is None:
        sweep = np.arange(0.05, 0.96, 0.05)
    probs = np.asarray(probs)
    truth = np.asarray(truth)
    thresholds = {}
    for j, name in enumerate(SCAM_LABELS):
        if truth[:, j].sum() == 0:
            thresholds[name] = 0.5
            continue
        table = []
        for t in sweep:
            pred = (probs[:, j] > t).astype(int)
            p, _r, f1, _ = precision_recall_fscore_support(
                truth[:, j], pred, average="binary", zero_division=0
            )
            table.append((float(t), float(f1), float(p)))
        eligible = [row for row in table if row[2] >= precision_floor]
        if eligible:
            best = max(eligible, key=lambda row: row[1])
        else:
            nonzero = [row for row in table if row[1] > 0]
            best = max(nonzero, key=lambda row: row[2]) if nonzero else (0.5, 0.0, 0.0)
        thresholds[name] = best[0]
    return thresholds
