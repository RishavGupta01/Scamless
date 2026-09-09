"""Metrics harness: per-label P/R/F1, macro-F1, false-positive rate.

fp_rate = share of records with NO true labels (safe) that get >= 1 predicted label.
This is the trust metric; it is optimized as hard as recall (spec section 7).
"""

import json
import pathlib

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from scamless.labels import SCAM_LABELS, labels_to_vector


def compute_metrics(df, preds: list[list[str]]) -> dict:
    y_true = np.array([labels_to_vector(list(r)) for r in df["labels"]])
    y_pred = np.array([labels_to_vector(list(p)) for p in preds])

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_label = {
        name: {"precision": float(p), "recall": float(r), "f1": float(f)}
        for name, p, r, f in zip(SCAM_LABELS, prec, rec, f1)
    }

    safe_mask = y_true.sum(axis=1) == 0
    flagged = y_pred.sum(axis=1) > 0
    fp_rate = (
        float((safe_mask & flagged).sum() / safe_mask.sum()) if safe_mask.any() else 0.0
    )

    # macro-F1 averages only over labels with support in the eval set;
    # zero-support categories are data gaps, not model failures
    support_mask = y_true.sum(axis=0) > 0
    macro_f1 = float(np.mean(f1[support_mask])) if support_mask.any() else 0.0
    return {
        "macro_f1": macro_f1,
        "false_positive_rate": fp_rate,
        "per_label": per_label,
        "n": int(len(df)),
    }


def save_metrics(metrics: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
