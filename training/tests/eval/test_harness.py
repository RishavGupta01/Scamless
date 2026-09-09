import pandas as pd

from scamless.eval.harness import compute_metrics


def _frame(texts, label_sets):
    return pd.DataFrame({"text": texts, "labels": label_sets})


def test_perfect_predictions():
    df = _frame(
        ["win money now", "meeting at 3pm", "verify account link"],
        [["generic_spam"], [], ["phishing"]],
    )
    preds = [
        ["generic_spam"],
        [],
        ["phishing"],
    ]
    metrics = compute_metrics(df, preds)
    assert metrics["macro_f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["per_label"]["generic_spam"]["f1"] == 1.0


def test_false_positive_rate_counts_safe_rows_with_any_positive():
    df = _frame(["meeting at 3pm", "invoice attached", "win money"], [[], [], ["generic_spam"]])
    preds = [["phishing"], [], ["generic_spam"]]
    metrics = compute_metrics(df, preds)
    # one of the two safe rows was falsely flagged
    assert metrics["false_positive_rate"] == 0.5


def test_safe_rows_do_not_count_in_scam_recall():
    df = _frame(["meeting at 3pm"], [[]])
    metrics = compute_metrics(df, [[]])
    assert metrics["macro_f1"] == 0.0  # no positives anywhere: zero division -> 0
    assert metrics["false_positive_rate"] == 0.0
