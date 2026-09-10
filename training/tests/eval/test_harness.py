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
    metrics = compute_metrics(df, preds, min_support=1)
    assert metrics["macro_f1"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["per_label"]["generic_spam"]["f1"] == 1.0
    assert metrics["scam_union"]["f1"] == 1.0


def test_false_positive_rate_counts_safe_rows_with_any_positive():
    df = _frame(["meeting at 3pm", "invoice attached", "win money"], [[], [], ["generic_spam"]])
    preds = [["phishing"], [], ["generic_spam"]]
    metrics = compute_metrics(df, preds)
    # one of the two safe rows was falsely flagged
    assert metrics["false_positive_rate"] == 0.5


def test_safe_rows_do_not_count_in_scam_recall():
    df = _frame(["meeting at 3pm"], [[]])
    metrics = compute_metrics(df, [[]], min_support=1)
    assert metrics["macro_f1"] == 0.0  # no positives anywhere: zero division -> 0
    assert metrics["false_positive_rate"] == 0.0


def test_scam_union_counts_subcategory_confusion_as_protection():
    # a phishing email flagged as generic_spam is still caught as a scam:
    # the union metric must credit it even though the sublabel is wrong
    df = _frame(
        ["phishing email one", "phishing email two", "normal office note"],
        [["phishing"], ["phishing"], []],
    )
    preds = [["generic_spam"], ["generic_spam"], []]
    metrics = compute_metrics(df, preds, min_support=1)
    assert metrics["scam_union"]["recall"] == 1.0
    assert metrics["per_label"]["phishing"]["f1"] == 0.0  # sublabel missed
    assert metrics["scam_union"]["precision"] == 1.0


def test_zero_support_labels_excluded_from_macro():
    # 5 rows of phishing (support < min_support) must not drag macro down
    df = _frame(
        [f"phishing mail number {i} with long text" for i in range(5)] + ["normal mail"],
        [["phishing"]] * 5 + [[]],
    )
    preds = [["phishing"]] * 5 + [[]]
    metrics = compute_metrics(df, preds, min_support=30)
    assert metrics["macro_f1"] == 0.0  # phishing excluded: no evaluable labels
    assert metrics["macro_f1_evaluable_labels"] == 0
    assert metrics["per_label"]["phishing"]["f1"] == 1.0  # still reported per-label
