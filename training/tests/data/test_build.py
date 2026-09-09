import pandas as pd

from scamless.data.build import dedupe, split_records


def _msgs():
    return [
        {"text": "win free prizes now click here", "labels": ["generic_spam"], "source": "a", "language": "en"},
        {"text": "win free prizes now click here", "labels": ["generic_spam"], "source": "b", "language": "en"},
        {"text": "please review the attached invoice for march", "labels": [], "source": "a", "language": "en"},
        {"text": "verify your account at this secure link", "labels": ["phishing"], "source": "c", "language": "en"},
    ] * 10


def test_dedupe_keeps_first_occurrence_per_text():
    out = dedupe(_msgs())
    assert len(out) == 3
    assert {r["source"] for r in out} == {"a", "c"}


def test_split_records_proportions_and_disjoint():
    df = pd.DataFrame(dedupe(_msgs()))
    train, val, test = split_records(df, seed=7)
    total = len(train) + len(val) + len(test)
    assert total == len(df)
    assert abs(len(val) - round(0.1 * len(df))) <= 1
    train_texts = set(train["text"])
    test_texts = set(test["text"])
    assert not (train_texts & test_texts)
