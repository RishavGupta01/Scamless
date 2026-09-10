import pandas as pd

from scamless.data import phishing_v2


def test_records_split_by_content_type():
    df = pd.DataFrame(
        {
            "content": [
                "http://secure-login-verify.example/account/update",
                "https://www.chase.com/personal/credit-cards",
                "Dear customer please verify your account within 24 hours now",
                "The quarterly planning meeting notes are attached for review",
                "tiny",
            ],
            "label": [3, 2, 1, 0, 1],
        }
    )
    messages, url_records = phishing_v2.records_from_dataframe(df)
    assert len(url_records) == 2
    assert {u["malicious"] for u in url_records} == {True, False}
    assert len(messages) == 2
    phishing_msgs = [m for m in messages if m["labels"] == ["phishing"]]
    assert len(phishing_msgs) == 1
    assert all(m["language"] == "en" for m in messages)


def test_collect_handles_empty_dir(tmp_path, monkeypatch):
    import scamless.data.phishing_v2 as pv

    monkeypatch.setattr(pv, "RAW", tmp_path)
    msgs, urls = pv.collect()
    assert msgs == []
    assert urls == []
