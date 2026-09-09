from scamless.data.phishing_hf import records_from_rows


def test_phishing_rows_mapped():
    rows = [
        {"text": "Dear customer, verify your account within 24 hours or it will be suspended.", "label": 1},
        {"text": "Please find attached the minutes from the Tuesday product review meeting.", "label": 0},
        {"text": "short", "label": 1},
    ]
    records = records_from_rows(rows)
    assert len(records) == 2
    assert records[0]["labels"] == ["phishing"]
    assert records[1]["labels"] == []
    assert records[0]["source"] == "hf_phishing_texts"
    assert records[0]["language"] == "en"


def test_label_one_is_phishing_not_benign():
    rows = [{"text": "urgent: confirm your identity by clicking this secure link today", "label": 1}]
    assert records_from_rows(rows)[0]["labels"] == ["phishing"]
