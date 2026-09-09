from scamless.data.enron import records_from_hf_rows


def test_records_from_hf_rows():
    rows = [
        {"label": 1, "text": "Meeting is at 3pm on Tuesday, please confirm attendance."},
        {"label": 0, "text": "Congratulations! You have been selected for a cash prize."},
        {"label": 0, "text": "short"},
    ]
    records = records_from_hf_rows(rows)
    assert len(records) == 2
    assert records[0]["labels"] == []
    assert records[1]["labels"] == ["generic_spam"]
    assert records[0]["source"] == "enron_spam"
