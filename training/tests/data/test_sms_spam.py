from scamless.data.sms_spam import parse_sms_lines


def test_parse_sms_spam_collection_lines():
    lines = [
        "ham\tGo until jurong point, crazy.. Available only in bugis n great world la",
        "spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005",
        "ham\tOk lar... Joking wif u oni...",
    ]
    records = parse_sms_lines(lines)
    assert len(records) == 3
    assert records[0]["labels"] == []
    assert records[1]["labels"] == ["generic_spam"]
    assert records[0]["source"] == "sms_spam_collection"
    assert "jurong point" in records[0]["text"]


def test_blank_lines_skipped():
    records = parse_sms_lines(["", "  ", "spam\tWINNER! text WIN to 80085"])
    assert len(records) == 1
    assert records[0]["labels"] == ["generic_spam"]


def test_text_stripped_and_tab_safe():
    records = parse_sms_lines(["ham\t  padded text  "])
    assert records[0]["text"] == "padded text"
