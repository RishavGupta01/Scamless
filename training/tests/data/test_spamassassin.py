from scamless.data.spamassassin import classify_dir, parse_message_bytes


def test_classify_dir_names():
    assert classify_dir("spam_2") == ["generic_spam"]
    assert classify_dir("easy_ham_2") == []
    assert classify_dir("hard_ham") == []


def test_parse_message_bytes():
    raw = (
        b"Subject: urgent offer\r\n\r\n"
        b"Click here to claim your prize money now\r\n"
    )
    record = parse_message_bytes(raw, dirname="spam_2", filename="00001.abc")
    assert record["labels"] == ["generic_spam"]
    assert record["source"] == "spamassassin"
    assert "claim your prize" in record["text"]


def test_short_bodies_dropped():
    raw = b"Subject: hi\r\n\r\nok\r\n"
    assert parse_message_bytes(raw, "spam_2", "x.eml") is None
