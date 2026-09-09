from scamless.data.spamassassin import classify_dir, enrich_labels, parse_message_bytes


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


def test_phishing_styled_spam_gets_both_labels():
    raw = (
        b"Subject: security alert\r\n\r\n"
        b"We detected unusual activity on your account. "
        b"Please verify your account within 24 hours or access will be limited. "
        b"Click here to login and confirm your identity now.\r\n"
    )
    record = parse_message_bytes(raw, "spam_2", "00042.abc")
    assert set(record["labels"]) == {"generic_spam", "phishing"}


def test_enrich_labels_ignores_non_spam():
    assert enrich_labels("verify your account now", []) == []
    assert enrich_labels("verify your account now", ["phishing"]) == ["phishing"]
