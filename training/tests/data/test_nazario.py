from scamless.data.nazario import parse_phishing_email


def test_parse_phishing_email():
    raw = (
        b"From: security@paypa1-example.com\r\n"
        b"Subject: Verify your account\r\n\r\n"
        b"Dear customer, we detected unusual activity. "
        b"Please verify your identity within 24 hours by clicking the link below. "
        b"Failure to comply will result in permanent suspension.\r\n"
    )
    record = parse_phishing_email(raw, filename="phish001.eml")
    assert record["labels"] == ["phishing"]
    assert record["source"] == "nazario"
    assert "unusual activity" in record["text"]


def test_short_body_dropped():
    raw = b"Subject: x\r\n\r\nverify now\r\n"
    assert parse_phishing_email(raw, "x.eml") is None
