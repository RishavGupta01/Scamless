from scamless.data.emails import extract_body


def test_extracts_plain_text_body():
    raw = (
        b"From: bank@alerts.example\r\n"
        b"Subject: Account alert\r\n"
        b"\r\n"
        b"Your account will be suspended.\r\n"
        b"Call us now.\r\n"
    )
    body = extract_body(raw)
    assert "Your account will be suspended." in body


def test_extracts_text_part_from_multipart():
    raw = (
        b"From: x@example\r\n"
        b"Subject: hi\r\n"
        b"MIME-Version: 1.0\r\n"
        b'Content-Type: multipart/alternative; boundary="BOUND"\r\n'
        b"\r\n"
        b"--BOUND\r\n"
        b"Content-Type: text/html\r\n"
        b"\r\n"
        b"<b>html part</b>\r\n"
        b"--BOUND\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"plain part here\r\n"
        b"--BOUND--\r\n"
    )
    body = extract_body(raw)
    assert "plain part here" in body


def test_returns_empty_string_when_no_body():
    raw = b"From: x@example\r\nSubject: only headers\r\n"
    assert extract_body(raw) == ""
