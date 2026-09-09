from scamless.eval.baseline import heuristic_predict


def test_scams_trigger_labels():
    assert "generic_spam" in heuristic_predict("WIN a free prize now!! text 80085")
    assert "otp_request" in heuristic_predict("share the OTP code 5521 to verify")
    assert "payment_pressure" in heuristic_predict("send $500 immediately or else")


def test_normal_text_is_safe():
    assert heuristic_predict("Lunch tomorrow at the usual place?") == []
    assert heuristic_predict("Here are the meeting notes from Tuesday.") == []


def test_obfuscated_scams_still_trigger():
    # zero-width chars between letters must not hide the scam
    text = "claim\u200byour\u200bprize"
    assert "generic_spam" in heuristic_predict(text)
