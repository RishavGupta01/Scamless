import pandas as pd

from scamless.data import multilingual_sms, seven_phishing
from scamless.data.clean import clean_text


def test_multilingual_sms_emits_one_record_per_language():
    csv_bytes = (
        ",labels,text,text_hi,text_de,text_fr\n"
        '0,ham,"Go until jurong point, crazy.. Available",,"Gehen Sie bis zum Punkt, verrueckt",\n'
        '1,spam,"Free entry in 2 a wkly comp to win FA Cup final","मुफ़्त प्रवेश 2 साप्ताहिक प्रतियोगिता","Freier Eintritt zur woechentlichen Verlosung","Entree libre pour gagner la finale"\n'
    ).encode()
    records = multilingual_sms.records_from_csv_bytes(csv_bytes)
    langs = {r["language"] for r in records}
    assert "en" in langs and "de" in langs and "hi" in langs
    # only hi/de have long-enough translations for the spam row; en has both rows
    spam_rows = [r for r in records if r["labels"] == ["generic_spam"]]
    assert spam_rows
    assert any(r["language"] == "hi" for r in spam_rows)
    assert any(r["language"] == "de" for r in spam_rows)


def test_multilingual_sms_short_translations_skipped():
    csv_bytes = (
        ",labels,text,text_hi\n"
        '1,spam,"Free entry win now claim prize","जीत"\n'
    ).encode()
    records = multilingual_sms.records_from_csv_bytes(csv_bytes)
    assert all(r["language"] == "en" for r in records)


def test_seven_phishing_label_semantics():
    df = pd.DataFrame(
        {
            "text": [
                "Dear customer we detected unusual activity please verify your account now",
                "Meeting notes from the quarterly review are attached for your reference",
                "short",
            ],
            "label": [1, 0, 1],
        }
    )
    records = seven_phishing.records_from_dataframe(df)
    assert len(records) == 2
    assert "phishing" in records[0]["labels"]
    assert records[1]["labels"] == []
    assert all(r["source"] == "hf_seven_phishing" for r in records)


def test_clean_text_strips_quoted_replies():
    text = (
        "URGENT verify your account within 24 hours or it will be suspended. "
        "\n-----Original Message-----\nFrom: someone old thread text here that keeps going"
    )
    cleaned = clean_text(text)
    assert "verify your account" in cleaned
    assert "Original Message" not in cleaned
    assert "old thread" not in cleaned


def test_clean_text_handles_html():
    text = "<p>Click here to confirm your identity</p><br/>Visit us now today"
    cleaned = clean_text(text)
    assert "<p>" not in cleaned
    assert "confirm your identity" in cleaned


def test_clean_text_returns_empty_for_degenerate():
    assert clean_text("<b></b>  \n\n  ") == ""
    assert clean_text(None) == ""
