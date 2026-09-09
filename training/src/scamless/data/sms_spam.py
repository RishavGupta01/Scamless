"""Parser for the UCI SMS Spam Collection format: '<label>\t<text>' per line."""

from scamless.data.schemas import make_message

SOURCE = "sms_spam_collection"


def parse_sms_lines(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        label, _, text = line.partition("\t")
        labels = ["generic_spam"] if label == "spam" else []
        records.append(make_message(text, labels, SOURCE, language="en"))
    return records
