"""Record shapes shared by every parser and the dataset builder.

MessageRecord: {"text": str, "labels": list[str], "source": str, "language": str}
UrlRecord: {"url": str, "malicious": bool, "source": str}
"""

MESSAGE_FIELDS = ["text", "labels", "source", "language"]
URL_FIELDS = ["url", "malicious", "source"]


def make_message(text: str, labels: list[str], source: str, language: str = "en") -> dict:
    return {
        "text": text.strip(),
        "labels": list(labels),
        "source": source,
        "language": language,
    }


def make_url(url: str, malicious: bool, source: str) -> dict:
    return {"url": url.strip(), "malicious": bool(malicious), "source": source}
