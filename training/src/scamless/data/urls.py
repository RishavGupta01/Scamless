"""URL corpora: OpenPhish community feed (malicious) and Majestic Million (benign).

Majestic Million substitutes the spec's Tranco top-1M because Tranco requires
manual registration; semantics are identical (ranked benign top sites).
"""

import csv
import io

from scamless.data.schemas import make_url


def parse_openphish_lines(lines: list[str]) -> list[dict]:
    records = []
    for line in lines:
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        records.append(make_url(url, malicious=True, source="openphish"))
    return records


def parse_majestic_csv(csv_text: str, top_n: int = 50000) -> list[dict]:
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for i, row in enumerate(reader):
        if i >= top_n:
            break
        domain = (row.get("Domain") or "").strip()
        if not domain:
            continue
        records.append(make_url(f"http://{domain}", malicious=False, source="majestic_million"))
    return records
