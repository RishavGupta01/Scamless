"""Dataset builder: parse all corpora, dedupe, split, write Parquet + report.

Usage: python -m scamless.data.build
"""

import collections
import json
import pathlib

import pandas as pd

from scamless.data import enron, nazario, phishing_hf, sms_spam, spamassassin, urls
from scamless.data.download import RAW

PROCESSED = RAW.parent / "processed"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def collect_messages() -> list[dict]:
    records: list[dict] = []

    sms_lines = (
        (RAW / "sms" / "SMSSpamCollection").read_text(encoding="utf-8", errors="replace").splitlines()
    )
    records.extend(sms_spam.parse_sms_lines(sms_lines))

    sa_root = RAW / "spamassassin"
    if sa_root.exists():
        for tar_dir in sorted(sa_root.iterdir()):
            if not tar_dir.is_dir():
                continue
            for f in sorted(tar_dir.rglob("*")):
                if f.is_file() and not f.name.startswith(("cmds", "dontbother", "ctstSkipped", "sbatch")):
                    rec = spamassassin.parse_message_bytes(f.read_bytes(), tar_dir.name, f.name)
                    if rec:
                        records.append(rec)

    nz_root = RAW / "nazario"
    if nz_root.exists():
        for f in sorted(nz_root.rglob("*.eml")):
            rec = nazario.parse_phishing_email(f.read_bytes(), f.name)
            if rec:
                records.append(rec)

    ph_json = RAW / "phishing_hf" / "texts.json"
    if ph_json.exists():
        records.extend(phishing_hf.records_from_json(ph_json.read_bytes()))
    else:
        print("hf_phishing_texts missing: run fetch_all first")

    try:
        from datasets import load_dataset

        ds = load_dataset(enron.SOURCE_HF_ID)
        rows = [dict(r) for split in ds.values() for r in split]
        records.extend(enron.records_from_hf_rows(rows))
    except Exception:  # noqa: BLE001 - enron is optional for a build
        print("enron skipped (offline or unavailable)")

    return records


def collect_urls() -> list[dict]:
    out: list[dict] = []
    op = RAW / "urls" / "openphish.txt"
    if op.exists():
        out.extend(
            urls.parse_openphish_lines(op.read_text(encoding="utf-8", errors="replace").splitlines())
        )
    mj = RAW / "urls" / "majestic_million.csv"
    if mj.exists():
        out.extend(
            urls.parse_majestic_csv(mj.read_text(encoding="utf-8", errors="replace"), top_n=50000)
        )
    return out


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in records:
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def split_records(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_val = round(SPLITS["val"] * n)
    n_test = round(SPLITS["test"] * n)
    test = shuffled.iloc[:n_test]
    val = shuffled.iloc[n_test : n_test + n_val]
    train = shuffled.iloc[n_test + n_val :]
    return train, val, test


def report(records: list[dict], url_records: list[dict], path: pathlib.Path) -> None:
    label_counts = collections.Counter(l for r in records for l in r["labels"])
    safe_count = sum(1 for r in records if not r["labels"])
    payload = {
        "messages_total": len(records),
        "messages_safe": safe_count,
        "messages_labeled": len(records) - safe_count,
        "label_counts": dict(label_counts.most_common()),
        "urls_total": len(url_records),
        "urls_malicious": sum(1 for u in url_records if u["malicious"]),
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    messages = dedupe(collect_messages())
    url_records = collect_urls()

    msg_df = pd.DataFrame(messages)
    train, val, test = split_records(msg_df)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(PROCESSED / "messages_train.parquet")
    val.to_parquet(PROCESSED / "messages_val.parquet")
    test.to_parquet(PROCESSED / "messages_test.parquet")
    pd.DataFrame(url_records).to_parquet(PROCESSED / "urls.parquet")

    report(messages, url_records, PROCESSED / "report.json")
    print(
        f"built: {len(train)} train / {len(val)} val / {len(test)} test messages, "
        f"{len(url_records)} urls"
    )


if __name__ == "__main__":
    main()
