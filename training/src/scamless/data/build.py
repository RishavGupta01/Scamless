"""Dataset builder: parse all corpora, dedupe, split, write Parquet + report.

Usage: python -m scamless.data.build
"""

import argparse
import collections
import json
import pathlib

import pandas as pd

from scamless.data import (
    enron,
    multilingual_sms,
    nazario,
    phishing_hf,
    phishing_v2,
    seven_phishing,
    sms_spam,
    spamassassin,
    urls,
)
from scamless.data.clean import clean_text
from scamless.data.download import RAW

PROCESSED = RAW.parent / "processed"
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


def collect_messages() -> list[dict]:
    records: list[dict] = []
    extra_url_records: list[dict] = []  # url rows harvested from message corpora
    skipped = 0

    def guarded_parse(parse_fn, *args) -> dict | None:
        nonlocal skipped
        try:
            return parse_fn(*args)
        except Exception:  # noqa: BLE001 - one corrupt file must not kill the build
            skipped += 1
            return None

    sms_path = RAW / "sms" / "SMSSpamCollection"
    if sms_path.exists():
        sms_lines = sms_path.read_text(encoding="utf-8", errors="replace").splitlines()
        records.extend(sms_spam.parse_sms_lines(sms_lines))
    else:
        print("sms corpus missing: run fetch_all first")

    sa_root = RAW / "spamassassin"
    if sa_root.exists():
        for tar_dir in sorted(sa_root.iterdir()):
            if not tar_dir.is_dir():
                continue
            for f in sorted(tar_dir.rglob("*")):
                if f.is_file() and not f.name.startswith(("cmds", "dontbother", "ctstSkipped", "sbatch")):
                    rec = guarded_parse(spamassassin.parse_message_bytes, f.read_bytes(), tar_dir.name, f.name)
                    if rec:
                        records.append(rec)

    nz_root = RAW / "nazario"
    if nz_root.exists():
        for f in sorted(nz_root.rglob("*.eml")):
            rec = guarded_parse(nazario.parse_phishing_email, f.read_bytes(), f.name)
            if rec:
                records.append(rec)

    ph_json = RAW / "phishing_hf" / "texts.json"
    if ph_json.exists():
        records.extend(phishing_hf.records_from_json(ph_json.read_bytes()))
    else:
        print("hf_phishing_texts missing: run fetch_all first")

    seven_dir = RAW / "seven_phishing"
    if seven_dir.exists() and any(seven_dir.glob("*.parquet")):
        records.extend(seven_phishing.collect())
    else:
        print("seven_phishing missing: run fetch_all first")

    ml_csv = RAW / "multilingual_sms" / "data-augmented.csv"
    if ml_csv.exists():
        records.extend(
            multilingual_sms.records_from_csv_bytes(ml_csv.read_bytes())
        )
    else:
        print("multilingual_sms missing: run fetch_all first")

    if (RAW / "phishing_v2").exists() and any((RAW / "phishing_v2").glob("*.parquet")):
        v2_messages, v2_urls = phishing_v2.collect()
        records.extend(v2_messages)
        extra_url_records.extend(v2_urls)
    else:
        print("phishing_v2 missing: run fetch_all first")

    try:
        from datasets import load_dataset

        ds = load_dataset(enron.SOURCE_HF_ID)
        rows = [dict(r) for split in ds.values() for r in split]
        records.extend(enron.records_from_hf_rows(rows))
    except Exception:  # noqa: BLE001 - enron is optional for a build
        print("enron skipped (offline or unavailable)")

    if skipped:
        print(f"skipped {skipped} unreadable source files")
    return records, extra_url_records


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
    dropped_empty = 0
    for r in records:
        cleaned = clean_text(r["text"])
        if not cleaned:
            dropped_empty += 1
            continue
        r["text"] = cleaned
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if dropped_empty:
        print(f"cleaning removed {dropped_empty} degenerate texts")
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


def merge_replay(train: pd.DataFrame, replay_globs: list[str]) -> pd.DataFrame:
    """Mix previously-trained data into the new train split (replay buffer).

    Prevents catastrophic forgetting when extending the model with new
    languages or categories: old examples re-enter training, test/val stay
    composed of only the fresh build. Accepts absolute paths, relative
    globs, and missing patterns (silently skipped so a missing replay file
    never blocks a rebuild).
    """
    frames = [train]
    for pattern in replay_globs:
        p = pathlib.Path(pattern)
        candidates = [p] if p.is_file() else sorted(pathlib.Path().glob(pattern))
        for path in candidates:
            old = pd.read_parquet(path)
            old["source"] = "replay:" + old["source"].astype(str)
            frames.append(old)
            print(f"replay: merged {len(old)} rows from {path}")
    return pd.concat(frames, ignore_index=True)


def report(records: list[dict], url_records: list[dict], path: pathlib.Path) -> None:
    label_counts = collections.Counter(l for r in records for l in r["labels"])
    source_counts = collections.Counter(r["source"] for r in records)
    language_counts = collections.Counter(r.get("language", "en") for r in records)
    safe_count = sum(1 for r in records if not r["labels"])
    payload = {
        "messages_total": len(records),
        "messages_safe": safe_count,
        "messages_labeled": len(records) - safe_count,
        "label_counts": dict(label_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "language_counts": dict(language_counts.most_common()),
        "urls_total": len(url_records),
        "urls_malicious": sum(1 for u in url_records if u["malicious"]),
    }
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-glob",
        nargs="*",
        default=[],
        help="glob of old message parquets to mix into the new train split",
    )
    args = parser.parse_args()

    messages, extra_urls = collect_messages()
    messages = dedupe(messages)
    url_records = collect_urls() + extra_urls

    msg_df = pd.DataFrame(messages)
    train, val, test = split_records(msg_df)
    if args.replay_glob:
        train = merge_replay(train, args.replay_glob)

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
