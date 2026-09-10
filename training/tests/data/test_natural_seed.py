import json
import pathlib

import pytest

from scamless.data import natural_seed


def test_seed_files_load_and_count():
    records = natural_seed.collect()
    assert len(records) >= 300, "expected a substantial natural seed corpus"
    langs = {r["language"] for r in records}
    assert {"en", "hinglish", "hi", "bn", "mr", "pa", "gu", "ta", "te", "es", "pt", "fr", "ar", "id"} <= langs


def test_every_seed_row_has_valid_labels():
    from scamless.labels import label_index

    records = natural_seed.collect()
    for r in records:
        for label in r["labels"]:
            label_index(label)  # raises on unknown


def test_seed_covers_all_nine_categories_in_english():
    records = [r for r in natural_seed.collect() if r["language"] == "en"]
    covered = {label for r in records for label in r["labels"]}
    assert covered == {
        "otp_request", "payment_pressure", "investment_crypto", "lottery_prize",
        "job_task", "gov_bank_impersonation", "delivery_scam", "advance_fee",
        "account_suspension", "phishing", "sextortion", "romance", "tech_support",
    }


def test_seed_has_hard_negatives_in_every_language():
    records = natural_seed.collect()
    safe_by_lang: dict[str, int] = {}
    for r in records:
        if not r["labels"]:
            safe_by_lang[r["language"]] = safe_by_lang.get(r["language"], 0) + 1
    for lang in ("en", "hinglish", "hi", "bn", "mr", "pa", "gu", "ta", "te", "es", "pt", "fr", "ar", "id"):
        assert safe_by_lang.get(lang, 0) >= 3, f"{lang} lacks hard negatives"


def test_malformed_seed_row_raises_loudly(tmp_path):
    bad_dir = tmp_path / "seed"
    bad_dir.mkdir()
    (bad_dir / "bad.jsonl").write_text('{"text": "ok text here", "labels": ["totally_fake_label"], "lang": "en"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        natural_seed.load_seed_files(bad_dir)


def test_all_files_are_valid_jsonl():
    seed_dir = pathlib.Path(natural_seed.SEED_DIR)
    for path in sorted(seed_dir.glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                obj = json.loads(line)
                assert "text" in obj and "labels" in obj and "lang" in obj, f"{path.name}:{line_no}"
