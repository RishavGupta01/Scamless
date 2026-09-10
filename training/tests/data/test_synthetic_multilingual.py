import random

from scamless.data.synthetic_multilingual import (
    CATEGORIES,
    HARD_NEGATIVES,
    TEMPLATES,
    generate_synthetic,
)


def test_deterministic_generation():
    a = generate_synthetic(cap_per_lang_category=20, seed=7)
    b = generate_synthetic(cap_per_lang_category=20, seed=7)
    assert [r["text"] for r in a] == [r["text"] for r in b]


def test_all_labels_are_valid_schema_names():
    from scamless.labels import label_index

    records = generate_synthetic(cap_per_lang_category=10, seed=3)
    for r in records:
        for label in r["labels"]:
            label_index(label)  # raises if unknown


def test_all_target_languages_have_templates_and_negatives():
    for lang in ("hi", "hinglish", "bn", "mr", "pa", "gu", "ta", "te", "es", "pt", "fr", "ar", "id"):
        assert lang in TEMPLATES, f"missing scam templates for {lang}"
        assert lang in HARD_NEGATIVES, f"missing hard negatives for {lang}"
        # every category must have at least one template per language
        for category in CATEGORIES:
            assert TEMPLATES[lang].get(category), f"{lang}/{category} has no templates"


def test_synthetic_mix_is_balanced_and_capped():
    records = generate_synthetic(cap_per_lang_category=30, seed=5)
    scams = [r for r in records if r["labels"]]
    safe = [r for r in records if not r["labels"]]
    assert scams and safe
    # hard negatives stay a meaningful share so FP rate is trained, not ignored
    assert len(safe) > 0.1 * len(records)
    # no language dominates the synthetic pool
    from collections import Counter

    per_lang = Counter(r["language"] for r in records)
    assert max(per_lang.values()) < 0.5 * len(records)


def test_slots_are_filled_no_leftover_placeholders():
    rng = random.Random(11)
    from scamless.data.synthetic_multilingual import _fill

    filled = _fill("{bank} OTP {otp} for {amount}, pay at {link} within {hours}h via {app}", rng)
    for placeholder in ("{bank}", "{otp}", "{amount}", "{link}", "{hours}", "{app}"):
        assert placeholder not in filled
