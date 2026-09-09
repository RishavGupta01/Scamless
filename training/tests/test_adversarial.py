import random

from scamless.aug.adversarial import (
    apply_all,
    homoglyph_substitute,
    insert_zero_width,
    leet_substitute,
)


def test_homoglyph_substitute_changes_latin_chars():
    out = homoglyph_substitute("paypal.com secure", random.Random(1))
    assert out != "paypal.com secure"
    assert "а" in out or "е" in out or "о" in out  # cyrillic lookalikes present


def test_zero_width_insert_is_invisible():
    out = insert_zero_width("urgent action", random.Random(1))
    assert out != "urgent action"
    assert out.replace("\u200b", "") == "urgent action"


def test_leet_substitute_maps_known_chars():
    out = leet_substitute("free money", random.Random(1))
    assert out != "free money"
    assert any(c in "431057" for c in out)  # at least one leet digit present


def test_apply_all_preserves_meaning_characters():
    text = "claim your prize now at the link"
    out = apply_all(text, random.Random(3))
    assert len(out) >= len(text)
    assert out != text
