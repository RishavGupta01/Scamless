"""Adversarial augmentations: homoglyphs, zero-width chars, leetspeak.

Applied at training time to a fraction of scam examples so the model is
robust to obfuscation attacks. Deterministic given the rng argument.
"""

import random

HOMOGLYPHS = {
    "a": "а",  # cyrillic a
    "e": "е",  # cyrillic e
    "o": "о",  # cyrillic o
    "p": "р",  # cyrillic p
    "c": "с",  # cyrillic c
    "y": "у",  # cyrillic y
    "x": "х",  # cyrillic x
    "i": "і",  # cyrillic i
}

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}

ZERO_WIDTH = "\u200b"


def homoglyph_substitute(text: str, rng: random.Random, rate: float = 0.15) -> str:
    return "".join(
        HOMOGLYPHS[c] if c in HOMOGLYPHS and rng.random() < rate else c for c in text
    )


def leet_substitute(text: str, rng: random.Random, rate: float = 0.2) -> str:
    return "".join(LEET[c] if c in LEET and rng.random() < rate else c for c in text)


def insert_zero_width(text: str, rng: random.Random, rate: float = 0.1) -> str:
    chars = []
    for c in text:
        chars.append(c)
        if c.isalpha() and rng.random() < rate:
            chars.append(ZERO_WIDTH)
    return "".join(chars)


def apply_all(text: str, rng: random.Random) -> str:
    text = leet_substitute(text, rng)
    text = homoglyph_substitute(text, rng)
    return insert_zero_width(text, rng)
