"""Generate the extension / brand icons (16, 32, 48, 128 px).

Draws the Scamless mark programmatically: dark rounded tile, accent shield,
inner check mark. Deterministic, no font dependencies, no assets to ship.

Usage: python training/scripts/generate_icons.py
Writes: apps/extension/icons/icon{16,32,48,128}.png
"""

import pathlib

from PIL import Image, ImageDraw

OUT = pathlib.Path(__file__).resolve().parents[2] / "apps" / "extension" / "icons"
SIZES = [16, 32, 48, 128]

BG = (11, 15, 20)
ACCENT = (78, 161, 255)
CHECK = (61, 220, 132)


def make(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    radius = size * 0.24
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=BG,
                        outline=ACCENT + (255,), width=max(1, size // 32))

    # shield silhouette
    cx = size / 2
    top = size * 0.20
    bottom = size * 0.82
    half = size * 0.27
    shield = [
        (cx - half, top),
        (cx + half, top),
        (cx + half, size * 0.55),
        (cx, bottom),
        (cx - half, size * 0.55),
    ]
    d.polygon(shield, fill=ACCENT + (255,))

    # check mark inside the shield
    lw = max(2, size // 12)
    a = (cx - half * 0.55, size * 0.48)
    b = (cx - half * 0.12, size * 0.60)
    c = (cx + half * 0.60, size * 0.32)
    d.line([a, b, c], fill=CHECK + (255,), width=lw)
    for pt in (a, b, c):
        r = lw / 2
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=CHECK + (255,))
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        img = make(size)
        path = OUT / f"icon{size}.png"
        img.save(path)
        print(f"wrote {path}")
    print("icons generated")


if __name__ == "__main__":
    main()
