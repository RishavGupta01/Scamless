"""Generate all Scamless brand assets programmatically.

Extension icons: 16/32/48/128 px shield mark.
Web assets: favicon-32, apple-touch-icon (180), og-image (1200x630)
with the brand name and tagline (font resolved from the system, with
matplotlib's bundled DejaVu as fallback).

Usage: python training/scripts/generate_icons.py
Writes: apps/extension/icons/*.png and apps/web/assets/*.png
"""

import pathlib

from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXT_OUT = ROOT / "apps" / "extension" / "icons"
WEB_OUT = ROOT / "apps" / "web" / "assets"

BG = (11, 15, 20)
SURFACE = (17, 22, 29)
ACCENT = (78, 161, 255)
ACCENT_LIGHT = (124, 196, 255)
CHECK = (61, 220, 132)
TEXT = (230, 237, 243)
TEXT_DIM = (139, 152, 165)


def find_bold_font(size: int):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    try:
        import matplotlib.font_manager as fm

        candidates.insert(0, fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")))
    except Exception:  # noqa: BLE001, S110 - font finding is best-effort, system paths cover the rest
        pass
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:  # noqa: BLE001, S112 - try the next candidate font
            continue
    return ImageFont.load_default()


def draw_shield(d: ImageDraw.ImageDraw, cx: float, top: float, width: float, height: float) -> None:
    """Shield silhouette with a check mark, sized relative to a bounding box."""
    half = width / 2
    shield = [
        (cx - half, top),
        (cx + half, top),
        (cx + half, top + height * 0.52),
        (cx, top + height),
        (cx - half, top + height * 0.52),
    ]
    d.polygon(shield, fill=ACCENT + (255,))
    lw = max(2, int(width * 0.14))
    a = (cx - half * 0.52, top + height * 0.52)
    b = (cx - half * 0.10, top + height * 0.68)
    c = (cx + half * 0.58, top + height * 0.34)
    d.line([a, b, c], fill=BG + (255,), width=lw)
    r = lw / 2
    for pt in (a, b, c):
        d.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=BG + (255,))


def make_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=size * 0.22, fill=BG + (255,))
    d.rounded_rectangle(
        [1, 1, size - 2, size - 2], radius=size * 0.21, outline=ACCENT + (255,),
        width=max(1, size // 48),
    )
    draw_shield(d, size / 2, size * 0.20, size * 0.54, size * 0.62)
    return img


def make_og_image(width: int = 1200, height: int = 630) -> Image.Image:
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    # subtle accent bar on the left edge
    d.rectangle([0, 0, 12, height], fill=ACCENT)

    # shield mark, large
    draw_shield(d, width * 0.20, height * 0.22, width * 0.16, height * 0.56)

    title_font = find_bold_font(96)
    sub_font = find_bold_font(38)
    dim_font = find_bold_font(30)

    x = width * 0.34
    d.text((x, height * 0.30), "SCAMLESS", font=title_font, fill=TEXT)
    d.text((x, height * 0.50), "On-device scam detection", font=sub_font, fill=ACCENT_LIGHT)
    d.text(
        (x, height * 0.62),
        "Private - explainable - multilingual",
        font=dim_font,
        fill=TEXT_DIM,
    )
    d.text(
        (x, height * 0.74),
        "No uploads. No server. Nothing leaves your device.",
        font=dim_font,
        fill=TEXT_DIM,
    )
    return img


def main() -> None:
    EXT_OUT.mkdir(parents=True, exist_ok=True)
    WEB_OUT.mkdir(parents=True, exist_ok=True)

    for size in [16, 32, 48, 128]:
        make_icon(size).save(EXT_OUT / f"icon{size}.png")
        print(f"wrote {EXT_OUT / f'icon{size}.png'}")

    make_icon(32).save(WEB_OUT / "favicon-32.png")
    print(f"wrote {WEB_OUT / 'favicon-32.png'}")

    make_icon(180).save(WEB_OUT / "apple-touch-icon.png")
    print(f"wrote {WEB_OUT / 'apple-touch-icon.png'}")

    make_og_image().save(WEB_OUT / "og-image.png")
    print(f"wrote {WEB_OUT / 'og-image.png'}")

    print("brand assets generated")


if __name__ == "__main__":
    main()
