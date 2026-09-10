"""Package the Chrome extension into a Web Store upload zip.

Zips apps/extension contents (manifest, background, offscreen, popup,
content, icons, vendored deps) into dist/scamless-extension-vX.Y.Z.zip -
the exact file the Chrome Web Store dashboard accepts.

Usage: python training/scripts/make_store_zip.py [--version 0.1.0]
"""

import argparse
import pathlib
import zipfile

HERE = pathlib.Path(__file__).resolve().parents[2] / "apps"
EXT = HERE / "extension"
EXCLUDE_NAMES = {"README.md", "fetch-deps.py"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="0.1.0")
    args = parser.parse_args()

    vendor = EXT / "vendor" / "transformers.min.js"
    if not vendor.exists():
        raise FileNotFoundError(
            "vendor/transformers.min.js missing - run apps/extension/fetch-deps.py first"
        )

    out = HERE.parent / "dist" / f"scamless-extension-v{args.version}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(EXT.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(EXT)
            if rel.parts[0] in EXCLUDE_NAMES or any(p.startswith(".") for p in rel.parts):
                continue
            zf.write(path, rel)

    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
