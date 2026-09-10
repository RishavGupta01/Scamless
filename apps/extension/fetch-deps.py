"""Vendor browser dependencies for the Chrome extension.

Downloads the transformers.js ESM bundle and copies the shared fusion engine
from the web app into apps/extension/ so the offscreen document can import
them locally - MV3 extension pages cannot load remote scripts (strict CSP),
and we never ship code we have not inspected.

Usage: python apps/extension/fetch-deps.py
"""

import pathlib
import shutil
import urllib.request

SOURCES = {
    "vendor/transformers.min.js": "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.1/dist/transformers.min.js",
}
HERE = pathlib.Path(__file__).resolve().parent


def main() -> None:
    for rel, url in SOURCES.items():
        dest = HERE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    # shared fusion engine lives in the web app; keep the copies in sync
    fusion_src = HERE.parent / "web" / "js" / "fusion.js"
    fusion_dest = HERE / "fusion.js"
    shutil.copy2(fusion_src, fusion_dest)
    print(f"  -> {fusion_dest} (copied from web app)")

    print("done - the extension is ready to load unpacked")


if __name__ == "__main__":
    main()
