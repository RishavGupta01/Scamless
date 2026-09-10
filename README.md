# Scamless

**On-device, multilingual scam detection. Paste any message, get a fused
verdict in milliseconds - nothing ever leaves the device.**

A scam detection core (fine-tuned multilingual MiniLM + signal-fusion engine),
a static web app, and a Chrome extension - all backend-free, all private by
architecture: there is no server to trust because there is no server.

```
+---------------------------------------------------------------+
|  training/            data pipeline -> fine-tune -> ONNX      |
|                       + eval harness + release gate           |
+------------------------------+--------------------------------+
                               |  int8 ONNX (~118 MB)
+------------------------------v--------------------------------+
|  apps/web (GitHub Pages)     |  apps/extension (Chrome MV3)   |
|  transformers.js in-browser  |  offscreen inference, badges,  |
|  scan + risk meter + why     |  context-menu + popup scanning |
+------------------------------+--------------------------------+
```

## Detection coverage

15 labels with per-label calibrated thresholds: phishing, investment/crypto,
romance, tech-support, lottery, job/task, government-bank impersonation, OTP
requests, payment pressure, advance-fee (419), sextortion, delivery scams,
account suspension, malicious links, generic spam.

The fusion engine layers deterministic signals on top of the model: URL
structure analysis (punycode, brand lookalikes, IP hosts, abusive TLDs),
homoglyph detection, OTP-pattern matching, digital-arrest and UPI-reversal
patterns, and urgency language - each visible in the "Why" panel so every
verdict is explainable.

Training coverage spans 22 languages across email and SMS corpora
(620k+ rows), with per-language hard negatives so legitimate bank alerts and
parcel notifications are not flagged.

## Repository layout

| Path | What it is |
|------|------------|
| `training/` | detection core: data pipeline, trainer, eval harness, release gate |
| `apps/web/` | static web app (GitHub Pages) |
| `apps/extension/` | Chrome extension (Manifest V3) |
| `docs/` | spec, plans, model hosting, store listing |
| `notebooks/` | Colab training notebook |

## Quickstart

Web + extension (end users): the site is served from GitHub Pages; the
extension is load-unpacked or installed from the store. Model artifacts are
hosted on the Hugging Face Hub - see `docs/model-hosting.md`.

Training (one command on Colab T4, disconnect-proof):

    python -m scamless.pipeline

Unit tests (no network, no GPU):

    python -m pytest training/tests

## Quality gates

Every model artifact must pass the release gate before it ships:
macro-F1 above threshold, false-positive rate below threshold, ONNX-parity
verified against training weights. The gate blocks the pipeline on failure -
broken models never reach users silently.

## Documentation

- Design spec: `docs/superpowers/specs/`
- Implementation plans: `docs/superpowers/plans/`
- Model hosting: `docs/model-hosting.md`
- Store listing: `docs/store-listing.md`

## License

Code: Apache-2.0 (see `LICENSE`). Dataset attributions and brand notices:
`NOTICE`. The Scamless name and mark are not covered by the code license.
