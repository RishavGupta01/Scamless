# Scamless

**On-device, multilingual scam detection. Paste any message, get a fused,
explainable verdict in milliseconds - nothing ever leaves the device.**

A scam detection core (fine-tuned multilingual MiniLM + signal-fusion
engine), a static web app, and a Chrome extension - all backend-free, all
private by architecture: there is no server to trust because there is no
server.

```
+---------------------------------------------------------------+
|  training/            data pipeline -> fine-tune -> ONNX      |
|                       + eval harness + release gate           |
+------------------------------+--------------------------------+
                               |  verified artifact (fp16/fp32)
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
homoglyph detection, OTP-pattern matching, digital-arrest and UPI-PIN
reversal patterns, and urgency language - each visible in the "Why" panel so
every point of the risk score traces to a named signal.

Training coverage spans 22 languages across email and SMS corpora
(250k+ rows), with per-language hard negatives so legitimate bank alerts and
parcel notifications are not flagged.

## Measured quality (v0.9 artifact, 28.6k held-out test rows)

| Metric | Value |
|--------|-------|
| Scam/phishing category F1 | 0.90-1.00 (OTP, lottery, advance-fee, job/task, generic spam, investment) |
| Test macro-F1 | 0.694 across 11 evaluable categories |
| False-positive rate | 5.2% (threshold-calibrated) |
| ONNX artifact parity | 0.0000 max probability drift vs training weights |
| Verdict latency | milliseconds in-browser |

Weak categories (payment pressure, gov impersonation recall, delivery) are
known data gaps with the fix queued: LLM-generated natural data at scale
through the existing pipeline. The release gate blocks silent regressions
while they close.

## Repository layout

| Path | What it is |
|------|------------|
| `training/` | detection core: data pipeline, trainer, eval harness, release gate |
| `apps/web/` | static web app (GitHub Pages) |
| `apps/extension/` | Chrome extension (Manifest V3) |
| `docs/go-live.md` | step-by-step launch guide |
| `docs/model-hosting.md` | publishing model artifacts to the Hugging Face Hub |
| `docs/store-listing.md` | Chrome Web Store listing and permission justifications |
| `docs/superpowers/specs/` | design specification |
| `docs/superpowers/plans/` | implementation plans |
| `notebooks/` | Colab training notebook |

## Quickstart

End users: the site is served from GitHub Pages; the extension installs from
the Chrome Web Store (or load-unpacked). See `docs/go-live.md` for the full
launch walkthrough.

Training (one command on Colab T4, disconnect-proof, checkpoints on Drive):

    python -m scamless.pipeline

Unit tests (no network, no GPU):

    python -m pytest training/tests

## Quality gates

Every model artifact must pass the release gate before it ships:

- macro-F1 over evaluable categories (support >= 30 test rows)
- union scam-detection F1 (the product metric: any scam flagged counts)
- false-positive rate on legitimate messages
- ONNX parity against training weights

The gate blocks the pipeline on failure - a regressed model never reaches
users silently. Gate values are launch-calibrated and tighten as data grows.

## Documentation

- Go-live guide: `docs/go-live.md`
- Design spec: `docs/superpowers/specs/`
- Implementation plans: `docs/superpowers/plans/`

## License

Code: Apache-2.0 (see `LICENSE`). Dataset attributions and brand notices:
`NOTICE`. The Scamless name and mark are not covered by the code license.
