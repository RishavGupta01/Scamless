# Scamless Training Core

Detection core for Scamless: data pipeline, training, evaluation, ONNX
export, and the release gate. Everything needed to produce a verified model
artifact from public corpora - on a free Colab T4, in about one hour.

## Quickstart

Unit tests run locally (CPU only, no network):

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -e "training[dev]"
    .venv\Scripts\python.exe -m pytest training/tests

Training runs on Google Colab (free T4 GPU) - see `docs/go-live.md`:

    1. Open notebooks/train_colab.ipynb in Google Colab
    2. Runtime > Change runtime type > T4 GPU
    3. Run all
    4. Artifacts land in Drive at MyDrive/scamless/artifacts

## The one-command pipeline

    python -m scamless.pipeline            # fetch, build, train, export, tune, eval, gate
    python -m scamless.pipeline --skip-build --skip-fetch   # retrain on existing data
    python -m scamless.pipeline --init-from artifacts/model_v1 --replay-glob "old/*.parquet"   # incremental
    python -m scamless.pipeline --spans-file spans.jsonl    # tactic-span training (phase 4 ready)

Every step runs as a fresh subprocess (immune to stale module caches), all
heavy steps log progress, weights save before validation, and mid-training
checkpoints land on Drive every 3000 steps for disconnect recovery.

## Commands

    pytest training/tests          # unit tests (no network, no GPU)
    ruff check training            # lint

## Pipeline stages

    fetch_all    parallel idempotent downloads (atomic writes, retries)
    build        parse 9 corpora, clean, dedupe, split, Parquet + report
    train        multilingual MiniLM + 15-label sigmoid head, fp16, OOM failsafe
    export_onnx  torch.onnx export, self-verifying quantization ladder
    run_eval     test-set metrics; --tune calibrates per-label thresholds
    gate         release gate: macro-F1, union scam F1, false-positive rate

## Quality gates

The gate blocks the pipeline (nonzero exit) unless:

- macro-F1 over evaluable categories (>= 30 test rows) meets macro_f1_min
- union scam-detection F1 meets scam_union_f1_min
- false-positive rate on legitimate messages stays under fp_rate_max

Gate values live in `src/scamless/eval/eval_config.json` (launch-calibrated
v0.9 values; tighten toward spec targets as data grows).

## Dataset (v0.9 build)

229,125 train / 28,641 val / 28,641 test messages + 153,555 URLs from nine
sources: seven-datasets email corpus (203k), multilingual SMS (21 languages),
Enron, SMS Spam Collection, SpamAssassin (marker-enriched), HF phishing texts,
phishing-v2 emails, plus template-generated and LLM-written multilingual
scam seeds with per-language hard negatives.

Label mix: generic_spam 102.6k, phishing 7.7k, otp_request 859, delivery 644,
advance_fee 552, payment_pressure 551, lottery 523, gov/bank 519, job 505,
investment 464, account suspension 336, romance/tech/sextortion seed-only.

## Layout

    src/scamless/labels.py      label schema (public contract, append-only)
    src/scamless/data/          parsers, downloaders, builder, cleaning, spans
    src/scamless/aug/           adversarial augmentation (homoglyph, zero-width, leet)
    src/scamless/model/         trainer, multi-task span head, ONNX export
    src/scamless/eval/          harness, baseline, threshold tuning, release gate
    src/scamless/pipeline.py    one-command orchestration
    notebooks/train_colab.ipynb Colab GPU training notebook
