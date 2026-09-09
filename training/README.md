# Scamless Training Core

Detection core for Scamless: data pipeline, training, eval, ONNX export.

## Quickstart

Unit tests run locally (CPU only, no network):

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -e "training[dev]"
    .venv\Scripts\python.exe -m pytest training/tests

Training runs on Google Colab (free T4 GPU):

    1. Open notebooks/train_colab.ipynb in Google Colab
    2. Runtime > Change runtime type > T4 GPU
    3. Run all
    4. Artifacts land in Drive at MyDrive/scamless/artifacts

## Commands

    pytest training/tests          # unit tests (no network)
    ruff check training            # lint

## Pipeline

One command runs everything (use this for every training run):

    python -m scamless.pipeline            # fetch, build, train, export, tune, eval, gate
    python -m scamless.pipeline --skip-build --skip-fetch   # retrain on existing data

Every step runs as a fresh subprocess, so a `git pull` before the run always
takes effect. The gate exits nonzero and blocks the pipeline if quality
regressed (macro-F1 below threshold or false-positive rate above threshold).

    fetch_all    download raw corpora (SMS, SpamAssassin, OpenPhish, Majestic, HF phishing texts)
    build        parse, dedupe, split, write Parquet + label report (--replay-glob for incremental runs)
    train        fine-tune multilingual MiniLM, 15-label sigmoid head, weights saved before validation
    export_onnx  torch.onnx export + int8 dynamic quantization
    run_eval     test-set metrics, --tune calibrates per-label thresholds on validation
    gate         release gate: macro-F1 and false-positive-rate thresholds

## Status

- Pipeline runs end to end (fetch, build, train, export, eval, gate)
- Trained on: 44,612 train / 5,576 val / 5,576 test messages + 50,300 URLs
- Known data gaps (Phase 2 fills via synthetic generation): otp_request, payment_pressure,
  investment_crypto, romance, tech_support, lottery_prize, job_task, gov_bank_impersonation,
  advance_fee, sextortion, delivery_scam, account_suspension, malicious_link
- Metrics: recorded after first Colab run

## Layout

    src/scamless/labels.py      label schema (public contract, append-only)
    src/scamless/data/          parsers, downloaders, dataset builder
    src/scamless/aug/           adversarial augmentation
    src/scamless/model/         trainer + ONNX export
    src/scamless/eval/          harness, baseline, eval runner, release gate
