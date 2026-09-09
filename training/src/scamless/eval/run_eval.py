"""Run the trained model (or heuristic baseline) on the test split; write metrics.json.

Usage:
  python -m scamless.eval.run_eval --mode baseline
  python -m scamless.eval.run_eval --mode onnx --model-dir artifacts/model_v1
"""

import argparse
import json
import pathlib

import numpy as np
import onnxruntime as ort
import pandas as pd
from transformers import AutoTokenizer

from scamless.data.download import RAW
from scamless.eval.baseline import heuristic_predict
from scamless.eval.harness import compute_metrics, save_metrics
from scamless.labels import SCAM_LABELS


def load_test_df() -> pd.DataFrame:
    processed = RAW.parent / "processed"
    df = pd.read_parquet(processed / "messages_test.parquet")
    return df.reset_index(drop=True)


def predict_baseline(df: pd.DataFrame) -> list[list[str]]:
    return [heuristic_predict(str(t)) for t in df["text"]]


def predict_onnx_probs(df: pd.DataFrame, model_dir: str, max_len: int = 256) -> np.ndarray:
    onnx_path = pathlib.Path(model_dir) / "onnx" / "model_int8.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    texts = [str(t) for t in df["text"]]
    probs_batches = []
    total = len(texts)
    for i in range(0, total, 64):
        enc = tokenizer(
            texts[i : i + 64],
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="np",
        )
        logits = sess.run(
            None,
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        probs_batches.append(1.0 / (1.0 + np.exp(-logits)))
        print(f"  inference {min(i + 64, total)}/{total}", flush=True)
    return np.concatenate(probs_batches, axis=0)


def load_thresholds(model_dir: str) -> dict:
    path = pathlib.Path(model_dir) / "onnx" / "thresholds.json"
    if path.exists():
        return json.loads(path.read_text())
    return {label: 0.5 for label in SCAM_LABELS}


def probs_to_labels(probs: np.ndarray, thresholds: dict) -> list[list[str]]:
    return [
        [SCAM_LABELS[j] for j in range(probs.shape[1]) if row[j] > thresholds[SCAM_LABELS[j]]]
        for row in probs
    ]


def predict_onnx(df: pd.DataFrame, model_dir: str, max_len: int = 256) -> list[list[str]]:
    probs = predict_onnx_probs(df, model_dir, max_len)
    return probs_to_labels(probs, load_thresholds(model_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "onnx"], default="baseline")
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="tune per-label thresholds on the val split before evaluating the test split",
    )
    args = parser.parse_args()

    if args.mode == "onnx" and args.tune:
        from scamless.data.download import RAW
        from scamless.eval.harness import (
            save_metrics as _save,
        )
        from scamless.eval.harness import (
            truth_matrix as _truth,
        )
        from scamless.eval.harness import (
            tune_thresholds,
        )

        val_df = pd.read_parquet(
            RAW.parent / "processed" / "messages_val.parquet"
        ).reset_index(drop=True)
        cache = pathlib.Path("artifacts/eval/val_probs.npy")
        if cache.exists():
            val_probs = np.load(cache)
            print(f"loaded cached val probabilities from {cache}")
        else:
            val_probs = predict_onnx_probs(val_df, args.model_dir)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, val_probs)
        val_truth = _truth(val_df)
        thresholds = tune_thresholds(val_probs, val_truth)
        threshold_path = pathlib.Path(args.model_dir) / "onnx" / "thresholds.json"
        threshold_path.write_text(json.dumps(thresholds, indent=2))
        val_preds = probs_to_labels(val_probs, thresholds)
        val_metrics = compute_metrics(val_df, val_preds)
        print("tuned thresholds:", thresholds)
        print(
            "val metrics (tuned):",
            json.dumps({k: val_metrics[k] for k in ("macro_f1", "false_positive_rate")}),
        )
        _save(val_metrics, "artifacts/eval/metrics_val.json")

    df = load_test_df()
    preds = (
        predict_baseline(df) if args.mode == "baseline" else predict_onnx(df, args.model_dir)
    )
    metrics = compute_metrics(df, preds)

    out = pathlib.Path("artifacts/eval/metrics.json")
    save_metrics(metrics, out)
    print(json.dumps({k: metrics[k] for k in ("macro_f1", "false_positive_rate")}, indent=2))
    print(f"metrics written to {out}")


if __name__ == "__main__":
    main()
