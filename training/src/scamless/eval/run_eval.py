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


def predict_torch_probs(df: pd.DataFrame, model_dir: str, max_len: int = 256) -> np.ndarray:
    """GPU inference from the torch weights: ~10x faster than CPU ONNX sweeps.

    Used for the large quality sweeps when CUDA is available; the exported
    ONNX artifact is still validated separately by the parity check below.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    texts = [str(t) for t in df["text"]]
    probs_batches = []
    B = 256
    with torch.no_grad():
        for i in range(0, len(texts), B):
            enc = tokenizer(
                texts[i : i + B], truncation=True, max_length=max_len, padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs_batches.append(torch.sigmoid(logits).cpu().numpy())
            print(f"  gpu inference {min(i + B, len(texts))}/{len(texts)}", flush=True)
    del model
    if device == "cuda":
        import torch as _t

        _t.cuda.empty_cache()
    return np.concatenate(probs_batches, axis=0)


def onnx_parity_check(df: pd.DataFrame, model_dir: str, torch_probs: np.ndarray, n: int = 256) -> float:
    """Confirm the exported int8 artifact matches the torch weights on a sample."""
    sub = df.iloc[:n].reset_index(drop=True)
    onnx_probs = predict_onnx_probs(sub, model_dir)
    diff = float(np.abs(onnx_probs - torch_probs[:n]).max())
    print(f"onnx parity check ({n} samples): max prob diff = {diff:.4f}", flush=True)
    return diff


def sweep_probs(df: pd.DataFrame, model_dir: str, max_len: int = 256) -> tuple[np.ndarray, str]:
    """Pick the fastest available inference path; fall back to CPU ONNX if the
    GPU/torch path diverges from the exported artifact beyond tolerance."""
    import torch

    if torch.cuda.is_available():
        probs = predict_torch_probs(df, model_dir, max_len)
        diff = onnx_parity_check(df, model_dir, probs)
        if diff <= 0.10:
            return probs, "torch-gpu"
        print(f"parity diff {diff:.4f} > 0.10 - falling back to CPU ONNX sweep", flush=True)
    return predict_onnx_probs(df, model_dir, max_len), "onnx-cpu"


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
            val_probs, _engine = sweep_probs(val_df, args.model_dir)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, val_probs)
        val_truth = _truth(val_df)
        from scamless.eval.gate import load_config

        eval_cfg = load_config(pathlib.Path(__file__).parent / "eval_config.json")
        thresholds = tune_thresholds(
            val_probs, val_truth, precision_floor=eval_cfg.get("precision_floor", 0.90)
        )
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
    if args.mode == "baseline":
        preds = predict_baseline(df)
    else:
        test_probs, _engine = sweep_probs(df, args.model_dir)
        preds = probs_to_labels(test_probs, load_thresholds(args.model_dir))
    metrics = compute_metrics(df, preds)

    out = pathlib.Path("artifacts/eval/metrics.json")
    save_metrics(metrics, out)
    print(json.dumps({k: metrics[k] for k in ("macro_f1", "false_positive_rate")}, indent=2))
    print(f"metrics written to {out}")


if __name__ == "__main__":
    main()
