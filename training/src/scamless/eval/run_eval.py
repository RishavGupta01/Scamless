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


def predict_onnx(df: pd.DataFrame, model_dir: str, max_len: int = 256) -> list[list[str]]:
    onnx_path = pathlib.Path(model_dir) / "onnx" / "model_int8.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    texts = [str(t) for t in df["text"]]
    preds = []
    for i in range(0, len(texts), 64):
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
        probs = 1.0 / (1.0 + np.exp(-logits))
        for row in probs:
            preds.append([SCAM_LABELS[j] for j in range(len(row)) if row[j] > 0.5])
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "onnx"], default="baseline")
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    args = parser.parse_args()

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
