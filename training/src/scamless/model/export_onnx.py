"""Export the fine-tuned model to ONNX, then apply int8 quantization.

Exports via torch.onnx.export directly (the optimum exporter path is
incompatible with torch >= 2.10's dynamo-first ONNX API). Quantization is
self-verifying: per-channel int8 is attempted first, and if the exported
artifact drifts from the fp32 model by more than 0.10 on probe probabilities,
the sensitive classifier head is excluded and quantization is retried. If
even that drifts, the fp32 model ships instead of a damaged int8 one.

Usage: python -m scamless.model.export_onnx --model-dir artifacts/model_v1
"""

import argparse
import pathlib
import shutil

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _embed_external_weights(model_path: pathlib.Path) -> None:
    model = onnx.load(str(model_path), load_external_data=True)
    changed = False
    for tensor in model.graph.initializer:
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            tensor.data_location = onnx.TensorProto.DEFAULT
            tensor.ClearField("external_data")
            changed = True
    if changed:
        onnx.save(model, str(model_path))


def export_fp32(model_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model.eval()

    dummy = tokenizer(
        ["example text"], return_tensors="pt", padding="max_length", max_length=128
    )
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy["input_ids"], dummy["attention_mask"]),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
    _embed_external_weights(out_path)
    for leftover in out_path.parent.glob(out_path.name + ".data"):
        leftover.unlink()


def _classifier_gemm_nodes(model_path: pathlib.Path) -> list[str]:
    """Gemm nodes feeding the classifier head (weight shape 15 x hidden).

    The head is tiny (5,760 weights) but sits directly on the logits:
    quantizing it shifts every probability. Excluding it costs nothing in
    size and removes the largest quantization error source.
    """
    model = onnx.load(str(model_path))
    init_dims = {i.name: list(i.dims) for i in model.graph.initializer}
    out = []
    for node in model.graph.node:
        if node.op_type != "Gemm" or len(node.input) < 2:
            continue
        dims = init_dims.get(node.input[1])
        if dims and len(dims) == 2 and dims[0] < 64 and dims[1] > 100:
            out.append(node.name)
    return out


def _max_prob_diff(fp32_path: pathlib.Path, int8_path: pathlib.Path, tokenizer) -> float:
    import onnxruntime as ort

    sess_a = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    texts = [
        "urgent verify your account within 24 hours click the link",
        "meeting notes from tuesday attached for review",
        "claim your prize now pay the fee",
        "your otp 4482 for login is valid 10 minutes",
        "quarterly budget planning attached please review",
    ] * 4
    enc = tokenizer(texts, truncation=True, max_length=128, padding="max_length", return_tensors="np")
    feed = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    la = sess_a.run(None, feed)[0]
    lb = sess_b.run(None, feed)[0]
    pa = 1.0 / (1.0 + np.exp(-la))
    pb = 1.0 / (1.0 + np.exp(-lb))
    return float(np.abs(pa - pb).max())


def export_and_quantize(model_dir: str) -> pathlib.Path:
    model_dir = pathlib.Path(model_dir)
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    fp32 = onnx_dir / "model.onnx"
    export_fp32(model_dir, fp32)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    int8 = onnx_dir / "model_int8.onnx"

    # strategy 1: per-channel int8 (standard; per-tensor destroyed logits by 0.51)
    quantize_dynamic(
        model_input=str(fp32),
        model_output=str(int8),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    diff = _max_prob_diff(fp32, int8, tokenizer)

    if diff > 0.10:
        print(f"per-channel parity diff {diff:.4f} > 0.10 - retrying with classifier excluded", flush=True)
        exclude = _classifier_gemm_nodes(fp32)
        if exclude:
            quantize_dynamic(
                model_input=str(fp32),
                model_output=str(int8),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
                nodes_to_exclude=exclude,
            )
            diff = _max_prob_diff(fp32, int8, tokenizer)

    print(f"int8 parity (max prob diff vs fp32): {diff:.4f}", flush=True)
    if diff > 0.10:
        print("WARNING: int8 parity above 0.10 - shipping the fp32 model instead")
        shutil.copy2(fp32, int8)

    print(
        f"fp32: {fp32.stat().st_size / 1e6:.1f} MB, "
        f"int8: {int8.stat().st_size / 1e6:.1f} MB"
    )
    return int8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    args = parser.parse_args()
    export_and_quantize(args.model_dir)


if __name__ == "__main__":
    main()
