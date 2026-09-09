"""Export the fine-tuned model to ONNX, then apply dynamic int8 quantization.

Exports via torch.onnx.export directly (the optimum exporter path is
incompatible with torch >= 2.10's dynamo-first ONNX API). A consolidation
step embeds external weights if any were produced, so the artifact is always
one self-contained .onnx file.

Usage: python -m scamless.model.export_onnx --model-dir artifacts/model_v1
"""

import argparse
import pathlib

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


def export_and_quantize(model_dir: str) -> pathlib.Path:
    model_dir = pathlib.Path(model_dir)
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    fp32 = onnx_dir / "model.onnx"
    export_fp32(model_dir, fp32)

    int8 = onnx_dir / "model_int8.onnx"
    quantize_dynamic(
        model_input=str(fp32),
        model_output=str(int8),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=False,
    )
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
