import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoTokenizer

from scamless.model.config import TrainConfig
from scamless.model.export_onnx import export_and_quantize
from scamless.model.train import train_model


def _tiny_df():
    rows = []
    for i in range(8):
        rows.append({"text": f"claim your free prize winner {i}", "labels": ["generic_spam"]})
        rows.append({"text": f"normal office note about taxes {i}", "labels": []})
    return pd.DataFrame(rows)


def test_exported_quantized_model_matches_pytorch_logits():
    cfg = TrainConfig(
        backbone="prajjwal1/bert-tiny",
        max_len=64,
        epochs=1,
        batch_size=8,
        lr=5e-5,
        seed=7,
        output_dir="artifacts/test_export",
    )
    model, tok, _ = train_model(_tiny_df(), _tiny_df(), cfg)
    model.save_pretrained(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)

    onnx_path = export_and_quantize(cfg.output_dir)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    text = "free prize claim now"
    enc = tok(
        [text], return_tensors="np", truncation=True, max_length=cfg.max_len, padding="max_length"
    )
    onnx_logits = sess.run(
        None,
        {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )[0]

    torch_enc = tok(
        [text], return_tensors="pt", truncation=True, max_length=cfg.max_len, padding="max_length"
    )
    model.eval()
    with torch.no_grad():
        torch_logits = model(**torch_enc).logits.numpy()

    # int8 dynamic quantization stays close; tolerate 0.15 logit drift
    assert np.abs(onnx_logits - torch_logits).max() < 0.15
