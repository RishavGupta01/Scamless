import pandas as pd
import torch

from scamless.model.config import TrainConfig
from scamless.model.train import build_dataset, train_model


def _tiny_df(n_pos=16, n_neg=16):
    rows = []
    for i in range(n_pos):
        rows.append({"text": f"claim your free prize winner number {i}", "labels": ["generic_spam"]})
    for i in range(n_neg):
        rows.append({"text": f"regular business sentence about invoice {i}", "labels": []})
    return pd.DataFrame(rows)


def test_train_smoke_runs_and_returns_metrics():
    cfg = TrainConfig(
        backbone="prajjwal1/bert-tiny",  # tiny English model: CPU smoke test only
        max_len=64,
        epochs=1,
        batch_size=8,
        lr=5e-5,
        seed=13,
        adversarial_rate=0.0,
        output_dir="artifacts/test_smoke",
    )
    model, _tok, metrics = train_model(_tiny_df(), _tiny_df(), cfg)
    assert metrics["val_macro_f1"] >= 0.0
    assert model is not None


def test_build_dataset_shapes():
    cfg = TrainConfig(
        backbone="prajjwal1/bert-tiny", max_len=64, epochs=1, batch_size=8, lr=5e-5, seed=1
    )
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(cfg.backbone)
    ds = build_dataset(_tiny_df(n_pos=4, n_neg=4), tok, cfg, adversarial_rate=0.0)
    item = ds[0]
    # dynamic padding: items are unpadded lists, collator produces tensors
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(item["labels"]) == 15
    assert len(item["input_ids"]) <= 64
    batch = ds.collate([ds[0], ds[1]])
    t = batch["input_ids"].shape[1]
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape == (2, t)
    # shorter item must be padded to the longer one
    assert t >= max(len(ds[0]["input_ids"]), len(ds[1]["input_ids"]))


def test_trained_model_checkpoint_roundtrip(tmp_path):
    cfg = TrainConfig(
        backbone="prajjwal1/bert-tiny",
        max_len=32,
        epochs=1,
        batch_size=8,
        lr=5e-5,
        seed=3,
        output_dir=str(tmp_path / "m"),
    )
    model, tok, _ = train_model(_tiny_df(n_pos=8, n_neg=8), _tiny_df(n_pos=4, n_neg=4), cfg)
    assert (tmp_path / "m" / "config.json").exists()
    # reload from disk must produce the same number of labels
    from transformers import AutoModelForSequenceClassification

    reloaded = AutoModelForSequenceClassification.from_pretrained(str(tmp_path / "m"))
    assert reloaded.config.num_labels == 15
    assert model is not None
    assert tok is not None


def test_dataset_without_spans_has_no_span_key():
    import transformers

    cfg = TrainConfig(backbone="prajjwal1/bert-tiny", max_len=32, epochs=1, batch_size=4, seed=1)
    tok = transformers.AutoTokenizer.from_pretrained(cfg.backbone)
    ds = build_dataset(_tiny_df(n_pos=2, n_neg=2), tok, cfg, adversarial_rate=0.0)
    assert not ds.has_spans()
    assert "spans" not in ds[0]
    assert torch.is_tensor(ds[0]["labels"])
