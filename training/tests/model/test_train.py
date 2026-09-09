import pandas as pd

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
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert len(item["labels"]) == 15
    assert item["input_ids"].shape == (64,)
