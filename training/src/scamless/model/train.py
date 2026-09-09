"""Fine-tune the multilingual MiniLM backbone with a 15-label sigmoid head.

Multi-label via BCEWithLogitsLoss. `bert-tiny` is only for the smoke test;
the default backbone in config.py is the shipped 118M multilingual MiniLM.
"""

import pathlib
import random

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from scamless.aug.adversarial import apply_all
from scamless.labels import NUM_SCAM_LABELS, labels_to_vector


class MultiLabelDataset(Dataset):
    def __init__(self, texts, label_vectors, tokenizer, max_len):
        self.enc = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding="max_length"
        )
        self.labels = label_vectors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.enc["input_ids"][idx]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


class WeightedTrainer(Trainer):
    """BCE loss with per-label pos_weight computed at setup."""

    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pos_weight = (
            self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.device), pos_weight=pos_weight
        )
        return (loss, outputs) if return_outputs else loss


def build_dataset(df, tokenizer, cfg, adversarial_rate: float) -> MultiLabelDataset:
    texts, vectors = [], []
    rng = random.Random(cfg.seed)
    for _, row in df.iterrows():
        text = str(row["text"])
        labels = list(row["labels"])
        if labels and adversarial_rate > 0 and rng.random() < adversarial_rate:
            text = apply_all(text, rng)
        texts.append(text)
        vectors.append(labels_to_vector(labels))
    return MultiLabelDataset(texts, vectors, tokenizer, cfg.max_len)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(train_df, val_df, cfg):
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone,
        num_labels=NUM_SCAM_LABELS,
        problem_type="multi_label_classification",
    )

    train_ds = build_dataset(train_df, tokenizer, cfg, cfg.adversarial_rate)
    val_ds = build_dataset(val_df, tokenizer, cfg, adversarial_rate=0.0)

    # pos_weight: inverse frequency per label, capped to keep loss stable
    vecs = np.array([labels_to_vector(list(r["labels"])) for _, r in train_df.iterrows()])
    pos = vecs.sum(axis=0)
    pos_weight = torch.tensor(
        np.clip((len(vecs) - pos) / np.maximum(pos, 1.0), 1.0, 10.0), dtype=torch.float
    )

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.lr,
        seed=cfg.seed,
        use_cpu=torch.cuda.is_available() is False,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, pos_weight=pos_weight
    )
    trainer.train()

    # persist weights BEFORE validation so a late-stage error never loses the run
    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # validation metrics with 0.5 threshold
    device = next(model.parameters()).device
    model.eval()
    logits = []
    eval_bs = cfg.batch_size * 4
    with torch.no_grad():
        for i in range(0, len(val_ds), eval_bs):
            batch = {
                k: torch.stack(
                    [val_ds[j][k] for j in range(i, min(i + eval_bs, len(val_ds)))]
                ).to(device)
                for k in ("input_ids", "attention_mask")
            }
            logits.append(model(**batch).logits)
        probs = torch.sigmoid(torch.cat(logits)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    truth = np.array([labels_to_vector(list(r["labels"])) for _, r in val_df.iterrows()])
    # macro-F1 over supported labels only (same definition as the eval harness)
    support = truth.sum(axis=0) > 0
    val_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))
    supported_f1 = float(np.mean(f1_score(truth, preds, average=None, zero_division=0)[support])) if support.any() else 0.0
    metrics = {"val_macro_f1": supported_f1, "val_macro_f1_all_labels": val_f1}
    return model, tokenizer, metrics


def main() -> None:
    import pathlib

    import pandas as pd

    from scamless.data.download import RAW
    from scamless.model.config import TrainConfig

    processed = RAW.parent / "processed"
    train_df = pd.read_parquet(processed / "messages_train.parquet")
    val_df = pd.read_parquet(processed / "messages_val.parquet")
    cfg = TrainConfig()
    model, tokenizer, metrics = train_model(train_df, val_df, cfg)
    out = pathlib.Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(metrics)


if __name__ == "__main__":
    main()
