"""Fine-tune the multilingual MiniLM backbone with scam + span heads.

Architecture (frozen contract for all future phases):
- Shared multilingual encoder (MiniLM-L12)
- Head A: 15-label sigmoid classifier (multi-label message verdict)
- Head B: token-level span tagger for TACTIC_TAGS (Phase 4 data plugs in
  without touching this file; until then spans are absent and head B loss
  is masked to zero)

Both losses are computed in one backward pass, so a future run that has span
annotations trains both skills simultaneously with zero architecture changes.
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

from scamless import labels as labels_schema
from scamless.aug.adversarial import apply_all


class MultiLabelDataset(Dataset):
    def __init__(self, texts, label_vectors, tokenizer, max_len, span_masks=None):
        self.enc = tokenizer(
            list(texts), truncation=True, max_length=max_len, padding="max_length"
        )
        self.labels = label_vectors
        # span_masks[i][j] = -100 (ignore) or a TACTIC_TAGS index per token.
        # None means this example has no span annotations -> all ignored.
        self.span_masks = span_masks

    def __len__(self):
        return len(self.labels)

    def has_spans(self) -> bool:
        return self.span_masks is not None and any(
            any(t != -100 for t in row) for row in self.span_masks
        )

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.enc["input_ids"][idx]),
            "attention_mask": torch.tensor(self.enc["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }
        if self.span_masks is not None:
            item["spans"] = torch.tensor(self.span_masks[idx], dtype=torch.long)
        return item


class MultiTaskTrainer(Trainer):
    """BCE message loss + (optional, masked) token CE span loss."""

    def __init__(self, *args, pos_weight=None, span_weight=0.5, num_tactic_tags=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight
        self._span_weight = span_weight
        self._num_tactic_tags = num_tactic_tags

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        spans = inputs.pop("spans", None)

        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits.to(labels.device)
        pos_weight = (
            self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(logits.device), pos_weight=pos_weight
        )

        if spans is not None:
            hidden = outputs.hidden_states[-1]  # (B, T, H); requires output_hidden_states
            # project hidden states to per-tag logits with a lazily-created linear head
            if not hasattr(self, "_span_head"):
                hidden_size = hidden.size(-1)
                self._span_head = torch.nn.Linear(hidden_size, self._num_tactic_tags).to(
                    hidden.device
                )
            span_logits = self._span_head(hidden)  # (B, T, tags)
            active = spans != -100
            if active.any():
                span_loss = torch.nn.functional.cross_entropy(
                    span_logits[active],
                    spans.to(span_logits.device)[active],
                    ignore_index=-100,
                )
                loss = loss + self._span_weight * span_loss

        return (loss, outputs) if return_outputs else loss


def build_dataset(df, tokenizer, cfg, adversarial_rate: float) -> MultiLabelDataset:
    texts, vectors, augmented = [], [], []
    rng = random.Random(cfg.seed)
    for _, row in df.iterrows():
        text = str(row["text"])
        labels = list(row["labels"])
        is_aug = bool(labels) and adversarial_rate > 0 and rng.random() < adversarial_rate
        if is_aug:
            text = apply_all(text, rng)
        texts.append(text)
        augmented.append(is_aug)
        vectors.append(labels_schema.labels_to_vector(labels))

    span_masks = None
    if getattr(cfg, "spans_file", ""):
        from scamless.data.spans import build_span_masks, load_spans

        span_records = load_spans(cfg.spans_file)
        # spans align to original text only; augmented rows lose their spans
        # (annotating warped text is invalid - offsets no longer match)
        unaugmented_texts = [
            t if not is_aug else "" for t, is_aug in zip(texts, augmented)
        ]
        span_masks = build_span_masks(unaugmented_texts, span_records, tokenizer, cfg.max_len)
        print(f"span annotations loaded from {cfg.spans_file}")

    return MultiLabelDataset(texts, vectors, tokenizer, cfg.max_len, span_masks=span_masks)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def expand_classifier_head(model, old_num_labels: int, new_num_labels: int) -> None:
    """Grow the classifier head when labels were appended (append-only contract).

    Old rows are preserved exactly; new rows start near-zero so old knowledge
    is untouched until the new labels get training signal.
    """
    if old_num_labels >= new_num_labels:
        return
    weight = model.classifier.weight.data
    bias = model.classifier.bias.data
    hidden = weight.size(1)
    new_weight = torch.zeros(new_num_labels, hidden, dtype=weight.dtype)
    new_bias = torch.zeros(new_num_labels, dtype=bias.dtype)
    new_weight[:old_num_labels] = weight
    new_bias[:old_num_labels] = bias
    model.classifier.weight = torch.nn.Parameter(new_weight)
    model.classifier.bias = torch.nn.Parameter(new_bias)


def load_model(cfg):
    """Load the base backbone, or a previous checkpoint with head expansion.

    The checkpoint is loaded with its ORIGINAL head width (preserving all old
    label rows), then the head is grown to the current schema width. Loading
    with ignore_mismatched_sizes would silently reinitialize the classifier
    and destroy previous label knowledge.
    """
    from transformers import AutoConfig

    if cfg.init_from:
        old_cfg = AutoConfig.from_pretrained(cfg.init_from)
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.init_from,
            num_labels=old_cfg.num_labels,
            problem_type="multi_label_classification",
            output_hidden_states=True,
        )
        expand_classifier_head(model, old_cfg.num_labels, labels_schema.NUM_SCAM_LABELS)
        print(
            f"resumed from {cfg.init_from}: classifier head "
            f"{old_cfg.num_labels} -> {labels_schema.NUM_SCAM_LABELS} labels"
        )
        return model
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.backbone,
        num_labels=labels_schema.NUM_SCAM_LABELS,
        problem_type="multi_label_classification",
        output_hidden_states=True,
    )


def train_model(train_df, val_df, cfg):
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    model = load_model(cfg)

    train_ds = build_dataset(train_df, tokenizer, cfg, cfg.adversarial_rate)
    val_ds = build_dataset(val_df, tokenizer, cfg, adversarial_rate=0.0)
    print(f"train examples: {len(train_ds)}, val examples: {len(val_ds)}", flush=True)

    # pos_weight: inverse frequency per label, capped to keep loss stable
    vecs = np.array([labels_schema.labels_to_vector(list(r["labels"])) for _, r in train_df.iterrows()])
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
        remove_unused_columns=False,
    )

    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        pos_weight=pos_weight,
        num_tactic_tags=len(labels_schema.TACTIC_TAGS),
    )
    trainer.train()

    # persist weights BEFORE validation so a late-stage error never loses the run
    out_dir = pathlib.Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    if hasattr(trainer, "_span_head"):
        torch.save(trainer._span_head.state_dict(), out_dir / "span_head.pt")

    # validation metrics with 0.5 threshold
    device = next(model.parameters()).device
    model.eval()
    logits = []
    eval_bs = cfg.batch_size * 4
    print(f"validating on {len(val_ds)} examples (device: {device})...", flush=True)
    with torch.no_grad():
        for i in range(0, len(val_ds), eval_bs):
            batch = {
                k: torch.stack(
                    [val_ds[j][k] for j in range(i, min(i + eval_bs, len(val_ds)))]
                ).to(device)
                for k in ("input_ids", "attention_mask")
            }
            logits.append(model(**batch).logits)
            if (i // eval_bs) % 10 == 0:
                print(f"  validation {min(i + eval_bs, len(val_ds))}/{len(val_ds)}", flush=True)
        probs = torch.sigmoid(torch.cat(logits)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    truth = np.array([labels_schema.labels_to_vector(list(r["labels"])) for _, r in val_df.iterrows()])
    # macro-F1 over supported labels only (same definition as the eval harness)
    support = truth.sum(axis=0) > 0
    val_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))
    supported_f1 = (
        float(np.mean(f1_score(truth, preds, average=None, zero_division=0)[support]))
        if support.any()
        else 0.0
    )
    metrics = {"val_macro_f1": supported_f1, "val_macro_f1_all_labels": val_f1}
    return model, tokenizer, metrics


def main() -> None:
    import pandas as pd

    from scamless.data.download import RAW
    from scamless.model.config import TrainConfig

    processed = RAW.parent / "processed"
    train_df = pd.read_parquet(processed / "messages_train.parquet")
    val_df = pd.read_parquet(processed / "messages_val.parquet")
    cfg = TrainConfig()
    _model, _tokenizer, metrics = train_model(train_df, val_df, cfg)
    out = pathlib.Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(metrics)


if __name__ == "__main__":
    main()
