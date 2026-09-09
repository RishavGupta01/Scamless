import pandas as pd
import torch
import transformers

from scamless.data.spans import align_spans, build_span_masks, load_spans
from scamless.labels import TACTIC_TAGS
from scamless.model.config import TrainConfig
from scamless.model.train import build_dataset, expand_classifier_head, train_model


def _tok():
    return transformers.AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")


def _tiny_df(n_pos=8, n_neg=8):
    rows = []
    for i in range(n_pos):
        rows.append({"text": f"claim your free prize winner number {i}", "labels": ["generic_spam"]})
    for i in range(n_neg):
        rows.append({"text": f"regular business sentence about invoice {i}", "labels": []})
    return pd.DataFrame(rows)


def test_align_spans_maps_characters_to_tokens():
    tok = _tok()
    text = "Act now! This is urgent."
    span = [{"start": 18, "end": 24, "tag": "urgency"}]  # the word "urgent"
    mask = align_spans(text, span, tok, max_len=16)
    hits = [i for i, v in enumerate(mask) if v == TACTIC_TAGS.index("urgency")]
    assert hits, "expected at least one token tagged urgency"
    assert all(v == -100 for i, v in enumerate(mask) if i not in hits)


def test_build_span_masks_defaults_to_ignore():
    tok = _tok()
    recs = [{"text": "Act now! This is urgent.", "spans": [{"start": 0, "end": 3, "tag": "urgency"}]}]
    masks = build_span_masks(["Act now! This is urgent.", "no annotations here"], recs, tok, 16)
    assert len(masks) == 2
    assert (masks[1] == -100).all()  # unannotated text: fully ignored


def test_load_spans_skips_blank_and_malformed(tmp_path):
    p = tmp_path / "spans.jsonl"
    p.write_text(
        '{"text": "a", "spans": [{"start": 0, "end": 1, "tag": "fear"}]}\n'
        "\n"
        '{"no_spans_field": true}\n',
        encoding="utf-8",
    )
    recs = load_spans(str(p))
    assert len(recs) == 1
    assert recs[0]["spans"][0]["tag"] == "fear"


def test_dataset_with_spans_file(tmp_path):
    p = tmp_path / "spans.jsonl"
    matched_text = "claim your free prize winner number 0"
    span_line = (
        '{"text": "claim your free prize winner number 0", '
        '"spans": [{"start": 6, "end": 10, "tag": "urgency"}]}'
    )
    p.write_text(span_line, encoding="utf-8")
    cfg = TrainConfig(
        backbone="prajjwal1/bert-tiny", max_len=32, epochs=1, batch_size=4,
        seed=1, spans_file=str(p),
    )
    tok = _tok()
    df = pd.DataFrame(
        [
            {"text": matched_text, "labels": ["generic_spam"]},
            {"text": "normal note about taxes", "labels": []},
        ]
    )
    ds = build_dataset(df, tok, cfg, adversarial_rate=0.0)
    assert ds.has_spans()
    item = ds[0]
    assert "spans" in item
    assert item["spans"].shape == (32,)
    assert (item["spans"] == -100).any()  # unannotated tokens ignored
    assert (item["spans"] != -100).any()  # annotated tokens present


def test_expand_classifier_head_preserves_old_rows():
    old = torch.nn.Linear(8, 5)
    model = torch.nn.Module()
    model.classifier = old

    class Holder:
        pass

    h = Holder()
    h.classifier = old
    new_weight_before = old.weight.data.clone()
    new_bias_before = old.bias.data.clone()

    expand_classifier_head(h, 5, 8)
    assert h.classifier.weight.shape == (8, 8)
    assert h.classifier.bias.shape == (8,)
    assert torch.equal(h.classifier.weight.data[:5], new_weight_before)
    assert torch.equal(h.classifier.bias.data[:5], new_bias_before)
    assert torch.all(h.classifier.weight.data[5:] == 0)


def test_train_with_init_from_expands_head(tmp_path):
    # step 1: train a 15-label model on the tiny df
    cfg1 = TrainConfig(
        backbone="prajjwal1/bert-tiny", max_len=32, epochs=1, batch_size=8,
        lr=5e-5, seed=5, output_dir=str(tmp_path / "v1"),
    )
    train_model(_tiny_df(), _tiny_df(), cfg1)
    assert (tmp_path / "v1" / "config.json").exists()

    # step 2: append a label to the schema, resume from v1
    import scamless.labels as L

    L.SCAM_LABELS.append("_test_new_label")
    L._INDEX["_test_new_label"] = len(L.SCAM_LABELS) - 1
    L.NUM_SCAM_LABELS = len(L.SCAM_LABELS)
    try:
        cfg2 = TrainConfig(
            backbone="prajjwal1/bert-tiny", max_len=32, epochs=1, batch_size=8,
            lr=5e-5, seed=6, output_dir=str(tmp_path / "v2"),
            init_from=str(tmp_path / "v1"),
        )
        model, _, metrics = train_model(_tiny_df(), _tiny_df(), cfg2)
        assert model.classifier.weight.shape[0] == L.NUM_SCAM_LABELS
        assert metrics["val_macro_f1"] >= 0.0
    finally:
        L.SCAM_LABELS.pop()
        L._INDEX.pop("_test_new_label", None)
        L.NUM_SCAM_LABELS = len(L.SCAM_LABELS)
