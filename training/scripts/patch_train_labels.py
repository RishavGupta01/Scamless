"""One-off patch: route train.py schema reads through the live labels module."""

import ast
import pathlib

p = pathlib.Path("src/scamless/model/train.py")
src = p.read_text(encoding="utf-8")

src = src.replace(
    "vectors.append(labels_to_vector(labels))",
    "vectors.append(labels_schema.labels_to_vector(labels))",
)
src = src.replace(
    "expand_classifier_head(model, old_cfg.num_labels, NUM_SCAM_LABELS)",
    "expand_classifier_head(model, old_cfg.num_labels, labels_schema.NUM_SCAM_LABELS)",
)
src = src.replace(
    'f"{old_cfg.num_labels} -> {NUM_SCAM_LABELS} labels"',
    'f"{old_cfg.num_labels} -> {labels_schema.NUM_SCAM_LABELS} labels"',
)
src = src.replace(
    "num_labels=NUM_SCAM_LABELS,",
    "num_labels=labels_schema.NUM_SCAM_LABELS,",
)
src = src.replace(
    'vecs = np.array([labels_to_vector(list(r["labels"])) for _, r in train_df.iterrows()])',
    'vecs = np.array([labels_schema.labels_to_vector(list(r["labels"])) for _, r in train_df.iterrows()])',
)
src = src.replace(
    "num_tactic_tags=len(TACTIC_TAGS),",
    "num_tactic_tags=len(labels_schema.TACTIC_TAGS),",
)
src = src.replace(
    'truth = np.array([labels_to_vector(list(r["labels"])) for _, r in val_df.iterrows()])',
    'truth = np.array([labels_schema.labels_to_vector(list(r["labels"])) for _, r in val_df.iterrows()])',
)

p.write_text(src, encoding="utf-8")
ast.parse(src)
print("patched, syntax ok")
