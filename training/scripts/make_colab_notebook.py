"""Generate notebooks/train_colab.ipynb. Run once from repo root."""

import json
import pathlib


def code(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


def md(src, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": src.strip("\n").splitlines(keepends=True),
    }


CELL1 = """# Scamless - Training Run (GPU)

Runtime > Change runtime type > **T4 GPU**, then Run all.
Clones the repo, builds the dataset, fine-tunes, exports int8 ONNX, evals, gates, saves artifacts to Google Drive."""

CELL2 = """%cd /content
!git clone https://github.com/RishavGupta01/Scamless.git scamless
%cd /content/scamless
!pip install -q -e "training[dev]"
import sys
sys.path.insert(0, "/content/scamless/training/src")"""

CELL4 = """!nvidia-smi
import torch
print("CUDA available:", torch.cuda.is_available())"""

CELL5 = """from google.colab import drive
import pathlib

drive.mount("/content/drive")
DRIVE_ART = pathlib.Path("/content/drive/MyDrive/scamless/artifacts")
DRIVE_ART.mkdir(parents=True, exist_ok=True)
print("artifacts will persist to", DRIVE_ART)"""

CELL6 = """!python -m scamless.data.fetch_all
!python -m scamless.data.build"""

CELL7 = """import pathlib
import sys
sys.path.insert(0, "/content/scamless/training/src")

import pandas as pd

from scamless.model.config import TrainConfig
from scamless.model.train import train_model

processed = pathlib.Path("training/data/processed")
cfg = TrainConfig(output_dir="artifacts/model_v1")
train_df = pd.read_parquet(processed / "messages_train.parquet")
val_df = pd.read_parquet(processed / "messages_val.parquet")

model, tokenizer, metrics = train_model(train_df, val_df, cfg)
model.save_pretrained(cfg.output_dir)
tokenizer.save_pretrained(cfg.output_dir)
print(metrics)"""

CELL8 = """!python -m scamless.model.export_onnx --model-dir artifacts/model_v1"""

CELL9 = """import json
import sys
sys.path.insert(0, "/content/scamless/training/src")

import pandas as pd

from scamless.eval.baseline import heuristic_predict
from scamless.eval.harness import compute_metrics

# trained model metrics + release gate (gate exits nonzero on failure)
!python -m scamless.eval.run_eval --mode onnx --model-dir artifacts/model_v1
!python -m scamless.eval.gate --metrics artifacts/eval/metrics.json

# heuristic baseline for comparison (does NOT overwrite metrics.json)
test_df = pd.read_parquet("training/data/processed/messages_test.parquet")
base = compute_metrics(test_df, [heuristic_predict(str(t)) for t in test_df["text"]])
print("baseline macro_f1:", base["macro_f1"], "fp_rate:", base["false_positive_rate"])
print("trained:", json.loads(open("artifacts/eval/metrics.json").read()))"""

CELL10 = """import shutil

shutil.copytree("artifacts", DRIVE_ART / "model_v1_run", dirs_exist_ok=True)
print("saved to", DRIVE_ART / "model_v1_run")"""


def main() -> None:
    cells = [
        md(CELL1, "cell-intro"),
        code(CELL2, "cell-clone"),
        code(CELL4, "cell-gpu"),
        code(CELL5, "cell-drive"),
        code(CELL6, "cell-data"),
        code(CELL7, "cell-train"),
        code(CELL8, "cell-export"),
        code(CELL9, "cell-eval"),
        code(CELL10, "cell-save"),
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    out = pathlib.Path("notebooks")
    out.mkdir(exist_ok=True)
    (out / "train_colab.ipynb").write_text(
        json.dumps(nb, indent=1), encoding="utf-8"
    )
    print("notebook written with", len(cells), "cells")


if __name__ == "__main__":
    main()
