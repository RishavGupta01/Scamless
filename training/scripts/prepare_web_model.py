"""Stage the exported model artifact for web hosting (and optionally upload).

Creates publish/web-model/ with exactly the file layout transformers.js
expects, plus the calibrated thresholds.json. Optionally uploads to the
Hugging Face Hub when --upload is given and HF_TOKEN is set.

Usage:
  python training/scripts/prepare_web_model.py --model-dir artifacts/model_v1
  python training/scripts/prepare_web_model.py --model-dir artifacts/model_v1 --upload USER/scamless-model-v1
"""

import argparse
import pathlib
import shutil

NEEDED = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]


def stage(model_dir: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "onnx").mkdir(exist_ok=True)

    missing = [f for f in NEEDED if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"artifact incomplete, missing: {missing}")

    for f in NEEDED:
        shutil.copy2(model_dir / f, out / f)

    int8 = model_dir / "onnx" / "model_int8.onnx"
    if not int8.exists():
        raise FileNotFoundError(f"{int8} not found - run the export step first")
    shutil.copy2(int8, out / "onnx" / "model_int8.onnx")

    thresholds = model_dir / "onnx" / "thresholds.json"
    if thresholds.exists():
        shutil.copy2(thresholds, out / "thresholds.json")
    else:
        print("warning: thresholds.json not found - the app will fall back to 0.5 per label")

    print(f"staged web model at {out}")


def upload(out: pathlib.Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model")
    print(f"uploaded to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    parser.add_argument("--out", default="publish/web-model")
    parser.add_argument("--upload", default="", help="HF repo id, e.g. USER/scamless-model-v1")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    stage(pathlib.Path(args.model_dir), out)
    if args.upload:
        upload(out, args.upload)


if __name__ == "__main__":
    main()
