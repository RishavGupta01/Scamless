"""Stage the exported model artifact for web hosting (and optionally upload).

Creates publish/web-model/ with exactly the file layout transformers.js
expects, plus the calibrated thresholds.json. Optionally uploads to the
Hugging Face Hub when --upload is given.

The upload namespace is resolved automatically: pass --upload scamless-model-v1
(just a name) and the repo is created under the token owner's account.

Usage:
  python training/scripts/prepare_web_model.py --model-dir artifacts/model_v1
  python training/scripts/prepare_web_model.py --model-dir artifacts/model_v1 --upload scamless-model-v1 --token hf_...
"""

import argparse
import pathlib
import shutil

# files the browser runtime requires
NEEDED = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]
# tokenizer format extras: present for WordPiece (vocab.txt) or SentencePiece
# (sentencepiece.bpe.model) tokenizers - staged only if they exist
OPTIONAL = ["vocab.txt", "sentencepiece.bpe.model", "tokenizer.model"]


def stage(model_dir: pathlib.Path, out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "onnx").mkdir(exist_ok=True)

    missing = [f for f in NEEDED if not (model_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"artifact incomplete, missing: {missing}")

    for f in NEEDED:
        shutil.copy2(model_dir / f, out / f)
    for f in OPTIONAL:
        if (model_dir / f).exists():
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


def upload(out: pathlib.Path, repo_name: str, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami(token=token)
    owner = who.get("name")
    if not owner:
        raise RuntimeError(
            "could not resolve your HF username from the token - "
            "make sure it is a Write token created under your account"
        )
    repo_id = f"{owner}/{repo_name}"
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model")
    print(f"uploaded to https://huggingface.co/{repo_id}")
    return repo_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    parser.add_argument("--out", default="publish/web-model")
    parser.add_argument("--upload", default="", help="repo name (created under your account) or full 'user/name' id")
    parser.add_argument("--token", default="", help="HF write token (or set HF_TOKEN env)")
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    stage(pathlib.Path(args.model_dir), out)
    if args.upload:
        token = args.token or os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError("provide --token or set HF_TOKEN to upload")
        repo_id = upload(out, args.upload, token)
        print(f"set MODEL_REPO in apps/web/js/config.js to: \"{repo_id}\"")


if __name__ == "__main__":
    import os

    main()
