"""One-command pipeline: build -> train -> export -> tune -> eval -> gate.

This is THE entry point for every future training run, including incremental
extensions (new languages, new scam categories). Adding data sources or
categories never changes this script: the builder self-describes via
report.json, the trainer reads whatever labels exist, the gate enforces
quality. A full refresh is: `python -m scamless.pipeline` (or the Colab cell).

Usage:
  python -m scamless.pipeline                 # full run, local paths
  python -m scamless.pipeline --skip-build    # data already built
"""

import argparse
import json
import pathlib
import subprocess
import sys

from scamless.data.download import RAW

PROCESSED = RAW.parent / "processed"
ARTIFACTS = pathlib.Path("artifacts")


def run(cmd: list[str]) -> None:
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-build", action="store_true", help="data already built")
    parser.add_argument("--skip-fetch", action="store_true", help="raw data already downloaded")
    parser.add_argument("--model-dir", default="artifacts/model_v1")
    args = parser.parse_args()

    if not args.skip_fetch:
        run([sys.executable, "-m", "scamless.data.fetch_all"])
    if not args.skip_build:
        run([sys.executable, "-m", "scamless.data.build"])

    report = json.loads((PROCESSED / "report.json").read_text())
    print(f"\nDataset: {json.dumps(report['label_counts'], indent=2)}")

    run([sys.executable, "-m", "scamless.model.train"])  # uses TrainConfig defaults

    run([sys.executable, "-m", "scamless.model.export_onnx", "--model-dir", args.model_dir])
    run([
        sys.executable, "-m", "scamless.eval.run_eval",
        "--mode", "onnx", "--model-dir", args.model_dir, "--tune",
    ])
    run([sys.executable, "-m", "scamless.eval.gate", "--metrics", "artifacts/eval/metrics.json"])

    metrics = json.loads((ARTIFACTS / "eval" / "metrics.json").read_text())
    print("\n================ PIPELINE COMPLETE ================")
    print(f"macro_f1:            {metrics['macro_f1']:.4f}")
    print(f"false_positive_rate: {metrics['false_positive_rate']:.4f}")
    print(f"artifacts:           {ARTIFACTS.resolve()}")
    print("(next: copy artifacts to Drive / release to production)")


if __name__ == "__main__":
    main()
