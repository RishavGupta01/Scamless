"""Release gate: exits nonzero when metrics violate thresholds.

Usage: python -m scamless.eval.gate --metrics artifacts/eval/metrics.json
"""

import argparse
import json
import pathlib
import sys


def load_config(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def gate_fails(metrics: dict, config: dict) -> list[str]:
    failures = []
    if metrics["macro_f1"] < config["macro_f1_min"]:
        failures.append(
            f"macro_f1 {metrics['macro_f1']:.3f} < required {config['macro_f1_min']}"
        )
    if metrics["false_positive_rate"] > config["fp_rate_max"]:
        failures.append(
            f"false_positive_rate {metrics['false_positive_rate']:.4f} > "
            f"allowed {config['fp_rate_max']}"
        )
    union_min = config.get("scam_union_f1_min", 0)
    union_f1 = metrics.get("scam_union", {}).get("f1", 0.0)
    if union_f1 < union_min:
        failures.append(f"scam_union_f1 {union_f1:.3f} < required {union_min}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="artifacts/eval/metrics.json")
    parser.add_argument(
        "--config", default=str(pathlib.Path(__file__).parent / "eval_config.json")
    )
    args = parser.parse_args()

    metrics = json.loads(pathlib.Path(args.metrics).read_text())
    config = load_config(pathlib.Path(args.config))
    failures = gate_fails(metrics, config)
    for f in failures:
        print(f"GATE FAIL: {f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
