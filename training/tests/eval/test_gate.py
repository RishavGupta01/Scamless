import json

from scamless.eval.gate import gate_fails, load_config


def test_gate_passes_when_metrics_meet_thresholds():
    metrics = {"macro_f1": 0.95, "false_positive_rate": 0.005}
    config = {"macro_f1_min": 0.9, "fp_rate_max": 0.01}
    assert gate_fails(metrics, config) == []


def test_gate_fails_lists_each_violation():
    metrics = {"macro_f1": 0.8, "false_positive_rate": 0.02}
    config = {"macro_f1_min": 0.9, "fp_rate_max": 0.01}
    failures = gate_fails(metrics, config)
    assert len(failures) == 2
    assert any("macro_f1" in f for f in failures)
    assert any("false_positive_rate" in f for f in failures)


def test_load_config_reads_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"macro_f1_min": 0.9, "fp_rate_max": 0.01}))
    assert load_config(p)["macro_f1_min"] == 0.9
