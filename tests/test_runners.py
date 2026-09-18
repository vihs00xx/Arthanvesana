import importlib.util
import json
from pathlib import Path

import pytest

from arthanvesana.data.parse import to_tidy

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_robustness_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_robustness")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i < 6 else "B",
            "direction": "L/R", "complete": i % 3 != 0,
            "symbols": ["001", f"{i + 100:03d}", "003"],
        }
        for i in range(12)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output),
        "--repeats", "2", "--min-site-inscriptions", "2",
    ])
    saved = json.loads((output / "robustness_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["manifest"]["seeds"] == [0, 1]
    assert saved["manifest"]["corpus_sha256"] == runner.sha256_file(corpus)
    assert len(saved["manifest"]["source_sha256"]) > 1
    for protocol in saved["repeated"].values():
        assert protocol["aggregate"]["n_ok"] == 2
        for run in protocol["runs"]:
            assert run["overlap_diagnostics"]["shared_split_groups"] == 0
    assert saved["held_out_sites"]["eligible_sites"] == ["A", "B"]
    report = (output / "robustness_report.txt").read_text(encoding="utf-8")
    assert "NOT confidence intervals" in report
    assert "context_vs_position" in report


def test_transition_probabilities_include_unplotted_destinations():
    runner = load_runner("run_stats")
    matrix = runner.transition_probabilities([["a", "b"], ["a", "c"]], ["a", "b"])
    assert matrix == [[0.0, 0.5], [0.0, 0.0]]


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("name", ["run_stats", "run_replication", "run_upgrade"])
def test_runner_synthetic_smoke(name, tmp_path, monkeypatch):
    runner = load_runner(name)
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i < 20 else "B",
            "direction": "L/R", "complete": True,
            "symbols": ["001", "002", f"{i + 100:03d}", "003", "004"] * 2,
        }
        for i in range(40)
    ]
    raw[0]["symbols"][3] = "000"
    raw[1]["complete"] = False
    raw[2]["direction"] = "OTHER"
    frame = to_tidy(raw)
    monkeypatch.setattr(runner.pd, "read_csv", lambda *args, **kwargs: frame)
    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "FIG", tmp_path / "figures")
    if hasattr(runner, "site_records"):
        original = runner.site_records
        monkeypatch.setattr(runner, "site_records", lambda df, **kwargs: original(df, min_inscriptions=1, **kwargs))
    runner.main()
    summary = json.loads((tmp_path / f"{name[4:]}_summary.json").read_text(encoding="utf-8"))
    assert summary["split"]["track"] == "artifact"
    assert summary["split"]["overlap"]["artifact"] == 0
    assert summary["split"]["overlap"]["record"] == 0
    assert summary["split"]["overlap"]["sequence"] == 0
    if name == "run_upgrade":
        from arthanvesana.replicate.assoc import analyze_pairs

        records = runner.analysis_records(frame, gap_policy="split", known_direction_only=True)
        expected = analyze_pairs([r["sequence"] for r in records], min_count=3)
        inference = summary["association_inference"]
        assert inference["n_tested"] == expected["n_observed"] == len(inference["pairs"])
        assert [r["p_adj"] for r in inference["pairs"]] == [r["p_adj"] for r in expected["pairs"]]
        assert any(r["count"] < 3 for r in inference["pairs"])
        assert "observed ordered adjacent pairs" in inference["family"]
        assert "dependent" in inference["caution"]
        report = (tmp_path / "upgrade_report.md").read_text(encoding="utf-8")
        assert "p<0.001" not in report
        assert "LLR-significant" not in report
        assert "exploratory q" in report
    elif name == "run_replication":
        report = (tmp_path / "replication_report.md").read_text(encoding="utf-8")
        assert "Merge counts are NOT tree heights" in report
