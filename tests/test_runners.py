import importlib.util
import json
from pathlib import Path

import pandas as pd
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


def test_transformer_runner_writes_reproducible_report(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    assert torch is not None
    runner = load_runner("run_transformer")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i < 6 else "B",
            "direction": "L/R", "complete": True,
            "symbols": ["001", f"{100 + (i % 2):03d}", "003"],
        }
        for i in range(12)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output),
        "--repeats", "1", "--max-epochs", "2", "--patience", "2",
    ])
    saved = json.loads((output / "transformer_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["n_ok"] == 1
    assert len(saved["per_seed"]) == 1
    run = saved["per_seed"][0]
    assert len(run["inner"]) == len(runner.GRID)
    assert run["selected_config"] in runner.GRID
    assert 1 <= run["selected_epochs"] <= 2
    assert run["transformer"]["n_masked"] == run["bigram"]["n_masked"]
    assert run["fit_vocab_size"] > 0
    report = (output / "transformer_report.txt").read_text(encoding="utf-8")
    assert "NOT confidence intervals" in report
    assert "nested" in report.lower()


def test_transformer_inner_selection_never_sees_outer_test(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    assert torch is not None
    runner = load_runner("run_transformer")
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i < 6 else "B",
            "direction": "L/R", "complete": True,
            "symbols": [f"{100 + (i % 4):03d}", "001", f"{200 + (i % 3):03d}"],
        }
        for i in range(16)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    seen = []
    original = runner._fit_valid_split

    def recording_split(train, seed):
        fit, valid = original(train, seed)
        seen.append((
            {r["inscription_id"] for r in fit},
            {r["inscription_id"] for r in valid},
        ))
        return fit, valid

    monkeypatch.setattr(runner, "_fit_valid_split", recording_split)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output),
        "--repeats", "1", "--max-epochs", "2", "--patience", "2",
    ])
    run = summary["per_seed"][0]
    test_ids = set(run["test_ids"])
    train_ids = set(run["train_ids"])
    assert seen
    for fit_ids, valid_ids in seen:
        assert fit_ids | valid_ids == train_ids
        assert not (fit_ids & test_ids)
        assert not (valid_ids & test_ids)


def test_stratified_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_stratified")
    raw = [
        {
            "id": str(i), "cisi": f"C-{i % 3}", "site": "A",
            "direction": "L/R", "complete": True,
            "symbols": ["001", f"{100 + (i % 4):03d}", "003"],
        }
        for i in range(24)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame([
        {"cisi": "C-0", "ext_id": "S0", "motif": "Bull1", "ext_direction": "R-L",
         "ext_line_count": 1, "ext_object_type": "unknown", "source": "test"},
        {"cisi": "C-1", "ext_id": "S1", "motif": "Gaur", "ext_direction": "R-L",
         "ext_line_count": 1, "ext_object_type": "unknown", "source": "test"},
        {"cisi": "C-2", "ext_id": "S2", "motif": "unknown", "ext_direction": "L-R",
         "ext_line_count": 1, "ext_object_type": "unknown", "source": "test"},
    ]).to_csv(metadata, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--metadata", str(metadata),
        "--output", str(output),
    ])
    saved = json.loads((output / "stratified_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["direction"]["agreement"] == 2 / 3
    assert set(summary["motif_groups"]) == {"bull", "other_known", "unknown"}
    report = (output / "stratified_report.txt").read_text(encoding="utf-8")
    assert "Motif groups" in report


def test_audit_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_audit")
    raw = [
        {"id": "a", "cisi": "C-1", "site": "A", "direction": "L/R",
         "complete": True, "symbols": ["001", "002"]},
        {"id": "b", "cisi": "C-2", "site": "A", "direction": "L/R",
         "complete": True, "symbols": ["003", "000"]},
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    external = tmp_path / "external.csv"
    pd.DataFrame([
        {"inscription_id": "S1", "cisi_number": "C-1", "sign_sequence": "G1 G2",
         "site": "A", "object_type": "u", "line_count": 1, "damaged": False,
         "reading_direction": "R-L", "motif": "u"},
        {"inscription_id": "S2", "cisi_number": "C-2", "sign_sequence": "G3",
         "site": "A", "object_type": "u", "line_count": 1, "damaged": False,
         "reading_direction": "R-L", "motif": "u"},
    ]).to_csv(external, index=False)
    mayig = tmp_path / "mayig.csv"
    pd.DataFrame([
        {"inscription_id": "M-1A", "sign_sequence": "P1 P2", "site": "s",
         "object_type": "seal", "line_count": 1, "damaged": False,
         "reading_direction": "L-R", "motif": "u", "mean_uncertainty": 0.0},
    ]).to_csv(mayig, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--external", str(external),
        "--mayig", str(mayig), "--output", str(output),
    ])
    saved = json.loads((output / "audit_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["family"]["n_compared"] == 2
    assert summary["family"]["exact_agreement"] == 0.5
    assert summary["family"]["mismatch_relations"] == {"exact": 1, "gap_placement_only": 1}
    assert summary["mayig"]["matched_inscriptions"] == 0
    report = (output / "audit_report.txt").read_text(encoding="utf-8")
    assert "Cross-corpus audit" in report


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
