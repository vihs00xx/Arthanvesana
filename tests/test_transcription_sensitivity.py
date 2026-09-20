import importlib.util
import json
from pathlib import Path

import pandas as pd

from arthanvesana.data.parse import to_tidy

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transcription_sensitivity_runner(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    # 6 distinct primary inscriptions with unique CISI values; external file has
    # 5 exact matches + 1 edge-difference + 1 duplicate-CISI exclusion.
    raw = []
    for i in range(7):
        raw.append({
            "id": str(i), "cisi": f"C-{i}", "site": "A" if i % 2 else "B",
            "direction": "L/R", "complete": True,
            "symbols": [f"{100 + i:03d}", "001", f"{200 + i:03d}"],
        })
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    ext_rows = []
    for i in range(5):
        # external R-L maps to our L/R label (0.992 cross-check in
        # run_stratified), so stored order == primary reading order
        ext_rows.append({
            "inscription_id": f"S{i}", "cisi_number": f"C-{i}",
            "sign_sequence": f"G{100 + i} G1 G{200 + i}",
            "site": "A", "object_type": "u", "line_count": 1,
            "damaged": False, "reading_direction": "R-L", "motif": "u",
        })
    # exact match for C-5, then a duplicate external row for the same CISI
    ext_rows.append({
        "inscription_id": "S5", "cisi_number": "C-5",
        "sign_sequence": "G105 G1 G205", "site": "B", "object_type": "u",
        "line_count": 1, "damaged": False, "reading_direction": "R-L",
        "motif": "u",
    })
    ext_rows.append({
        "inscription_id": "S-dup", "cisi_number": "C-5",
        "sign_sequence": "G1 G205", "site": "B", "object_type": "u",
        "line_count": 1, "damaged": False, "reading_direction": "R-L",
        "motif": "u",
    })
    # first-edge difference on its own CISI (external keeps only our last sign)
    ext_rows.append({
        "inscription_id": "S6", "cisi_number": "C-6",
        "sign_sequence": "G206", "site": "B", "object_type": "u",
        "line_count": 1, "damaged": False, "reading_direction": "R-L",
        "motif": "u",
    })
    external = tmp_path / "external.csv"
    pd.DataFrame(ext_rows).to_csv(external, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--external", str(external),
        "--output", str(output), "--repeats", "1",
    ])
    saved = json.loads(
        (output / "transcription_sensitivity_summary.json").read_text(encoding="utf-8")
    )
    assert saved == summary
    # 6 matched CISI (C-0..C-4 exact + C-6 first-edge); C-5 excluded (duplicate)
    assert summary["matched_cisi"] == 6
    assert summary["exclusions"]["duplicate"] == 1
    assert summary["exclusions"]["missing"] == 0
    # both transcriptions must be present
    assert summary["primary"]["n_runs"] == 1
    assert summary["external"]["n_runs"] == 1
    # aligned test sets: the external and primary use the same test ids
    assert summary["runs"][0]["primary_ok"] and summary["runs"][0]["external_ok"]
    # relation breakdown includes exact and first-edge labels
    labels = set(summary["by_relation"])
    assert "exact" in labels and "first_edge" in labels
    report = (output / "transcription_sensitivity_report.txt").read_text(encoding="utf-8")
    assert "NOT independent" in report
    assert "does contextual information outperform" in report


def test_relation_classification():
    runner = load_runner("run_transcription_sensitivity")
    assert runner.relation(["1", "2", "3"], ["1", "2", "3"]) == "exact"
    assert runner.relation(["1", "2", "3"], ["1", "3"]) == "other"
    assert runner.relation(["1", "2", "3"], ["3"]) == "first_edge"
    assert runner.relation(["1", "2", "3"], ["1"]) == "last_edge"


def test_missing_external_file_reports_dependency(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    raw = [{
        "id": "0", "cisi": "C-0", "site": "A", "direction": "L/R",
        "complete": True, "symbols": ["001", "002"],
    }]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    try:
        runner.main([
            "--corpus", str(corpus),
            "--external", str(tmp_path / "nope.csv"),
            "--output", str(tmp_path / "out"),
        ])
    except FileNotFoundError as exc:
        assert "indus_website_real_corpus.csv" in str(exc)
    else:
        raise AssertionError("missing external file must raise FileNotFoundError")

