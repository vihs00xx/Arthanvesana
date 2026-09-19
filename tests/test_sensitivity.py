import importlib.util
import json
from pathlib import Path

from arthanvesana.data.parse import to_tidy

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sensitivity_runner_writes_reproducible_report(tmp_path):
    runner = load_runner("run_sensitivity")
    raw = []
    # 12 L/R inscriptions, half complete; unique sequences keep split groups atomic.
    for i in range(12):
        raw.append({
            "id": str(i), "cisi": str(i), "site": "A" if i < 6 else "B",
            "direction": "L/R", "complete": i % 2 == 0,
            "symbols": ["001", f"{100 + i:03d}", "003"],
        })
    # 2 R/L and 2 OTHER inscriptions exercise the direction factor cells.
    for j, direction in enumerate(("R/L", "R/L", "OTHER", "OTHER")):
        raw.append({
            "id": f"x{j}", "cisi": f"x{j}", "site": "B",
            "direction": direction, "complete": j % 2 == 0,
            "symbols": ["002", f"{200 + j:03d}", "004"],
        })
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    output = tmp_path / "results"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(output), "--repeats", "1",
    ])
    saved = json.loads((output / "sensitivity_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert saved["manifest"]["seeds"] == [0]
    assert saved["manifest"]["gap_policy"] == "split"
    assert len(saved["manifest"]["matrix"]) == 8
    assert saved["manifest"]["corpus_sha256"] == runner.sha256_file(corpus)

    cells = saved["cells"]
    assert len(cells) == 8
    factor_keys = {
        (c["factors"]["transcription"], c["factors"]["direction"], c["factors"]["completeness"])
        for c in cells
    }
    assert len(factor_keys) == 8
    for cell in cells:
        assert cell["status"] in {"ok", "skipped"}
        assert cell["n_records"] > 0
        for run in cell["runs"]:
            assert run["status"] in {"ok", "skipped"}
            if run["status"] != "ok":
                assert run["reason"]
            assert run["split"]["track"] == "artifact"
        assert cell["aggregate"]["n_runs"] == 1

    # The synthetic corpus is large enough that every cell splits successfully.
    assert all(cell["status"] == "ok" for cell in cells)
    for cell in cells:
        agg = cell["aggregate"]
        assert agg["n_ok"] == 1
        for model in ("context", "frequency", "position"):
            assert 0.0 <= agg["models"][model]["top_1"]["mean"] <= 1.0
        for comparison in ("context_vs_frequency", "context_vs_position"):
            delta = agg["paired_deltas"][comparison]["top_1"]
            assert delta["n_runs"] == 1

    # Direction and completeness factors actually change the record counts.
    counts = {cell["label"]: cell["n_records"] for cell in cells}
    assert counts["normalized | known | all"] == 14
    assert counts["normalized | any | all"] == 16
    assert counts["normalized | known | complete-only"] == 7

    report = (output / "sensitivity_report.txt").read_text(encoding="utf-8")
    assert "SD describes split variability, NOT confidence intervals." in report
    assert "not verified archaeological restorations" in report
    assert "headline claim" in report
