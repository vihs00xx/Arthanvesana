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


def _corpus(tmp_path, n=30):
    raw = [
        {
            "id": str(i), "cisi": str(i), "site": "A" if i % 2 else "B",
            "direction": "L/R", "complete": True,
            "symbols": [f"{200 + i:03d}", f"{100 + (i % 5):03d}", "003", "004", "005"],
        }
        for i in range(n)
    ]
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    return corpus


def test_crossfit_full_out_of_fold_coverage(tmp_path):
    runner = load_runner("run_ngram_inference")
    corpus = _corpus(tmp_path)
    out = tmp_path / "r"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(out),
        "--folds", "5", "--permutations", "200", "--bootstrap", "200",
    ])
    saved = json.loads((out / "ngram_inference_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    total_tokens = 30 * 5
    scored = sum(g["n_tokens"] for g in saved["groups"] if g["mean_paired_diff"] is not None)
    assert scored == total_tokens
    assert saved["n_groups"] == len(saved["groups"])
    assert saved["leakage_group_violations"] == 0
    p = saved["primary"]["randomization_p"]
    assert p >= 1.0 / (200 + 1)
    assert "ci95" in saved["primary"]
    assert "secondary_token_weighted" in saved
    report = (out / "ngram_inference_report.txt").read_text(encoding="utf-8")
    assert "never 0" in report
    assert "group" in report.lower()


def test_fold_assignment_deterministic():
    runner = load_runner("run_ngram_inference")
    groups = [
        {"group_id": f"g{k}", "indices": (k,), "n_tokens": (k % 4) + 1}
        for k in range(20)
    ]
    a1 = runner.assign_folds(groups, 5, seed=7)
    a2 = runner.assign_folds(groups, 5, seed=7)
    assert a1 == a2
    assert set(a1.values()) <= set(range(5))


def test_duplicate_tokens_in_group_not_multiplied(tmp_path):
    runner = load_runner("run_ngram_inference")
    raw = []
    # 10 genuinely distinct inscriptions (distinct sequences, distinct artifacts)
    for i in range(10):
        raw.append({
            "id": f"u{i}", "cisi": f"u{i}", "site": "A", "direction": "L/R",
            "complete": True, "symbols": [f"{300 + i:03d}", f"{400 + i:03d}", "003"],
        })
    # a duplicated pair sharing an exact sequence -> merged into one group
    raw.append({"id": "d1", "cisi": "d1", "site": "B", "direction": "L/R",
                "complete": True, "symbols": ["007", "008", "009"]})
    raw.append({"id": "d2", "cisi": "d2", "site": "B", "direction": "L/R",
                "complete": True, "symbols": ["007", "008", "009"]})
    corpus = tmp_path / "c.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    out = tmp_path / "r"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(out),
        "--folds", "2", "--permutations", "100", "--bootstrap", "100",
    ])
    # 10 distinct + 1 merged duplicate-group = 11 groups, not 12 records
    assert summary["n_groups"] == 11
    dup = [g for g in summary["groups"] if g["n_spans"] == 2]
    assert len(dup) == 1
    assert dup[0]["n_tokens"] == 6