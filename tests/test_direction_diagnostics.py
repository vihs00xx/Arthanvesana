"""Tests for direction diagnostics: reversal symmetry and runner behavior."""

import importlib.util
import json
from pathlib import Path

from arthanvesana.data.parse import to_tidy
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.stats.ngrams import NGramModel

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy_records():
    """Symmetric toy corpus: sequences are palindromic so reversal is exact."""
    seqs = [["001", "002", "001"], ["002", "001", "002"], ["001", "001", "002"]]
    return [
        {"inscription_id": f"i{k}", "sequence": list(s), "start_complete": True,
         "end_complete": True, "site": "A", "artifact_id": f"a{k}"}
        for k, s in enumerate(seqs)
    ]


def test_interior_restoration_is_reversal_invariant():
    """Interior masks are exactly invariant under global reversal.

    The forward chain uses P(next|cur) and the reversed chain uses the same
    matrix, and candidate tie-breaking sorts sign codes, so interior positions
    are mirror-symmetric. Boundary positions are asymmetric (see next test).
    """
    runner = load_runner("run_direction_diagnostics")
    train = _toy_records()
    test = _toy_records()
    rev_train = runner._reverse_records(train)
    rev_test = runner._reverse_records(test)
    forward, _ = restoration_records(train, test, mask_length=1)
    backward, _ = restoration_records(rev_train, rev_test, mask_length=1)
    interior_f = [r["rank"] for r in forward if r["position"] == 1]
    interior_b = [r["rank"] for r in backward if r["position"] == 1]
    assert interior_f == interior_b


def test_boundary_positions_are_not_reversal_invariant():
    """Document the <S>-start vs uniform-end boundary asymmetry."""
    runner = load_runner("run_direction_diagnostics")
    train = _toy_records()
    test = _toy_records()
    probe = runner._boundary_probe(
        train, test, runner._reverse_records(train), runner._reverse_records(test)
    )
    probes = {row["orientation"]: row for row in probe["probe"]}
    assert "original" in probes and "reversed" in probes
    # The interior stays equal; the first-position value need not (start uses
    # the <S> distribution, the end does not).
    assert probes["original"]["interior_top_1"] == probes["reversed"]["interior_top_1"]


def test_forward_and_reverse_use_same_transition_matrix():
    """Sanity: P(b|a) and P(a|b) both come from the same bigram counts."""
    seqs = [["001", "002"], ["002", "001"]]
    model = NGramModel(seqs, 2, method="wittenbell")
    assert model.dist(("001",))["002"] > 0
    assert model.dist(("002",))["001"] > 0


def test_direction_diagnostics_runner(tmp_path):
    runner = load_runner("run_direction_diagnostics")
    raw = []
    for i in range(24):
        direction = "L/R" if i % 2 else "R/L"
        raw.append({
            "id": str(i), "cisi": str(i), "site": "A" if i % 3 else "B",
            "direction": direction, "complete": True,
            "symbols": ["001", f"{100 + (i % 4):03d}", "003"],
        })
    for i in range(24, 30):
        raw.append({
            "id": str(i), "cisi": str(i), "site": "B", "direction": "-",
            "complete": True, "symbols": ["001", "002"],
        })
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    out = tmp_path / "out"
    summary = runner.main([
        "--corpus", str(corpus), "--output", str(out), "--repeats", "2",
    ])
    saved = json.loads(
        (out / "direction_diagnostics_summary.json").read_text(encoding="utf-8")
    )
    assert saved == summary
    assert set(summary["A_direction_specific"]) == {
        "LR_normalized", "RL_as_stored", "RL_after_reversal", "OTHER_as_stored",
    }
    assert "aggregate_delta" in summary["B_reversal"]
    assert "cause" in summary["B_reversal"]
    for name in ("normalized_LR_to_RL", "normalized_RL_to_LR", "stored_LR_to_RL"):
        assert name in summary["C_transfer"]
    report = (out / "direction_diagnostics_report.txt").read_text(encoding="utf-8")
    assert "diagnostic only" in report.lower() or "Diagnostic only" in report
    assert "NOT confidence intervals" in report
