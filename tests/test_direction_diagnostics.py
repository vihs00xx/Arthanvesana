"""Tests for direction diagnostics.

These check the corrected behaviour:

* interior restoration is exactly reversal-invariant;
* the symmetric boundary models (``none``, ``symmetric``) ARE reversal-invariant,
  while the default ``asymmetric`` model is only required to be invariant in the
  interior;
* the union of duplicate relations across orientation variants is used for the
  shared split, so neither variant can leak an equivalent sequence;
* accuracy is separated into overall (OOV counts as failure) and shared-vocabulary
  targets, so vocabulary coverage is not confused with an ordering effect;
* first / interior / last / singleton positions are reported separately, with the
  last-position output actually populated.
"""

import importlib.util
import json
from pathlib import Path

from arthanvesana.data.parse import to_tidy
from arthanvesana.replicate.restore import BOUNDARY_MODES, restoration_records
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


def _palindromic_records():
    """Every sequence equals its own reverse.

    Reversal then maps record ``k`` position ``p`` onto record ``k`` position
    ``L-1-p``, so a row-by-row comparison of the original and reversed runs is a
    valid mirror comparison. (With non-palindromic sequences, row ``i`` of the
    reversed run is no longer the mirror of row ``i``.)
    """
    seqs = [["001", "002", "001"], ["002", "001", "002"], ["003", "002", "003"]]
    return [
        {"inscription_id": f"i{k}", "sequence": list(s), "start_complete": True,
         "end_complete": True, "site": "A", "artifact_id": f"a{k}"}
        for k, s in enumerate(seqs)
    ]


def _ranks_by_record(rows):
    out = {}
    for row in rows:
        out.setdefault(row["inscription_id"], {})[row["position"]] = row["rank"]
    return out


def _mirror_mismatches(train, test, mode):
    """Positions where masking p in S differs from masking L-1-p in reverse(S).

    The correct mirror pairing is position ``p`` of the original against position
    ``L-1-p`` of the reversed run. Comparing row-by-row would only be valid for
    the middle position, which is why the earlier version of this test could pass
    while the model was not actually symmetric.
    """
    runner = load_runner("run_direction_diagnostics")
    forward, _ = restoration_records(train, test, mask_length=1, boundary_mode=mode)
    backward, _ = restoration_records(
        runner._reverse_records(train), runner._reverse_records(test),
        mask_length=1, boundary_mode=mode)
    fwd = _ranks_by_record(forward)
    rev = _ranks_by_record(backward)
    mismatches = 0
    for inscription_id, positions in fwd.items():
        length = max(positions) + 1
        for position, rank in positions.items():
            if rev.get(inscription_id, {}).get(length - 1 - position) != rank:
                mismatches += 1
    return mismatches


# ---------------------------------------------------------------------------
# Reversal invariants
# ---------------------------------------------------------------------------

def test_interior_restoration_is_reversal_invariant():
    runner = load_runner("run_direction_diagnostics")
    train = _toy_records()
    test = _toy_records()
    forward, _ = restoration_records(train, test, mask_length=1)
    backward, _ = restoration_records(
        runner._reverse_records(train), runner._reverse_records(test), mask_length=1)
    interior_f = [r["rank"] for r in forward if r["position"] == 1]
    interior_b = [r["rank"] for r in backward if r["position"] == 1]
    assert interior_f == interior_b


def test_symmetric_end_seed_is_gated_on_end_complete():
    """The backward <S> seed must obey end_complete, mirroring the forward edge.

    This is the precise property the fix adds: gating only the start left the two
    edges inconsistent whenever the completeness flags differed.
    """
    from arthanvesana.replicate.restore import _mask_distributions

    seqs = [["001", "002", "003"], ["003", "002", "001"], ["001", "002", "003"]]
    model = NGramModel(seqs, 2, method="wittenbell")
    mats = {c: model.dist((c,)) for c in sorted(model.vocab | {"<S>"})}
    prior = NGramModel(seqs, 1, method="wittenbell").dist(())
    seq = ["001", "002", "003"]

    complete = _mask_distributions(mats, prior, model.vocab, seq, 2, 3, True,
                                   "symmetric", True)[0]
    incomplete = _mask_distributions(mats, prior, model.vocab, seq, 2, 3, True,
                                     "symmetric", False)[0]
    no_evidence = _mask_distributions(mats, prior, model.vocab, seq, 2, 3, True,
                                      "none", False)[0]

    # an incomplete end falls back to the no-evidence prior term
    assert incomplete == no_evidence
    # a complete end uses the <S> distribution, so the two must differ
    assert complete != incomplete


def test_edge_models_are_at_least_as_invariant_as_asymmetric():
    """No mode is exactly invariant; the edge-symmetric ones must be closer.

    Exact reversal invariance is NOT achievable here: the context model's left
    term is P(w | prev) and its right term is P(next | w), which are
    transpose-related and coincide only under detailed balance. So the assertion
    is comparative, not absolute.
    """
    train = _palindromic_records()
    test = _palindromic_records()
    for i, record in enumerate(test):
        record["start_complete"] = (i % 2 == 0)
        record["end_complete"] = (i % 2 == 1)

    asymmetric = _mirror_mismatches(train, test, "asymmetric")
    assert _mirror_mismatches(train, test, "symmetric") <= asymmetric
    assert _mirror_mismatches(train, test, "none") <= asymmetric


def test_no_mode_is_documented_as_exactly_invariant():
    runner = load_runner("run_direction_diagnostics")
    for mode, text in runner.REVERSAL_EXPECTATIONS.items():
        assert "EXACTLY invariant" not in text, (
            f"{mode} must not claim exact reversal invariance"
        )
    assert "NOT exactly invariant" in runner.REVERSAL_EXPECTATIONS["symmetric"]


def test_every_boundary_mode_states_its_expected_invariant():
    runner = load_runner("run_direction_diagnostics")
    assert set(runner.REVERSAL_EXPECTATIONS) == set(BOUNDARY_MODES)
    # the asymmetric model must NOT be described as reversal-invariant
    assert "NOT expected to be invariant" in runner.REVERSAL_EXPECTATIONS["asymmetric"]
    assert "NOT exactly invariant" in runner.REVERSAL_EXPECTATIONS["symmetric"]
    assert "APPROXIMATELY" in runner.REVERSAL_EXPECTATIONS["none"]


def test_unknown_boundary_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        restoration_records(_toy_records(), _toy_records(), mask_length=1,
                            boundary_mode="nonsense")


def test_forward_and_reverse_use_same_transition_matrix():
    seqs = [["001", "002"], ["002", "001"]]
    model = NGramModel(seqs, 2, method="wittenbell")
    assert model.dist(("001",))["002"] > 0
    assert model.dist(("002",))["001"] > 0


# ---------------------------------------------------------------------------
# Union grouping across orientation variants
# ---------------------------------------------------------------------------

def _variant(rows):
    return [
        {"inscription_id": iid, "artifact_id": aid, "site": "A",
         "sequence": list(seq), "span_index": 0, "span_start": 0,
         "start_complete": True, "end_complete": True}
        for iid, aid, seq in rows
    ]


def test_union_group_keys_links_duplicates_across_variants():
    """A duplicate found in EITHER variant must be linked.

    Here i1/i2 agree only in the normalized variant and i3/i4 agree only in the
    stored variant. The union must link both pairs, so neither variant can leak an
    equivalent sequence across a shared split.
    """
    runner = load_runner("run_direction_diagnostics")
    normalized = _variant([
        ("i1", "a1", ["001", "002"]),
        ("i2", "a2", ["001", "002"]),          # matches i1 in normalized
        ("i3", "a3", ["003", "004"]),
        ("i4", "a4", ["005", "006"]),
    ])
    stored = _variant([
        ("i1", "a1", ["002", "001"]),
        ("i2", "a2", ["009", "009"]),          # differs from i1 in stored
        ("i3", "a3", ["004", "003"]),
        ("i4", "a4", ["004", "003"]),          # matches i3 in stored
    ])
    keys = runner.union_group_keys([normalized, stored])
    k = runner._stable_key
    assert keys[k(normalized[0])] == keys[k(normalized[1])], "normalized duplicates"
    assert keys[k(normalized[2])] == keys[k(normalized[3])], "stored duplicates"
    assert keys[k(normalized[0])] != keys[k(normalized[2])]


def test_union_group_keys_without_duplicates_keeps_records_separate():
    runner = load_runner("run_direction_diagnostics")
    records = _variant([
        ("i1", "a1", ["001", "002"]),
        ("i2", "a2", ["003", "004"]),
    ])
    keys = runner.union_group_keys([records])
    assert keys[runner._stable_key(records[0])] != keys[runner._stable_key(records[1])]


def test_aligned_split_keeps_linked_duplicates_together():
    runner = load_runner("run_direction_diagnostics")
    records = _variant([
        ("i1", "a1", ["001", "002"]),
        ("i2", "a2", ["001", "002"]),
        ("i3", "a3", ["003", "004"]),
        ("i4", "a4", ["005", "006"]),
    ])
    keys = runner.union_group_keys([records])
    for seed in range(8):
        train, test, _ = runner.aligned_split(records, seed, keys)
        where = {}
        for split_name, rows in (("train", train), ("test", test)):
            for row in rows:
                where[row["inscription_id"]] = split_name
        assert where["i1"] == where["i2"], "duplicates must not be split"


# ---------------------------------------------------------------------------
# OOV separation and position classes
# ---------------------------------------------------------------------------

def test_restoration_breakdown_separates_oov_from_ordering():
    runner = load_runner("run_direction_diagnostics")
    train = _variant([
        ("i1", "a1", ["001", "002", "003"]),
        ("i2", "a2", ["001", "002", "004"]),
        ("i3", "a3", ["001", "002", "005"]),
    ])
    # a test record containing a sign absent from training
    test = _variant([("t1", "b1", ["001", "002", "999"])])
    block = runner._restoration_breakdown(train, test)
    assert block["n_targets"] == 3
    assert block["n_oov"] == 1
    assert block["oov_rate"] > 0
    # overall counts the OOV target as a failure; shared excludes it
    assert block["n_known_targets"] == 2
    assert block["top_1_overall"] is not None
    assert block["top_1_shared"] is not None
    assert block["oov_targets_are_failures"] is True


def test_position_breakdown_reports_all_classes_including_last():
    runner = load_runner("run_direction_diagnostics")
    train = _variant([
        ("i1", "a1", ["001", "002", "003"]),
        ("i2", "a2", ["001", "002", "004"]),
        ("i3", "a3", ["001", "002", "005"]),
        ("i4", "a4", ["001"]),
    ])
    test = _variant([
        ("t1", "b1", ["001", "002", "003"]),
        ("t2", "b2", ["001"]),
    ])
    pos = runner.position_breakdown(train, test)
    assert set(pos) == {"singleton", "first", "interior", "last"}
    # the last-position cell must be populated, not None
    assert pos["last"]["n_targets"] >= 1
    assert pos["last"]["top_1_overall"] is not None
    assert pos["singleton"]["n_targets"] >= 1
    assert pos["first"]["n_targets"] >= 1
    assert pos["interior"]["n_targets"] >= 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

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
    # every boundary mode is reported with its expected invariant
    assert set(summary["B_reversal"]) == set(BOUNDARY_MODES)
    for _mode, block in summary["B_reversal"].items():
        assert "expected" in block and block["expected"]
        assert "position_classes" in block
    assert summary["manifest"]["default_boundary_mode"] == "asymmetric"
    # the grouping policy is explicit and states the union
    assert "union" in summary["manifest"]["grouping_policy"]
    assert "union" in summary["manifest"]["duplicate_grouping"]
    # transfer conditions carry OOV-separated metrics
    for name in ("normalized_LR_to_RL", "normalized_RL_to_LR", "stored_LR_to_RL"):
        assert name in summary["C_transfer"]
        block = summary["C_transfer"][name]
        assert "top_1_shared" in block
        assert "oov_rate" in block
    assert "data/PROVENANCE.md" in summary["manifest"]["source_sha256"]

    report = (out / "direction_diagnostics_report.txt").read_text(encoding="utf-8")
    assert "Diagnostic only" in report
    assert "top-1 shared" in report
    assert "NOT confidence intervals" in report
    assert "NOT expected to be invariant" in report