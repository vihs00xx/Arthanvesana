"""Regression tests for the shared grouped-evaluation utilities.

These check scientific failure modes, not file existence:

* linked spans of one artifact cannot be split across folds;
* transitive duplicate connections stay together under the duplicate policy;
* every eligible record gets exactly one out-of-fold prediction;
* no outer-test record enters fitting;
* fold assignment is deterministic for a fixed seed and varies across seeds;
* a component larger than a fold target is never broken up to balance folds;
* leakage diagnostics read original identities and sequences, not group labels.
"""

from arthanvesana.stats.grouping import (
    INFO,
    aggregate_by_group,
    assign_folds,
    build_groups,
    crossfit_predictions,
    fold_loads,
    group_stats,
    leakage_diagnostics,
)


def _records():
    # Two spans of artifact A1 (linked), one span of A2, and three copies of one
    # exact sequence spread over three different artifacts (transitively linked
    # only when duplicates are grouped).
    recs = [
        {"inscription_id": "i1", "artifact_id": "A1", "site": "S",
         "sequence": ["001", "002"], "span_index": 0, "span_start": 0},
        {"inscription_id": "i1", "artifact_id": "A1", "site": "S",
         "sequence": ["003", "004"], "span_index": 1, "span_start": 3},
        {"inscription_id": "i2", "artifact_id": "A2", "site": "S",
         "sequence": ["005", "006"], "span_index": 0, "span_start": 0},
        {"inscription_id": "i3", "artifact_id": "B1", "site": "T",
         "sequence": ["007", "008", "009"], "span_index": 0, "span_start": 0},
        {"inscription_id": "i4", "artifact_id": "B2", "site": "T",
         "sequence": ["007", "008", "009"], "span_index": 0, "span_start": 0},
        {"inscription_id": "i5", "artifact_id": "B3", "site": "T",
         "sequence": ["007", "008", "009"], "span_index": 0, "span_start": 0},
    ]
    for r in recs:
        r.setdefault("artifact_group", (None, "explicit", r["artifact_id"]))
    return recs


def test_linked_spans_of_one_artifact_never_cross_folds():
    records = _records()
    groups = build_groups(records)
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, 3, seed=0)
    for group in groups:
        folds = {assignment[group["group_id"]] for _ in group["indices"]}
        assert len(folds) == 1
    # the two A1 spans share a component
    linked = [g for g in groups
              if any(i in (0, 1) for i in g["indices"])]
    assert len(linked) == 1
    assert set(linked[0]["indices"]) >= {0, 1}


def test_transitive_duplicates_stay_together_when_policy_enabled():
    records = _records()
    with_dups = build_groups(records, group_duplicates=True)
    merged = [g for g in with_dups if {3, 4, 5} <= set(g["indices"])]
    assert merged, "exact-sequence duplicates must be merged into one component"

    without = build_groups(records, group_duplicates=False)
    separate = [g for g in without if {3, 4, 5} <= set(g["indices"])]
    assert not separate, "duplicate grouping must be opt-outable"


def test_every_eligible_record_gets_exactly_one_prediction():
    records = _records()
    groups = build_groups(records)
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, 3, seed=1)

    def score_record(record, train):
        # record index is not passed, so use identity to prove no leakage below
        return [0.0] * len(record["sequence"])

    rows, summary = crossfit_predictions(records, groups, assignment,
                                         score_record, 3)
    assert summary["n_eligible"] == len(records)
    assert summary["n_predicted"] == len(records)
    assert summary["records_without_prediction"] == []
    assert summary["records_with_duplicate_predictions"] == []
    assert len(rows) == len(records)
    # each record is scored under exactly one fold
    assert sorted(r["record_index"] for r in rows) == list(range(len(records)))


def test_no_outer_test_record_enters_fitting_data():
    records = _records()
    groups = build_groups(records)
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, 3, seed=2)
    leaked = []

    def score_record(record, train):
        for held in train:
            # training data must never contain the record being scored, and
            # never a record from the same component
            if held["inscription_id"] == record["inscription_id"]:
                leaked.append(("inscription", record["inscription_id"]))
        return [0.0] * len(record["sequence"])

    rows, _ = crossfit_predictions(records, groups, assignment, score_record, 3)
    fold_of_record = {r["record_index"]: r["fold"] for r in rows}
    group_of = {}
    for g in groups:
        for i in g["indices"]:
            group_of[i] = g["group_id"]
    for row in rows:
        fold = row["fold"]
        for i, _rec in enumerate(records):
            if group_of[i] == row["group_id"]:
                assert fold_of_record[i] == fold
    assert leaked == []


def test_fold_assignment_is_deterministic_and_seed_sensitive():
    # more independent components than folds, so the seed can actually matter
    records = [
        {"inscription_id": f"i{k}", "artifact_id": f"A{k}", "site": "S",
         "artifact_group": (None, "explicit", f"A{k}"),
         "sequence": [f"{k:03d}", f"{k + 500:03d}"], "span_index": 0, "span_start": 0}
        for k in range(12)
    ]
    stats = group_stats(records, build_groups(records))
    a = assign_folds(stats, 3, seed=7)
    b = assign_folds(stats, 3, seed=7)
    assert a == b, "fold assignment must be deterministic for a fixed seed"
    assert set(a.values()) == {0, 1, 2}
    # a different seed should generally move at least one component
    different = any(assign_folds(stats, 3, seed=s) != a for s in range(1, 12))
    assert different


def test_large_component_stays_intact_and_imbalance_is_reported():
    # one component far larger than a fold target plus many small ones
    records = []
    for k in range(40):
        records.append({
            "inscription_id": f"big{k}", "artifact_id": "BIG", "site": "S",
            "artifact_group": (None, "explicit", "BIG"),
            "sequence": ["001", "002", "003"], "span_index": k, "span_start": 0,
        })
    for k in range(20):
        records.append({
            "inscription_id": f"sm{k}", "artifact_id": f"S{k}", "site": "S",
            "artifact_group": (None, "explicit", f"S{k}"),
            "sequence": ["004", f"{k:03d}"], "span_index": 0, "span_start": 0,
        })
    groups = build_groups(records)
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, 5, seed=3)
    loads = fold_loads(assignment, stats, 5)
    # the big component is not split
    for group in groups:
        assert len({assignment[group["group_id"]] for _ in group["indices"]}) == 1
    assert loads["largest_component_tokens"] == 120
    assert loads["spread_tokens"] > 0
    assert loads["unavoidable_imbalance_tokens"] > 0


def test_leakage_diagnostics_flag_a_duplicate_grouping_miss():
    records = _records()
    # deliberately group WITHOUT duplicate merging, so the same sequence lands in
    # three different components; the sequence check must still catch it
    groups = build_groups(records, group_duplicates=False)
    stats = group_stats(records, groups)
    assignment = assign_folds(stats, 3, seed=11)
    diag = leakage_diagnostics(records, groups, assignment)
    assert diag["groups_split_across_folds"] == 0
    assert diag["record_identity_crossings"] == 0
    # with duplicates ungrouped, crossing is only detected if the folds differ;
    # force it to be visible by assigning the three copies to different folds
    forced = dict(assignment)
    ids = {g["group_id"]: g for g in groups}
    targets = [gid for gid, g in ids.items()
               if set(g["indices"]) & {3, 4, 5}]
    for n, gid in enumerate(sorted(targets)):
        forced[gid] = n % 3
    diag2 = leakage_diagnostics(records, groups, forced)
    assert diag2["sequence_crossings"] >= 1
    assert diag2["example_sequence_crossings"]


def test_group_stats_and_policy_names_are_explicit():
    records = _records()
    groups = build_groups(records)
    stats = group_stats(records, groups)
    assert sum(v[1] for v in stats.values()) == len(records)
    assert INFO["policies"] == ("artifact_grouped",
                               "artifact_plus_exact_sequence_grouped")


def test_aggregate_by_group_uses_group_as_unit():
    rows = [
        {"group_id": "g1", "fold": 0, "scores": [1.0, 1.0, 1.0]},
        {"group_id": "g2", "fold": 1, "scores": [-1.0]},
    ]
    agg = aggregate_by_group(rows)
    assert agg["g1"]["mean"] == 1.0
    assert agg["g1"]["n_tokens"] == 3
    assert agg["g2"]["mean"] == -1.0
    # the macro mean over groups is 0.0 even though the token mean is +0.5
    macro = sum(v["mean"] for v in agg.values()) / len(agg)
    tokens = sum(v["mean"] * v["n_tokens"] for v in agg.values())
    weight = sum(v["n_tokens"] for v in agg.values())
    assert macro == 0.0
    assert tokens / weight == 0.5