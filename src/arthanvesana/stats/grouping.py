"""Shared grouped-evaluation utilities.

One place for the operations that real-data inference, the synthetic
calibration pipeline, the compact-model comparison, and the diagnostic runners
all need:

* grouping records into indivisible connected components (delegated to the
  authoritative implementation in :mod:`arthanvesana.stats.sampling`, never
  reimplemented here);
* deterministic, imbalance-aware fold assignment;
* per-record out-of-fold scoring;
* aggregation to group level;
* leakage diagnostics computed from original record identities and sequences,
  not from already-created group labels.

Grouping policies are explicit. ``POLICY_ARTIFACT`` keeps linked spans of one
artifact and exact-sequence duplicates together. ``POLICY_ARTIFACT_SEQUENCE`` is
the same connected-component relation (artifact, inscription and exact-sequence
links); the two names exist so a caller must state which policy produced a
score, because scores under different policies are not interchangeable.
"""

from __future__ import annotations

import random
from collections import Counter

from arthanvesana.data.parse import identity_value
from arthanvesana.stats.sampling import connected_groups

#: Linked spans of one artifact, plus exact-sequence duplicates, stay together.
POLICY_ARTIFACT = "artifact_grouped"
#: Same relation as POLICY_ARTIFACT; named separately so reports must declare it.
POLICY_ARTIFACT_SEQUENCE = "artifact_plus_exact_sequence_grouped"

POLICIES = (POLICY_ARTIFACT, POLICY_ARTIFACT_SEQUENCE)

#: Machine-readable description of the grouping conventions, for reports.
INFO = {
    "policies": POLICIES,
    "default_track": "artifact",
    "default_group_duplicates": True,
    "relations": ["artifact", "inscription", "exact sequence"],
    "note": ("Components are connected by artifact, inscription and exact-sign "
             "sequence identity; every component is indivisible and moves "
             "between folds as a unit. Scores produced under different grouping "
             "policies are not directly interchangeable."),
}


def group_tokens(records: list[dict], groups: list[dict]) -> dict:
    """Map group id -> total token count, and attach it as ``n_tokens``."""
    totals = {}
    for group in groups:
        gid = group["group_id"]
        totals[gid] = sum(len(records[i]["sequence"]) for i in group["indices"])
        group["n_tokens"] = totals[gid]
    return totals


def group_stats(records: list[dict], groups: list[dict]) -> dict:
    """Map group id -> (n_tokens, n_spans) for fold balancing."""
    out = {}
    for group in groups:
        gid = group["group_id"]
        tokens = sum(len(records[i]["sequence"]) for i in group["indices"])
        out[gid] = (tokens, len(group["indices"]))
    return out


def assign_folds(group_stats_map, n_folds: int, seed: int) -> dict:
    """Assign whole components to folds, largest component first.

    Groups are indivisible: a component is never split across folds. Processing
    the largest components first (longest-processing-time first) keeps fold token
    loads as close as the component sizes allow, and seeded tie-breaking makes
    the assignment deterministic for a fixed ``seed`` while still varying across
    seeds. Returns ``group_id -> fold`` and preserves every group exactly once.

    ``group_stats_map`` maps group id -> ``(n_tokens, n_spans)``.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    tie_break = random.Random(seed)
    keys = {gid: tie_break.random() for gid in group_stats_map}
    order = sorted(
        group_stats_map,
        # larger components first; ties broken by a seeded pseudo-random key
        key=lambda gid: (-group_stats_map[gid][0], -group_stats_map[gid][1],
                         keys[gid], str(gid)),
    )
    load = [(0, 0) for _ in range(n_folds)]
    assignment = {}
    for gid in order:
        tokens, spans = group_stats_map[gid]
        fold = min(range(n_folds), key=lambda f: (load[f][0], load[f][1], f))
        assignment[gid] = fold
        load[fold] = (load[fold][0] + tokens, load[fold][1] + spans)
    return assignment


def fold_loads(assignment: dict, group_stats_map: dict, n_folds: int) -> dict:
    """Realized load, imbalance, and over-target components per fold.

    ``unavoidable_imbalance_tokens`` is the best achievable spread given
    indivisible components: the largest single component minus the average fold
    target when that component alone exceeds a fold's share.
    """
    loads = {f: {"n_groups": 0, "n_spans": 0, "n_tokens": 0} for f in range(n_folds)}
    for gid, fold in assignment.items():
        tokens, spans = group_stats_map[gid]
        loads[fold]["n_groups"] += 1
        loads[fold]["n_spans"] += spans
        loads[fold]["n_tokens"] += tokens
    total = sum(v[0] for v in group_stats_map.values())
    target = total / n_folds if n_folds else total
    largest = max((v[0] for v in group_stats_map.values()), default=0)
    tokens_by_fold = [loads[f]["n_tokens"] for f in range(n_folds)]
    return {
        "folds": {str(f): v for f, v in loads.items()},
        "target_tokens_per_fold": target,
        "min_tokens": min(tokens_by_fold, default=0),
        "max_tokens": max(tokens_by_fold, default=0),
        "spread_tokens": (max(tokens_by_fold) - min(tokens_by_fold))
        if tokens_by_fold else 0,
        "largest_component_tokens": largest,
        "unavoidable_imbalance_tokens": max(0.0, largest - target),
    }


def leakage_diagnostics(records: list[dict], groups: list[dict],
                        assignment: dict) -> dict:
    """Check that whole components, and their original identities, stay put.

    Verifies three things independently:

    1. every group lands in exactly one fold (no component split);
    2. no original *record identity* appears in two folds;
    3. no original *sequence* appears in two folds, which catches a duplicate
       that grouping failed to merge.

    The checks read the original record metadata, not the group labels, so a
    grouping bug cannot hide behind a consistent label.
    """
    fold_of_group = assignment
    group_of_record = {}
    for group in groups:
        for i in group["indices"]:
            group_of_record[i] = group["group_id"]

    record_fold = {}
    sequence_fold = {}
    artifact_fold = {}
    ungrouped = 0
    split_groups = 0
    for i, record in enumerate(records):
        gid = group_of_record.get(i)
        if gid is None:
            ungrouped += 1
            continue
        fold = fold_of_group.get(gid)
        if fold is None:
            ungrouped += 1
            continue
        record_fold.setdefault(identity_value(record.get("inscription_id")), set()).add(fold)
        sequence_fold.setdefault(tuple(record["sequence"]), set()).add(fold)
        artifact = record.get("artifact_group")
        if artifact is None:
            artifact = identity_value(record.get("artifact_id"))
        if artifact is not None:
            artifact_fold.setdefault(str(artifact), set()).add(fold)

    for group in groups:
        folds = {fold_of_group.get(group["group_id"]) for _ in group["indices"]}
        if len(folds) > 1:
            split_groups += 1

    def crossing(table):
        return sorted(str(k) for k, v in table.items() if k is not None and len(v) > 1)

    return {
        "n_records": len(records),
        "n_groups": len(groups),
        "ungrouped_records": ungrouped,
        "groups_split_across_folds": split_groups,
        "record_identity_crossings": len(crossing(record_fold)),
        "sequence_crossings": len(crossing(sequence_fold)),
        "artifact_crossings": len(crossing(artifact_fold)),
        "example_sequence_crossings": crossing(sequence_fold)[:5],
        "example_record_crossings": crossing(record_fold)[:5],
    }


def crossfit_predictions(records: list[dict], groups: list[dict],
                         assignment: dict, score_record, n_folds: int):
    """Give every eligible record exactly one out-of-fold prediction.

    ``score_record(record, train_records) -> list[float] | None`` returns one
    score per token of the record under a model fitted on ``train_records``, or
    ``None`` when the record cannot be scored. Records in a fold are never part
    of their own fitting data, because a record's fold comes from its component.

    Returns ``(rows, summary)`` where each row carries the record index, its
    group id, its fold, and its per-token scores. ``summary`` counts records that
    received zero predictions and records that received more than one.
    """
    group_of_record = {}
    for group in groups:
        for i in group["indices"]:
            group_of_record[i] = group["group_id"]

    rows = []
    for fold in range(n_folds):
        train = [r for i, r in enumerate(records)
                 if group_of_record.get(i) is not None
                 and assignment[group_of_record[i]] != fold]
        for i, record in enumerate(records):
            gid = group_of_record.get(i)
            if gid is None or assignment[gid] != fold:
                continue
            scores = score_record(record, train)
            rows.append({
                "record_index": i,
                "group_id": gid,
                "fold": fold,
                "n_tokens": len(record["sequence"]),
                "scores": list(scores) if scores is not None else None,
            })

    counts = Counter(row["record_index"] for row in rows)
    eligible = sum(1 for i in range(len(records)) if i in group_of_record)
    return rows, {
        "n_records": len(records),
        "n_eligible": eligible,
        "n_predicted": len(counts),
        "records_without_prediction": sorted(set(range(len(records))) - set(counts)),
        "records_with_duplicate_predictions": sorted(
            i for i, c in counts.items() if c > 1),
        "n_unscored_rows": sum(1 for row in rows if row["scores"] is None),
    }


def aggregate_by_group(rows: list[dict], per_token_diff=None) -> dict:
    """Aggregate per-record out-of-fold scores to the group level.

    Returns ``{group_id: {"mean": ..., "n_tokens": ..., "fold": ...}}``. The
    group mean is the unit of the macro estimand; the token count is the weight
    of the token-weighted estimand. Keeping both from the same rows is what makes
    the two estimands auditable.
    """
    pooled: dict = {}
    for row in rows:
        if row["scores"] is None:
            continue
        values = (per_token_diff(row) if per_token_diff is not None
                  else list(row["scores"]))
        if not values:
            continue
        bucket = pooled.setdefault(row["group_id"], {
            "values": [], "fold": row["fold"], "n_records": 0,
        })
        bucket["values"].extend(values)
        bucket["n_records"] += 1
    return {
        gid: {
            "mean": sum(b["values"]) / len(b["values"]),
            "n_tokens": len(b["values"]),
            "n_records": b["n_records"],
            "fold": b["fold"],
        }
        for gid, b in pooled.items()
    }


def build_groups(records: list[dict], *, track: str = "artifact",
                 group_duplicates: bool = True) -> list[dict]:
    """Thin, explicit wrapper over the authoritative grouping implementation."""
    return connected_groups(records, track=track, group_duplicates=group_duplicates)