"""Audit the connected artifact/inscription/duplicate components used for splits.

Explains why grouped test sets vary in size: exact-sequence connectivity can
merge many records into very large components. Uses connected_groups (the same
grouping code path as split_records) so grouping is not reimplemented.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, identity_value, sha256_file
from arthanvesana.stats.sampling import connected_groups

N_LARGEST = 20
N_FOLDS = 5


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, round(q / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def component_stats(records, indices):
    spans = [records[i] for i in indices]
    inscriptions = {identity_value(r.get("inscription_id")) for r in spans}
    artifacts = set()
    for r in spans:
        a = r.get("artifact_group")
        if a is not None:
            artifacts.add(str(a))
        else:
            aid = identity_value(r.get("artifact_id"))
            if aid is not None:
                artifacts.add(aid)
    sites = {identity_value(r.get("site")) for r in spans} - {None}
    sequences = {tuple(r["sequence"]) for r in spans}
    tokens = sum(len(r["sequence"]) for r in spans)
    lengths = sorted(len(r["sequence"]) for r in spans)
    return {
        "n_inscriptions": len({i for i in inscriptions if i is not None}),
        "n_spans": len(spans),
        "n_tokens": tokens,
        "n_artifacts": len(artifacts),
        "n_sites": len(sites),
        "n_sequences": len(sequences),
        "length_min": lengths[0] if lengths else 0,
        "length_median": _percentile(lengths, 50),
        "length_max": lengths[-1] if lengths else 0,
        "sites": sorted(sites),
    }


def explain_component(stat):
    """Classify which identity relation most plausibly created the component."""
    if stat["n_spans"] <= 1:
        return "single span"
    if stat["n_artifacts"] == 1 and stat["n_inscriptions"] == 1:
        return "single artifact/inscription (multi-span)"
    if stat["n_artifacts"] == 1:
        return "shared artifact identity"
    if stat["n_sequences"] < stat["n_spans"] and stat["n_inscriptions"] > 1:
        return "exact-sequence duplicates across inscriptions/artifacts"
    if stat["n_inscriptions"] > 1 and stat["n_artifacts"] > 1:
        return "chained artifact+sequence connectivity"
    return "shared inscription identity"



def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit connected split components")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "group_audit")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    groups = connected_groups(records, track="artifact", group_duplicates=True)

    total_tokens = sum(len(r["sequence"]) for r in records)
    fold_target = total_tokens / args.folds if args.folds else total_tokens

    components = []
    for g in groups:
        stat = component_stats(records, g["indices"])
        stat["group_id"] = g["group_id"]
        stat["cause"] = explain_component(stat)
        components.append(stat)

    token_sizes = sorted(c["n_tokens"] for c in components)
    span_sizes = sorted(c["n_spans"] for c in components)
    largest = sorted(components, key=lambda c: (-c["n_tokens"], c["group_id"]))[:N_LARGEST]
    largest_tokens = largest[0]["n_tokens"] if largest else 0
    oversized = [c for c in components if c["n_tokens"] > fold_target]

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "src/arthanvesana/stats/sampling.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "sampling.py"
                ),
                "scripts/run_group_audit.py": sha256_file(Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "track": "artifact", "group_duplicates": True,
            "gap_policy": "split", "known_direction_only": True,
            "n_folds_target": args.folds,
        },
        "n_records": len(records),
        "n_components": len(components),
        "total_tokens": total_tokens,
        "fold_target_tokens": fold_target,
        "component_token_percentiles": {
            "median": _percentile(token_sizes, 50),
            "p90": _percentile(token_sizes, 90),
            "p95": _percentile(token_sizes, 95),
            "p99": _percentile(token_sizes, 99),
            "max": token_sizes[-1] if token_sizes else 0,
        },
        "component_span_percentiles": {
            "median": _percentile(span_sizes, 50),
            "p90": _percentile(span_sizes, 90),
            "p95": _percentile(span_sizes, 95),
            "p99": _percentile(span_sizes, 99),
            "max": span_sizes[-1] if span_sizes else 0,
        },
        "largest_component_token_fraction": (
            largest_tokens / total_tokens if total_tokens else 0.0
        ),
        "largest_components": largest,
        "n_oversized_components": len(oversized),
        "oversized_group_ids": [c["group_id"] for c in oversized],
        "warning": (
            f"{len(oversized)} component(s) exceed one fold's target token size "
            f"({fold_target:.0f}); folds cannot be perfectly balanced."
            if oversized else
            "No component exceeds one fold's target token size."
        ),
        "interpretation": (
            "Component-size variation across grouped test sets is driven by "
            "exact-sequence connectivity merging many records into large "
            "components. Grouping is NOT weakened to balance folds, because that "
            "would leak duplicate/linked records across train and test."
        ),
    }

    lines = [
        "Grouped-partition (connected component) audit",
        "",
        "Grouping uses the exact split_records code path (artifact/inscription/"
        "exact-sequence connectivity). Group ids are deterministic; every record",
        "belongs to exactly one component. Grouping is not weakened to balance folds.",
        "",
        f"Records (spans): {summary['n_records']}; total tokens: {total_tokens}",
        f"Connected components: {summary['n_components']}",
        f"Target size of one {args.folds}-fold: {fold_target:.0f} tokens",
        "",
        "Component token sizes: "
        f"median {summary['component_token_percentiles']['median']}, "
        f"p90 {summary['component_token_percentiles']['p90']}, "
        f"p95 {summary['component_token_percentiles']['p95']}, "
        f"p99 {summary['component_token_percentiles']['p99']}, "
        f"max {summary['component_token_percentiles']['max']}",
        "Component span sizes: "
        f"median {summary['component_span_percentiles']['median']}, "
        f"p90 {summary['component_span_percentiles']['p90']}, "
        f"p95 {summary['component_span_percentiles']['p95']}, "
        f"p99 {summary['component_span_percentiles']['p99']}, "
        f"max {summary['component_span_percentiles']['max']}",
        "",
        f"Largest component holds {summary['largest_component_token_fraction']:.3f} "
        "of all tokens.",
        "",
        f"Largest {len(largest)} components (by tokens):",
        "group_id | spans | tokens | inscriptions | artifacts | sites | sequences | cause",
    ]
    for c in largest:
        lines.append(
            f"{c['group_id']} | {c['n_spans']} | {c['n_tokens']} | "
            f"{c['n_inscriptions']} | {c['n_artifacts']} | {c['n_sites']} | "
            f"{c['n_sequences']} | {c['cause']}"
        )
    lines.extend([
        "",
        f"WARNING: {summary['warning']}",
        "",
        summary["interpretation"],
    ])
    report = "\n".join(lines) + "\n"

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "group_audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "group_audit_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
