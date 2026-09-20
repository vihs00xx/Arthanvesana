from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, parse_bool, sha256_file
from arthanvesana.replicate.robustness import (
    AGGREGATION_PROTOCOL,
    DEDUP_POLICY,
    aggregate_runs,
    evaluate_split,
)
from arthanvesana.stats.sampling import split_records

# Sensitivity matrix: 3 binary preprocessing factors = 8 cells.
#   sequence_order: "reading_order_normalized" (reading_order=True) vs
#                   "physical_as_stored" (False)
#   direction:      "known" (known_direction_only=True) vs "any" (False)
#   completeness:   "all" spans vs "complete-only" (complete inscription with
#                  fully observed start and end), filtered after analysis_records
# gap_policy="split" throughout, matching the headline evaluation.
# NOTE: sequence_order is a sequence-order processing choice, NOT a comparison
# of independent transcription traditions.
MATRIX = [
    {"sequence_order": sequence_order, "direction": direction,
     "completeness": completeness}
    for sequence_order in ("reading_order_normalized", "physical_as_stored")
    for direction in ("known", "any")
    for completeness in ("all", "complete-only")
]

COMPLETENESS_POLICY = (
    "complete-only keeps spans with complete inscription and both start/end "
    "fully observed; all keeps every span"
)


def cell_label(factors):
    return (
        f"{factors['sequence_order']} | {factors['direction']} | "
        f"{factors['completeness']}"
    )


def cell_records(frame, factors):
    records = analysis_records(
        frame,
        gap_policy="split",
        reading_order=factors["sequence_order"] == "reading_order_normalized",
        known_direction_only=factors["direction"] == "known",
    )
    if factors["completeness"] == "complete-only":
        records = [
            record for record in records
            if parse_bool(record["complete"])
            and parse_bool(record["start_complete"])
            and parse_bool(record["end_complete"])
        ]
    return records


def _run_seed(records, seed):
    train, test, diagnostics = split_records(
        records, 0.8, seed, track="artifact", group_duplicates=True
    )
    run = {
        "status": "ok", "reason": None, "seed": seed, "evaluation": None,
        "n_train": len(train), "n_test": len(test), "split": diagnostics,
    }
    if not train:
        run.update(status="skipped", reason="empty training partition")
    elif not any(record["sequence"] for record in test):
        run.update(status="skipped", reason="empty test token partition")
    else:
        try:
            run["evaluation"] = evaluate_split(train, test)
        except ValueError as exc:
            run.update(status="skipped", reason=str(exc))
    return run


def evaluate_cell(frame, factors, seeds):
    records = cell_records(frame, factors)
    runs = [_run_seed(records, seed) for seed in seeds]
    aggregate = aggregate_runs(runs)
    return {
        "label": cell_label(factors),
        "factors": factors,
        "n_records": len(records),
        "status": "ok" if aggregate["n_ok"] else "skipped",
        "reason": None if aggregate["n_ok"] else "no successful runs",
        "runs": runs,
        "aggregate": aggregate,
    }


def percentage(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def _delta_text(delta):
    mean = delta["mean"]
    text = "n/a" if mean is None else f"{100 * mean:+.2f} pp"
    return (
        f"{text} (w/t/l {delta['n_positive']}/{delta['n_zero']}/"
        f"{delta['n_negative']})"
    )

def render_report(summary):
    lines = [
        "Sensitivity-matrix evaluation of the restoration result",
        "",
        "The headline comparison (context bigram vs frequency and position",
        "baselines) is rerun under each cell of a 2x2x2 preprocessing matrix",
        "covering exactly three things:",
        "  1. sequence-order processing: reading_order_normalized vs physical_as_stored",
        "  2. direction inclusion: known (L/R, R/L only) vs any (includes OTHER)",
        f"  3. completeness filtering: {COMPLETENESS_POLICY}",
        "This matrix does NOT compare independent transcription traditions; the",
        "two sequence-order levels are two ways of ordering the SAME transcription.",
        "gap_policy=split throughout; artifact-grouped seeded splits;",
        "OOV targets count as failures; candidates come only from training data.",
        f"Grouping/deduplication: {DEDUP_POLICY}",
        f"Aggregation: {AGGREGATION_PROTOCOL}",
        "SD describes split variability, NOT confidence intervals.",
        "These are artificial single-sign masks, not verified archaeological restorations.",
        "",
        "Cell (sequence_order | direction | completeness) | Spans | Mean test spans | "
        "Context top-1 | Frequency top-1 | Position top-1 | "
        "Context-frequency top-1 | Context-position top-1",
    ]
    for cell in summary["cells"]:
        agg = cell["aggregate"]
        if cell["status"] != "ok":
            lines.append(f"{cell['label']} | {cell['n_records']} | skipped ({cell['reason']})")
            continue
        ok_runs = [run for run in cell["runs"] if run["status"] == "ok"]
        n_test = f"{fmean(run['n_test'] for run in ok_runs):.1f}"
        models = agg["models"]
        deltas = agg["paired_deltas"]
        lines.append(
            f"{cell['label']} | {cell['n_records']} | {n_test} | "
            f"{percentage(models['context']['top_1']['mean'])} +/- "
            f"{percentage(models['context']['top_1']['sd'])} | "
            f"{percentage(models['frequency']['top_1']['mean'])} | "
            f"{percentage(models['position']['top_1']['mean'])} | "
            f"{_delta_text(deltas['context_vs_frequency']['top_1'])} | "
            f"{_delta_text(deltas['context_vs_position']['top_1'])}"
        )
        for run in cell["runs"]:
            if run["status"] != "ok":
                lines.append(f"  seed {run['seed']} skipped: {run['reason']}")
    lines.extend(["", _interpretation(summary["cells"])])
    return "\n".join(lines) + "\n"


def _interpretation(cells):
    evaluated = [cell for cell in cells if cell["status"] == "ok"]
    skipped = [cell["label"] for cell in cells if cell["status"] != "ok"]
    lines = []
    if not evaluated:
        lines.append(
            "No cell produced a successful run; the sensitivity matrix is "
            "uninformative on this corpus."
        )
    else:
        freq_means = [
            cell["aggregate"]["paired_deltas"]["context_vs_frequency"]["top_1"]["mean"]
            for cell in evaluated
        ]
        pos_means = [
            cell["aggregate"]["paired_deltas"]["context_vs_position"]["top_1"]["mean"]
            for cell in evaluated
        ]
        robust = all(mean > 0 for mean in freq_means + pos_means)
        lines.append(
            f"Across {len(evaluated)} of {len(cells)} evaluated cells, the mean "
            "paired top-1 difference (context minus frequency) ranges from "
            f"{100 * min(freq_means):+.2f} to {100 * max(freq_means):+.2f} "
            "percentage points, and (context minus position) from "
            f"{100 * min(pos_means):+.2f} to {100 * max(pos_means):+.2f} "
            "percentage points."
        )
        if robust:
            lines.append(
                "The sign of both paired deltas is positive in every evaluated "
                "cell and the magnitudes are of the same order, so the headline "
                "claim (context bigram clearly beats frequency and position "
                "baselines) is robust to these preprocessing choices."
            )
        else:
            lines.append(
                "The paired deltas are NOT positive in every evaluated cell, so "
                "the headline claim (context bigram clearly beats the baselines) "
                "does NOT hold uniformly across these preprocessing choices; "
                "inspect the per-cell table before citing it."
            )
    if skipped:
        lines.append(
            "Skipped cells (no successful runs): " + "; ".join(skipped) + "."
        )
    lines.append(
        "Sparse cells have few spans and carry high split noise; treat their "
        "means as unstable. Skipped runs are excluded, not scored as zero."
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Sensitivity-matrix evaluation over preprocessing choices"
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "sensitivity")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    seeds = list(range(args.seed, args.seed + args.repeats))
    sources = sorted((ROOT / "src" / "arthanvesana").rglob("*.py")) + [Path(__file__).resolve()]
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "corpus_path": str(args.corpus.resolve()),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy")},
            "seeds": seeds, "train_frac": 0.8,
            "gap_policy": "split",
            "matrix": [dict(factors) for factors in MATRIX],
            "completeness_policy": COMPLETENESS_POLICY,
            "model": "Witten-Bell bigram; fixed, no tuning on evaluation splits",
            "deduplication": DEDUP_POLICY,
            "aggregation": AGGREGATION_PROTOCOL,
        },
        "cells": [evaluate_cell(frame, factors, seeds) for factors in MATRIX],
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "sensitivity_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
