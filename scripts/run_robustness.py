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

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.replicate.robustness import (
    AGGREGATION_PROTOCOL,
    DEDUP_POLICY,
    MODEL_KEYS,
    POSITION_BUCKET_PROTOCOL,
    leave_one_site_out,
    repeated_evaluation,
)


def percentage(value):
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_report(summary):
    lines = [
        "Restoration baseline and robustness evaluation",
        "",
        "All models predict the same masked positions, including singleton spans.",
        "OOV targets count as failures; candidates come only from training data.",
        "Context model: fixed Witten-Bell bigram, using available left/right signs.",
        "Frequency baseline: training sign counts, without context or position.",
        f"Position baseline: {POSITION_BUCKET_PROTOCOL}",
        f"Grouping/deduplication: {DEDUP_POLICY}",
        f"Aggregation: {AGGREGATION_PROTOCOL}",
        "Means +/- SD describe split variability, NOT confidence intervals.",
        "These are artificial single-sign masks, not verified archaeological restorations.",
        "",
    ]
    for protocol, result in summary["repeated"].items():
        agg = result["aggregate"]
        lines.extend([
            f"Protocol: {protocol}; {agg['n_ok']}/{agg['n_runs']} successful runs",
            "Model | Top-1 mean +/- SD | Top-5 mean | Top-10 mean | MRR mean",
        ])
        for model in MODEL_KEYS:
            block = agg["models"][model]
            mrr = block["mrr"]["mean"]
            mrr_text = "n/a" if mrr is None else f"{mrr:.4f}"
            lines.append(
                f"{model} | {percentage(block['top_1']['mean'])} +/- "
                f"{percentage(block['top_1']['sd'])} | "
                f"{percentage(block['top_5']['mean'])} | "
                f"{percentage(block['top_10']['mean'])} | {mrr_text}"
            )
        for comparison, metrics in agg["paired_deltas"].items():
            delta = metrics["top_1"]
            mean = delta["mean"]
            text = "n/a" if mean is None else f"{100 * mean:+.2f} percentage points"
            lines.append(
                f"{comparison}: mean top-1 difference {text}; "
                f"wins/ties/losses {delta['n_positive']}/{delta['n_zero']}/{delta['n_negative']}."
            )
        lines.append(f"Mean OOV target fraction: {percentage(agg['oov_fraction']['mean'])}")
        for run in result["runs"]:
            if run["status"] != "ok":
                lines.append(f"Seed {run['seed']} skipped: {run['reason']}")
        lines.append("")
    held = summary["held_out_sites"]
    lines.extend([
        f"Leave-one-site-out (minimum {held['min_inscriptions']} inscriptions)",
        "Other-site records connected to any held-site artifact/sequence group are purged.",
        "Unknown-site records are excluded from training; training sizes vary by site.",
        "Site | Train spans | Test targets | Purged spans | OOV | Frequency top-1 | Position top-1 | Context top-1",
    ])
    for run in held["runs"]:
        if run["status"] != "ok":
            lines.append(f"{run['site']}: skipped ({run['reason']})")
            continue
        models = run["evaluation"]["models"]
        scores = " | ".join(percentage(models[m]["top_1"]) for m in MODEL_KEYS)
        lines.append(
            f"{run['site']} | {run['n_train']} | {run['n_masked']} | "
            f"{run['n_purged']} | {percentage(run['oov_fraction'])} | {scores}"
        )
    if held["status"] == "skipped":
        lines.append(held["reason"])
    lines.extend([
        "",
        "Site differences also reflect vocabulary coverage, training size and artifact mix;",
        "they do not establish dialects or languages. Full per-run metrics, paired differences,",
        "partition identities and leakage diagnostics are in robustness_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training-only restoration baselines and robustness evaluation")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "robustness")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--min-site-inscriptions", type=int, default=100)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not 0 < args.train_frac < 1:
        parser.error("--train-frac must be strictly between 0 and 1")
    if args.min_site_inscriptions < 1:
        parser.error("--min-site-inscriptions must be positive")
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    seeds = list(range(args.seed, args.seed + args.repeats))
    sources = sorted((ROOT / "src" / "arthanvesana").rglob("*.py")) + [Path(__file__).resolve()]
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "corpus_path": str(args.corpus.resolve()),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy")},
            "seeds": seeds, "train_frac": args.train_frac,
            "gap_policy": "split", "known_direction_only": True,
            "n_spans": len(records),
            "n_inscriptions": len({r['inscription_id'] for r in records}),
            "model": "Witten-Bell bigram; fixed, no tuning on evaluation splits",
            "position_baseline": POSITION_BUCKET_PROTOCOL,
            "deduplication": DEDUP_POLICY,
            "aggregation": AGGREGATION_PROTOCOL,
        },
        "repeated": repeated_evaluation(records, seeds=seeds, train_frac=args.train_frac),
        "held_out_sites": leave_one_site_out(records, min_inscriptions=args.min_site_inscriptions),
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "robustness_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
