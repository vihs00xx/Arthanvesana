"""Compact structural models vs the bigram under grouped nested evaluation.

Models: three position models (exact length/index, relative normalized-position
bins, and a completeness-aware exact variant) and a discrete HMM whose state
count and initialization are selected on an inner *grouped* split of each outer
training partition. Primary metric: held-out log loss (bits/token), with
perplexity as a readable transformation.

Two explicit budget experiments are reported, because comparing a capped HMM
against a full-data bigram is not a fair model-family comparison:

* ``matched`` — every model is trained on the SAME capped training subset,
  selected with whole components; realized sizes are reported.
* ``full`` — every model is trained on the full outer-training partition, except
  the HMM when a tractability cap is set; that limitation is stated, not hidden.

The HMM states are an economical latent-state representation of positional
sequence structure. They are NOT words, phrases, grammatical roles, or semantic
classes.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, stdev

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.replicate.compact import (
    POSITION_MODES,
    crossfit_compact,
)

DEFAULT_STATE_COUNTS = (2, 3, 4, 5)
DEFAULT_INITS = ("frequency_slice", "uniform")
DEFAULT_ALPHAS = (0.1, 1.0, 10.0)
MODEL_NAMES = ("bigram", "position_exact", "position_relative",
               "position_exact_complete", "hmm")


def _aggregate(runs):
    aggregate = {}
    for name in MODEL_NAMES:
        vals = [r[name]["bits_per_token"] for r in runs
                if r[name]["bits_per_token"] is not None]
        ppls = [r[name]["perplexity"] for r in runs
                if r[name]["perplexity"] is not None]
        oovs = [r[name]["oov_rate"] for r in runs
                if r[name]["oov_rate"] is not None]
        aggregate[name] = {
            "bits_per_token_mean": fmean(vals) if vals else None,
            "bits_per_token_sd": (stdev(vals) if len(vals) > 1 else 0.0) if vals else None,
            "perplexity_mean": fmean(ppls) if ppls else None,
            "oov_rate_mean": fmean(oovs) if oovs else None,
            "n_runs": len(vals),
        }
    aggregate["selected_hmm_states"] = [s for r in runs for s in r["selected_hmm_states"]]
    aggregate["selected_hmm_inits"] = [s for r in runs for s in r["selected_hmm_inits"]]
    aggregate["selected_position_alpha"] = [
        a for r in runs for a in r["selected_position_alpha"]]
    aggregate["hmm_n_parameters"] = [p for r in runs for p in r["hmm_n_parameters"]]
    return aggregate


def _budget_block(budget, cap_records, seeds, records):
    runs = []
    for seed in seeds:
        result = crossfit_compact(
            records, n_folds=5, seed=seed,
            state_counts=DEFAULT_STATE_COUNTS, inits=DEFAULT_INITS,
            alphas=DEFAULT_ALPHAS, hmm_iter=8,
            budget=budget,
            cap_records=cap_records,
        )
        runs.append({"seed": seed, **result})
    return {"budget": budget, "cap_records": cap_records,
            "runs": runs, "aggregate": _aggregate(runs)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact structural models")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "compact_models")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budgets", nargs="*", default=["matched", "full"],
                        choices=["matched", "full"])
    parser.add_argument("--cap-records", type=int, default=600,
                        help=("record cap used for the matched budget and, in the "
                              "full budget, for the HMM only"))
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    seeds = list(range(args.seed, args.seed + args.repeats))

    blocks = {}
    for budget in args.budgets:
        blocks[budget] = _budget_block(budget, args.cap_records, seeds, records)

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "src/arthanvesana/replicate/compact.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "replicate" / "compact.py"),
                "src/arthanvesana/stats/grouping.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "grouping.py"),
                "scripts/run_compact_models.py": sha256_file(Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": seeds,
            "n_folds": 5,
            "budgets": args.budgets,
            "cap_records": args.cap_records,
            "state_counts": list(DEFAULT_STATE_COUNTS),
            "hmm_inits": list(DEFAULT_INITS),
            "position_alphas": list(DEFAULT_ALPHAS),
            "position_modes": list(POSITION_MODES),
            "grouping_policy": ("artifact_grouped: connected artifact / inscription / "
                                "exact-sequence components, duplicates merged"),
            "design": ("outer folds are connected-component grouped; HMM state count "
                       "and initialization, and position smoothing strength, are "
                       "selected on an inner GROUPED split of each outer TRAIN "
                       "partition only; the outer test never enters selection"),
            "oov_policy": ("training-only vocabulary; <UNK> always present with a "
                           "pseudo-count; likelihood evaluated over mapped <UNK> "
                           "outcomes in every model; one unseen sign never discards "
                           "the known tokens of its sequence"),
            "restoration_policy": ("restoration scoring is separate: an unseen original "
                                   "sign can never count as a correct restoration, and "
                                   "<UNK> is never a candidate"),
        },
        "budgets": blocks,
        "interpretation": (
            "A compact model matching the bigram with far fewer parameters is an "
            "economical latent-state representation of positional sequence "
            "structure. The states are not words, phrases, grammatical roles, or "
            "semantic classes, and no linguistic interpretation is claimed."),
    }

    lines = [
        "Compact structural models vs the bigram (grouped nested evaluation)",
        "",
        "Outer folds are connected-component grouped. HMM state count and",
        "initialization, and the position smoothing strength, are selected on an",
        "inner GROUPED split of each outer TRAIN partition and refit on the",
        "permitted outer train before the single outer-test scoring.",
        "",
        "Unknown signs: one shared training-only vocabulary policy. <UNK> is always",
        "present with a pseudo-count and likelihood is evaluated over mapped <UNK>",
        "outcomes in every model, so one unseen sign never discards the known-token",
        "contributions of its sequence. Restoration scoring is separate: an unseen",
        "original sign can never count as a correct restoration.",
        "",
        "TWO BUDGET EXPERIMENTS (a capped HMM vs a full-data bigram is not a fair",
        "model-family comparison):",
    ]
    for budget in args.budgets:
        block = blocks[budget]
        agg = block["aggregate"]
        realized = [r["n_records"] for run in block["runs"]
                    for r in run["training_budgets"]]
        lines.append("")
        lines.append(f"=== budget: {budget} ===")
        if budget == "matched":
            lines.append(f"All models trained on the SAME capped subset "
                         f"(cap {args.cap_records} records, whole components).")
        else:
            lines.append("All models trained on the full outer-training partition,")
            lines.append("except the HMM, which is capped for tractability. The")
            lines.append("limitation is stated rather than hidden.")
        if realized:
            lines.append(f"Realized per-fold training sizes: min {min(realized)}, "
                         f"max {max(realized)}, folds {len(realized)}")
        lines.append("")
        lines.append("Model | bits/token mean +/- SD | perplexity | OOV rate | n_runs")
        for name in MODEL_NAMES:
            blk = agg[name]
            if blk["bits_per_token_mean"] is None:
                lines.append(f"{name} | skipped | - | - | 0")
                continue
            lines.append(
                f"{name} | {blk['bits_per_token_mean']:.4f} +/- "
                f"{blk['bits_per_token_sd']:.4f} | {blk['perplexity_mean']:.2f} | "
                f"{blk['oov_rate_mean']:.4f} | {blk['n_runs']}"
            )
        lines.append(f"Selected HMM states: {agg['selected_hmm_states']}")
        lines.append(f"Selected HMM inits: {agg['selected_hmm_inits']}")
        lines.append(f"Selected position alpha: {agg['selected_position_alpha']}")
        lines.append(f"HMM parameters per fold: {agg['hmm_n_parameters']}")
        hmm_diag = [d for run in block["runs"] for d in run["hmm_diagnostics"]]
        converged = [t["converged"] for d in hmm_diag for t in d["trials"]]
        if converged:
            lines.append(f"HMM inner-selection trials converged: "
                         f"{sum(converged)}/{len(converged)}")
        leakage = [run["leakage"]["groups_split_across_folds"] for run in block["runs"]]
        seq_cross = [run["leakage"]["sequence_crossings"] for run in block["runs"]]
        lines.append(f"Leakage (components split across folds): {leakage}; "
                     f"sequence crossings: {seq_cross}")
    lines.extend([
        "",
        "SD describes split variability, NOT confidence intervals.",
        "",
        summary["interpretation"],
    ])
    report = "\n".join(lines) + "\n"

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "compact_models_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (args.output / "compact_models_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()