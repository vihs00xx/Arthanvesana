"""Compact structural models vs the bigram under grouped nested evaluation.

Models: a smoothed relative-position model and a discrete HMM (2-5 states, with
the state count selected on an inner split of each outer training partition).
Primary metric: held-out log loss (bits/token) with perplexity as a readable
transformation. Parameter counts and selected state counts are reported.

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
from arthanvesana.replicate.compact import crossfit_compact

DEFAULT_STATE_COUNTS = (2, 3, 4, 5)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compact structural models")
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "compact_models")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    seqs = [r["sequence"] for r in records if r["sequence"]]
    seeds = list(range(args.seed, args.seed + args.repeats))

    runs = []
    for seed in seeds:
        result = crossfit_compact(
            seqs, n_folds=5, seed=seed, state_counts=DEFAULT_STATE_COUNTS
        )
        runs.append({"seed": seed, **result})

    models = ("bigram", "relative_position", "hmm")
    aggregate = {}
    for name in models:
        vals = [r[name]["bits_per_token"] for r in runs
                if r[name]["bits_per_token"] is not None]
        ppls = [r[name]["perplexity"] for r in runs
                if r[name]["perplexity"] is not None]
        aggregate[name] = {
            "bits_per_token_mean": fmean(vals) if vals else None,
            "bits_per_token_sd": (stdev(vals) if len(vals) > 1 else 0.0) if vals else None,
            "perplexity_mean": fmean(ppls) if ppls else None,
            "n_runs": len(vals),
        }
    aggregate["selected_hmm_states"] = [
        s for r in runs for s in r["selected_hmm_states"]
    ]
    aggregate["hmm_n_parameters"] = [
        p for r in runs for p in r["hmm_n_parameters"]
    ]
    aggregate["relative_position_n_parameters"] = (
        runs[0]["relative_position_n_parameters"] if runs else None
    )

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "src/arthanvesana/replicate/compact.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "replicate" / "compact.py"),
                "scripts/run_compact_models.py": sha256_file(Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": seeds, "n_folds": 5,
            "state_counts": list(DEFAULT_STATE_COUNTS),
            "design": ("grouped outer folds; HMM state count selected on an inner "
                       "split of each outer TRAIN partition"),
            "oov_policy": "unseen signs take a uniform-sign penalty of log2(V) bits",
        },
        "runs": runs,
        "aggregate": aggregate,
        "interpretation": (
            "A compact model matching the bigram with far fewer parameters is an "
            "economical latent-state representation of positional sequence "
            "structure. The states are not words, phrases, grammatical roles, or "
            "semantic classes, and no linguistic interpretation is claimed."
        ),
    }

    lines = [
        "Compact structural models vs the bigram (grouped nested evaluation)",
        "",
        "Outer folds are connected-component grouped; the HMM state count is",
        "selected on an inner split of each outer TRAIN partition and refit on the",
        "full outer train before the single outer-test scoring. Unseen signs take a",
        "uniform-sign penalty of log2(V) bits. SD describes split variability, NOT",
        "confidence intervals.",
        "",
        "Model | bits/token mean +/- SD | perplexity | parameters",
    ]
    params = {
        "bigram": "see MKN bigram",
        "relative_position": aggregate["relative_position_n_parameters"],
        "hmm": (fmean(aggregate["hmm_n_parameters"])
                if aggregate["hmm_n_parameters"] else None),
    }
    for name in models:
        blk = aggregate[name]
        if blk["bits_per_token_mean"] is None:
            lines.append(f"{name} | skipped | - | -")
            continue
        lines.append(
            f"{name} | {blk['bits_per_token_mean']:.4f} +/- "
            f"{blk['bits_per_token_sd']:.4f} | {blk['perplexity_mean']:.2f} | "
            f"{params[name]}"
        )
    lines.extend([
        "",
        f"Selected HMM state counts across folds: {aggregate['selected_hmm_states']}",
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