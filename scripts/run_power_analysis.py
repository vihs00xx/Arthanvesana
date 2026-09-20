"""Synthetic power and specificity calibration for the n-gram inference.

Passes synthetic corpora (matched to empirical properties) through the same
grouped cross-fitted bigram-vs-trigram pipeline as the real corpus, at several
corpus sizes and controlled higher-order effect strengths. Reports
false-positive rate under zero effect, power under known effects, effect bias,
and Monte Carlo standard errors. Default replicates are reduced for
tractability; use --replicates 100 for the full calibration.

Synthetic generators are calibration instruments, not models of the Indus
production process or of natural language.
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
from arthanvesana.simulate import (
    crossfit_effect,
    empirical_profile,
    gen_markov,
    gen_position_only,
    gen_shuffled_real,
    gen_trigram_mixture,
    gen_unigram,
)

SCENARIOS = ("unigram", "position_only", "markov", "shuffled_real")
TRI_LAMBDAS = (0.0, 0.25, 0.5, 1.0)


def _scenario_seqs(name, profile, real_records, n, seed, lam=0.0):
    if name == "unigram":
        return gen_unigram(profile, n, seed)
    if name == "position_only":
        return gen_position_only(profile, n, seed)
    if name == "markov":
        return gen_markov(profile, n, seed)
    if name == "shuffled_real":
        return gen_shuffled_real(real_records, n, seed)
    if name == "trigram_mixture":
        return gen_trigram_mixture(profile, n, seed, lam)
    raise ValueError(f"unknown scenario {name}")


def _mcse(vals):
    v = stdev(vals) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return v


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic power calibration")
    parser.add_argument("--corpus", type=Path,
        default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path,
        default=ROOT / "outputs" / "power_analysis")
    parser.add_argument("--replicates", type=int, default=24)
    parser.add_argument("--sizes", type=float, nargs="*", default=[0.5, 1.0, 2.0])
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    profile = empirical_profile(records)
    n_emp = profile["n_inscriptions"]
    total_tokens = sum(profile["lengths"])

    cells = []
    for scenario in SCENARIOS:
        for size in args.sizes:
            n = max(10, int(n_emp * size))
            for rep in range(args.replicates):
                seed = args.seed * 1_000_000 + rep
                seqs = _scenario_seqs(scenario, profile, records, n, seed)
                cells.append({
                    "scenario": scenario, "lambda": 0.0, "size": size,
                    "replicate": rep, "n_inscriptions": n,
                    **crossfit_effect(seqs, 5, seed, args.permutations,
                                       args.bootstrap),
                })
    for lam in TRI_LAMBDAS:
        for size in args.sizes:
            n = max(10, int(n_emp * size))
            for rep in range(args.replicates):
                seed = args.seed * 1_000_000 + 500_000 + rep
                seqs = _scenario_seqs("trigram_mixture", profile, records, n, seed,
                                       lam=lam)
                cells.append({
                    "scenario": "trigram_mixture", "lambda": lam, "size": size,
                    "replicate": rep, "n_inscriptions": n,
                    **crossfit_effect(seqs, 5, seed, args.permutations,
                                       args.bootstrap),
                })

    def summarize(subset):
        effects = [c["macro_effect"] for c in subset if c["macro_effect"] is not None]
        detections = [c["p"] is not None and c["p"] < 0.05 for c in subset]
        return {
            "n_runs": len(subset),
            "mean_effect": fmean(effects) if effects else None,
            "mcse_effect": _mcse(effects) if len(effects) > 1 else None,
            "detection_rate": fmean(detections) if detections else None,
        }

    grid = {}
    for c in cells:
        grid.setdefault(f"{c['scenario']}|lam={c['lambda']}|size={c['size']}", []).append(c)
    summary_grid = {k: summarize(v) for k, v in sorted(grid.items())}

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "src/arthanvesana/simulate/generators.py":
                    sha256_file(ROOT / "src" / "arthanvesana" / "simulate" / "generators.py"),
                "src/arthanvesana/simulate/pipeline.py":
                    sha256_file(ROOT / "src" / "arthanvesana" / "simulate" / "pipeline.py"),
                "scripts/run_power_analysis.py": sha256_file(Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seed": args.seed, "replicates": args.replicates,
            "sizes": args.sizes, "permutations": args.permutations,
            "bootstrap": args.bootstrap,
            "empirical_tokens": total_tokens, "empirical_inscriptions": n_emp,
            "matched": {
                "exact": ["inscription count", "length distribution",
                          "vocabulary size", "unigram frequencies"],
                "approximate": ["higher-order transitions", "duplicate structure"],
            },
            "note": ("generators are calibration instruments, not models of the "
                     "Indus production process or of natural language; "
                     "default replicates reduced for tractability"),
        },
        "grid": summary_grid,
        "cells": cells,
    }

    lines = ["Synthetic power and specificity calibration", "",
        f"Replicates per cell: {args.replicates}; sizes: {args.sizes}; "
        f"permutations: {args.permutations}.",
        "Detection rate at lambda=0 is the false-positive rate; at lambda>0 it is power.",
        "", "cell | n_runs | mean effect (bits/token) | MCSE | detection rate"]
    for key, blk in summary_grid.items():
        if blk["n_runs"] == 0 or blk["mean_effect"] is None:
            continue
        lines.append(f"{key} | {blk['n_runs']} | "
                     f"{blk['mean_effect']:+.5f} | {blk['mcse_effect']:.5f} | "
                     f"{blk['detection_rate']:.3f}")
    lines.extend(["",
        "Interpretation: read each row as at this corpus size, the grouped "
        "cross-fitted pipeline detects a controlled higher-order effect of this "
        "strength with approximately this probability. The observed real-corpus "
        "macro effect (+0.028 bits/token) can be compared with the detection "
        "boundary of the lambda grid at 1x size. Generators are calibration "
        "instruments; no claim about the Indus production process follows."])
    report = "\n".join(lines) + "\n"

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "power_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (args.output / "power_analysis_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()