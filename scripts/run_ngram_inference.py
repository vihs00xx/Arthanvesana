"""Paired bigram-vs-trigram held-out log-loss inference.

Trains modified Kneser-Ney bigram and trigram models on identical
artifact-grouped train/test splits and scores every token position of every
held-out sequence under both models (full-sequence log-loss, not the
single-mask task). Per-token paired differences d_i = log2 P_trigram -
log2 P_bigram are tested with a seeded paired permutation (sign-flip) test,
per split and pooled across splits.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import split_records

TRAIN_FRAC = 0.8


def paired_logprobs(model: NGramModel, seqs: list[list[str]]) -> list[float]:
    """log2 P(token | context) at every position of every sequence."""
    out: list[float] = []
    for seq in seqs:
        mapped = model._map(seq)
        for i in range(len(mapped)):
            context = tuple((["<S>"] * (model.n - 1) + mapped[:i])[-(model.n - 1):])
            prob = model.dist(context)[mapped[i]]
            if prob <= 0.0:
                raise ValueError(
                    f"Zero probability for token {mapped[i]!r} in context "
                    f"{context!r}; log-loss undefined."
                )
            out.append(math.log2(prob))
    return out


def paired_permutation_p(
    diffs: np.ndarray, n_permutations: int, seed: int
) -> float:
    """Sign-flip permutation p for H0: models exchangeable at each position."""
    observed = abs(float(diffs.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    n = diffs.size
    for _ in range(n_permutations):
        signs = rng.choice((-1.0, 1.0), size=n)
        if abs(float(diffs.dot(signs)) / n) >= observed:
            exceed += 1
    return exceed / n_permutations


def mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }



def run_split(
    records: list[dict], seed: int, n_permutations: int
) -> tuple[dict, np.ndarray]:
    train_records, test_records, split = split_records(
        records, TRAIN_FRAC, seed, track="artifact", group_duplicates=True
    )
    train = [r["sequence"] for r in train_records]
    test = [r["sequence"] for r in test_records]
    if not train or not test:
        raise ValueError(f"Nonempty artifact-grouped partitions required: {split}")
    bigram = NGramModel(train, 2, method="mkn")
    trigram = NGramModel(train, 3, method="mkn")
    logp_bi = np.array(paired_logprobs(bigram, test))
    logp_tri = np.array(paired_logprobs(trigram, test))
    diffs = logp_tri - logp_bi
    run = {
        "seed": seed,
        "mean_logloss_bigram": -float(logp_bi.mean()),
        "mean_logloss_trigram": -float(logp_tri.mean()),
        "mean_paired_diff": float(diffs.mean()),
        "perm_p": paired_permutation_p(diffs, n_permutations, seed),
        "n_tokens": int(diffs.size),
        "n_train_spans": len(train),
        "n_test_spans": len(test),
    }
    return run, diffs


def render_report(summary: dict) -> str:
    lines = [
        "Paired bigram vs trigram held-out log-loss inference",
        "",
        "Full-sequence per-token log-loss: every token position of every held-out",
        "span is scored under both modified Kneser-Ney models on identical",
        "artifact-grouped splits; this is NOT the single-mask restoration task.",
        "Paired difference d_i = log2 P_trigram - log2 P_bigram at the same",
        "position; a positive mean d means the trigram assigns higher probability",
        "(lower loss) than the bigram at identical positions.",
        "SD describes split variability, NOT confidence intervals.",
        "Overlapping tokens within a sequence are dependent (neighboring scores",
        "share context), so the permutation p assumes token-level exchangeability",
        "of the paired differences and should be read with that caveat.",
        "",
    ]
    agg = summary["aggregate"]
    bi = agg["mean_logloss_bigram"]
    tri = agg["mean_logloss_trigram"]
    diff = agg["mean_paired_diff"]
    lines.extend([
        f"Bigram log-loss: {bi['mean']:.4f} +/- {bi['sd']:.4f} bits/token",
        f"Trigram log-loss: {tri['mean']:.4f} +/- {tri['sd']:.4f} bits/token",
        f"Mean paired diff (trigram - bigram): {diff['mean']:+.5f} +/- "
        f"{diff['sd']:.5f} bits/token",
        "",
        "Permutation test (sign flips of d_i under exchangeability)",
        f"permutations per test: {summary['manifest']['n_permutations']}",
        "seed | n_tokens | mean d | perm p",
    ])
    for run in summary["runs"]:
        lines.append(
            f"{run['seed']} | {run['n_tokens']} | "
            f"{run['mean_paired_diff']:+.5f} | {run['perm_p']:.4f}"
        )
    pooled = summary["pooled"]
    lines.extend([
        "",
        f"Pooled across all runs: {pooled['n_tokens']} tokens, "
        f"mean d {pooled['mean_paired_diff']:+.5f}, "
        f"perm p {pooled['perm_p']:.4f}.",
        "A positive pooled mean d with small p supports the trigram assigning",
        "higher held-out probability than the bigram; interpret against the",
        "dependence caveat above.",
    ])
    return "\n".join(lines) + "\n"




def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Paired bigram-vs-trigram held-out log-loss with a paired "
        "permutation test."
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ngram_inference")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.permutations < 1:
        parser.error("--permutations must be positive")

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    seeds = list(range(args.seed, args.seed + args.repeats))

    runs = []
    all_diffs = []
    for seed in seeds:
        run, diffs = run_split(records, seed, args.permutations)
        runs.append(run)
        all_diffs.append(diffs)
    pooled_diffs = np.concatenate(all_diffs)
    pooled = {
        "n_tokens": int(pooled_diffs.size),
        "mean_paired_diff": float(pooled_diffs.mean()),
        "perm_p": paired_permutation_p(pooled_diffs, args.permutations, args.seed),
        "n_permutations": args.permutations,
    }
    aggregate = {
        key: mean_sd([run[key] for run in runs])
        for key in ("mean_logloss_bigram", "mean_logloss_trigram", "mean_paired_diff", "perm_p")
    }

    sources = sorted((ROOT / "src" / "arthanvesana").rglob("*.py")) + [Path(__file__).resolve()]
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "corpus_path": str(args.corpus.resolve()),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
            "python": platform.python_version(),
            "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy")},
            "seeds": seeds,
            "train_frac": TRAIN_FRAC,
            "gap_policy": "split",
            "known_direction_only": True,
            "n_spans": len(records),
            "n_inscriptions": len({r["inscription_id"] for r in records}),
            "n_permutations": args.permutations,
            "model": "modified Kneser-Ney bigram vs trigram; fixed, no tuning on evaluation splits",
            "paired_difference": "d_i = log2 P_trigram(token|context) - log2 P_bigram(token|context) at every held-out position",
        },
        "runs": runs,
        "aggregate": aggregate,
        "pooled": pooled,
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ngram_inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "ngram_inference_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
