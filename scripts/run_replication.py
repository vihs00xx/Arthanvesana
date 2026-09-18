from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records
from arthanvesana.replicate.llr import bigram_llr, trigram_llr
from arthanvesana.replicate.region import cross_perplexity, site_records
from arthanvesana.replicate.restore import restoration_accuracy
from arthanvesana.replicate.segment import segmentation_merge_counts
from arthanvesana.replicate.zipf import (
    beginner_ender_coverage,
    mandelbrot_fit,
    rank_frequencies,
)
from arthanvesana.stats.entropy import conditional_entropy, mutual_information, unigram_entropy
from arthanvesana.stats.ngrams import NGramModel, top_ngrams
from arthanvesana.stats.sampling import shuffled_corpus, split_records

SEED = 0
OUT = ROOT / "outputs" / "replication"
FIG = OUT / "figures"

YADAV_PPL = [68.82, 26.69, 26.09, 25.26, 25.26]
EBUDS_H1 = 6.68
EBUDS_MI = 2.24


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        ROOT / "data" / "processed" / "corpus.csv",
        encoding="utf-8",
        dtype={"sign_code": str},
    )
    records = analysis_records(df, gap_policy="split", known_direction_only=True)
    train_records, test_records, split = split_records(records, track="artifact", seed=SEED)
    seqs = [r["sequence"] for r in records]
    train = [r["sequence"] for r in train_records]
    test = [r["sequence"] for r in test_records]
    complete_seqs = [r["sequence"] for r in records if r["start_complete"] and r["end_complete"]]
    if any("000" in s for s in seqs):
        raise ValueError("placeholder sign 000 leaked into analysis spans")
    if not train or not test:
        raise ValueError(f"Nonempty artifact-grouped partitions required: {split}")
    vocab_size = len({sign for seq in seqs for sign in seq})

    h1 = unigram_entropy(seqs)
    h2, _ = conditional_entropy(seqs, 1)
    mi = mutual_information(seqs)

    ppl = {}
    for method in ("laplace", "wittenbell", "interp"):
        curve = []
        for n in (1, 2, 3, 4, 5):
            curve.append(NGramModel(train, n, method=method).perplexity(test))
        ppl[method] = curve

    llr2 = bigram_llr(seqs)
    llr3 = trigram_llr(seqs)
    freq2 = [tuple(t) for t, _ in top_ngrams(seqs, 2, 20)]
    sig2 = [pair for pair, _, _ in llr2[:20]]
    overlap = len(set(freq2) & set(sig2))

    freqs = rank_frequencies(seqs)
    zm = mandelbrot_fit(freqs)
    be = beginner_ender_coverage(complete_seqs)

    restore = restoration_accuracy(train_records, test_records)

    sites = site_records(df, gap_policy="split", known_direction_only=True)
    region_report = cross_perplexity(
        sites, track="artifact", seed=SEED, return_report=True,
        common_vocabulary=True, equal_train_size=True,
    )
    region = region_report["matrix"]

    pair_score = {pair: v for pair, _, v in llr2}
    long_seqs = [s for s in seqs if len(s) >= 10]
    real_h = segmentation_merge_counts(long_seqs, pair_score)
    null_h = segmentation_merge_counts(
        shuffled_corpus(long_seqs, seed=SEED, n_replicates=1)[0], pair_score
    )

    def ratio(hs):
        return sum(r / L for L, r in hs) / len(hs) if hs else 0.0

    summary = {
        "seed": SEED,
        "split": split,
        "methods": {
            "gap_policy": "split", "known_direction_only": True,
            "model_unit": "contiguous observed span; model starts are not inscription boundaries",
            "positional_subset": "both start_complete and end_complete",
            "n_complete_spans": len(complete_seqs),
            "mi": "joint adjacent-pair marginals, not H1 minus H2",
            "published_comparisons": "descriptive only; different corpora and protocols",
        },
        "r1_entropy": {
            "ours": {"h1": h1, "mi": mi, "h2": h2, "vocab": vocab_size,
                     "uniform_baseline": math.log2(vocab_size)},
            "ebuds_published": {"h1": EBUDS_H1, "mi": EBUDS_MI,
                                "vocab": 377,
                                "uniform_baseline": math.log2(377)},
        },
        "r2_perplexity": ppl,
        "r2_yadav_published": YADAV_PPL,
        "r3_llr": {
            "freq_sig_overlap_top20": overlap,
            "top_sig_bigrams": [
                [list(p), c, v] for p, c, v in llr2[:20]
            ],
            "top_sig_trigrams": [
                [list(t), c, v] for t, c, v in llr3[:20]
            ],
        },
        "r4_zipf": {
            "a": zm["a"],
            "b": zm["b"],
            "c": zm["c"],
            "r_squared": zm["r_squared"],
            "beginner_80": be["beginner_80"],
            "ender_80": be["ender_80"],
            "yadav_beginner_80": 82,
            "yadav_ender_80": 23,
        },
        "r5_restoration": restore,
        "r5_yadav_top1": 0.75,
        "r6_region": region,
        "r6_region_report": region_report,
        "r7_segmentation": {
            "n_long": len(long_seqs),
            "n_long_complete_both_ends": sum(
                r["start_complete"] and r["end_complete"]
                for r in records if len(r["sequence"]) >= 10
            ),
            "mean_merges_per_length_real": ratio(real_h),
            "mean_merges_per_length_null": ratio(null_h),
            "merge_counts_real": real_h,
            "merge_counts_null": null_h,
        },
    }
    with open(OUT / "replication_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    make_figures(summary, freqs, zm)

    lines = []
    lines.append("# Replication sprint report")
    lines.append("")
    lines.append(f"Corpus: ICIT-derived, {len(seqs)} known-direction observed spans, "
                 f"{vocab_size} signs. Published checkpoints use the EBUDS corpus "
                 "(Mahadevan-derived, 377 signs appearing).")
    lines.append("")
    lines.append("Methods: known-direction records split at gaps, with artifact-, "
                 "inscription- and duplicate-linked holdout groups. Models use "
                 "contiguous observed spans; artificial model starts are not "
                 f"inscription boundaries. Positional coverage uses {len(complete_seqs)} "
                 "records with both ends complete. Published checkpoints are "
                 "descriptive only, from different corpora and protocols.")
    lines.append("")
    lines.append("## R1 entropy vs languages")
    lines.append("")
    lines.append(f"Ours: H1 {h1:.2f}, MI {mi:.2f}, H2 {h2:.2f} "
                 f"(uniform baseline {math.log2(713):.2f}).")
    lines.append(f"EBUDS published: H1 {EBUDS_H1}, MI {EBUDS_MI} "
                 f"(uniform baseline {math.log2(377):.2f}).")
    lines.append("MI differences across corpora with different inventories and "
                 "length distributions are descriptive; they do not license "
                 "cross-corpus linguistic claims.")
    lines.append("")
    lines.append("## R2 perplexity by order and smoothing")
    lines.append("")
    lines.append("| n | laplace | witten-bell | interp | yadav |")
    lines.append("|---|---|---|---|---|")
    for i in range(5):
        lines.append(f"| {i + 1} | {ppl['laplace'][i]:.2f} | "
                     f"{ppl['wittenbell'][i]:.2f} | {ppl['interp'][i]:.2f} | "
                     f"{YADAV_PPL[i]:.2f} |")
    lines.append("")
    lines.append(f"Best bigram/trigram (interp): {ppl['interp'][1]:.2f} / "
                 f"{ppl['interp'][2]:.2f}. Vocabulary-size-normalized comparisons "
                 "across corpora are not reported; perplexity levels are not "
                 "comparable across different sign inventories and protocols.")
    lines.append("")
    lines.append("## R3 significant vs frequent pairs")
    lines.append("")
    lines.append(f"Overlap of top-20 frequent and top-20 LLR-significant "
                 f"bigrams: {overlap}/20.")
    lines.append("Top LLR bigrams: " +
                 ", ".join(f"{p[0]}-{p[1]}" for p, _, _ in llr2[:10]))
    lines.append("")
    lines.append("## R4 beginners, enders, zipf")
    lines.append("")
    lines.append(f"Signs covering 80%: beginners {be['beginner_80']} "
                 f"(published 82), enders {be['ender_80']} (published 23).")
    lines.append(f"Zipf-Mandelbrot fit R2 = {zm['r_squared']:.4f} "
                 f"(a={zm['a']:.2f}, b={zm['b']:.3f}, c={zm['c']:.2f}).")
    lines.append("")
    lines.append("## R5 restoration")
    lines.append("")
    lines.append(f"Bigram fill-in accuracy: top-1 {restore['top_1']:.3f}, "
                 f"top-5 {restore['top_5']:.3f}, top-10 {restore['top_10']:.3f} "
                 f"over {restore['n_masked']} masked positions "
                 f"(published top-1 ~0.75).")
    lines.append("")
    lines.append("## R6 regional cross-perplexity")
    lines.append("")
    for tr, row in region.items():
        lines.append(tr + ": " + ", ".join(f"{k}={v:.1f}" for k, v in row.items()))
    lines.append("")
    lines.append("R6 uses equal-size artifact-grouped training sets with a common "
                 "vocabulary; held-out cells are comparable within this protocol, "
                 "not with published scores. Site-level conclusions are descriptive.")
    lines.append("")
    lines.append("## R7 segmentation merge counts")
    lines.append("")
    lines.append(f"{len(long_seqs)} known-direction spans of length >= 10. Mean merge "
                 f"rounds per unit length: real "
                 f"{summary['r7_segmentation']['mean_merges_per_length_real']:.3f}, "
                 f"shuffled "
                 f"{summary['r7_segmentation']['mean_merges_per_length_null']:.3f}.")
    lines.append("Merge counts are NOT tree heights; they do not identify "
                 "linguistic units.")
    lines.append("")
    with open(OUT / "replication_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def make_figures(summary, freqs, zm):
    xs = [1, 2, 3, 4, 5]
    plt.figure()
    for method in ("laplace", "wittenbell", "interp"):
        plt.plot(xs, summary["r2_perplexity"][method], marker="o", label=method)
    plt.plot(xs, summary["r2_yadav_published"], marker="s", linestyle="--",
             label="yadav-2010")
    plt.xlabel("n-gram order")
    plt.ylabel("held-out perplexity")
    plt.title("Perplexity curves vs published")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "ppl_curves.png", dpi=120)
    plt.close()

    ranks = np.arange(1, len(freqs) + 1)
    plt.figure()
    plt.loglog(ranks, freqs, marker=".", linestyle="none", label="observed")
    plt.loglog(ranks, zm["predicted"], label="mandelbrot fit")
    plt.xlabel("rank")
    plt.ylabel("frequency")
    plt.title("Zipf-Mandelbrot fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "zipf_fit.png", dpi=120)
    plt.close()

    region = summary["r6_region"]
    names = sorted(region)
    mat = np.array([[region[tr][te] for te in names] for tr in names])
    plt.figure()
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="cross-perplexity")
    plt.xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("test site")
    plt.ylabel("train site")
    plt.title("Regional cross-perplexity (bigram)")
    plt.tight_layout()
    plt.savefig(FIG / "region.png", dpi=120)
    plt.close()

    real = summary["r7_segmentation"]["merge_counts_real"]
    null = summary["r7_segmentation"]["merge_counts_null"]
    plt.figure()
    if real:
        plt.scatter([L for L, _ in real], [r for _, r in real], label="real")
    if null:
        plt.scatter([L for L, _ in null], [r for _, r in null], label="shuffled")
    plt.xlabel("inscription length")
    plt.ylabel("merge rounds")
    plt.title("Segmentation merge counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "heights.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description="Replication analyses on known-direction, gap-split spans with artifact-grouped holdout.").parse_args()
    main()
