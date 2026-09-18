from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records
from arthanvesana.stats.entropy import conditional_entropy, unigram_entropy
from arthanvesana.stats.ngrams import NGramModel, top_ngrams
from arthanvesana.stats.position import positional_profile
from arthanvesana.stats.sampling import (
    deduplicated,
    shuffled_corpus,
    split_records,
)

SEED = 0
OUT = ROOT / "outputs" / "stats"
FIG = OUT / "figures"


def perplexity_by_length(models, test):
    buckets: dict = {}
    for seq in test:
        buckets.setdefault(len(seq), []).append(seq)
    rows = []
    for length in sorted(buckets):
        row = {"length": length, "n": len(buckets[length])}
        for name, model in models.items():
            row[name] = model.perplexity(buckets[length])
        rows.append(row)
    return rows


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
    if any("000" in seq for seq in seqs):
        raise ValueError("placeholder sign 000 leaked into analysis spans")
    if not train or not test:
        raise ValueError(f"Nonempty artifact-grouped partitions required: {split}")

    uniq = deduplicated(seqs)
    strain, stest = deduplicated(train), deduplicated(test)

    models = {f"{n}gram": NGramModel(train, n) for n in (1, 2, 3)}
    smodels = {f"{n}gram": NGramModel(strain, n) for n in (1, 2, 3)}
    perplexities = {name: m.perplexity(test) for name, m in models.items()}
    perplexities_dedup = {name: m.perplexity(stest) for name, m in smodels.items()}
    by_length = perplexity_by_length(models, test)

    h1 = unigram_entropy(seqs)
    h2, h2mm = conditional_entropy(seqs, 1)
    h3, h3mm = conditional_entropy(seqs, 2)

    nulls = shuffled_corpus(seqs, seed=SEED, n_replicates=20)
    null_h1 = [unigram_entropy(n) for n in nulls]
    null_h2 = [conditional_entropy(n, 1)[0] for n in nulls]
    null_h3 = [conditional_entropy(n, 2)[0] for n in nulls]

    def mean_sd(xs):
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return m, v**0.5

    null_stats = {
        "h1": mean_sd(null_h1),
        "h2": mean_sd(null_h2),
        "h3": mean_sd(null_h3),
    }

    profile = positional_profile(complete_seqs, min_count=5)
    prefix_like = sorted(profile, key=lambda r: -r["p_begin"])[:10]
    suffix_like = sorted(profile, key=lambda r: -r["p_end"])[:10]

    summary = {
        "seed": SEED,
        "methods": {
            "gap_policy": "split",
            "known_direction_only": True,
            "model_unit": "contiguous observed span; model starts are not inscription boundaries",
            "positional_subset": "both start_complete and end_complete",
            "n_complete_spans": len(complete_seqs),
            "deduplication": "within each artifact-grouped partition, after splitting",
        },
        "split": split,
        "split_dedup": {"source_split": split, "n_train": len(strain), "n_test": len(stest)},
        "n_train": len(train),
        "n_test": len(test),
        "n_unique_sequences": len(uniq),
        "perplexity": perplexities,
        "perplexity_dedup": perplexities_dedup,
        "perplexity_by_length": by_length,
        "entropy": {
            "h1": h1,
            "h2_plugin": h2,
            "h2_mm": h2mm,
            "h3_plugin": h3,
            "h3_mm": h3mm,
        },
        "null_entropy": {
            k: {"mean": m, "sd": s} for k, (m, s) in null_stats.items()
        },
        "top_unigrams": top_ngrams(seqs, 1, 20),
        "top_bigrams": top_ngrams(seqs, 2, 20),
        "top_trigrams": [list(t) + [c] for t, c in top_ngrams(seqs, 3, 20)],
        "prefix_like": prefix_like,
        "suffix_like": suffix_like,
    }
    with open(OUT / "stats_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    make_figures(seqs, complete_seqs, summary)

    lines = []
    lines.append("# Phase 2 — statistical baseline report")
    lines.append("")
    lines.append(f"Train spans: {len(train)}; test spans: {len(test)}; "
                 f"unique sequences: {len(uniq)}.")
    lines.append("Methods: known-direction records split at missing signs; "
                 "artifact-, inscription- and duplicate-linked records stay together. "
                 "Models use contiguous observed spans, with artificial model starts, "
                 "not inferred inscription boundaries. Deduplication is within the same partitions.")
    lines.append("")
    lines.append("## Held-out perplexity (Laplace k=1)")
    lines.append("")
    lines.append("| model | all data | deduplicated |")
    lines.append("|---|---|---|")
    for name in ("1gram", "2gram", "3gram"):
        lines.append(f"| {name} | {perplexities[name]:.2f} | "
                     f"{perplexities_dedup[name]:.2f} |")
    lines.append("")
    lines.append("## Entropy (bits)")
    lines.append("")
    lines.append(f"H1 = {h1:.3f}; H(s2|s1) = {h2:.3f} (MM {h2mm:.3f}); "
                 f"H(s3|s1,s2) = {h3:.3f} (MM {h3mm:.3f}).")
    lines.append("")
    m1, s1 = null_stats["h1"]
    m2, s2 = null_stats["h2"]
    m3, s3 = null_stats["h3"]
    lines.append("Shuffled nulls (20 replicates, mean +/- sd): "
                 f"H1 {m1:.3f} +/- {s1:.4f}; H2 {m2:.3f} +/- {s2:.4f}; "
                 f"H3 {m3:.3f} +/- {s3:.4f}.")
    lines.append("Unigram entropy is shuffle-invariant by construction. "
                 "Within-span shuffles provide a descriptive order control, "
                 "not evidence of linguistic units.")
    lines.append("")
    lines.append("## Positional extremes (min 5 occurrences)")
    lines.append("")
    lines.append("Most beginning-biased: " +
                 ", ".join(f"{r['sign']} ({r['p_begin']:.2f})" for r in prefix_like))
    lines.append("")
    lines.append("Most end-biased: " +
                 ", ".join(f"{r['sign']} ({r['p_end']:.2f})" for r in suffix_like))
    lines.append("")
    lines.append(f"Positional analyses use only {len(complete_seqs)} known-direction "
                 "records with both ends complete; split-span edges are excluded. "
                 "Length-1 inscriptions count toward both beginning and end. "
                 "Beginning/end bias does not establish prefixes or suffixes.")
    lines.append("")
    with open(OUT / "stats_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def transition_probabilities(seqs, signs):
    pairs = Counter(pair for seq in seqs for pair in zip(seq, seq[1:]))
    totals = Counter()
    for (source, _), count in pairs.items():
        totals[source] += count
    return [
        [pairs[(a, b)] / totals[a] if totals[a] else 0.0 for b in signs]
        for a in signs
    ]


def make_figures(seqs, complete_seqs, summary):
    uni = Counter(s for seq in seqs for s in seq)
    top_signs = [s for s, _ in uni.most_common(30)]

    counts = sorted(uni.values(), reverse=True)
    plt.figure()
    plt.loglog(range(1, len(counts) + 1), counts, marker=".", linestyle="none")
    plt.xlabel("rank")
    plt.ylabel("frequency")
    plt.title("Sign rank-frequency (Zipf)")
    plt.tight_layout()
    plt.savefig(FIG / "zipf.png", dpi=120)
    plt.close()

    import numpy as np

    mat = np.asarray(transition_probabilities(seqs, top_signs))
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="P(next | current)")
    plt.xticks(range(len(top_signs)), top_signs, rotation=90, fontsize=7)
    plt.yticks(range(len(top_signs)), top_signs, fontsize=7)
    plt.xlabel("next sign")
    plt.ylabel("current sign")
    plt.title("Bigram transition matrix (top 30 signs)")
    plt.tight_layout()
    plt.savefig(FIG / "transitions.png", dpi=120)
    plt.close()

    top25 = [s for s, _ in uni.most_common(25)]
    rows = [r for r in positional_profile(complete_seqs, 5) if r["sign"] in top25]
    order = {s: i for i, s in enumerate(top25)}
    rows.sort(key=lambda r: order[r["sign"]])
    pb = [r["p_begin"] for r in rows]
    pm = [r["p_middle"] for r in rows]
    pe = [r["p_end"] for r in rows]
    x = np.arange(len(rows))
    plt.figure(figsize=(12, 4))
    plt.bar(x - 0.25, pb, width=0.25, label="begin")
    plt.bar(x, pm, width=0.25, label="middle")
    plt.bar(x + 0.25, pe, width=0.25, label="end")
    plt.xticks(x, [r["sign"] for r in rows], rotation=90, fontsize=7)
    plt.ylabel("fraction of positional slots")
    plt.title("Positional profile (both ends complete; top 25 signs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "positional.png", dpi=120)
    plt.close()

    names = ["1gram", "2gram", "3gram"]
    vals = [summary["perplexity"][n] for n in names]
    dvals = [summary["perplexity_dedup"][n] for n in names]
    x = np.arange(len(names))
    plt.figure()
    plt.bar(x - 0.2, vals, width=0.4, label="all data")
    plt.bar(x + 0.2, dvals, width=0.4, label="deduplicated")
    plt.xticks(x, names)
    plt.ylabel("perplexity")
    plt.title("Held-out perplexity by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "perplexity.png", dpi=120)
    plt.close()

    labels = ["H1", "H2", "H3"]
    real = [summary["entropy"]["h1"], summary["entropy"]["h2_plugin"], summary["entropy"]["h3_plugin"]]
    nullm = [summary["null_entropy"]["h1"]["mean"], summary["null_entropy"]["h2"]["mean"], summary["null_entropy"]["h3"]["mean"]]
    nulls = [summary["null_entropy"]["h1"]["sd"], summary["null_entropy"]["h2"]["sd"], summary["null_entropy"]["h3"]["sd"]]
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x - 0.2, real, width=0.4, label="real corpus")
    plt.bar(x + 0.2, nullm, width=0.4, yerr=nulls, capsize=4, label="shuffled null")
    plt.xticks(x, labels)
    plt.ylabel("bits")
    plt.title("Entropy: real vs shuffled")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "entropy_null.png", dpi=120)
    plt.close()

    by_len = summary["perplexity_by_length"]
    xs = [r["length"] for r in by_len]
    plt.figure()
    for n in names:
        plt.plot(xs, [r[n] for r in by_len], marker="o", label=n)
    plt.xlabel("test observed span length")
    plt.ylabel("perplexity")
    plt.title("Perplexity by observed span length")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "perplexity_by_length.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description="Statistical baselines on known-direction, gap-split spans with artifact-grouped holdout.").parse_args()
    main()
