from __future__ import annotations

import argparse
import json
import random
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
from arthanvesana.replicate.assoc import analyze_pairs, ranked_pairs
from arthanvesana.replicate.entropy2 import (
    chao_shen,
    shrinkage_entropy,
    unigram_counts,
)
from arthanvesana.replicate.llr import bigram_llr
from arthanvesana.replicate.metrics import (
    bootstrap_ci,
    ece,
    js_divergence,
    mrr,
)
from arthanvesana.replicate.region import site_records
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.replicate.segment import greedy_segmentation
from arthanvesana.replicate.segment_morfessor import (
    boundaries,
    boundary_f1,
    segment,
)
from arthanvesana.replicate.segment_morfessor import train as train_morfessor
from arthanvesana.stats.entropy import conditional_entropy, unigram_entropy
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import (
    sample_sequence,
    shuffled_corpus,
    split_records,
)

SEED = 0
OUT = ROOT / "outputs" / "upgrade"
FIG = OUT / "figures"


def artifact_clusters(records):
    groups = {}
    for record in records:
        artifact = record.get("artifact_group")
        key = ("artifact", tuple(artifact)) if artifact is not None else ("inscription", record["inscription_id"])
        groups.setdefault(key, []).append(record)
    return list(groups.values())


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
    if any("000" in s for s in seqs):
        raise ValueError("placeholder sign 000 leaked into analysis spans")
    if not train or not test:
        raise ValueError(f"Nonempty artifact-grouped partitions required: {split}")

    mkn_curve = [NGramModel(train, n, method="mkn").perplexity(test) for n in (1, 2, 3, 4, 5)]
    wb_curve = [NGramModel(train, n, method="wittenbell").perplexity(test) for n in (1, 2, 3, 4, 5)]

    counts = unigram_counts(seqs)
    entropy_est = {
        "plugin": unigram_entropy(seqs),
        "chao_shen": chao_shen(counts),
        "shrinkage": shrinkage_entropy(counts),
    }
    entropy_clusters = artifact_clusters(records)
    h2_boot = bootstrap_ci(
        entropy_clusters,
        lambda groups: conditional_entropy([r["sequence"] for group in groups for r in group], 1)[0],
        n_reps=200, seed=SEED,
    )
    h2_boot.update(
        original=conditional_entropy(seqs, 1)[0],
        interpretation="exploratory 95% percentile range; not bias-corrected",
        resampling_unit="artifact_group, falling back to inscription_id",
        n_clusters=len(entropy_clusters),
    )

    association = analyze_pairs(seqs, alpha=0.05, min_count=3, exact=True)
    association.update(
        family="all distinct observed ordered adjacent pairs in known-direction, gap-split spans; no count or LLR filtering before BH",
        test="one-sided Fisher exact (greater), Benjamini-Hochberg adjusted p-values (q)",
        caution="The family selects observed pairs and excludes unobserved pairs; overlapping windows are dependent. BH q thresholds are exploratory, not guaranteed FDR control or confirmation of linguistic units.",
    )
    ranked = [row for row in association["pairs"] if row["retained"]]
    llr2 = bigram_llr(seqs)
    llr_top = {pair for pair, _, _ in llr2[:20]}
    effect_top = {(r["pair"][0], r["pair"][1]) for r in ranked[:20]}

    restored, skipped = restoration_records(train_records, test_records)
    ranks = [r["rank"] for r in restored]
    restore_metrics = {
        "top_1": sum(r == 1 for r in ranks) / len(ranks) if ranks else 0.0,
        "mrr": mrr(ranks),
        "top_label_ece": ece([r["top_p"] for r in restored], [r["hit"] for r in restored]),
        "n_masked": len(restored),
        "n_skipped_oov": skipped,
        "n_oov": sum(r["rank"] is None for r in restored),
        "oov_policy": "included as failures; reciprocal rank zero",
    }
    restore_clusters = artifact_clusters(restored)
    mrr_boot = bootstrap_ci(
        restore_clusters,
        lambda groups: mrr([r["rank"] for group in groups for r in group]),
        n_reps=1000, seed=SEED,
    )
    mrr_boot.update(
        original=restore_metrics["mrr"],
        resampling_unit="artifact_group, falling back to inscription_id",
        n_clusters=len(restore_clusters),
        interpretation="95% cluster percentile range, conditional on fitted model and split",
    )

    sites = site_records(df, gap_policy="split", known_direction_only=True)
    dice_by_site = {}
    for site, rows in sites.items():
        top = ranked_pairs([r["sequence"] for r in rows], min_count=2)[:30]
        dice_by_site[site] = {tuple(r["pair"]): r["logdice"] for r in top}

    gen_model = NGramModel(train, 2, method="wittenbell")
    train_lens = [len(s) for s in train]
    gen_lens = random.Random(SEED).choices(train_lens, k=6000)
    generated = [
        sample_sequence(gen_model, max_len=L, seed=SEED + i)
        for i, L in enumerate(gen_lens)
    ]
    gen_uni = Counter(s for g in generated for s in g)
    gen_bi = Counter()
    for g in generated:
        for a, b in zip(g, g[1:]):
            gen_bi[(a, b)] += 1
    real_uni = unigram_counts(seqs)
    real_bi = Counter()
    for s in seqs:
        for a, b in zip(s, s[1:]):
            real_bi[(a, b)] += 1
    js_uni = js_divergence(real_uni, gen_uni)
    js_bi = js_divergence(real_bi, gen_bi)
    floor_uni = js_divergence(
        Counter(s for s in train for s in s),
        Counter(s for s in test for s in s),
    )
    floor_bi = js_divergence(
        Counter(p for s in train for p in zip(s, s[1:])),
        Counter(p for s in test for p in zip(s, s[1:])),
    )

    pair_score = {pair: v for pair, _, v in llr2}
    long_seqs = [s for s in seqs if 5 <= len(s) <= 12]
    mor_model = train_morfessor(seqs)
    agree, agree_null = [], []
    greedy_cuts, mor_cuts = [], []
    null_seqs = shuffled_corpus(long_seqs, seed=SEED, n_replicates=1)[0]
    for s, ns in zip(long_seqs, null_seqs):
        greedy, _ = greedy_segmentation(s, pair_score, 10.83)
        mor = segment(mor_model, s)
        agree.append(boundary_f1(greedy, mor))
        greedy_cuts.append(len(boundaries(greedy)))
        mor_cuts.append(len(boundaries(mor)))
        greedy_n, _ = greedy_segmentation(ns, pair_score, 10.83)
        mor_n = segment(mor_model, ns)
        agree_null.append(boundary_f1(greedy_n, mor_n))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    summary = {
        "seed": SEED,
        "split": split,
        "methods": {
            "gap_policy": "split", "known_direction_only": True,
            "model_unit": "contiguous observed span; model starts are not inscription boundaries",
            "restoration_boundaries": "record completeness respected",
            "segmentation": "descriptive in-sample algorithm agreement, not linguistic units",
            "regional_logdice": "descriptive full-span corpus ranking, not held-out inference",
            "generative_check": "descriptive full-span corpus comparison; train-test divergence is not a pure sampling floor",
        },
        "mkn_perplexity": mkn_curve,
        "wb_perplexity": wb_curve,
        "entropy_estimators": entropy_est,
        "h2_bootstrap": h2_boot,
        "association_inference": association,
        "effect_ranking": {
            "n_exploratory": len(ranked),
            "overlap_llr_top20": len(llr_top & effect_top),
            "top_by_npmi": [
                {
                    "pair": list(r["pair"]),
                    "count": r["count"],
                    "npmi": r["npmi"],
                    "logdice": r["logdice"],
                }
                for r in ranked[:20]
            ],
        },
        "restoration": restore_metrics,
        "restoration_mrr_bootstrap": mrr_boot,
        "regional_logdice_sites": sorted(dice_by_site),
        "generative_check": {
            "js_unigram": js_uni,
            "js_bigram": js_bi,
            "noise_floor_unigram": floor_uni,
            "noise_floor_bigram": floor_bi,
        },
        "segmentation_agreement": {
            "n_compared": len(agree),
            "mean_f1_real": mean(agree),
            "mean_f1_null": mean(agree_null),
            "mean_greedy_cuts": mean(greedy_cuts),
            "mean_morfessor_cuts": mean(mor_cuts),
        },
    }
    with open(OUT / "upgrade_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    make_figures(summary)

    lines = []
    lines.append("# Methods upgrade report")
    lines.append("")
    lines.append("## Smoothing: modified kneser-ney")
    lines.append("")
    lines.append("Perplexity n=1..5, MKN: " +
                 ", ".join(f"{v:.2f}" for v in mkn_curve))
    lines.append("Perplexity n=1..5, Witten-Bell: " +
                 ", ".join(f"{v:.2f}" for v in wb_curve))
    lines.append("Methods: known-direction records split at gaps, with artifact-, "
                 "inscription- and duplicate-linked holdout groups. Models use observed "
                 "spans; artificial model starts are not inscription boundaries. "
                 "Vocabulary-normalized cross-corpus conclusions are not warranted.")
    lines.append("")
    lines.append("## Entropy estimators")
    lines.append("")
    lines.append(f"H1 plugin {entropy_est['plugin']:.3f}, Chao-Shen "
                 f"{entropy_est['chao_shen']:.3f}, shrinkage "
                 f"{entropy_est['shrinkage']:.3f}.")
    lines.append(f"H2 original sample value: {h2_boot['original']:.3f}; "
                 f"exploratory cluster bootstrap 95% percentile range "
                 f"[{h2_boot['lo']:.3f}, {h2_boot['hi']:.3f}] "
                 f"(mean {h2_boot['mean']:.3f}).")
    lines.append("The range reflects resampling variability across artifact groups "
                 "conditional on this split, not bias-corrected uncertainty about "
                 "H2, and should not be read as confirmation of linguistic units.")
    lines.append("")
    lines.append("## Effect-size pair ranking")
    lines.append("")
    lines.append(f"{association['n_tested']} observed pairs tested with {association['test']}. "
                 f"{len(ranked)} meet exploratory q <= {association['alpha']} and "
                 f"count >= {association['min_count']}; overlap with LLR top-20: "
                 f"{len(llr_top & effect_top)}/20.")
    lines.append("Family: " + association["family"] + ".")
    lines.append(association["caution"])
    lines.append("All observed-family p-values and BH q-values (p_adj), including "
                 "pairs below the count threshold, are in upgrade_summary.json "
                 "under association_inference.pairs.")
    lines.append("Top by NPMI: " +
                 ", ".join(f"{r['pair'][0]}-{r['pair'][1]} ({r['npmi']:.2f})"
                           for r in ranked[:10]))
    lines.append("NPMI 1.00 entries are rare co-occurrence artifacts of small "
                 "counts admitted by min_count=3; no linguistic-unit reading "
                 "is implied.")
    lines.append("")
    lines.append("## Restoration metrics")
    lines.append("")
    lines.append(f"Top-1 {restore_metrics['top_1']:.3f}, MRR "
                 f"{restore_metrics['mrr']:.3f} "
                 f"(95% CI [{mrr_boot['lo']:.3f}, {mrr_boot['hi']:.3f}]), "
                 f"ECE {restore_metrics['top_label_ece']:.3f}.")
    lines.append("Restoration masks are evaluated within observed spans only; "
                 "OOV targets count as failures (reciprocal rank zero) rather "
                 "than being excluded.")
    lines.append("")
    lines.append("## Generative check")
    lines.append("")
    lines.append(f"JS divergence real vs bigram-generated: unigram "
                 f"{js_uni:.4f} (train-test reference {floor_uni:.4f}), bigram "
                 f"{js_bi:.4f} (train-test reference {floor_bi:.4f}). 0 = "
                 f"identical; the reference divergences include model-free "
                 f"split and site effects, so they overstate a pure sampling floor.")
    lines.append("")
    lines.append("## Segmentation agreement")
    lines.append("")
    seg = summary["segmentation_agreement"]
    lines.append(f"Greedy-LLR (descriptive score cutoff 10.83, not a p-value threshold) places "
                 f"{seg['mean_greedy_cuts']:.2f} cuts per span; "
                 f"Morfessor places {seg['mean_morfessor_cuts']:.2f}. "
                 f"Boundary F1: real {seg['mean_f1_real']:.3f}, shuffled "
                 f"{seg['mean_f1_null']:.3f} over {seg['n_compared']} "
                 f"spans of length 5-12.")
    lines.append("Both segmenters are fit on the same observed spans; agreement "
                 "measures algorithmic consistency, not recovered linguistic units.")
    lines.append("")
    with open(OUT / "upgrade_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


def make_figures(summary):
    xs = [1, 2, 3, 4, 5]
    plt.figure()
    plt.plot(xs, summary["wb_perplexity"], marker="o", label="witten-bell")
    plt.plot(xs, summary["mkn_perplexity"], marker="s", label="mod-kneser-ney")
    plt.xlabel("n-gram order")
    plt.ylabel("held-out perplexity")
    plt.title("Smoothing upgrade")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "mkn_curves.png", dpi=120)
    plt.close()

    est = summary["entropy_estimators"]
    plt.figure()
    plt.bar(list(est), list(est.values()))
    plt.ylabel("bits")
    plt.title("H1 by estimator")
    plt.tight_layout()
    plt.savefig(FIG / "entropy_est.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    argparse.ArgumentParser(description="Upgraded analyses on known-direction, gap-split spans with artifact-grouped holdout.").parse_args()
    main()
