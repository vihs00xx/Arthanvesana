"""Methods upgrade runner. Usage from repo root: .venv\\Scripts\\python scripts\\run_upgrade.py
Writes outputs/upgrade/upgrade_summary.json, upgrade_report.md and figures:
Modified Kneser-Ney perplexity, modern entropy estimates, effect-size pair
ranking, restoration metrics with calibration, regional log-Dice comparison,
Morfessor segmentation agreement, and generative predictive checks.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import inscription_sequences
from arthanvesana.replicate.assoc import ranked_pairs
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
    normalized_perplexity,
)
from arthanvesana.replicate.region import site_sequences
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
    train_test_split,
)

SEED = 0
OUT = ROOT / "outputs" / "upgrade"
FIG = OUT / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        ROOT / "data" / "processed" / "corpus.csv",
        encoding="utf-8",
        dtype={"sign_code": str},
    )
    seqs = inscription_sequences(df)
    if any("000" in s for s in seqs):
        raise ValueError("placeholder sign 000 leaked into gated sequences")
    train, test = train_test_split(seqs, train_frac=0.8, seed=SEED)

    mkn_curve = [NGramModel(train, n, method="mkn").perplexity(test) for n in (1, 2, 3, 4, 5)]
    wb_curve = [NGramModel(train, n, method="wittenbell").perplexity(test) for n in (1, 2, 3, 4, 5)]

    counts = unigram_counts(seqs)
    entropy_est = {
        "plugin": unigram_entropy(seqs),
        "chao_shen": chao_shen(counts),
        "shrinkage": shrinkage_entropy(counts),
    }
    h2_boot = bootstrap_ci(
        seqs, lambda s: conditional_entropy(s, 1)[0], n_reps=200, seed=SEED
    )

    ranked = ranked_pairs(seqs)
    llr2 = bigram_llr(seqs)
    llr_top = {pair for pair, _, _ in llr2[:20]}
    effect_top = {(r["pair"][0], r["pair"][1]) for r in ranked[:20]}

    records, skipped = restoration_records(train, test)
    ranks = [r["rank"] for r in records]
    restore_metrics = {
        "top_1": sum(1 for r in ranks if r == 1) / len(ranks),
        "mrr": mrr(ranks),
        "ece": ece([r["top_p"] for r in records], [r["hit"] for r in records]),
        "n_masked": len(records),
        "n_skipped_oov": skipped,
    }
    mrr_boot = bootstrap_ci(records, lambda s: mrr([r["rank"] for r in s]), n_reps=1000, seed=SEED)

    sites = site_sequences(df)
    dice_by_site = {}
    for site, sseqs in sites.items():
        top = ranked_pairs(sseqs, min_count=2)[:30]
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
        "mkn_perplexity": mkn_curve,
        "wb_perplexity": wb_curve,
        "mkn_normalized_trigram": normalized_perplexity(mkn_curve[2], 713),
        "entropy_estimators": entropy_est,
        "h2_bootstrap": h2_boot,
        "effect_ranking": {
            "n_significant": len(ranked),
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
    lines.append(f"MKN trigram normalized: "
                 f"{summary['mkn_normalized_trigram']:.4f} (interp was 0.0678).")
    lines.append("")
    lines.append("## Entropy estimators")
    lines.append("")
    lines.append(f"H1 plugin {entropy_est['plugin']:.3f}, Chao-Shen "
                 f"{entropy_est['chao_shen']:.3f}, shrinkage "
                 f"{entropy_est['shrinkage']:.3f}.")
    lines.append(f"H2 bootstrap 95% CI: [{h2_boot['lo']:.3f}, "
                 f"{h2_boot['hi']:.3f}] (mean {h2_boot['mean']:.3f}).")
    lines.append("Note: the bootstrap mean sits below the full-sample H2 "
                 "(3.48) because resampling shrinks the effective sample and "
                 "plugin entropy underestimates more on smaller samples. The "
                 "interval reflects precision of the estimator, not accuracy.")
    lines.append("")
    lines.append("## Effect-size pair ranking")
    lines.append("")
    lines.append(f"{len(ranked)} LLR-significant pairs (p<0.001); overlap "
                 f"with LLR top-20: {len(llr_top & effect_top)}/20.")
    lines.append("Top by NPMI: " +
                 ", ".join(f"{r['pair'][0]}-{r['pair'][1]} ({r['npmi']:.2f})"
                           for r in ranked[:10]))
    lines.append("NPMI 1.00 tops are rare exclusive bonds (signs occurring "
                 "only together); min_count=3 admits them by design.")
    lines.append("")
    lines.append("## Restoration metrics")
    lines.append("")
    lines.append(f"Top-1 {restore_metrics['top_1']:.3f}, MRR "
                 f"{restore_metrics['mrr']:.3f} "
                 f"(95% CI [{mrr_boot['lo']:.3f}, {mrr_boot['hi']:.3f}]), "
                 f"ECE {restore_metrics['ece']:.3f}.")
    lines.append("")
    lines.append("## Generative check")
    lines.append("")
    lines.append(f"JS divergence real vs bigram-generated: unigram "
                 f"{js_uni:.4f} (noise floor {floor_uni:.4f}), bigram "
                 f"{js_bi:.4f} (noise floor {floor_bi:.4f}). 0 = identical; "
                 f"the floors show how much divergence finite sampling alone "
                 f"produces.")
    lines.append("")
    lines.append("## Segmentation agreement")
    lines.append("")
    seg = summary["segmentation_agreement"]
    lines.append(f"Greedy-LLR (p<0.001 threshold) places "
                 f"{seg['mean_greedy_cuts']:.2f} cuts per inscription; "
                 f"Morfessor places {seg['mean_morfessor_cuts']:.2f}. "
                 f"Boundary F1: real {seg['mean_f1_real']:.3f}, shuffled "
                 f"{seg['mean_f1_null']:.3f} over {seg['n_compared']} "
                 f"inscriptions.")
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
    main()
