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
from arthanvesana.replicate.clusters import (
    adjusted_rand,
    cluster_purity,
    hierarchical_labels,
    kmeans_sweep,
    purity_permutation_null,
)
from arthanvesana.replicate.embeddings import (
    cosine_neighbors,
    embedding_restoration_ranks,
    neighbor_jaccard,
    skipgram_restoration_ranks,
    train_embeddings,
    train_skipgram,
)
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.stats.position import positional_profile
from arthanvesana.stats.sampling import split_records

WINDOWS = (1, 2)
DIMS = (25, 50, 100)
SG_WINDOWS = (1, 2)
SG_DIMS = (25, 50)
K_RANGE = list(range(2, 13))
TOP_K = (1, 5, 10)
MODELS = ("bigram", "ppmi", "skipgram")


def _metrics(ranks):
    total = len(ranks)
    out = {
        f"top_{k}": sum(r is not None and r <= k for r in ranks) / total
        for k in TOP_K
    }
    out["mrr"] = sum(1.0 / r for r in ranks if r) / total
    out["n_masked"] = total
    out["n_oov"] = sum(r is None for r in ranks)
    return out


def _paired(first, base):
    out = {}
    for k in TOP_K:
        out[f"top_{k}"] = fmean(
            (a is not None and a <= k) - (b is not None and b <= k)
            for a, b in zip(first, base)
        )
    out["mrr"] = fmean(
        (1.0 / a if a else 0.0) - (1.0 / b if b else 0.0)
        for a, b in zip(first, base)
    )
    return out


def _summarize(values):
    return {
        "mean": fmean(values), "sd": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values), "n_runs": len(values),
    }


def _summarize_opt(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None, "n_runs": 0}
    return _summarize(values)


def _role_labels(records):
    complete = [r["sequence"] for r in records if r["start_complete"] and r["end_complete"]]
    roles = {}
    for row in positional_profile(complete, min_count=5):
        if row["p_begin"] >= 0.5:
            roles[row["sign"]] = "begin"
        elif row["p_end"] >= 0.5:
            roles[row["sign"]] = "end"
        else:
            roles[row["sign"]] = "middle"
    return roles


def _cluster_block(vectors, vocab, roles, seed=0):
    order = [s for s in vocab if s in roles]
    mat = vectors[[vocab.index(s) for s in order]]
    sweep = kmeans_sweep(mat, [k for k in K_RANGE if k < len(order)], seed)
    best = sweep["best_k"]
    role_map = {s: roles[s] for s in order}
    labels = sweep["labels"][best]
    hier = hierarchical_labels(mat, best)
    return {
        "n_signs": len(order),
        "sweep": sweep["sweep"],
        "best_k": best,
        "kmeans_labels": dict(zip(order, labels)),
        "kmeans_purity": cluster_purity(labels, role_map),
        "kmeans_null": purity_permutation_null(labels, role_map),
        "hierarchical_labels": dict(zip(order, hier)),
        "hierarchical_purity": cluster_purity(hier, role_map),
    }


def _plot_projection(fig_dir, name, points, order, roles):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"begin": "tab:blue", "middle": "tab:gray", "end": "tab:red"}
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for role, color in colors.items():
        idx = [i for i, s in enumerate(order) if roles.get(s) == role]
        if idx:
            plt.scatter(points[idx, 0], points[idx, 1], s=8, c=color, label=role)
    plt.legend(markerscale=3)
    plt.title(f"{name} colored by positional role")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.png", dpi=120)
    plt.close()


def render_report(summary):
    lines = [
        "Sign embeddings (PPMI/SVD + skip-gram) evaluation",
        "",
        "Symmetric co-occurrence windows over known-direction, gap-split spans.",
        "PPMI uses 0.75 context smoothing with truncated SVD; skip-gram is",
        "gensim Word2Vec (sg=1, negative=5, single worker, fixed seeds).",
        "Restoration scores candidates by reconstructed association with the",
        "observed in-window neighbors. Same artifact-grouped splits,",
        "same masked positions, and OOV-as-failure scoring as the bigram.",
        "SD describes split variability, NOT confidence intervals.",
        "",
        "PPMI config sweep on seed 0 (top-1):",
    ]
    for row in summary["config_sweep"]:
        lines.append(
            f"window {row['window']}, dim {row['dim']}: "
            f"ppmi {row['embedding_top1']:.4f} vs bigram {row['bigram_top1']:.4f}"
        )
    lines.append("Skip-gram config sweep on seed 0 (top-1):")
    for row in summary["skipgram_sweep"]:
        lines.append(
            f"window {row['window']}, dim {row['dim']}: "
            f"skipgram {row['embedding_top1']:.4f} vs bigram {row['bigram_top1']:.4f}"
        )
    lines.extend([
        f"Selected PPMI: window {summary['best_config']['window']}, "
        f"dim {summary['best_config']['dim']}; selected skip-gram: window "
        f"{summary['skipgram_best']['window']}, dim {summary['skipgram_best']['dim']}.",
        "",
        f"10-seed comparison ({summary['n_ok']}/{summary['n_runs']} successful runs):",
        "Model | Top-1 mean +/- SD | Top-5 mean | Top-10 mean | MRR mean",
    ])
    for model in MODELS:
        block = summary["aggregate"][model]
        lines.append(
            f"{model} | {block['top_1']['mean']:.4f} +/- {block['top_1']['sd']:.4f} | "
            f"{block['top_5']['mean']:.4f} | {block['top_10']['mean']:.4f} | "
            f"{block['mrr']['mean']:.4f}"
        )
    for name in ("paired_ppmi", "paired_skipgram", "paired_sg_ppmi"):
        delta = summary["aggregate"][name]["top_1"]
        lines.append(
            f"{name}: mean top-1 difference {delta['mean']:+.4f}; "
            f"wins/ties/losses {delta['n_positive']}/{delta['n_zero']}/{delta['n_negative']}."
        )
    lines.extend([
        "",
        "Clustering (full-data vectors, roles from both-ends-complete spans):",
    ])
    for model in ("ppmi", "skipgram"):
        block = summary["clustering"][model]
        null = block["kmeans_null"]
        lines.extend([
            f"{model}: best k={block['best_k']} over {block['n_signs']} signs; "
            f"k-means purity {block['kmeans_purity']['purity']:.3f} "
            f"(null mean {null['null_mean']:.3f}, p={null['empirical_p']:.3f}); "
            f"hierarchical purity {block['hierarchical_purity']['purity']:.3f}.",
        ])
    cross = summary["clustering"]["cross_method_ari"]
    stab = summary["clustering"]["stability_ari"]
    def _ari_text(value):
        return "n/a" if value is None else f"{value:.3f}"
    lines.extend([
        f"Cross-method ARI (PPMI vs skip-gram clusters): {cross:.3f}.",
        f"Split-stability ARI vs full-data clustering: PPMI {_ari_text(stab['ppmi']['mean'])}, "
        f"skip-gram {_ari_text(stab['skipgram']['mean'])}.",
        f"Neighbor stability (top-10 Jaccard): {summary['stability']['mean_jaccard']:.3f}.",
        "Purity near the null mean, or low ARI, means clusters should not be",
        "read as stable functional groups.",
        "",
        "Full per-run metrics, paired differences, split identities, cluster",
        "assignments, and neighborhoods are in embeddings_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="PPMI/SVD and skip-gram sign embeddings vs bigram")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "embeddings")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)

    seeds = list(range(args.seed, args.seed + args.repeats))
    train0, test0, _ = split_records(records, 0.8, seeds[0], track="artifact", group_duplicates=True)
    bigram0 = [r["rank"] for r in restoration_records(train0, test0, mask_length=1)[0]]
    sweep = []
    for window in WINDOWS:
        for dim in DIMS:
            emb = embedding_restoration_ranks(train0, test0, window, dim)
            assert len(emb) == len(bigram0)
            sweep.append({
                "window": window, "dim": dim,
                "embedding_top1": _metrics(emb)["top_1"],
                "bigram_top1": _metrics(bigram0)["top_1"],
            })
    best = max(sweep, key=lambda r: r["embedding_top1"])
    window, dim = best["window"], best["dim"]
    sg_sweep = []
    for sg_window in SG_WINDOWS:
        for sg_dim in SG_DIMS:
            ranks = skipgram_restoration_ranks(train0, test0, sg_window, sg_dim, args.seed)
            assert len(ranks) == len(bigram0)
            sg_sweep.append({
                "window": sg_window, "dim": sg_dim,
                "embedding_top1": _metrics(ranks)["top_1"],
                "bigram_top1": _metrics(bigram0)["top_1"],
            })
    sg_best = max(sg_sweep, key=lambda r: r["embedding_top1"])
    sg_window, sg_dim = sg_best["window"], sg_best["dim"]

    runs = []
    for seed in seeds:
        train, test, diagnostics = split_records(
            records, 0.8, seed, track="artifact", group_duplicates=True
        )
        bigram = [r["rank"] for r in restoration_records(train, test, mask_length=1)[0]]
        ppmi = embedding_restoration_ranks(train, test, window, dim)
        skipgram = skipgram_restoration_ranks(train, test, sg_window, sg_dim, seed)
        runs.append({
            "seed": seed, "split": diagnostics,
            "train_ids": sorted({r["inscription_id"] for r in train}),
            "test_ids": sorted({r["inscription_id"] for r in test}),
            "bigram": _metrics(bigram), "ppmi": _metrics(ppmi),
            "skipgram": _metrics(skipgram),
            "paired_ppmi": _paired(ppmi, bigram),
            "paired_skipgram": _paired(skipgram, bigram),
            "paired_sg_ppmi": _paired(skipgram, ppmi),
        })

    aggregate = {
        model: {k: _summarize([r[model][k] for r in runs]) for k in ("top_1", "top_5", "top_10", "mrr")}
        for model in MODELS
    }
    for name in ("paired_ppmi", "paired_skipgram", "paired_sg_ppmi"):
        block = {}
        for k in ("top_1", "top_5", "top_10", "mrr"):
            values = [r[name][k] for r in runs]
            block[k] = _summarize(values) | {
                "n_positive": sum(v > 0 for v in values),
                "n_zero": sum(v == 0 for v in values),
                "n_negative": sum(v < 0 for v in values),
            }
        aggregate[name] = block

    full_ppmi = train_embeddings([r["sequence"] for r in records], window, dim)
    full_sg = train_skipgram(
        [r["sequence"] for r in records], sg_window, sg_dim, args.seed
    )
    full_neighbors = {
        sign: [n for n, _ in cosine_neighbors(full_ppmi, sign)]
        for sign in full_ppmi["vocab"]
    }
    overlaps = []
    stability_ari = {"ppmi": [], "skipgram": []}
    for seed in seeds:
        train, _, _ = split_records(records, 0.8, seed, track="artifact", group_duplicates=True)
        split_model = train_embeddings([r["sequence"] for r in train], window, dim)
        for sign in split_model["vocab"]:
            if sign in full_neighbors:
                overlaps.append(neighbor_jaccard(
                    [n for n, _ in cosine_neighbors(split_model, sign)],
                    full_neighbors[sign],
                ))

    roles = _role_labels(records)
    clustering = {}
    for name, model in (("ppmi", full_ppmi), ("skipgram", full_sg)):
        block = _cluster_block(model["vectors"], model["vocab"], roles, args.seed)
        clustering[name] = block
    ppmi_order = [s for s in full_ppmi["vocab"] if s in roles]
    sg_order = [s for s in full_sg["vocab"] if s in roles]
    shared = [s for s in ppmi_order if s in set(sg_order)]
    clustering["cross_method_ari"] = adjusted_rand(
        [clustering["ppmi"]["kmeans_labels"][s] for s in shared],
        [clustering["skipgram"]["kmeans_labels"][s] for s in shared],
    )
    for seed in seeds:
        train, _, _ = split_records(records, 0.8, seed, track="artifact", group_duplicates=True)
        seqs = [r["sequence"] for r in train]
        for name, trainer, kwargs in (
            ("ppmi", train_embeddings, {"window": window, "dim": dim}),
            ("skipgram", train_skipgram, {"window": sg_window, "dim": sg_dim, "seed": seed}),
        ):
            split_model = trainer(seqs, **kwargs)
            common = [s for s in split_model["vocab"] if s in roles]
            if len(common) < 3:
                stability_ari[name].append(None)
                continue
            mat = split_model["vectors"][[split_model["vocab"].index(s) for s in common]]
            best_k = clustering[name]["best_k"]
            k = min(best_k, len(common) - 1)
            split_labels = kmeans_sweep(mat, [k], seed)["labels"][k]
            ref = [clustering[name]["kmeans_labels"][s] for s in common]
            stability_ari[name].append(adjusted_rand(split_labels, ref))
    clustering["stability_ari"] = {
        name: _summarize_opt(values) for name, values in stability_ari.items()
    }

    import umap
    from sklearn.decomposition import PCA

    fig_dir = args.output / "figures"
    for name, model in (("ppmi", full_ppmi), ("skipgram", full_sg)):
        order = list(model["vocab"])
        _plot_projection(
            fig_dir, f"pca_{name}",
            PCA(n_components=2, random_state=args.seed).fit_transform(model["vectors"]),
            order, roles,
        )
        _plot_projection(
            fig_dir, f"umap_{name}",
            umap.UMAP(n_components=2, random_state=args.seed).fit_transform(model["vectors"]),
            order, roles,
        )

    sources = sorted((ROOT / "src" / "arthanvesana").rglob("*.py")) + [Path(__file__).resolve()]
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy", "gensim", "scikit-learn")},
            "seeds": seeds, "gap_policy": "split", "known_direction_only": True,
            "config_grid": {"windows": list(WINDOWS), "dims": list(DIMS)},
            "skipgram_grid": {"windows": list(SG_WINDOWS), "dims": list(SG_DIMS)},
            "k_range": K_RANGE,
        },
        "config_sweep": sweep,
        "best_config": {"window": window, "dim": dim},
        "skipgram_sweep": sg_sweep,
        "skipgram_best": {"window": sg_window, "dim": sg_dim},
        "n_runs": len(runs), "n_ok": len(runs),
        "runs": runs, "aggregate": aggregate,
        "stability": {
            "mean_jaccard": fmean(overlaps), "min_jaccard": min(overlaps),
            "n_signs": len(full_neighbors),
        },
        "full_data_neighborhoods": full_neighbors,
        "roles": roles,
        "clustering": clustering,
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "embeddings_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "embeddings_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()
