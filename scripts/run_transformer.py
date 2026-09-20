"""Masked-sign transformer vs bigram under FULLY NESTED model selection.

For every outer grouped split: reserve the outer test partition; carve a
inner fit/validation split from the outer TRAIN partition only; evaluate every
grid configuration on inner validation; select the configuration AND stopping
epoch by inner-validation top-1; then refit the selected configuration on the
whole outer-train partition for that epoch count (no outer-test access, no
outer-test early stopping); evaluate once on the outer test. Configuration
selection is therefore reproducible and never touches the outer test records.

The transformer vocabulary is built from fitting data only; unseen signs are
<UNK> context tokens, <UNK> is never a candidate, and OOV test targets are
failures. SD describes split variability, NOT confidence intervals.
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
from arthanvesana.replicate.masked_lm import masked_lm_ranks, train_masked_lm
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.stats.sampling import split_records

GRID = [
    {"layers": 1, "dim": 32, "heads": 2, "dropout": 0.1},
    {"layers": 2, "dim": 32, "heads": 2, "dropout": 0.1},
    {"layers": 1, "dim": 64, "heads": 4, "dropout": 0.1},
    {"layers": 2, "dim": 64, "heads": 4, "dropout": 0.2},
]
TOP_K = (1, 5, 10)


def _metrics(ranks):
    total = len(ranks)
    out = {f"top_{k}": sum(r is not None and r <= k for r in ranks) / total for k in TOP_K}
    out["mrr"] = sum(1.0 / r for r in ranks if r) / total
    out["n_masked"] = total
    out["n_oov"] = sum(r is None for r in ranks)
    out["oov_fraction"] = out["n_oov"] / total if total else 0.0
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


def _fit_valid_split(train, seed):
    fit, valid, _ = split_records(
        train, 0.8, seed, track="artifact", group_duplicates=True
    )
    if not fit or not valid:
        cut = max(1, len(train) // 2)
        fit, valid = train[:cut], train[cut:]
    return fit, valid


def _inner_select(train, seed, max_epochs, patience):
    """Select config and stopping epoch using INNER data only."""
    fit_records, valid_records = _fit_valid_split(train, seed)
    fit = [r["sequence"] for r in fit_records]
    valid = [r["sequence"] for r in valid_records]
    inner = []
    for config in GRID:
        bundle = train_masked_lm(
            fit, valid, max_epochs=max_epochs, patience=patience, seed=seed, **config
        )
        ranks = masked_lm_ranks(bundle, valid)
        inner.append({
            "config": config,
            "valid_top_1": _metrics(ranks)["top_1"],
            "epochs_run": bundle["epochs_run"],
            "n_parameters": bundle["n_parameters"],
            "fit_vocab_size": bundle["fit_vocab_size"],
            "valid_oov_rate": bundle["valid_oov_rate"],
        })
    best = max(inner, key=lambda row: (row["valid_top_1"], -row["n_parameters"]))
    return inner, best["config"], best["epochs_run"], len(fit_records), len(valid_records)


def _refit_and_test(train, test, config, epochs, seed):
    """Refit the selected config on the FULL outer-train, then test once."""
    train_seqs = [r["sequence"] for r in train]
    bundle = train_masked_lm(
        train_seqs, None, max_epochs=max(1, epochs), patience=epochs + 1,
        seed=seed, **config,
    )
    ranks = masked_lm_ranks(bundle, [r["sequence"] for r in test])
    return bundle, ranks

def render_report(summary):
    lines = [
        "Masked-sign transformer (tiny encoder) vs bigram [fully nested selection]",
        "",
        "Per outer grouped split: a grouped inner fit/validation split is carved from",
        "the outer TRAIN partition only; every grid configuration is scored on inner",
        "validation; the best configuration AND its stopping epoch are selected; the",
        "selected configuration is then refit on the FULL outer train for that many",
        "epochs and evaluated once on the outer test. The outer test never enters "
        "selection or early stopping. The vocabulary is built from fitting data "
        "only; unseen signs are <UNK> context tokens, <UNK> is never a candidate, "
        "and OOV test targets count as failures. "
        "SD describes split variability, NOT confidence intervals.",
        "",
        f"Outer splits ({summary['n_ok']}/{summary['n_runs']} successful):",
        "seed | selected config | epoch | valid top-1 | test top-1 | fit vocab | test OOV",
    ]
    for run in summary["per_seed"]:
        cfg = run["selected_config"]
        lines.append(
            f"{run['seed']} | L{cfg['layers']} d{cfg['dim']} h{cfg['heads']} "
            f"do{cfg['dropout']} | {run['selected_epochs']} | "
            f"{run['selected_valid_top_1']:.4f} | {run['transformer']['top_1']:.4f} | "
            f"{run['fit_vocab_size']} | {run['transformer']['oov_fraction']:.4f}"
        )
    lines.extend([
        "",
        "Inner selection results per outer split (config -> valid top-1, epochs):",
    ])
    for run in summary["per_seed"]:
        rows = ", ".join(
            f"L{r['config']['layers']}d{r['config']['dim']}={r['valid_top_1']:.4f}@{r['epochs_run']}"
            for r in run["inner"]
        )
        lines.append(f"  seed {run['seed']}: {rows}")
    lines.extend([
        "",
        "Aggregate comparison:",
        "Model | Top-1 mean +/- SD | Top-5 mean | Top-10 mean | MRR mean",
    ])
    for model in ("bigram", "transformer"):
        block = summary["aggregate"][model]
        lines.append(
            f"{model} | {block['top_1']['mean']:.4f} +/- {block['top_1']['sd']:.4f} | "
            f"{block['top_5']['mean']:.4f} | {block['top_10']['mean']:.4f} | "
            f"{block['mrr']['mean']:.4f}"
        )
    delta = summary["aggregate"]["paired"]["top_1"]
    lines.extend([
        f"Transformer minus bigram top-1: {delta['mean']:+.4f} "
        f"(wins/ties/losses {delta['n_positive']}/{delta['n_zero']}/{delta['n_negative']}).",
        f"Mean epochs run: {summary['aggregate']['epochs_run']['mean']:.1f}; "
        f"mean parameters: {summary['aggregate']['n_parameters']['mean']:.0f}.",
        "",
        "Full per-split inner selection results, OOV diagnostics, and split",
        "identities are in transformer_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Masked-sign transformer vs bigram under fully nested selection"
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "transformer")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    seeds = list(range(args.seed, args.seed + args.repeats))

    runs = []
    for seed in seeds:
        train, test, diagnostics = split_records(
            records, 0.8, seed, track="artifact", group_duplicates=True
        )
        bigram = [r["rank"] for r in restoration_records(train, test, mask_length=1)[0]]
        inner, config, epochs, n_fit, n_valid = _inner_select(
            train, seed, args.max_epochs, args.patience
        )
        bundle, ranks = _refit_and_test(train, test, config, epochs, seed)
        assert len(ranks) == len(bigram)
        runs.append({
            "seed": seed, "split": diagnostics,
            "train_ids": sorted({r["inscription_id"] for r in train}),
            "test_ids": sorted({r["inscription_id"] for r in test}),
            "inner": inner,
            "n_inner_fit": n_fit, "n_inner_valid": n_valid,
            "selected_config": config, "selected_epochs": epochs,
            "selected_valid_top_1": max(row["valid_top_1"] for row in inner),
            "fit_vocab_size": bundle["fit_vocab_size"],
            "valid_oov_rate": bundle["valid_oov_rate"],
            "test_oov_rate": _metrics(ranks)["oov_fraction"],
            "bigram": _metrics(bigram), "transformer": _metrics(ranks),
            "paired": _paired(ranks, bigram),
            "epochs_run": bundle["epochs_run"],
            "n_parameters": bundle["n_parameters"],
        })

    aggregate = {
        model: {k: _summarize([r[model][k] for r in runs])
                for k in ("top_1", "top_5", "top_10", "mrr")}
        for model in ("bigram", "transformer")
    }
    paired = {}
    for k in ("top_1", "top_5", "top_10", "mrr"):
        values = [r["paired"][k] for r in runs]
        paired[k] = _summarize(values) | {
            "n_positive": sum(v > 0 for v in values),
            "n_zero": sum(v == 0 for v in values),
            "n_negative": sum(v < 0 for v in values),
        }
    aggregate["paired"] = paired
    aggregate["epochs_run"] = _summarize([r["epochs_run"] for r in runs])
    aggregate["n_parameters"] = _summarize([r["n_parameters"] for r in runs])
    selected_counts = {}
    for run in runs:
        key = json.dumps(run["selected_config"], sort_keys=True)
        selected_counts[key] = selected_counts.get(key, 0) + 1

    import torch

    sources = sorted((ROOT / "src" / "arthanvesana").rglob("*.py")) + [Path(__file__).resolve()]
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy", "torch")},
            "torch_version": torch.__version__,
            "seeds": seeds, "gap_policy": "split", "known_direction_only": True,
            "grid": GRID, "max_epochs": args.max_epochs, "patience": args.patience,
            "selection": "fully nested per outer split (inner grouped fit/valid on outer train only)",
        },
        "per_seed": runs,
        "selected_config_counts": selected_counts,
        "n_runs": len(runs), "n_ok": len(runs),
        "aggregate": aggregate,
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transformer_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "transformer_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()