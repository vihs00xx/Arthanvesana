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


def _fit_valid_split(train, seed):
    fit, valid, _ = split_records(
        train, 0.8, seed, track="artifact", group_duplicates=True
    )
    if not fit or not valid:
        cut = max(1, len(train) // 2)
        fit, valid = train[:cut], train[cut:]
    return fit, valid


def _train_eval(train, test, config, seed, max_epochs, patience):
    fit_records, valid_records = _fit_valid_split(train, seed + 1000)
    fit = [r["sequence"] for r in fit_records]
    valid = [r["sequence"] for r in valid_records]
    bundle = train_masked_lm(
        fit, valid, max_epochs=max_epochs, patience=patience, seed=seed, **config
    )
    ranks = masked_lm_ranks(bundle, [r["sequence"] for r in test])
    return bundle, ranks


def render_report(summary):
    lines = [
        "Masked-sign transformer (tiny encoder) vs bigram",
        "",
        "One random position masked per span per epoch; early stopping on a",
        "grouped validation split; the vocabulary mapping covers fit plus",
        "validation signs, while test-only signs count as failures and OOV",
        "neighbors are shown to the model as masks. Same artifact-grouped",
        "splits and masked positions as the bigram. Parameter counts are",
        "reported so the capacity mismatch is explicit. SD describes split",
        "variability, NOT confidence intervals.",
        "",
        "Config selection on seed 0 by GROUPED VALIDATION top-1 (grouped",
        "split carved from training; test untouched during selection):",
    ]
    for row in summary["config_selection"]:
        config = row["config"]
        lines.append(
            f"layers {config['layers']}, dim {config['dim']}, heads {config['heads']}, "
            f"dropout {config['dropout']}: valid top-1 {row['valid_top1']:.4f}, "
            f"params {row['n_parameters']}, epochs {row['epochs_run']}"
        )
    best = summary["best_config"]
    lines.extend([
        f"Selected: layers {best['layers']}, dim {best['dim']}, heads {best['heads']}, "
        f"dropout {best['dropout']}.",
        f"Selected config test top-1 on seed 0 (reporting only, after "
        f"selection): {summary['selected_test_top1']:.4f}.",
        "",
        f"10-seed comparison ({summary['n_ok']}/{summary['n_runs']} successful runs):",
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
        "Full per-run metrics, paired differences, split identities, and",
        "training diagnostics are in transformer_summary.json.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Tiny masked-sign transformer vs bigram")
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
    train0, test0, _ = split_records(records, 0.8, seeds[0], track="artifact", group_duplicates=True)
    fit0, valid0 = _fit_valid_split(train0, seeds[0])
    bigram0 = [r["rank"] for r in restoration_records(fit0, valid0, mask_length=1)[0]]
    selection = []
    for config in GRID:
        bundle, valid_ranks = _train_eval(
            fit0, valid0, config, seeds[0], args.max_epochs, args.patience
        )
        assert len(valid_ranks) == len(bigram0)
        selection.append({
            "config": config, "valid_top1": _metrics(valid_ranks)["top_1"],
            "n_parameters": bundle["n_parameters"],
            "epochs_run": bundle["epochs_run"],
            "best_valid_loss": bundle["best_valid_loss"],
        })
    best_config = max(selection, key=lambda r: r["valid_top1"])["config"]
    _, selected_test_ranks = _train_eval(
        fit0, test0, best_config, seeds[0] + 500, args.max_epochs, args.patience
    )
    selected_test_top1 = _metrics(selected_test_ranks)["top_1"]

    runs = []
    for seed in seeds:
        train, test, diagnostics = split_records(
            records, 0.8, seed, track="artifact", group_duplicates=True
        )
        bigram = [r["rank"] for r in restoration_records(train, test, mask_length=1)[0]]
        bundle, ranks = _train_eval(
            train, test, best_config, seed, args.max_epochs, args.patience
        )
        runs.append({
            "seed": seed, "split": diagnostics,
            "train_ids": sorted({r["inscription_id"] for r in train}),
            "test_ids": sorted({r["inscription_id"] for r in test}),
            "bigram": _metrics(bigram), "transformer": _metrics(ranks),
            "paired": _paired(ranks, bigram),
            "epochs_run": bundle["epochs_run"],
            "n_parameters": bundle["n_parameters"],
            "best_valid_loss": bundle["best_valid_loss"],
        })

    aggregate = {
        model: {k: _summarize([r[model][k] for r in runs]) for k in ("top_1", "top_5", "top_10", "mrr")}
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
        },
        "config_selection": selection,
        "best_config": best_config,
        "selected_test_top1": selected_test_top1,
        "n_runs": len(runs), "n_ok": len(runs),
        "runs": runs, "aggregate": aggregate,
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
