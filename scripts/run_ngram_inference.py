"""Grouped cross-fitted bigram-vs-trigram held-out log-loss inference.

Design (unchanged in spirit from the earlier grouped runner, strengthened here):

* connected artifact/inscription/exact-sequence components from the authoritative
  grouping implementation are indivisible groups assigned to folds;
* every record gets exactly one out-of-fold prediction per cross-fitting run;
* token differences are aggregated within each group BEFORE inference;
* the **paired difference** is ``d = log2 P_trigram - log2 P_bigram`` in
  bits/token, so positive favours the trigram.

Two estimands are reported and they answer different questions:

* **macro (unweighted) group mean** — answers "does the trigram help a typical
  connected group?" Every component counts once regardless of size. This is the
  primary estimand.
* **token-weighted mean** — answers "does the trigram help a typical token?"
  Large components dominate. This is secondary. The two are not interchangeable
  and are never averaged together.

The complete cross-fitting procedure is repeated under several predefined fold
seeds as a sensitivity analysis. Repeated predictions across seeds are **never**
pooled as independent observations: each seed is reported separately, and the
primary result uses the full-data estimate at the primary seed.

The largest connected component is reported on its own, in the full estimate,
and in an explicitly labelled sensitivity analysis excluding it. The full-data
estimate remains the primary result; the large group is not dropped to improve
significance.

Limits of the reported intervals: they resample *fixed* out-of-fold scores, so
they capture group-level sampling variability but NOT the uncertainty from
retraining the models, and the two fitted models share training data within a
fold. They are not a substitute for whole-pipeline simulation, which the
calibration runner provides.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, identity_value, sha256_file
from arthanvesana.stats.grouping import (
    assign_folds as shared_assign_folds,
)
from arthanvesana.stats.grouping import (
    fold_loads,
    group_stats,
    leakage_diagnostics,
)
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import connected_groups

N_FOLDS = 5
DEFAULT_FOLD_SEEDS = (0, 1, 2, 3, 4)
LENGTH_BANDS = ((1, 2, "1-2"), (3, 4, "3-4"), (5, 7, "5-7"), (8, 10 ** 9, "8+"))
#: Absolute training-frequency bands, so no single band swallows the corpus.
FREQUENCY_BANDS = (
    (1, 1, "freq=1"),
    (2, 4, "freq=2-4"),
    (5, 19, "freq=5-19"),
    (20, 99, "freq=20-99"),
    (100, 10 ** 9, "freq>=100"),
)


def assign_folds(groups, n_folds, seed):
    """Backwards-compatible wrapper over the shared fold assignment.

    ``groups`` is the ``connected_groups`` output form (dicts with ``indices``).
    Balancing is by tokens then spans, largest component first, with seeded
    tie-breaking.
    """
    stats = {}
    for g in groups:
        tokens = g.get("n_tokens")
        if tokens is None:
            tokens = len(g["indices"])
        stats[g["group_id"]] = (tokens, len(g["indices"]))
    return shared_assign_folds(stats, n_folds, seed)


def token_logprobs(model, seqs):
    out = []
    for seq in seqs:
        mapped = model._map(seq)
        row = []
        for i in range(len(mapped)):
            context = tuple((["<S>"] * (model.n - 1) + mapped[:i])[-(model.n - 1):])
            prob = model.dist(context)[mapped[i]]
            if prob <= 0.0:
                raise ValueError(
                    f"Zero probability for token {mapped[i]!r} in context {context!r}."
                )
            row.append(math.log2(prob))
        out.append(row)
    return out


def signflip_p(group_means, permutations, seed):
    """Two-sided group-level sign-flip p, plus-one corrected; never 0.

    Retained for continuity and reported as a *descriptive* statistic. It is NOT
    cited as evidence for higher-order structure: its null assumes the group
    differences are symmetric about zero, which a first-order generator does not
    guarantee (an over-parameterized fitted trigram genuinely loses). Whether any
    rejection rule is calibrated is a separate question answered by whole-pipeline
    simulation, not by this p-value.
    """
    observed = abs(float(group_means.mean()))
    rng = np.random.default_rng(seed)
    n = group_means.size
    exceed = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=n)
        if abs(float(group_means.dot(signs)) / n) >= observed:
            exceed += 1
    return (exceed + 1) / (permutations + 1)


def cluster_boot(group_means, group_sizes, replicates, seed, weighted):
    gm = np.asarray(group_means, dtype=float)
    gs = np.asarray(group_sizes, dtype=float)
    rng = np.random.default_rng(seed)
    n = gm.size
    ests = np.empty(replicates)
    for r in range(replicates):
        idx = rng.integers(0, n, size=n)
        if weighted:
            w = gs[idx]
            ests[r] = float((gm[idx] * w).sum() / w.sum())
        else:
            ests[r] = gm[idx].mean()
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))


def length_band(n):
    for lo, hi, label in LENGTH_BANDS:
        if lo <= n <= hi:
            return label
    return LENGTH_BANDS[-1][2]


def frequency_band(freq):
    """Absolute training-frequency band for a token.

    Absolute bands are used rather than terciles because the corpus is dominated
    by a few very frequent signs: terciles of the token-frequency distribution put
    ~90% of tokens in a single bucket, which reports almost nothing. Absolute
    bands are directly interpretable and keep every band populated.
    """
    if freq <= 0:
        return "unseen_in_training"
    for lo, hi, label in FREQUENCY_BANDS:
        if lo <= freq <= hi:
            return label
    return FREQUENCY_BANDS[-1][2]


def crossfit(records, groups, assignment, n_folds):
    """Cross-fit both models; return group rows and per-token score rows.

    Frequency bands are defined from the TRAINING tokens of each fold, and length
    bands use the clear labels ``1-2``/``3-4``/``5-7``/``8+``. Per-record score
    rows keep the original inscription id, group id, and fold so any stratum can
    be audited back to the source records.
    """
    record_fold = {}
    record_group = {}
    for g in groups:
        for i in g["indices"]:
            record_fold[i] = assignment[g["group_id"]]
            record_group[i] = g["group_id"]
    group_tokens = {g["group_id"]: {"bi": [], "tri": []} for g in groups}
    fold_sizes = {f: {"n_groups": 0, "n_spans": 0, "n_tokens": 0} for f in range(n_folds)}
    token_rows = []
    freq_by_band = {}
    length_by_band = {}
    for f in range(n_folds):
        train = [records[i]["sequence"] for i in range(len(records))
                 if record_fold[i] != f]
        test_idx = [i for i in range(len(records)) if record_fold[i] == f]
        if not train or not test_idx:
            continue
        train_freq = Counter(s for seq in train for s in seq)
        bigram = NGramModel(train, 2, method="mkn")
        trigram = NGramModel(train, 3, method="mkn")
        by_group = {}
        for i in test_idx:
            by_group.setdefault(record_group[i], []).append(i)
        scored = {}
        for gid, idxs in by_group.items():
            seqs = [records[i]["sequence"] for i in idxs]
            lb = token_logprobs(bigram, seqs)
            lt = token_logprobs(trigram, seqs)
            for i, row_b, row_t in zip(idxs, lb, lt):
                diffs = [t - b for b, t in zip(row_b, row_t)]
                group_tokens[gid]["bi"].extend(row_b)
                group_tokens[gid]["tri"].extend(row_t)
                token_rows.append({
                    "record_index": i,
                    "inscription_id": records[i]["inscription_id"],
                    "artifact_id": records[i].get("artifact_id"),
                    "group_id": gid,
                    "fold": f,
                    "n_tokens": len(diffs),
                    "diffs": diffs,
                })
                scored[i] = (row_b, row_t)
        # per-token frequency bands (from TRAINING counts only) and length bands,
        # reusing the per-record scores already computed above
        for i in test_idx:
            pair = scored.get(i)
            if pair is None:
                continue
            row_b, row_t = pair
            seq = records[i]["sequence"]
            length_by_band.setdefault(length_band(len(seq)), []).extend(
                t - b for b, t in zip(row_b, row_t))
            for tok, b, t in zip(seq, row_b, row_t):
                band = frequency_band(train_freq.get(tok, 0))
                freq_by_band.setdefault(band, []).append(t - b)
    group_rows = []
    for g in groups:
        gid = g["group_id"]
        idxs = g["indices"]
        sites = {identity_value(records[i].get("site")) for i in idxs} - {None}
        bi = group_tokens[gid]["bi"]
        tri = group_tokens[gid]["tri"]
        d = [t - b for b, t in zip(bi, tri)]
        group_rows.append({
            "group_id": gid,
            "fold": assignment[gid],
            "n_inscriptions": len({records[i]["inscription_id"] for i in idxs}),
            "n_spans": len(idxs),
            "n_tokens": len(d),
            "mean_logloss_bigram": -float(np.mean(bi)) if bi else None,
            "mean_logloss_trigram": -float(np.mean(tri)) if tri else None,
            "mean_paired_diff": float(np.mean(d)) if d else None,
            "n_sites": len(sites),
            "n_artifacts": len({records[i].get("artifact_id") for i in idxs}),
            "site_counts": dict(Counter(
                identity_value(records[i].get("site")) or "unknown" for i in idxs)),
            "artefact_type_counts": dict(Counter(
                identity_value(records[i].get("artefact_type")) or "unknown"
                for i in idxs)),
        })
        f = assignment[gid]
        fold_sizes[f]["n_groups"] += 1
        fold_sizes[f]["n_spans"] += len(idxs)
        fold_sizes[f]["n_tokens"] += len(d)
    band_order = [label for _lo, _hi, label in FREQUENCY_BANDS] + ["unseen_in_training"]
    strata = {
        "by_training_frequency_band": {
            k: {"mean": float(np.mean(freq_by_band[k])),
                "n_tokens": len(freq_by_band[k])}
            for k in band_order if freq_by_band.get(k)
        },
        "by_length_band": {
            k: {"mean": float(np.mean(v)), "n_tokens": len(v)}
            for k, v in sorted(length_by_band.items())
        },
    }
    return group_rows, fold_sizes, token_rows, strata


def _summarise(group_rows, token_rows, args, seed):
    ok = [r for r in group_rows if r["mean_paired_diff"] is not None]
    gm = np.array([r["mean_paired_diff"] for r in ok], dtype=float)
    gs = np.array([r["n_tokens"] for r in ok], dtype=float)
    macro = float(gm.mean()) if gm.size else None
    weighted = float((gm * gs).sum() / gs.sum()) if gm.size else None
    return {
        "fold_seed": seed,
        "n_groups": len(group_rows),
        "n_groups_scored": len(ok),
        "macro_group_effect": macro,
        "macro_ci95": list(cluster_boot(gm, gs, args.bootstrap, seed, False))
        if gm.size else None,
        "token_weighted_effect": weighted,
        "token_weighted_ci95": list(cluster_boot(gm, gs, args.bootstrap, seed, True))
        if gm.size else None,
        "randomization_p": signflip_p(gm, args.permutations, seed) if gm.size else None,
        "fraction_groups": {
            "favor_trigram": float((gm > 0).mean()) if gm.size else None,
            "favor_bigram": float((gm < 0).mean()) if gm.size else None,
            "zero": float((gm == 0).mean()) if gm.size else None,
        },
        "n_tokens_scored": int(gs.sum()) if gm.size else 0,
        "n_records_scored": len(token_rows),
        "row_means": gm,
        "row_tokens": gs,
    }


def _largest_component_analysis(group_rows, args, seed):
    ok = [r for r in group_rows if r["mean_paired_diff"] is not None]
    if not ok:
        return {"available": False}
    largest = max(ok, key=lambda r: (r["n_tokens"], r["group_id"]))
    rest = [r for r in ok if r["group_id"] != largest["group_id"]]
    gm_all = np.array([r["mean_paired_diff"] for r in ok], dtype=float)
    gs_all = np.array([r["n_tokens"] for r in ok], dtype=float)
    gm_rest = np.array([r["mean_paired_diff"] for r in rest], dtype=float)
    gs_rest = np.array([r["n_tokens"] for r in rest], dtype=float)
    total_tokens = float(gs_all.sum())

    def weighted(gm, gs):
        return float((gm * gs).sum() / gs.sum()) if gm.size else None

    return {
        "available": True,
        "group_id": largest["group_id"],
        "n_tokens": largest["n_tokens"],
        "n_spans": largest["n_spans"],
        "token_fraction": largest["n_tokens"] / total_tokens if total_tokens else None,
        "own_effect": largest["mean_paired_diff"],
        "including": {
            "label": "primary result; the large component is NOT removed",
            "macro_group_effect": float(gm_all.mean()),
            "token_weighted_effect": weighted(gm_all, gs_all),
            "n_groups": int(gm_all.size),
        },
        "excluding": {
            "label": ("EXPLICITLY LABELLED SENSITIVITY ANALYSIS; not the primary "
                      "result and not a replacement for it"),
            "macro_group_effect": float(gm_rest.mean()) if gm_rest.size else None,
            "token_weighted_effect": weighted(gm_rest, gs_rest),
            "macro_ci95": list(cluster_boot(gm_rest, gs_rest, args.bootstrap, seed,
                                            False)) if gm_rest.size else None,
            "n_groups": int(gm_rest.size),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Grouped cross-fitted bigram-vs-trigram log-loss inference."
    )
    parser.add_argument("--corpus", type=Path,
                        default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "ngram_inference")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--fold-seeds", type=int, nargs="*",
                        default=list(DEFAULT_FOLD_SEEDS),
                        help="predefined fold seeds for the sensitivity analysis")
    parser.add_argument("--primary-seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--save-token-scores", action="store_true", default=True)
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be >= 2")
    if args.primary_seed not in args.fold_seeds:
        parser.error("--primary-seed must appear in --fold-seeds")

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    stats = group_stats(records, groups)

    per_seed = {}
    token_rows_primary = None
    strata_primary = None
    loads_primary = None
    leakage_primary = None
    for seed in args.fold_seeds:
        assignment = shared_assign_folds(stats, args.folds, seed)
        group_rows, fold_sizes, token_rows, strata = crossfit(
            records, groups, assignment, args.folds)
        block = _summarise(group_rows, token_rows, args, seed)
        block["fold_sizes"] = {str(f): v for f, v in fold_sizes.items()}
        diag = leakage_diagnostics(records, groups, assignment)
        block["leakage"] = diag
        loads = fold_loads(assignment, stats, args.folds)
        block["fold_loads"] = loads
        block["largest_component"] = _largest_component_analysis(group_rows, args, seed)
        per_seed[str(seed)] = block
        if seed == args.primary_seed:
            token_rows_primary = token_rows
            strata_primary = strata
            loads_primary = loads
            leakage_primary = diag
            primary_rows = group_rows

    summaries = []
    for seed in args.fold_seeds:
        blk = per_seed[str(seed)]
        summaries.append({
            "fold_seed": seed,
            "macro_group_effect": blk["macro_group_effect"],
            "macro_ci95": blk["macro_ci95"],
            "token_weighted_effect": blk["token_weighted_effect"],
            "token_weighted_ci95": blk["token_weighted_ci95"],
            "randomization_p": blk["randomization_p"],
            "n_groups_scored": blk["n_groups_scored"],
            "n_tokens_scored": blk["n_tokens_scored"],
        })
    macro_seeds = [s["macro_group_effect"] for s in summaries
                   if s["macro_group_effect"] is not None]
    token_seeds = [s["token_weighted_effect"] for s in summaries
                   if s["token_weighted_effect"] is not None]

    p_block = per_seed[str(args.primary_seed)]
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "src/arthanvesana/stats/sampling.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "sampling.py"),
                "src/arthanvesana/stats/grouping.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "grouping.py"),
                "src/arthanvesana/stats/ngrams.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "ngrams.py"),
                "src/arthanvesana/data/parse.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "data" / "parse.py"),
                "scripts/run_ngram_inference.py": sha256_file(Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": list(args.fold_seeds),
            "primary_fold_seed": args.primary_seed,
            "folds": args.folds,
            "permutations": args.permutations,
            "bootstrap": args.bootstrap,
            "model": "modified Kneser-Ney bigram vs trigram, grouped cross-fitting",
            "grouping_policy": ("artifact_grouped: connected artifact / inscription / "
                                "exact-sequence components, duplicates merged"),
            "paired_difference": "d = log2 P_trigram - log2 P_bigram (bits/token)",
            "primary_estimand": ("macro (unweighted) group mean: does the trigram help a "
                                 "typical connected group?"),
            "secondary_estimand": ("token-weighted mean: does the trigram help a "
                                   "typical token? large components dominate"),
            "estimands_note": ("The two estimands answer different questions and are "
                               "never averaged together."),
            "pvalue_status": ("the randomization p-value is descriptive only; the sign-flip "
                              "null assumes symmetry about zero, which a first-order "
                              "generator does not guarantee, and its calibration is a "
                              "separate whole-pipeline simulation question"),
            "interval_status": ("cluster bootstrap over fixed out-of-fold group scores: "
                                "captures group sampling variability, NOT retraining "
                                "uncertainty; the two models share training data within "
                                "a fold"),
        },
        "n_records": len(records),
        "n_groups": len(primary_rows),
        "n_groups_scored": p_block["n_groups_scored"],
        "n_tokens_scored": p_block["n_tokens_scored"],
        "fold_sizes": p_block["fold_sizes"],
        "fold_loads": loads_primary,
        "leakage_group_violations": p_block["leakage"]["groups_split_across_folds"],
        "leakage_diagnostics": leakage_primary,
        "primary": {
            "fold_seed": args.primary_seed,
            "macro_group_effect": p_block["macro_group_effect"],
            "ci95": p_block["macro_ci95"],
            "randomization_p": p_block["randomization_p"],
        },
        "secondary_token_weighted": {
            "fold_seed": args.primary_seed,
            "effect": p_block["token_weighted_effect"],
            "ci95": p_block["token_weighted_ci95"],
        },
        "fraction_groups": p_block["fraction_groups"],
        "fold_seed_sensitivity": {
            "seeds": summaries,
            "macro_mean_over_seeds": (sum(macro_seeds) / len(macro_seeds))
            if macro_seeds else None,
            "macro_min_over_seeds": min(macro_seeds) if macro_seeds else None,
            "macro_max_over_seeds": max(macro_seeds) if macro_seeds else None,
            "token_weighted_mean_over_seeds": (sum(token_seeds) / len(token_seeds))
            if token_seeds else None,
            "note": ("each seed is an independent complete cross-fitting run; repeated "
                     "predictions across seeds are NOT pooled as independent "
                     "observations, and no seed is averaged into the primary result"),
        },
        "largest_component": p_block["largest_component"],
        "stratified": {
            **strata_primary,
            "by_artefact_type": _artefact_type_table(primary_rows),
            "note": ("Frequency bands come from each fold's TRAINING tokens only. "
                     "Sample sizes are reported per stratum; small strata are "
                     "descriptive only. All tables are exploratory."),
        },
        "groups": primary_rows,
        "token_scores_file": "ngram_inference_token_scores.json",
    }

    lines = _render(summary, args, summaries)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ngram_inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False,
                   default=_json_default), encoding="utf-8")
    if args.save_token_scores and token_rows_primary is not None:
        (args.output / "ngram_inference_token_scores.json").write_text(
            json.dumps(token_rows_primary, ensure_ascii=False, allow_nan=False),
            encoding="utf-8")
    (args.output / "ngram_inference_report.txt").write_text(lines, encoding="utf-8")
    print(lines)
    return summary


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def _artefact_type_table(rows):
    table = {}
    for r in rows:
        counts = r.get("artefact_type_counts") or {}
        if not counts:
            key = "unknown"
        else:
            key = max(counts, key=counts.get)
        table.setdefault(key, []).append(r["mean_paired_diff"])
    return {
        k: {"mean": float(np.mean(v)), "n_groups": len(v)}
        for k, v in sorted(table.items()) if v and v[0] is not None
    }


def _render(summary, args, summaries):
    m = summary["manifest"]
    lines = [
        "Grouped cross-fitted bigram-vs-trigram log-loss inference",
        "",
        f"Grouping policy: {m['grouping_policy']}",
        f"Paired difference: {m['paired_difference']}",
        "Connected components are indivisible groups assigned to folds, largest",
        "component first, with seeded tie-breaking. Each record gets exactly one",
        "out-of-fold prediction per cross-fitting run.",
        "",
        "TWO ESTIMANDS, TWO QUESTIONS:",
        f"  primary   macro group mean  -> {m['primary_estimand']}",
        f"  secondary token-weighted    -> {m['secondary_estimand']}",
        "They are NOT interchangeable and are never averaged together.",
        "",
        f"Records: {summary['n_records']}; groups: {summary['n_groups']}; "
        f"groups scored: {summary['n_groups_scored']}; "
        f"tokens scored: {summary['n_tokens_scored']}.",
        f"Folds: {args.folds}; primary fold seed: {args.primary_seed}.",
        "Fold sizes (groups/spans/tokens): " + "; ".join(
            f"fold {f}={v['n_groups']}/{v['n_spans']}/{v['n_tokens']}"
            for f, v in sorted(summary["fold_sizes"].items())
        ),
        f"Leakage (components in >1 fold): {summary['leakage_group_violations']}",
        f"Sequence crossings between folds (original sequences, not labels): "
        f"{summary['leakage_diagnostics']['sequence_crossings']}",
        f"Fold token loads: min {summary['fold_loads']['min_tokens']}, "
        f"max {summary['fold_loads']['max_tokens']}, "
        f"target {summary['fold_loads']['target_tokens_per_fold']:.0f}, "
        f"unavoidable imbalance {summary['fold_loads']['unavoidable_imbalance_tokens']:.0f}",
        "",
        f"PRIMARY (macro group mean, fold seed {args.primary_seed}):",
        f"  mean paired diff = {summary['primary']['macro_group_effect']:+.5f} bits/token",
        f"  95% cluster-bootstrap interval = "
        f"[{summary['primary']['ci95'][0]:+.5f}, {summary['primary']['ci95'][1]:+.5f}]",
        f"  descriptive randomization p = {summary['primary']['randomization_p']:.4f} "
        f"(permutations={args.permutations}, +1 correction; never 0)",
        "  WARNING: this p-value is descriptive only and must not be cited as",
        "  evidence. Its calibration is a whole-pipeline simulation question.",
        "",
        f"SECONDARY (token-weighted, fold seed {args.primary_seed}):",
        f"  weighted mean diff = "
        f"{summary['secondary_token_weighted']['effect']:+.5f} bits/token",
        f"  95% cluster-bootstrap interval = "
        f"[{summary['secondary_token_weighted']['ci95'][0]:+.5f}, "
        f"{summary['secondary_token_weighted']['ci95'][1]:+.5f}]",
        "",
        "INTERVAL LIMITS: these intervals resample fixed out-of-fold group scores.",
        "They capture group-level sampling variability but NOT retraining",
        "uncertainty, and the two fitted models share training data within a fold.",
        "",
        "FOLD-SEED SENSITIVITY (independent complete cross-fitting runs; never",
        "pooled, never averaged into the primary result):",
        "seed | macro group effect | macro CI | token-weighted | token-weight CI",
    ]
    for s in summaries:
        ci = s["macro_ci95"]
        wci = s["token_weighted_ci95"]
        lines.append(
            f"{s['fold_seed']} | {s['macro_group_effect']:+.5f} | "
            f"[{ci[0]:+.5f}, {ci[1]:+.5f}] | {s['token_weighted_effect']:+.5f} | "
            f"[{wci[0]:+.5f}, {wci[1]:+.5f}]"
        )
    fs = summary["fold_seed_sensitivity"]
    lines.append(
        f"macro effect across seeds: mean {fs['macro_mean_over_seeds']:+.5f}, "
        f"range [{fs['macro_min_over_seeds']:+.5f}, {fs['macro_max_over_seeds']:+.5f}]"
    )
    lc = summary["largest_component"]
    lines.extend(["", "LARGEST CONNECTED COMPONENT:"])
    if lc.get("available"):
        lines.append(
            f"  group {lc['group_id']}: {lc['n_tokens']} tokens "
            f"({lc['token_fraction']:.1%} of scored tokens), {lc['n_spans']} spans")
        lines.append(f"  its own effect: {lc['own_effect']:+.5f} bits/token")
        lines.append(
            f"  INCLUDING it (primary): macro {lc['including']['macro_group_effect']:+.5f}, "
            f"token-weighted {lc['including']['token_weighted_effect']:+.5f} "
            f"over {lc['including']['n_groups']} groups")
        lines.append(
            f"  EXCLUDING it (labelled sensitivity analysis): macro "
            f"{lc['excluding']['macro_group_effect']:+.5f}, token-weighted "
            f"{lc['excluding']['token_weighted_effect']:+.5f} "
            f"over {lc['excluding']['n_groups']} groups")
        lines.append("  The full-data estimate remains primary; the large group is not")
        lines.append("  removed to improve significance.")
    else:
        lines.append("  not available")
    lines.extend(["", "Effect by length band (exploratory; per-token):"])
    for band, blk in summary["stratified"]["by_length_band"].items():
        lines.append(f"  length {band}: {blk['mean']:+.5f} over {blk['n_tokens']} tokens")
    lines.append("Effect by TRAINING-DATA token-frequency band (exploratory; per-token):")
    for band, blk in summary["stratified"]["by_training_frequency_band"].items():
        lines.append(f"  {band}: {blk['mean']:+.5f} over {blk['n_tokens']} tokens")
    lines.append("Effect by artefact type (exploratory; per-component mean):")
    for art, blk in summary["stratified"]["by_artefact_type"].items():
        lines.append(f"  {art}: {blk['mean']:+.5f} over {blk['n_groups']} groups")
    lines.extend(["", summary["stratified"]["note"], ""])
    lines.append(
        "Provisional status: the higher-order (trigram) evidence is PROVISIONAL. It "
        "must be read against the separately calibrated structural test produced by "
        "scripts/run_power_analysis.py, not against the descriptive p-value above.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()