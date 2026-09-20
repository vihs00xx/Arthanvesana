"""Grouped cross-fitted bigram-vs-trigram held-out log-loss inference.

Replaces the earlier token-level design, which was statistically invalid
(token-level sign flips, pooled overlapping splits, repeated records). Here the
connected artifact/inscription/duplicate components from split_records are
indivisible groups assigned to folds; every record gets exactly one out-of-fold
prediction; token differences are aggregated within each group BEFORE
inference; and the primary test is a group-level sign-flip randomization with
p = (exceedances + 1) / (permutations + 1), so p is never 0.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.parse import analysis_records, identity_value, sha256_file
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import connected_groups

N_FOLDS = 5


def assign_folds(groups, n_folds, seed):
    order = [g["group_id"] for g in groups]
    random.Random(seed).shuffle(order)
    by_id = {g["group_id"]: g for g in groups}
    fold_tokens = [0] * n_folds
    fold_spans = [0] * n_folds
    assignment = {}
    for gid in order:
        g = by_id[gid]
        n_tok = g.get("n_tokens", 0)
        n_span = len(g["indices"])
        fold = min(range(n_folds), key=lambda f: (fold_tokens[f], fold_spans[f], f))
        assignment[gid] = fold
        fold_tokens[fold] += n_tok
        fold_spans[fold] += n_span
    return assignment


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


def crossfit(records, groups, assignment, n_folds):
    record_fold = {}
    record_group = {}
    for g in groups:
        for i in g["indices"]:
            record_fold[i] = assignment[g["group_id"]]
            record_group[i] = g["group_id"]
    group_tokens = {g["group_id"]: {"bi": [], "tri": []} for g in groups}
    fold_sizes = {f: {"n_groups": 0, "n_spans": 0, "n_tokens": 0} for f in range(n_folds)}
    for f in range(n_folds):
        train = [records[i]["sequence"] for i in range(len(records)) if record_fold[i] != f]
        test_idx = [i for i in range(len(records)) if record_fold[i] == f]
        bigram = NGramModel(train, 2, method="mkn")
        trigram = NGramModel(train, 3, method="mkn")
        by_group = {}
        for i in test_idx:
            by_group.setdefault(record_group[i], []).append(i)
        for gid, idxs in by_group.items():
            seqs = [records[i]["sequence"] for i in idxs]
            lb = token_logprobs(bigram, seqs)
            lt = token_logprobs(trigram, seqs)
            for row_b, row_t in zip(lb, lt):
                group_tokens[gid]["bi"].extend(row_b)
                group_tokens[gid]["tri"].extend(row_t)
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
            "artefact_counts": dict(Counter(
                identity_value(records[i].get("artefact_type")) or "unknown" for i in idxs)),
        })
        f = assignment[gid]
        fold_sizes[f]["n_groups"] += 1
        fold_sizes[f]["n_spans"] += len(idxs)
        fold_sizes[f]["n_tokens"] += len(d)
    return group_rows, fold_sizes


def signflip_p(group_means, permutations, seed):
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

def _band(value, edges):
    for e in edges:
        if value <= e:
            return f"<={e}"
    return f">{edges[-1]}"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Grouped cross-fitted bigram-vs-trigram log-loss inference."
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "ngram_inference")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds must be >= 2")

    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    records = analysis_records(frame, gap_policy="split", known_direction_only=True)
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    ntok = {}
    for g in groups:
        ntok[g["group_id"]] = sum(len(records[i]["sequence"]) for i in g["indices"])
    for g in groups:
        g["n_tokens"] = ntok[g["group_id"]]
    assignment = assign_folds(groups, args.folds, args.seed)
    group_rows, fold_sizes = crossfit(records, groups, assignment, args.folds)

    ok = [r for r in group_rows if r["mean_paired_diff"] is not None]
    gm = np.array([r["mean_paired_diff"] for r in ok], dtype=float)
    gs = np.array([r["n_tokens"] for r in ok], dtype=float)
    macro_effect = float(gm.mean())
    macro_ci = cluster_boot(gm, gs, args.bootstrap, args.seed, weighted=False)
    weighted_effect = float((gm * gs).sum() / gs.sum())
    weighted_ci = cluster_boot(gm, gs, args.bootstrap, args.seed, weighted=True)
    p_val = signflip_p(gm, args.permutations, args.seed)
    frac_pos = float((gm > 0).mean())
    frac_neg = float((gm < 0).mean())
    frac_zero = float((gm == 0).mean())
    ppl_bi = float(2 ** np.mean([r["mean_logloss_bigram"] for r in ok]))
    ppl_tri = float(2 ** np.mean([r["mean_logloss_trigram"] for r in ok]))

    # leakage diagnostic: no group id appears in two folds
    fold_seen = {}
    leakage = 0
    for r in group_rows:
        if r["group_id"] in fold_seen and fold_seen[r["group_id"]] != r["fold"]:
            leakage += 1
        fold_seen[r["group_id"]] = r["fold"]

    # Stratified (exploratory) macro effects.
    freq = Counter(s for r in records for s in r["sequence"])
    median_freq = float(np.median(list(freq.values())))

    def freq_band(group_idxs):
        toks = [s for i in group_idxs for s in records[i]["sequence"]]
        if not toks:
            return "unknown"
        mean_f = float(np.mean([freq[s] for s in toks]))
        return "high_freq" if mean_f >= median_freq else "low_freq"

    by_length = {}
    by_freqband = {}
    by_site = {}
    by_artifact = {}
    gid_to_idxs = {g["group_id"]: g["indices"] for g in groups}
    for r in ok:
        idxs = gid_to_idxs[r["group_id"]]
        mean_len = float(np.mean([len(records[i]["sequence"]) for i in idxs]))
        len_band = _band(mean_len, [2, 4, 7])
        fb = freq_band(idxs)
        site = max(r["site_counts"], key=r["site_counts"].get) if r["site_counts"] else "unknown"
        art = (max(r["artefact_counts"], key=r["artefact_counts"].get)
               if r["artefact_counts"] else "unknown")
        for table, key in ((by_length, len_band), (by_freqband, fb),
                           (by_site, site), (by_artifact, art)):
            table.setdefault(key, []).append(r["mean_paired_diff"])
    stratified = {
        "by_length": {k: {"mean": float(np.mean(v)), "n_groups": len(v)}
                      for k, v in sorted(by_length.items())},
        "by_frequency_band": {k: {"mean": float(np.mean(v)), "n_groups": len(v)}
                              for k, v in sorted(by_freqband.items())},
        "by_site": {k: {"mean": float(np.mean(v)), "n_groups": len(v)}
                    for k, v in sorted(by_site.items())},
        "by_artefact_type": {k: {"mean": float(np.mean(v)), "n_groups": len(v)}
                             for k, v in sorted(by_artifact.items())},
        "note": ("Exploratory macro means of per-group paired diff within each "
                 "stratum; small strata carry high variability."),
    }

    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "source_sha256": {
                "src/arthanvesana/stats/sampling.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "sampling.py"),
                "src/arthanvesana/stats/ngrams.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "stats" / "ngrams.py"),
                "scripts/run_ngram_inference.py": sha256_file(Path(__file__).resolve()),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seed": args.seed, "folds": args.folds,
            "permutations": args.permutations, "bootstrap": args.bootstrap,
            "model": "modified Kneser-Ney bigram vs trigram, grouped 5-fold cross-fitting",
            "estimand": "macro (unweighted) group mean of per-token paired log-loss diff",
            "pvalue": "group-level sign-flip, (exceedances+1)/(permutations+1), never 0",
        },
        "n_records": len(records),
        "n_groups": len(group_rows),
        "n_groups_scored": len(ok),
        "fold_sizes": {str(f): v for f, v in fold_sizes.items()},
        "leakage_group_violations": leakage,
        "primary": {
            "macro_group_effect": macro_effect,
            "ci95": list(macro_ci),
            "randomization_p": p_val,
        },
        "secondary_token_weighted": {
            "effect": weighted_effect,
            "ci95": list(weighted_ci),
        },
        "fraction_groups": {
            "favor_trigram": frac_pos, "favor_bigram": frac_neg, "zero": frac_zero,
        },
        "perplexity": {"bigram": ppl_bi, "trigram": ppl_tri},
        "stratified": stratified,
        "groups": group_rows,
    }
    lines = [
        "Grouped cross-fitted bigram-vs-trigram log-loss inference",
        "",
        "Connected artifact/inscription/duplicate components are indivisible",
        "groups assigned to folds (greedy, balanced by tokens then spans). Each",
        "record gets exactly one out-of-fold prediction; token differences are",
        "aggregated within each group before inference. The earlier token-level",
        "pooled permutation test is removed as the headline because tokens within",
        "groups are dependent and overlapping splits repeated records.",
        "SD/intervals are over independent connected groups, not tokens.",
        "",
        f"Groups scored: {len(ok)} of {len(group_rows)}; records: {len(records)}; "
        f"folds: {args.folds}.",
        "Fold sizes (groups/spans/tokens): "
        + "; ".join(
            f"fold {f}={fold_sizes[f]['n_groups']}/"
            f"{fold_sizes[f]['n_spans']}/{fold_sizes[f]['n_tokens']}"
            for f in sorted(fold_sizes)
        ),
        f"Leakage (groups in >1 fold): {leakage}",
        "",
        "PRIMARY (macro group-level):",
        f"  mean paired diff (trigram - bigram) = {macro_effect:+.5f} bits/token",
        f"  95% cluster-bootstrap interval = [{macro_ci[0]:+.5f}, {macro_ci[1]:+.5f}]",
        f"  group-level randomization p = {p_val:.4f} "
        f"(permutations={args.permutations}, +1 correction; never 0)",
        "SECONDARY (token-weighted):",
        f"  weighted mean diff = {weighted_effect:+.5f} bits/token",
        f"  95% cluster-bootstrap interval = [{weighted_ci[0]:+.5f}, {weighted_ci[1]:+.5f}]",
        "",
        f"Perplexity: bigram {ppl_bi:.2f}, trigram {ppl_tri:.2f}",
        f"Groups favoring trigram: {frac_pos:.3f}; bigram: {frac_neg:.3f}; "
        f"zero: {frac_zero:.3f}",
        "",
        "A positive macro effect with a 95% interval above zero and a small",
        "group-level p supports a small trigram advantage; overlapping tokens",
        "within a group remain dependent, which is why inference is at the group",
        "level. This is descriptive of predictive information, not of linguistic",
        "units.",
        "",
        "Effect by sequence-length band (exploratory):",
    ]
    for band, blk in stratified["by_length"].items():
        lines.append(
            f"  length {band}: mean diff {blk['mean']:+.5f} over {blk['n_groups']} groups"
        )
    lines.append("Effect by sign-frequency band (exploratory):")
    for band, blk in stratified["by_frequency_band"].items():
        lines.append(
            f"  {band}: mean diff {blk['mean']:+.5f} over {blk['n_groups']} groups"
        )
    lines.append("Effect by site (exploratory):")
    for site, blk in stratified["by_site"].items():
        lines.append(
            f"  {site}: mean diff {blk['mean']:+.5f} over {blk['n_groups']} groups"
        )
    lines.append("Effect by artefact type (exploratory):")
    for art, blk in stratified["by_artefact_type"].items():
        lines.append(
            f"  {art}: mean diff {blk['mean']:+.5f} over {blk['n_groups']} groups"
        )
    lines.append("")
    lines.append(stratified["note"])
    report = "\n".join(lines) + "\n"

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ngram_inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (args.output / "ngram_inference_report.txt").write_text(report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()