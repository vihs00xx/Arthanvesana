"""Cross-fitted bigram/trigram evaluation for synthetic corpora."""

from __future__ import annotations

import math
import random

import numpy as np

from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import connected_groups


def records_from_seqs(seqs):
    """Wrap plain sequences in minimal grouped records (one artifact each)."""
    return [
        {"inscription_id": f"i{i}", "artifact_id": f"a{i}",
         "artifact_group": (None, "explicit", f"a{i}"), "site": "S",
         "sequence": list(seq), "span_index": 0, "span_start": 0,
         "start_complete": True, "end_complete": True}
        for i, seq in enumerate(seqs)
    ]


def _logprobs(model, seqs):
    out = []
    for seq in seqs:
        mapped = model._map(seq)
        for i in range(len(mapped)):
            context = tuple((["<S>"] * (model.n - 1) + mapped[:i])[-(model.n - 1):])
            prob = model.dist(context)[mapped[i]]
            if prob <= 0:
                prob = 1e-12
            out.append(math.log2(prob))
    return out


def assign_folds(group_stats, n_folds, seed):
    """Greedy fold assignment balancing tokens then spans; deterministic."""
    order = list(group_stats)
    random.Random(seed).shuffle(order)
    load = [(0, 0) for _ in range(n_folds)]
    assignment = {}
    for gid in order:
        tokens, spans = group_stats[gid]
        fold = min(range(n_folds), key=lambda f: (load[f][0], load[f][1], f))
        assignment[gid] = fold
        load[fold] = (load[fold][0] + tokens, load[fold][1] + spans)
    return assignment


def crossfit_effect(seqs, n_folds=5, seed=0, permutations=2000, bootstrap=2000):
    """Run the grouped cross-fitted bigram-vs-trigram inference on sequences.

    Returns the primary estimand block: macro group effect, its cluster
    bootstrap CI, and the group-level sign-flip p (never 0). Group means are
    the independent units; duplicating tokens inside one group changes group
    size but not the number of groups.
    """
    records = records_from_seqs(seqs)
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    assignment = assign_folds(
        {g["group_id"]: (
            sum(len(records[i]["sequence"]) for i in g["indices"]), len(g["indices"]))
         for g in groups},
        n_folds, seed,
    )
    diffs, bi, tri = _per_group_scores(records, groups, assignment, n_folds)
    group_means = [
        sum(diffs[g["group_id"]]) / len(diffs[g["group_id"]])
        for g in groups if diffs.get(g["group_id"])
    ]
    gm = np.asarray(group_means, dtype=float)
    if gm.size == 0:
        return {"macro_effect": None, "ci95": None, "p": None, "n_groups": 0}
    all_bi = [x for vals in bi.values() for x in vals]
    all_tri = [x for vals in tri.values() for x in vals]
    return {
        "macro_effect": float(gm.mean()),
        "ci95": list(_cluster_boot(gm, bootstrap, seed)),
        "p": _signflip_p(gm, permutations, seed),
        "n_groups": int(gm.size),
        "weighted_effect": _weighted_effect(groups, diffs),
        "logloss_bigram": -float(np.mean(all_bi)) if all_bi else None,
        "logloss_trigram": -float(np.mean(all_tri)) if all_tri else None,
    }


def _per_group_scores(records, groups, assignment, n_folds):
    gid_of = {}
    for g in groups:
        for i in g["indices"]:
            gid_of[i] = g["group_id"]
    diffs, bi, tri = {}, {}, {}
    for f in range(n_folds):
        train = [records[i]["sequence"] for i in range(len(records))
                 if assignment[gid_of[i]] != f]
        test_idx = [i for i in range(len(records)) if assignment[gid_of[i]] == f]
        if not train or not test_idx:
            continue
        bigram = NGramModel(train, 2, method="mkn")
        trigram = NGramModel(train, 3, method="mkn")
        for i in test_idx:
            lb = _logprobs(bigram, [records[i]["sequence"]])
            lt = _logprobs(trigram, [records[i]["sequence"]])
            gid = gid_of[i]
            diffs.setdefault(gid, []).extend(t - b for b, t in zip(lb, lt))
            bi.setdefault(gid, []).extend(lb)
            tri.setdefault(gid, []).extend(lt)
    return diffs, bi, tri


def _weighted_effect(groups, diffs):
    num = den = 0.0
    for g in groups:
        d = diffs.get(g["group_id"])
        if d:
            num += sum(d)
            den += len(d)
    return num / den if den else None


def _signflip_p(gm, permutations, seed):
    rng = np.random.default_rng(seed)
    observed = abs(float(gm.mean()))
    exceed = 0
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=gm.size)
        if abs(float(gm.dot(signs)) / gm.size) >= observed:
            exceed += 1
    return (exceed + 1) / (permutations + 1)


def _cluster_boot(gm, replicates, seed):
    rng = np.random.default_rng(seed)
    n = gm.size
    ests = np.empty(replicates)
    for r in range(replicates):
        ests[r] = gm[rng.integers(0, n, size=n)].mean()
    return float(np.percentile(ests, 2.5)), float(np.percentile(ests, 97.5))
