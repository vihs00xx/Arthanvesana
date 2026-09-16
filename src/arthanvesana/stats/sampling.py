"""Seeded corpus sampling: within-inscription shuffles (null model preserving
lengths and unigram counts), train/test splits, and exact deduplication.
"""

from __future__ import annotations

import random


def shuffled_corpus(
    seqs: list[list[str]], seed: int = 0, n_replicates: int = 20
) -> list[list[list[str]]]:
    replicates = []
    for r in range(n_replicates):
        rng = random.Random(seed + r)
        replicates.append([rng.sample(seq, len(seq)) for seq in seqs])
    return replicates


def train_test_split(
    seqs: list[list[str]], train_frac: float = 0.8, seed: int = 0
) -> tuple[list[list[str]], list[list[str]]]:
    rng = random.Random(seed)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    cut = int(len(seqs) * train_frac)
    train_idx = set(idx[:cut])
    train = [s for i, s in enumerate(seqs) if i in train_idx]
    test = [s for i, s in enumerate(seqs) if i not in train_idx]
    return train, test


def deduplicated(seqs: list[list[str]]) -> list[list[str]]:
    seen = set()
    unique = []
    for seq in seqs:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique.append(seq)
    return unique


def sample_sequence(model, max_len: int = 20, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    seq: list[str] = []
    context = ("<S>",) * (model.n - 1) if model.n > 1 else ()
    while len(seq) < max_len:
        dist = model.dist(context)
        r = rng.random()
        acc = 0.0
        pick = None
        for w, p in dist.items():
            acc += p
            if r <= acc:
                pick = w
                break
        if pick is None or pick == "<UNK>":
            break
        seq.append(pick)
        if model.n > 1:
            context = tuple((list(context) + [pick])[1:])
    return seq
