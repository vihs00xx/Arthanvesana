"""Synthetic corpus generation matched to empirical properties.

Generators reproduce: inscription count, the empirical sequence-length
distribution, vocabulary size, and the heavy-tailed empirical sign-frequency
distribution. Matched exactly: inscription count, length distribution (sampled
from the empirical length list), vocabulary size, unigram frequencies.
Matched approximately: higher-order structure (empirical bigram / trigram
transitions) and duplicate component structure. No generator represents
natural language or the true Indus production process; they are calibration
instruments only.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def empirical_profile(records):
    """Extract the empirical properties a generator needs."""
    seqs = [r["sequence"] for r in records if r["sequence"]]
    lengths = [len(s) for s in seqs]
    unigram = Counter(s for seq in seqs for s in seq)
    bigram = Counter()
    trigram = Counter()
    for seq in seqs:
        padded = ["<S>", "<S>"] + seq
        for i in range(2, len(padded)):
            bigram[(padded[i - 1], padded[i])] += 1
            trigram[(padded[i - 2], padded[i - 1], padded[i])] += 1
    starters = Counter(seq[0] for seq in seqs if seq)
    enders = Counter(seq[-1] for seq in seqs if seq)
    return {
        "n_inscriptions": len(seqs),
        "lengths": lengths,
        "unigram": unigram,
        "bigram": bigram,
        "trigram": trigram,
        "starters": dict(starters),
        "enders": dict(enders),
    }


def _draw(signs, weights, rng):
    return signs[int(rng.choice(len(signs), p=weights))]


def _sample_unigram(unigram, rng):
    signs = sorted(unigram)
    weights = np.array([unigram[s] for s in signs], dtype=float)
    return signs, weights / weights.sum()


def gen_unigram(profile, n_inscriptions, seed):
    rng = np.random.default_rng(seed)
    signs, weights = _sample_unigram(profile["unigram"], rng)
    lengths = profile["lengths"]
    return [
        [_draw(signs, weights, rng) for _ in range(lengths[int(rng.integers(0, len(lengths)))])]
        for _ in range(n_inscriptions)
    ]


def gen_position_only(profile, n_inscriptions, seed):
    """Position-conditioned sampler: first token over-samples empirical
    starters, last over-samples empirical enders, middle = unigram."""
    rng = np.random.default_rng(seed)
    marginal = _sample_unigram(profile["unigram"], rng)
    marginal = dict(zip(marginal[0], marginal[1]))
    lengths = profile["lengths"]
    out = []
    for _ in range(n_inscriptions):
        length = lengths[int(rng.integers(0, len(lengths)))]
        seq = []
        for pos in range(length):
            if pos == 0:
                pool = profile.get("starters") or marginal
            elif pos == length - 1:
                pool = profile.get("enders") or marginal
            else:
                pool = marginal
            names = sorted(pool)
            w = np.array([pool[n] for n in names], dtype=float)
            seq.append(names[int(rng.choice(len(names), p=w / w.sum()))])
        out.append(seq)
    return out


def gen_markov(profile, n_inscriptions, seed):
    """First-order Markov chain on the empirical bigram transitions."""
    rng = np.random.default_rng(seed)
    bigram = profile["bigram"]
    unigram = profile["unigram"]
    lengths = profile["lengths"]
    out = []
    for _ in range(n_inscriptions):
        length = lengths[int(rng.integers(0, len(lengths)))]
        seq = []
        prev = "<S>"
        for _pos in range(length):
            followers = {w: c for (p, w), c in bigram.items() if p == prev}
            if not followers:
                signs, weights = _sample_unigram(unigram, rng)
                seq.append(_draw(signs, weights, rng))
                continue
            names = sorted(followers)
            w = np.array([followers[n] for n in names], dtype=float)
            seq.append(names[int(rng.choice(len(names), p=w / w.sum()))])
            prev = seq[-1]
        out.append(seq)
    return out



def gen_trigram_mixture(profile, n_inscriptions, seed, lam):
    """Controlled higher-order generator.

    Samples from P(w | h2, h1) = (1-lam) * P_bi(w|h1) + lam * P_tri(w|h2,h1).
    lam = 0 reproduces the first-order Markov generator (zero higher-order
    effect); increasing lam adds known higher-order strength.
    """
    rng = np.random.default_rng(seed)
    bigram = profile["bigram"]
    trigram = profile["trigram"]
    unigram = profile["unigram"]
    lengths = profile["lengths"]
    out = []
    for _ in range(n_inscriptions):
        length = lengths[int(rng.integers(0, len(lengths)))]
        seq = []
        h1 = h2 = "<S>"
        for _pos in range(length):
            followers_bi = {w: c for (p, w), c in bigram.items() if p == h1}
            followers_tri = {w: c for (a, b, w), c in trigram.items()
                             if a == h2 and b == h1}
            names = sorted(set(followers_bi) | set(followers_tri) or set(unigram))
            probs = np.zeros(len(names))
            tot_bi = sum(followers_bi.values()) or 1
            tot_tri = sum(followers_tri.values()) or 1
            for j, w in enumerate(names):
                probs[j] = ((1 - lam) * followers_bi.get(w, 0) / tot_bi
                            + lam * followers_tri.get(w, 0) / tot_tri)
            total = probs.sum()
            if total <= 0:
                signs, weights = _sample_unigram(unigram, rng)
                seq.append(_draw(signs, weights, rng))
            else:
                seq.append(names[int(rng.choice(len(names), p=probs / total))])
            h2, h1 = h1, seq[-1]
        out.append(seq)
    return out


def gen_hmm_slots(profile, n_inscriptions, seed, n_states=3):
    """Positional/slot HMM: state determined by relative position; emissions
    from per-state frequency-sliced vocabularies. Returns (sequences,
    true_states) so cluster recovery is checkable."""
    rng = np.random.default_rng(seed)
    unigram = profile["unigram"]
    lengths = profile["lengths"]
    order = [s for s, _ in sorted(unigram.items(), key=lambda kv: -kv[1])]
    slices = np.array_split(np.arange(len(order)), n_states)
    pools = [sorted(order[j] for j in slice_) for slice_ in slices]
    out, true_states = [], []
    for _ in range(n_inscriptions):
        length = lengths[int(rng.integers(0, len(lengths)))]
        seq, states = [], []
        for pos in range(length):
            frac = pos / max(1, length - 1)
            state = 0 if frac < 0.25 else (n_states - 1 if frac > 0.75 else n_states // 2)
            pool = pools[state]
            w = np.array([unigram[s] for s in pool], dtype=float)
            seq.append(pool[int(rng.choice(len(pool), p=w / w.sum()))])
            states.append(state)
        out.append(seq)
        true_states.append(states)
    return out, true_states


def gen_shuffled_real(records, n_inscriptions, seed):
    """Within-sequence shuffles of real sequences (order null)."""
    rng = np.random.default_rng(seed)
    seqs = [r["sequence"] for r in records if r["sequence"]]
    out = []
    for _ in range(n_inscriptions):
        seq = list(seqs[int(rng.integers(0, len(seqs)))])
        rng.shuffle(seq)
        out.append(seq)
    return out
