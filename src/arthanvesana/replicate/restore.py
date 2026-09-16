"""Missing-sign restoration benchmark. Each position is masked and ranked by
bidirectional bigram score P(m|left)*P(right|m) under a Witten-Bell model;
reports top-k hit rates. The bigram matrix is precomputed once, so ranking
is pure lookup.
"""

from __future__ import annotations

from arthanvesana.stats.ngrams import NGramModel


def _matrix(model: NGramModel) -> dict:
    mats = {}
    for ctx in list(model.vocab) + ["<S>"]:
        mats[ctx] = model.dist((ctx,))
    return mats


def restore_rank(
    mats: dict, vocab: set, seq: list[str], pos: int
) -> int | None:
    if not seq:
        return None
    target = seq[pos]
    if target not in vocab or target == "<UNK>":
        return None
    prev = seq[pos - 1] if pos > 0 else "<S>"
    prev = prev if prev in mats else "<UNK>"
    nxt = seq[pos + 1] if pos < len(seq) - 1 else None
    nxt = nxt if nxt is None or nxt in vocab else "<UNK>"
    d_prev = mats[prev]
    scored = []
    for cand in vocab:
        if cand == "<UNK>":
            continue
        s = d_prev.get(cand, 0.0)
        if nxt is not None:
            s *= mats[cand].get(nxt, 0.0)
        scored.append((s, cand))
    scored.sort(key=lambda t: -t[0])
    for rank, (_, cand) in enumerate(scored, start=1):
        if cand == target:
            return rank
    return None


def restoration_records(
    train: list[list[str]], test: list[list[str]]
) -> tuple[list[dict], int]:
    model = NGramModel(train, 2, method="wittenbell")
    mats = _matrix(model)
    records = []
    skipped = 0
    for seq in test:
        if len(seq) < 2:
            continue
        for pos in range(len(seq)):
            if seq[pos] not in model.vocab:
                skipped += 1
                continue
            prev = seq[pos - 1] if pos > 0 else "<S>"
            prev = prev if prev in mats else "<UNK>"
            nxt = seq[pos + 1] if pos < len(seq) - 1 else None
            nxt = nxt if nxt is None or nxt in model.vocab else "<UNK>"
            d_prev = mats[prev]
            scored = []
            for cand in model.vocab:
                if cand == "<UNK>":
                    continue
                s = d_prev.get(cand, 0.0)
                if nxt is not None:
                    s *= mats[cand].get(nxt, 0.0)
                scored.append((s, cand))
            scored.sort(key=lambda t: -t[0])
            total = sum(s for s, _ in scored) or 1.0
            rank = None
            p_true = 0.0
            for r, (s, cand) in enumerate(scored, start=1):
                if cand == seq[pos]:
                    rank = r
                    p_true = s / total
            records.append(
                {
                    "length": len(seq),
                    "rank": rank,
                    "p_true": p_true,
                    "top_p": scored[0][0] / total if scored else 0.0,
                    "hit": rank == 1,
                }
            )
    return records, skipped


def restoration_accuracy(
    train: list[list[str]], test: list[list[str]], top_k: tuple = (1, 5, 10)
) -> dict:
    records, skipped = restoration_records(train, test)
    hits = {k: 0 for k in top_k}
    by_length: dict[int, list[int]] = {}
    for rec in records:
        by_length.setdefault(rec["length"], []).append(rec["rank"])
        for k in top_k:
            if rec["rank"] is not None and rec["rank"] <= k:
                hits[k] += 1
    total = len(records)
    out = {f"top_{k}": hits[k] / total if total else 0.0 for k in top_k}
    out["n_masked"] = total
    out["n_skipped_oov"] = skipped
    out["by_length"] = {
        L: sum(1 for r in rs if r == 1) / len(rs) for L, rs in by_length.items()
    }
    return out
