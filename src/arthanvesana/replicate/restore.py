from __future__ import annotations

from arthanvesana.data.parse import parse_bool
from arthanvesana.stats.ngrams import NGramModel


def _matrix(model: NGramModel) -> dict:
    return {ctx: model.dist((ctx,)) for ctx in sorted(model.vocab | {"<S>"})}


def _as_records(items: list) -> list[dict]:
    return [
        item if isinstance(item, dict) else {
            "sequence": item, "inscription_id": f"sequence-{i}",
            "start_complete": True, "end_complete": True,
        }
        for i, item in enumerate(items)
    ]


def _mask_distributions(
    mats: dict, prior: dict, vocab: set, seq: list[str],
    start: int, stop: int, start_complete: bool,
) -> list[dict]:
    words = sorted(vocab)
    if start:
        prev = seq[start - 1] if seq[start - 1] in mats else "<UNK>"
        initial = mats[prev]
    else:
        initial = mats["<S>"] if start_complete else prior
    nxt = seq[stop] if stop < len(seq) else None
    if nxt is not None and nxt not in vocab:
        nxt = "<UNK>"

    def normalized(dist):
        total = sum(dist.values())
        return {w: p / total if total else 0.0 for w, p in dist.items()}

    forward = [normalized(initial)]
    for _ in range(stop - start - 1):
        forward.append(normalized({
            w: sum(forward[-1][v] * mats[v].get(w, 0.0) for v in words)
            for w in words
        }))
    backward = [{w: mats[w].get(nxt, 0.0) if nxt is not None else 1.0 for w in words}]
    for _ in range(stop - start - 1):
        backward.append(normalized({
            w: sum(mats[w].get(v, 0.0) * backward[-1][v] for v in words)
            for w in words
        }))
    return [
        normalized({w: left.get(w, 0.0) * right[w] for w in words})
        for left, right in zip(forward, reversed(backward))
    ]


def restore_rank(
    mats: dict, vocab: set, seq: list[str], pos: int
) -> int | None:
    if not seq:
        return None
    if seq[pos] not in vocab or seq[pos] == "<UNK>":
        return None
    dist = _mask_distributions(mats, mats["<UNK>"], vocab, seq, pos, pos + 1, True)[0]
    candidates = sorted(vocab - {"<UNK>"}, key=lambda w: (-dist[w], w))
    return candidates.index(seq[pos]) + 1


def restoration_records(
    train: list, test: list, *, mask_length: int = 1, mask_stride: int = 1,
) -> tuple[list[dict], int]:
    if not isinstance(mask_length, int) or mask_length < 1:
        raise ValueError("mask_length must be a positive integer")
    if not isinstance(mask_stride, int) or mask_stride < 1:
        raise ValueError("mask_stride must be a positive integer")
    train_records = _as_records(train)
    test_records = _as_records(test)
    sequences = [r["sequence"] for r in train_records]
    if not any(sequences):
        raise ValueError("Restoration requires nonempty training data")
    model = NGramModel(sequences, 2, method="wittenbell")
    starts = model.followers[1].get(("<S>",))
    if starts is not None:
        for record in train_records:
            if record["sequence"] and not parse_bool(record.get("start_complete", True)):
                sign = record["sequence"][0]
                starts[sign] -= 1
                if starts[sign] == 0:
                    del starts[sign]
    mats = _matrix(model)
    prior = NGramModel(sequences, 1, method="wittenbell").dist(())
    records = []
    for source in test_records:
        seq = source["sequence"]
        for start in range(0, len(seq) - mask_length + 1, mask_stride):
            stop = start + mask_length
            distributions = _mask_distributions(
                mats, prior, model.vocab, seq, start, stop,
                parse_bool(source.get("start_complete", True)),
            )
            for pos, dist in zip(range(start, stop), distributions):
                candidates = sorted(model.vocab - {"<UNK>"}, key=lambda w: (-dist[w], w))
                target = seq[pos]
                oov = target not in model.vocab or target == "<UNK>"
                rank = None if oov else candidates.index(target) + 1
                records.append({
                    "inscription_id": source["inscription_id"],
                    "artifact_id": source.get("artifact_id"),
                    "artifact_group": source.get("artifact_group"),
                    "site": source.get("site"),
                    "split_group": source.get("split_group", source["inscription_id"]),
                    "span_index": source.get("span_index", 0),
                    "start_complete": parse_bool(source.get("start_complete", True)),
                    "end_complete": parse_bool(source.get("end_complete", True)),
                    "position": pos,
                    "mask_start": start, "mask_end": stop,
                    "mask_length": mask_length,
                    "length": len(seq), "target": target, "oov": oov,
                    "rank": rank, "p_true": 0.0 if oov else dist[target],
                    "prediction": candidates[0] if candidates else None,
                    "top_p": dist[candidates[0]] if candidates else 0.0,
                    "unknown_p": dist.get("<UNK>", 0.0),
                    "hit": rank == 1,
                })
    return records, 0


def restoration_accuracy(
    train: list, test: list, top_k: tuple = (1, 5, 10),
    *, mask_length: int = 1, mask_stride: int = 1,
) -> dict:
    records, skipped = restoration_records(
        train, test, mask_length=mask_length, mask_stride=mask_stride
    )
    hits = {k: 0 for k in top_k}
    by_length: dict[int, list] = {}
    for rec in records:
        by_length.setdefault(rec["length"], []).append(rec["rank"])
        for k in top_k:
            if rec["rank"] is not None and rec["rank"] <= k:
                hits[k] += 1
    total = len(records)
    out = {f"top_{k}": hits[k] / total if total else 0.0 for k in top_k}
    out["n_masked"] = total
    out["n_skipped_oov"] = skipped
    out["n_oov"] = sum(r["oov"] for r in records)
    out["mask_length"] = mask_length
    out["by_length"] = {
        length: sum(r == 1 for r in ranks) / len(ranks)
        for length, ranks in by_length.items()
    }
    return out
