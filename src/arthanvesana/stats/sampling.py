"""Seeded corpus sampling: within-inscription shuffles (null model preserving
lengths and unigram counts), train/test splits, and exact deduplication.
"""

from __future__ import annotations

import random

from arthanvesana.data.parse import identity_value


def shuffled_corpus(
    seqs: list[list[str]], seed: int = 0, n_replicates: int = 20
) -> list[list[list[str]]]:
    replicates = []
    for r in range(n_replicates):
        rng = random.Random(seed + r)
        replicates.append([rng.sample(seq, len(seq)) for seq in seqs])
    return replicates


def train_test_split(
    seqs: list[list[str]], train_frac: float = 0.8, seed: int = 0,
    *, group_keys: list | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    if not 0 <= train_frac <= 1:
        raise ValueError("train_frac must be between 0 and 1")
    if group_keys is not None:
        if len(group_keys) != len(seqs):
            raise ValueError("group_keys must have one key per sequence")
        records = [
            {"inscription_id": str(i), "sequence": seq, "group_key": key}
            for i, (seq, key) in enumerate(zip(seqs, group_keys))
        ]
        train, test, _ = split_records(
            records, train_frac, seed, group_keys=group_keys
        )
        train_ids = {int(r["inscription_id"]) for r in train}
        return (
            [s for i, s in enumerate(seqs) if i in train_ids],
            [s for i, s in enumerate(seqs) if i not in train_ids],
        )
    rng = random.Random(seed)
    idx = list(range(len(seqs)))
    rng.shuffle(idx)
    cut = int(len(seqs) * train_frac)
    train_idx = set(idx[:cut])
    train = [s for i, s in enumerate(seqs) if i in train_idx]
    test = [s for i, s in enumerate(seqs) if i not in train_idx]
    return train, test


def _freeze(value):
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def _record_keys(record: dict) -> dict:
    inscription = identity_value(record.get("inscription_id"))
    if inscription is None:
        raise ValueError("Every record requires an inscription_id")
    site = identity_value(record.get("site"))
    artifact = record.get("artifact_group")
    if artifact is None:
        artifact_id = identity_value(record.get("artifact_id"))
        artifact = (None, "explicit", artifact_id) if artifact_id else None
    return {
        "record": inscription,
        "sequence": tuple(record["sequence"]),
        "artifact": _freeze(artifact),
        "site": site,
    }


def _assign_groups(
    records: list[dict], track: str, group_duplicates: bool, group_keys: list | None
) -> tuple[list[list[int]], list[dict], dict]:
    """Shared union-find grouping used by split_records and connected_groups.

    Returns (groups, keys, group_ids) where groups is the deterministically
    ordered list of record-index lists and group_ids maps each record index to
    its stable component label (the minimum record identity in the component).
    This is the single source of artifact/inscription/duplicate grouping.
    """
    keys = [_record_keys(r) for r in records]
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    fields = {"record", track}
    if group_duplicates:
        fields.add("sequence")
    for i, row in enumerate(keys):
        links = [(field, row[field]) for field in sorted(fields) if row[field] is not None]
        if group_keys is not None and group_keys[i] is not None:
            links.append(("custom", _freeze(group_keys[i])))
        for link in links:
            if link in seen:
                parent[root(i)] = root(seen[link])
            else:
                seen[link] = i
    components = {}
    for i in range(len(records)):
        components.setdefault(root(i), []).append(i)

    def order_key(i):
        return (keys[i]["record"], records[i].get("span_index", 0),
                records[i].get("span_start", 0), keys[i]["sequence"])

    groups = sorted(
        (sorted(indices, key=order_key) for indices in components.values()),
        key=lambda indices: tuple(order_key(i) for i in indices),
    )
    group_ids = {}
    for indices in groups:
        label = min(keys[i]["record"] for i in indices)
        for i in indices:
            group_ids[i] = label
    return groups, keys, group_ids


def connected_groups(
    records: list[dict], *, track: str = "artifact", group_duplicates: bool | None = None,
    group_keys: list | None = None,
) -> list[dict]:
    """Expose the connected components used for grouped splits, with stable ids.

    Uses the exact grouping code path as split_records, so audit/inference code
    does not reimplement grouping. Each component gets a deterministic id (the
    minimum record identity among its members). Every record belongs to exactly
    one component.
    """
    if track not in {"record", "sequence", "artifact", "site"}:
        raise ValueError("track must be record, sequence, artifact, or site")
    if group_keys is not None and len(group_keys) != len(records):
        raise ValueError("group_keys must have one key per record")
    if group_duplicates is None:
        group_duplicates = track in {"sequence", "artifact"}
    groups, _, group_ids = _assign_groups(records, track, group_duplicates, group_keys)
    return [
        {"group_id": group_ids[indices[0]], "indices": tuple(indices)}
        for indices in groups
    ]


def split_records(
    records: list[dict], train_frac: float = 0.8, seed: int = 0,
    *, track: str = "record", group_duplicates: bool | None = None,
    group_keys: list | None = None,
) -> tuple[list[dict], list[dict], dict]:
    if track not in {"record", "sequence", "artifact", "site"}:
        raise ValueError("track must be record, sequence, artifact, or site")
    if not 0 <= train_frac <= 1:
        raise ValueError("train_frac must be between 0 and 1")
    if group_keys is not None and len(group_keys) != len(records):
        raise ValueError("group_keys must have one key per record")
    if group_duplicates is None:
        group_duplicates = track in {"sequence", "artifact"}
    groups, keys, group_ids = _assign_groups(records, track, group_duplicates, group_keys)
    random.Random(seed).shuffle(groups)
    cut = int(len(groups) * train_frac)
    train_indices = {i for indices in groups[:cut] for i in indices}

    def order_key(i):
        return (keys[i]["record"], records[i].get("span_index", 0),
                records[i].get("span_start", 0), keys[i]["sequence"])

    ordered = sorted(range(len(records)), key=order_key)
    train = [dict(records[i], split_group=group_ids[i]) for i in ordered if i in train_indices]
    test = [dict(records[i], split_group=group_ids[i]) for i in ordered if i not in train_indices]
    overlap = {}
    for field in ("record", "sequence", "artifact", "site"):
        left = {keys[i][field] for i in train_indices if keys[i][field] is not None}
        right = {keys[i][field] for i in ordered if i not in train_indices and keys[i][field] is not None}
        overlap[field] = len(left & right)
    diagnostics = {
        "track": track, "seed": seed, "train_frac": train_frac,
        "group_duplicates": group_duplicates,
        "n_records": len(records), "n_groups": len(groups),
        "n_train_groups": cut, "n_test_groups": len(groups) - cut,
        "n_train": len(train), "n_test": len(test),
        "actual_train_frac": len(train) / len(records) if records else 0.0,
        "largest_group": max(map(len, groups), default=0),
        "overlap": overlap,
        "missing_artifacts": sum(k["artifact"] is None for k in keys),
        "missing_sites": sum(k["site"] is None for k in keys),
        "warnings": ["empty partition; grouping constraints retained"] if not train or not test else [],
    }
    return train, test, diagnostics


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
