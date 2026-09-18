from __future__ import annotations

import math
import random

import pandas as pd

from arthanvesana.data.parse import analysis_records, normalize_direction, parse_bool
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import split_records


class _SiteSequences(list):
    def __init__(self, records):
        super().__init__(r["sequence"] for r in records)
        self.records = records


def site_records(
    df: pd.DataFrame, min_inscriptions: int = 100, *,
    gap_policy: str = "split", known_direction_only: bool = True,
) -> dict[str, list[dict]]:
    sites = {}
    for record in analysis_records(
        df, gap_policy=gap_policy, known_direction_only=known_direction_only
    ):
        if record["site"] is not None:
            sites.setdefault(record["site"], []).append(record)
    return {
        site: records for site, records in sorted(sites.items())
        if len({r["inscription_id"] for r in records}) >= min_inscriptions
    }


def site_sequences(
    df: pd.DataFrame, min_inscriptions: int = 100, *,
    gap_policy: str = "split", known_direction_only: bool = True,
) -> dict[str, list]:
    return {
        site: _SiteSequences(records) for site, records in site_records(
            df, min_inscriptions, gap_policy=gap_policy,
            known_direction_only=known_direction_only,
        ).items()
    }


def cross_perplexity(
    sites: dict[str, list], method: str = "wittenbell", seed: int = 0, *,
    train_frac: float = 0.8, track: str = "record",
    group_duplicates: bool | None = None,
    common_vocabulary: bool = True, vocabulary: set[str] | None = None,
    equal_train_size: bool | int = False, return_report: bool = False,
    known_direction_only: bool = True,
) -> dict:
    names = sorted(sites)
    records = []
    for name in names:
        items = getattr(sites[name], "records", sites[name])
        for i, item in enumerate(items):
            record = dict(item) if isinstance(item, dict) else {
                "sequence": item, "inscription_id": f"{name}:{i}",
                "start_complete": True,
            }
            known = parse_bool(record.get("reading_order_known", True))
            if "direction" in record:
                known = known and normalize_direction(record["direction"]) != "OTHER"
            if known_direction_only and not known:
                continue
            record["site"] = name
            records.append(record)
    train, test, diagnostics = split_records(
        records, train_frac, seed, track=track, group_duplicates=group_duplicates
    )
    trains = {name: [r for r in train if r["site"] == name] for name in names}
    tests = {name: [r for r in test if r["site"] == name] for name in names}
    before_sizes = {name: len({r["inscription_id"] for r in rows}) for name, rows in trains.items()}
    if equal_train_size is not False:
        if equal_train_size is True:
            size = min(before_sizes.values(), default=0)
        elif isinstance(equal_train_size, int) and equal_train_size >= 0:
            size = equal_train_size
        else:
            raise ValueError("equal_train_size must be a nonnegative integer or boolean")
        if any(count < size for count in before_sizes.values()):
            raise ValueError("equal_train_size exceeds a site's training inscriptions")
        for name in names:
            ids = sorted({r["inscription_id"] for r in trains[name]})
            random.Random(seed).shuffle(ids)
            selected = set(ids[:size])
            trains[name] = [r for r in trains[name] if r["inscription_id"] in selected]
    training_vocab = {sign for rows in trains.values() for r in rows for sign in r["sequence"]}
    support = set(vocabulary) if vocabulary is not None else training_vocab
    if vocabulary is not None and not training_vocab <= support:
        raise ValueError("vocabulary must contain all training signs")
    shared = common_vocabulary or vocabulary is not None
    support = support | {"<UNK>"}
    if "<S>" in support:
        raise ValueError("vocabulary cannot include <S>")
    matrix = {}
    model_vocabularies = {}
    for name in names:
        sequences = [r["sequence"] for r in trains[name]]
        row = {}
        if not any(sequences):
            matrix[name] = {site: float("nan") for site in names}
            model_vocabularies[name] = sorted(support) if shared else ["<UNK>"]
            continue
        model = NGramModel(sequences, 2, method=method)
        if shared:
            model.vocab = set(support)
            model.vocab_order = tuple(sorted(support))
            model.vsize = len(support)
        starts = model.followers[1].get(("<S>",))
        if starts is not None:
            for record in trains[name]:
                if record["sequence"] and not parse_bool(record.get("start_complete", True)):
                    sign = record["sequence"][0]
                    starts[sign] -= 1
                    if starts[sign] == 0:
                        del starts[sign]
        prior = NGramModel(sequences, 1, method=method)
        if shared:
            prior.vocab = set(support)
            prior.vocab_order = tuple(sorted(support))
            prior.vsize = len(support)
        for site in names:
            log_sum = 0.0
            tokens = 0
            for record in tests[site]:
                seq = record["sequence"]
                if not seq:
                    continue
                mapped = model._map(seq)
                for pos, sign in enumerate(mapped):
                    if pos == 0 and not parse_bool(record.get("start_complete", True)):
                        dist = prior.dist(())
                    else:
                        dist = model.dist((mapped[pos - 1] if pos else "<S>",))
                    probability = dist[sign]
                    log_sum += math.log2(probability) if probability > 0 else -math.inf
                    tokens += 1
            row[site] = 2.0 ** (-log_sum / tokens) if tokens else float("nan")
        matrix[name] = row
        model_vocabularies[name] = sorted(model.vocab)
    if not return_report:
        return matrix
    return {
        "matrix": matrix, "split": diagnostics,
        "common_vocabulary": shared,
        "vocabulary": sorted(support) if shared else None,
        "vocabulary_source": "provided" if vocabulary is not None else "training_union",
        "model_vocabularies": model_vocabularies,
        "equal_train_size": equal_train_size,
        "training_size_unit": "inscriptions",
        "sites": {
            name: {
                "n_train_before_equalizing": before_sizes[name],
                "n_train_inscriptions": len({r["inscription_id"] for r in trains[name]}),
                "n_train_spans": len(trains[name]), "n_test_spans": len(tests[name]),
                "train_ids": sorted({r["inscription_id"] for r in trains[name]}),
                "test_ids": sorted({r["inscription_id"] for r in tests[name]}),
                "n_test_oov": sum(s not in support for r in tests[name] for s in r["sequence"]),
            } for name in names
        },
    }
