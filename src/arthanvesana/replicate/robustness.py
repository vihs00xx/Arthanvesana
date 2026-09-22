from __future__ import annotations

from collections import Counter
from statistics import fmean, stdev

from arthanvesana.data.parse import identity_value, parse_bool
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.stats.sampling import split_records

DEFAULT_SEEDS = range(10)
TOP_K = (1, 5, 10)
METRIC_KEYS = ("top_1", "top_5", "top_10", "mrr")
MODEL_KEYS = ("frequency", "position", "context")
PROTOCOLS = ("gap_split", "complete", "unique", "complete_unique")
POSITION_BUCKET_PROTOCOL = (
    "Training-only buckets (observed span length, zero-based index, "
    "start_complete, end_complete); rank the entire training sign vocabulary "
    "by descending bucket count, descending global count, then ascending sign. "
    "Unseen bucket tokens follow observed tokens in global frequency order; "
    "unseen cells use global frequency. Observed-span positions are not genuine "
    "inscription edges when boundaries are damaged. Missing flags are unknown "
    "(false); <UNK> is never a candidate."
)
DEDUP_POLICY = (
    "Group full records by artifact/inscription/exact-sequence connectivity "
    "before filtering. Deduplicate exact sequences within each partition, "
    "preferring both complete flags without gaps, then more complete flags, "
    "then no gap, then sorted inscription_id, span_index, span_start, sequence, "
    "site and artifact identity. Complete filtering precedes deduplication."
)
AGGREGATION_PROTOCOL = (
    "Unweighted macro summaries over successful runs; sample standard deviation "
    "(null for fewer than two runs), no confidence intervals for overlapping "
    "repeated splits. Positive paired deltas mean context beats the baseline. "
    "Skipped runs are excluded, not scored as zero."
)


def _global_counts(records):
    counts = Counter()
    for record in records:
        counts.update(sign for sign in record["sequence"] if sign != "<UNK>")
    return counts


def _frequency_rank_map(counts):
    ordered = sorted(counts, key=lambda sign: (-counts[sign], sign))
    return {sign: rank for rank, sign in enumerate(ordered, start=1)}


def _position_rank_fn(train_records, global_counts):
    buckets = {}
    for record in train_records:
        seq = record["sequence"]
        length = len(seq)
        start = parse_bool(record.get("start_complete"))
        end = parse_bool(record.get("end_complete"))
        for pos, sign in enumerate(seq):
            if sign in global_counts:
                buckets.setdefault((length, pos, start, end), Counter())[sign] += 1
    cache = {}
    global_ranks = _frequency_rank_map(global_counts)

    def rank_map(key):
        if key not in buckets:
            return global_ranks
        if key not in cache:
            bucket = buckets[key]
            ordered = sorted(
                global_counts,
                key=lambda sign: (-bucket.get(sign, 0), -global_counts[sign], sign),
            )
            cache[key] = {sign: rank for rank, sign in enumerate(ordered, start=1)}
        return cache[key]

    return rank_map


def _metric_block(ranks, n_oov):
    total = len(ranks)
    block = {}
    for k in TOP_K:
        block[f"top_{k}"] = (
            sum(rank is not None and rank <= k for rank in ranks) / total
            if total else 0.0
        )
    block["mrr"] = (
        sum(1.0 / rank for rank in ranks if rank is not None) / total
        if total else 0.0
    )
    block["n_masked"] = total
    block["n_oov"] = n_oov
    return block


def _paired_block(ctx_ranks, base_ranks):
    total = len(ctx_ranks)
    if not total:
        return {f"top_{k}": 0.0 for k in TOP_K} | {"mrr": 0.0, "n_pairs": 0}
    block = {}
    for k in TOP_K:
        block[f"top_{k}"] = fmean(
            (ctx is not None and ctx <= k) - (base is not None and base <= k)
            for ctx, base in zip(ctx_ranks, base_ranks)
        )
    block["mrr"] = fmean(
        (1.0 / ctx if ctx is not None else 0.0)
        - (1.0 / base if base is not None else 0.0)
        for ctx, base in zip(ctx_ranks, base_ranks)
    )
    block["n_pairs"] = total
    return block


def evaluate_split_records(train_records, test_records):
    """Per-token restoration ranks, grouped by test record.

    Performs exactly the computation :func:`evaluate_split` does — one training
    pass, one context restoration pass — but returns one row per TEST RECORD
    instead of aggregate metrics. Callers can therefore aggregate over arbitrary
    subgroups (transcription agreement classes, sites, length bands) without
    retraining, which is what makes subgroup reporting affordable.

    Tokens whose target sign is outside the training vocabulary get
    ``rank_frequency is None`` (an OOV failure); the same target is an OOV
    failure for every model.
    """
    train_records = [_normalized_record(record) for record in train_records]
    test_records = [_normalized_record(record) for record in test_records]
    if not train_records:
        raise ValueError("evaluate_split_records requires nonempty training records")
    if not any(record["sequence"] for record in test_records):
        raise ValueError("evaluate_split_records requires nonempty test tokens")
    counts = _global_counts(train_records)
    if not counts:
        raise ValueError("evaluate_split_records requires training candidate signs")
    freq_rank = _frequency_rank_map(counts)
    pos_rank = _position_rank_fn(train_records, counts)
    context_rows, _ = restoration_records(train_records, test_records, mask_length=1)

    rows = []
    cursor = 0
    for record in test_records:
        seq = record["sequence"]
        length = len(seq)
        start = parse_bool(record.get("start_complete", True))
        end = parse_bool(record.get("end_complete", True))
        tokens = []
        for pos, sign in enumerate(seq):
            context = context_rows[cursor]
            cursor += 1
            tokens.append({
                "position": pos,
                "target": sign,
                "rank_frequency": freq_rank.get(sign),
                "rank_position": pos_rank((length, pos, start, end)).get(sign),
                "rank_context": context["rank"],
            })
        rows.append({"record": record, "tokens": tokens})
    if cursor != len(context_rows):
        raise ValueError("context evaluation produced misaligned positions")
    return {
        "n_train_records": len(train_records),
        "n_test_records": len(test_records),
        "candidate_vocab_size": len(counts),
        "rows": rows,
    }


def evaluate_split(train_records, test_records):
    per = evaluate_split_records(train_records, test_records)
    freq_ranks = [t["rank_frequency"] for row in per["rows"] for t in row["tokens"]]
    pos_ranks = [t["rank_position"] for row in per["rows"] for t in row["tokens"]]
    ctx_ranks = [t["rank_context"] for row in per["rows"] for t in row["tokens"]]

    n_oov = sum(rank is None for rank in freq_ranks)
    models = {
        "frequency": _metric_block(freq_ranks, n_oov),
        "position": _metric_block(pos_ranks, n_oov),
        "context": _metric_block(ctx_ranks, n_oov),
    }
    paired_deltas = {
        "context_vs_frequency": _paired_block(ctx_ranks, freq_ranks),
        "context_vs_position": _paired_block(ctx_ranks, pos_ranks),
    }
    return {
        "n_train_records": per["n_train_records"],
        "n_test_records": per["n_test_records"],
        "candidate_vocab_size": per["candidate_vocab_size"],
        "n_masked": len(freq_ranks),
        "n_oov": n_oov,
        "oov_fraction": n_oov / len(freq_ranks) if freq_ranks else 0.0,
        "models": models,
        "paired_deltas": paired_deltas,
    }


def _normalized_record(record):
    return dict(
        record,
        start_complete=parse_bool(record.get("start_complete")),
        end_complete=parse_bool(record.get("end_complete")),
        has_gap=parse_bool(record.get("has_gap", False)),
    )


def _record_sort_key(record):
    return (
        str(record.get("inscription_id")),
        record.get("span_index", 0),
        record.get("span_start", 0),
        tuple(record["sequence"]),
    )


def _complete_filter(records):
    return [
        record for record in records
        if parse_bool(record.get("start_complete"))
        and parse_bool(record.get("end_complete"))
        and not parse_bool(record.get("has_gap", False))
    ]


def _dedup_sort_key(record):
    start = parse_bool(record.get("start_complete"))
    end = parse_bool(record.get("end_complete"))
    gap = parse_bool(record.get("has_gap", False))
    return (
        not (start and end and not gap), -(int(start) + int(end)), gap,
        _record_sort_key(record), identity_value(record.get("site")) or "",
        repr(record.get("artifact_group")),
        identity_value(record.get("artifact_id")) or "", start, end,
    )


def _dedup_filter(records):
    chosen = {}
    for record in sorted(records, key=_dedup_sort_key):
        chosen.setdefault(tuple(record["sequence"]), record)
    return sorted(chosen.values(), key=_record_sort_key)


def _apply_protocol(protocol, records):
    if protocol == "gap_split":
        return list(records)
    if protocol == "complete":
        return _complete_filter(records)
    if protocol == "unique":
        return _dedup_filter(records)
    if protocol == "complete_unique":
        return _dedup_filter(_complete_filter(records))
    raise ValueError("protocol must be gap_split, complete, unique, or complete_unique")


def _summarize(values):
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None, "n_runs": 0}
    return {
        "mean": fmean(values),
        "sd": stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "n_runs": len(values),
    }


def _sign_counts(values):
    return {
        "n_positive": sum(v > 0 for v in values),
        "n_zero": sum(v == 0 for v in values),
        "n_negative": sum(v < 0 for v in values),
    }


def aggregate_runs(runs):
    ok = [run for run in runs if run.get("status") == "ok"]
    aggregate = {
        "n_runs": len(runs),
        "n_ok": len(ok),
        "n_skipped": len(runs) - len(ok),
    }
    aggregate["aggregation_policy"] = AGGREGATION_PROTOCOL
    if not ok:
        aggregate["reason"] = "no successful runs"
    models = {}
    for model in MODEL_KEYS:
        block = {
            key: _summarize([run["evaluation"]["models"][model][key] for run in ok])
            for key in METRIC_KEYS
        }
        block["n_masked"] = _summarize(
            [run["evaluation"]["models"][model]["n_masked"] for run in ok]
        )
        block["n_oov"] = _summarize(
            [run["evaluation"]["models"][model]["n_oov"] for run in ok]
        )
        models[model] = block
    paired = {}
    for baseline, delta_key in (
        ("context_vs_frequency", "context_vs_frequency"),
        ("context_vs_position", "context_vs_position"),
    ):
        block = {}
        for key in METRIC_KEYS:
            values = [run["evaluation"]["paired_deltas"][delta_key][key] for run in ok]
            block[key] = _summarize(values)
            block[key].update(_sign_counts(values))
        paired[baseline] = block
    aggregate["models"] = models
    aggregate["paired_deltas"] = paired
    aggregate["oov_fraction"] = _summarize(
        [run["evaluation"]["oov_fraction"] for run in ok]
    )
    aggregate["candidate_vocab_size"] = _summarize(
        [run["evaluation"]["candidate_vocab_size"] for run in ok]
    )
    return aggregate


def _partition_ids(records):
    return sorted({str(record["inscription_id"]) for record in records})


def _span_ids(records):
    return [
        {"inscription_id": str(record["inscription_id"]),
         "span_index": record.get("span_index", 0),
         "span_start": record.get("span_start", 0),
         "split_group": record.get("split_group")}
        for record in sorted(records, key=_record_sort_key)
    ]


def _overlap(train, test):
    return {
        "shared_inscription_ids": len(set(_partition_ids(train)) & set(_partition_ids(test))),
        "shared_split_groups": len(
            {r["split_group"] for r in train} & {r["split_group"] for r in test}
        ),
        "shared_sequences": len(
            {tuple(r["sequence"]) for r in train} & {tuple(r["sequence"]) for r in test}
        ),
        "shared_sites": len(
            {identity_value(r.get("site")) for r in train}
            & {identity_value(r.get("site")) for r in test} - {None}
        ),
    }


def _run(train, test):
    counts = _global_counts(train)
    n_masked = sum(len(r["sequence"]) for r in test)
    n_oov = sum(sign not in counts for r in test for sign in r["sequence"])
    run = {
        "status": "ok", "reason": None, "evaluation": None,
        "n_train": len(train), "n_test": len(test),
        "n_train_inscriptions": len(_partition_ids(train)),
        "n_test_inscriptions": len(_partition_ids(test)),
        "train_ids": _partition_ids(train), "test_ids": _partition_ids(test),
        "train_spans": _span_ids(train), "test_spans": _span_ids(test),
        "candidate_vocab_size": len(counts), "n_masked": n_masked, "n_oov": n_oov,
        "oov_fraction": n_oov / n_masked if n_masked else None,
        "overlap_diagnostics": _overlap(train, test),
    }
    if not counts:
        run.update(status="skipped", reason="empty training candidate vocabulary")
    elif not n_masked:
        run.update(status="skipped", reason="empty test token partition")
    else:
        run["evaluation"] = evaluate_split(train, test)
    return run


def repeated_evaluation(records, *, seeds=DEFAULT_SEEDS, train_frac=0.8):
    records = list(records)
    if not 0 <= train_frac <= 1:
        raise ValueError("train_frac must be between 0 and 1")
    results = {protocol: {"runs": [], "aggregate": None} for protocol in PROTOCOLS}
    for seed in seeds:
        train_full, test_full, diagnostics = split_records(
            records, train_frac, seed, track="artifact", group_duplicates=True
        )
        for protocol in PROTOCOLS:
            train = _apply_protocol(protocol, train_full)
            test = _apply_protocol(protocol, test_full)
            run = _run(train, test)
            run.update(protocol=protocol, seed=seed, split=diagnostics)
            results[protocol]["runs"].append(run)
    for result in results.values():
        result["aggregate"] = aggregate_runs(result["runs"])
    return results


def leave_one_site_out(records, *, min_inscriptions=100):
    if isinstance(min_inscriptions, bool) or not isinstance(min_inscriptions, int) or min_inscriptions < 1:
        raise ValueError("min_inscriptions must be a positive integer")
    records = list(records)
    site_inscriptions = {}
    for record in records:
        site = identity_value(record.get("site"))
        if site is not None:
            site_inscriptions.setdefault(site, set()).add(str(record["inscription_id"]))
    eligible = sorted(
        site for site, ids in site_inscriptions.items() if len(ids) >= min_inscriptions
    )
    pooled, _, grouping = split_records(
        records, 1.0, track="artifact", group_duplicates=True
    )
    runs = []
    for site in eligible:
        held_groups = {
            record["split_group"]
            for record in pooled if identity_value(record.get("site")) == site
        }
        test = [
            record for record in pooled if identity_value(record.get("site")) == site
        ]
        train = []
        n_purged = 0
        n_unknown_site_excluded = 0
        for record in pooled:
            record_site = identity_value(record.get("site"))
            if record_site is None:
                n_unknown_site_excluded += 1
                continue
            if record_site == site:
                continue
            if record["split_group"] in held_groups:
                n_purged += 1
                continue
            train.append(record)
        run = _run(train, test)
        run.update(
            site=site, n_purged=n_purged,
            n_unknown_site_excluded=n_unknown_site_excluded,
        )
        runs.append(run)
    return {
        "min_inscriptions": min_inscriptions,
        "eligible_sites": eligible,
        "site_inscription_counts": {site: len(ids) for site, ids in sorted(site_inscriptions.items())},
        "grouping": grouping,
        "status": "ok" if any(run["status"] == "ok" for run in runs) else "skipped",
        "reason": None if any(run["status"] == "ok" for run in runs) else "no eligible successful site runs",
        "purge_policy": "n_purged counts other known-site spans linked to held-site groups; unknown-site exclusions are counted separately",
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
