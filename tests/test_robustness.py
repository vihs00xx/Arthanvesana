import copy
import json
import math

import pytest

from arthanvesana.replicate import robustness as rb
from arthanvesana.replicate.restore import restoration_records
from arthanvesana.stats.sampling import split_records


def record(name, sequence, **metadata):
    return {
        "inscription_id": name, "sequence": list(sequence),
        "start_complete": True, "end_complete": True, "has_gap": False,
        **metadata,
    }


def toy_records():
    return [record(str(i), ["a", str(i)], site=f"s{i % 2}") for i in range(12)]


def test_frequency_and_position_mathematical_rankings():
    train = [record("a", "ba"), record("b", "ca"), record("c", "cccc")]
    counts = rb._global_counts(train)
    assert rb._frequency_rank_map(counts) == {"c": 1, "a": 2, "b": 3}
    position = rb._position_rank_fn(train, counts)
    assert position((2, 0, True, True)) == {"c": 1, "b": 2, "a": 3}
    assert position((2, 1, True, True)) == {"a": 1, "c": 2, "b": 3}
    assert position((9, 0, True, True)) == {"c": 1, "a": 2, "b": 3}
    assert position((2, 1, False, False)) == {"c": 1, "a": 2, "b": 3}
    assert position((2, 1, True, True)) is position((2, 1, True, True))
    result = rb.evaluate_split(train, [record("test", "ba")])
    assert result["models"]["frequency"]["mrr"] == pytest.approx((1 / 3 + 1 / 2) / 2)
    assert result["models"]["position"]["mrr"] == pytest.approx((1 / 2 + 1) / 2)
    assert result["models"]["position"]["top_1"] == 0.5


def test_all_training_candidates_and_full_mrr_beyond_top_ten():
    signs = [f"s{i:02}" for i in range(12)]
    train = [record("all", signs)]
    ranks = rb._position_rank_fn(train, rb._global_counts(train))
    assert len(ranks((12, 0, True, True))) == 12
    assert ranks((12, 0, True, True))["s11"] == 12
    result = rb.evaluate_split(train, [record("test", ["s11"])])
    for model in ("frequency", "position"):
        assert result["models"][model]["top_10"] == 0
        assert result["models"][model]["mrr"] == 1 / 12


def test_singletons_oov_and_context_share_all_targets():
    train = [record("train", "ab")]
    test = [record("one", "a"), record("new", "z"), record("pair", "bz")]
    result = rb.evaluate_split(train, test)
    assert result["candidate_vocab_size"] == 2
    assert result["n_masked"] == 4
    assert result["n_oov"] == 2
    assert result["oov_fraction"] == 0.5
    rows, _ = restoration_records(train, test)
    assert result["models"]["context"] == rb._metric_block([r["rank"] for r in rows], 2)
    for metrics in result["models"].values():
        assert metrics["n_masked"] == 4
        assert metrics["n_oov"] == 2
        assert metrics["top_5"] == metrics["top_10"] == 0.5
    for baseline in ("frequency", "position"):
        delta = result["paired_deltas"][f"context_vs_{baseline}"]
        assert delta["n_pairs"] == 4
        for key in rb.METRIC_KEYS:
            assert delta[key] == pytest.approx(
                result["models"]["context"][key] - result["models"][baseline][key]
            )


def test_ties_are_sign_sorted_and_test_truth_never_fits(monkeypatch):
    train = [record("b", "b"), record("a", "a")]
    original = copy.deepcopy(train)
    calls = []
    real = rb.restoration_records

    def spy(fitted, test, **kwargs):
        calls.append(copy.deepcopy(fitted))
        return real(fitted, test, **kwargs)

    monkeypatch.setattr(rb, "restoration_records", spy)
    for sign, expected in (("a", 1.0), ("b", 0.5), ("z", 0.0)):
        result = rb.evaluate_split(train, [record("target", sign, site="secret")])
        assert result["candidate_vocab_size"] == 2
        for model in rb.MODEL_KEYS:
            assert result["models"][model]["mrr"] == expected
    assert calls == [original] * 3
    assert train == original
    assert rb.evaluate_split(train, [record("t", "b")]) == rb.evaluate_split(
        list(reversed(train)), [record("t", "b")]
    )


def test_reserved_unknown_is_failure_for_every_model():
    result = rb.evaluate_split(
        [record("train", ["a", "<UNK>"])], [record("test", ["<UNK>"])]
    )
    assert result["candidate_vocab_size"] == 1
    for metrics in result["models"].values():
        assert metrics["mrr"] == 0
        assert metrics["n_oov"] == 1


def test_completeness_strings_and_missing_boundaries():
    train = [record("intact", "a"), record("damaged", "b", start_complete="False")]
    ranker = rb._position_rank_fn(train, rb._global_counts(train))
    assert ranker((1, 0, False, True))["b"] == 1
    assert ranker((1, 0, True, True))["a"] == 1
    missing = {"inscription_id": "missing", "sequence": ["a"]}
    assert rb._complete_filter([missing]) == []
    records = train + [record("gap", "c", has_gap="True")]
    assert rb._complete_filter(records) == [train[0]]
    result = rb.evaluate_split(train, [missing])
    rows, _ = restoration_records(train, [dict(missing, start_complete=False, end_complete=False)])
    assert result["models"]["context"]["mrr"] == 1 / rows[0]["rank"]


def test_full_linkage_before_filtering_and_once_per_seed(monkeypatch):
    records = [
        record("a", "x", artifact_id="obj"),
        record("bridge", "y", artifact_id="obj", has_gap=True),
        record("c", "y", artifact_id="other"),
        record("d", "z"), record("e", "w"),
    ]
    calls = []

    def spy(source, *args, **kwargs):
        calls.append(copy.deepcopy(source))
        return split_records(source, *args, **kwargs)

    monkeypatch.setattr(rb, "split_records", spy)
    result = rb.repeated_evaluation(records, seeds=iter(range(3)), train_frac=0.5)
    assert calls == [records] * 3
    for protocol in rb.PROTOCOLS:
        assert len(result[protocol]["runs"]) == 3
        for run in result[protocol]["runs"]:
            assert run["split"]["n_groups"] == 3
            assert run["overlap_diagnostics"]["shared_split_groups"] == 0
            assert run["overlap_diagnostics"]["shared_sequences"] == 0
            partition = set(run["train_ids"])
            retained = set(run["train_ids"] + run["test_ids"])
            if {"a", "c"} <= retained:
                assert ("a" in partition) == ("c" in partition)
            if protocol in ("complete", "complete_unique"):
                assert "bridge" not in retained


def test_dedup_intact_representative_and_reproducible_span_identity():
    damaged = record("a", "x", start_complete=False, span_index=1)
    intact = record("z", "x", span_index=2)
    assert rb._dedup_filter([damaged, intact]) == [intact]
    assert rb._dedup_filter([intact, damaged]) == [intact]
    assert rb._dedup_filter([record("b", "x"), record("a", "x")])[0]["inscription_id"] == "a"
    records = toy_records() + [damaged, intact]
    result = rb.repeated_evaluation(records, seeds=[0], train_frac=0.5)
    run = result["unique"]["runs"][0]
    assert "a" not in run["train_ids"] + run["test_ids"]
    assert any(s["inscription_id"] == "z" and s["span_index"] == 2
               for s in run["train_spans"] + run["test_spans"])


def test_default_ten_seeds_deterministic_and_json_serializable():
    records = toy_records()
    original = copy.deepcopy(records)
    result = rb.repeated_evaluation(records)
    assert result == rb.repeated_evaluation(list(reversed(records)))
    assert records == original
    json.dumps(result, allow_nan=False)
    for protocol in rb.PROTOCOLS:
        assert [run["seed"] for run in result[protocol]["runs"]] == list(range(10))
        assert result[protocol]["aggregate"]["n_ok"] == 10
        for run in result[protocol]["runs"]:
            assert run["n_train_inscriptions"] == len(run["train_ids"])
            assert run["n_test_inscriptions"] == len(run["test_ids"])


def test_site_threshold_counts_inscriptions_not_spans():
    records = [record("one", [str(i)], site="small", span_index=i) for i in range(4)]
    records += [record("two", "x", site="big"), record("three", "y", site="big")]
    result = rb.leave_one_site_out(records, min_inscriptions=2)
    assert result["eligible_sites"] == ["big"]
    assert result["site_inscription_counts"] == {"big": 2, "small": 1}
    assert result["runs"][0]["n_train_inscriptions"] == 1
    assert result["runs"][0]["n_train"] == 4


def test_site_transitive_purge_including_unknown_bridge(monkeypatch):
    records = [
        record("held", "a", site="held", artifact_id="one"),
        record("bridge", "b", site=None, artifact_id="one"),
        record("duplicate", "b", site="other", artifact_id="two"),
        record("transitive", "c", site="other", artifact_id="two"),
        record("safe", "d", site="other"),
        record("unknown-record", "e", site="unknown"),
    ]
    calls = []
    real = rb.evaluate_split

    def spy(train, test):
        calls.append((train, test))
        return real(train, test)

    monkeypatch.setattr(rb, "evaluate_split", spy)
    result = rb.leave_one_site_out(records, min_inscriptions=1)
    held = next(run for run in result["runs"] if run["site"] == "held")
    assert held["train_ids"] == ["safe"]
    assert held["test_ids"] == ["held"]
    assert held["n_purged"] == 2
    assert held["n_unknown_site_excluded"] == 2
    assert all(value == 0 for value in held["overlap_diagnostics"].values())
    assert held["candidate_vocab_size"] == 1
    assert held["oov_fraction"] == 1.0
    for train, test in calls:
        site = test[0]["site"]
        assert all(r["site"] != site and r["site"] not in (None, "unknown") for r in train)
    assert result == rb.leave_one_site_out(list(reversed(records)), min_inscriptions=1)


def test_aggregate_macro_sample_sd_wins_and_skips():
    runs = []
    for value, size in ((0.0, 100), (1.0, 1), (0.5, 10)):
        evaluation = {
            "models": {model: dict.fromkeys(rb.METRIC_KEYS, value) | {"n_masked": size, "n_oov": 0}
                       for model in rb.MODEL_KEYS},
            "paired_deltas": {f"context_vs_{model}": dict.fromkeys(rb.METRIC_KEYS, value - 0.5)
                              for model in ("frequency", "position")},
            "candidate_vocab_size": 2, "oov_fraction": 0.0,
        }
        runs.append({"status": "ok", "evaluation": evaluation})
    runs.append({"status": "skipped", "evaluation": None})
    result = rb.aggregate_runs(runs)
    assert result["n_runs"] == 4
    assert result["n_ok"] == 3
    assert result["n_skipped"] == 1
    assert result["models"]["context"]["top_1"] == {
        "mean": 0.5, "sd": 0.5, "min": 0, "max": 1, "n_runs": 3,
    }
    delta = result["paired_deltas"]["context_vs_frequency"]["mrr"]
    assert delta == {
        "mean": 0, "sd": 0.5, "min": -0.5, "max": 0.5, "n_runs": 3,
        "n_positive": 1, "n_zero": 1, "n_negative": 1,
    }
    assert rb.aggregate_runs(runs[:1])["models"]["context"]["mrr"]["sd"] is None
    assert math.isclose(result["models"]["context"]["n_masked"]["mean"], 37)


@pytest.mark.parametrize("records", [[], [record("empty", [])], [record("only", "a")]])
def test_empty_repeated_partitions_are_transparent(records):
    result = rb.repeated_evaluation(records, seeds=[0])
    for block in result.values():
        run = block["runs"][0]
        assert run["status"] == "skipped" and run["reason"]
        assert run["evaluation"] is None
        assert block["aggregate"]["n_ok"] == 0
        assert block["aggregate"]["models"]["context"]["mrr"]["mean"] is None
    json.dumps(result, allow_nan=False)


def test_empty_site_and_filtered_test_skips():
    result = rb.leave_one_site_out([], min_inscriptions=1)
    assert result["status"] == "skipped" and result["reason"]
    result = rb.leave_one_site_out([record("a", "a", site="s")], min_inscriptions=1)
    assert result["runs"][0]["status"] == "skipped"
    assert all(v == 0 for v in result["runs"][0]["overlap_diagnostics"].values())
    result = rb.repeated_evaluation(toy_records(), seeds=[0], train_frac=1)
    assert all(block["runs"][0]["status"] == "skipped" for block in result.values())
    damaged = [dict(r, start_complete=False) for r in toy_records()]
    result = rb.repeated_evaluation(damaged, seeds=[0])
    assert result["gap_split"]["runs"][0]["status"] == "ok"
    assert result["complete"]["runs"][0]["status"] == "skipped"


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5])
def test_invalid_site_threshold(threshold):
    with pytest.raises(ValueError, match="min_inscriptions"):
        rb.leave_one_site_out([], min_inscriptions=threshold)


def test_invalid_evaluation_inputs_and_empty_seed_iterator():
    with pytest.raises(ValueError, match="training"):
        rb.evaluate_split([], [record("a", "a")])
    with pytest.raises(ValueError, match="test"):
        rb.evaluate_split([record("a", "a")], [])
    with pytest.raises(ValueError, match="training"):
        rb.evaluate_split([record("a", [])], [record("b", "b")])
    with pytest.raises(ValueError, match="train_frac"):
        rb.repeated_evaluation([], seeds=[], train_frac=2)
    assert all(block["runs"] == [] for block in rb.repeated_evaluation([], seeds=[]).values())
