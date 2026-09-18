from pathlib import Path

import pandas as pd
import pytest

from arthanvesana.data.parse import (
    EXPECTED_SHA256,
    analysis_records,
    gate,
    inscription_sequences,
    load_raw,
    normalize_direction,
    parse_bool,
    sequence_in_reading_order,
    sha256_file,
    to_tidy,
)
from arthanvesana.data.validate import validate_raw, validate_tidy

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "sanitized_corpus.json"


def test_raw_file_hash():
    assert sha256_file(RAW).lower() == EXPECTED_SHA256.lower()


def test_load_and_validate_raw():
    records = load_raw(RAW, verify=True)
    report = validate_raw(records)
    assert report == {"records": 5704, "unique_ids": 5704}


def test_load_rejects_tampered_hash(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_raw(bad, verify=True)


def test_tidy_matches_published_counts():
    df = to_tidy(load_raw(RAW, verify=False))
    report = validate_tidy(df)
    assert report["token_rows"] == 19976
    assert report["analyzable_inscriptions"] == 5536
    assert report["gated_sign_tokens"] == 18065
    assert report["distinct_signs_gated"] == 713


def test_normalize_direction():
    assert normalize_direction("L/R") == "L/R"
    assert normalize_direction("R/L") == "R/L"
    assert normalize_direction("R/L ") == "R/L"
    assert normalize_direction("R/l") == "R/L"
    for other in ("-", "NR", "BUS", "SYM", "T/B", None, ""):
        assert normalize_direction(other) == "OTHER"


def test_reading_order_reversal():
    assert sequence_in_reading_order(["a", "b", "c"], "R/L") == ["c", "b", "a"]
    assert sequence_in_reading_order(["a", "b", "c"], "L/R") == ["a", "b", "c"]
    assert sequence_in_reading_order(["a", "b"], "OTHER") == ["a", "b"]
    assert gate(["a", "000", "b"]) == ["a", "b"]


def test_sequences_reconstruction():
    df = to_tidy(load_raw(RAW, verify=False))
    seqs = inscription_sequences(df)
    assert len(seqs) == 5536
    assert all(len(s) > 0 for s in seqs)
    assert sum(len(s) for s in seqs) == 18065
    first = df[df["inscription_id"] == "INDUS-0001"].sort_values("position")
    assert first["sign_code"].tolist() == ["410", "017"]
    assert seqs[0] == ["410", "017"] or isinstance(seqs[0], list)


def test_csv_roundtrip():
    csv = ROOT / "data" / "processed" / "corpus.csv"
    df = pd.read_csv(csv, encoding="utf-8", dtype={"sign_code": str})
    assert len(df) == 19976
    assert list(df.columns)[:3] == ["inscription_id", "cisi", "site"]
    assert "000" in set(df["sign_code"])
    seqs = inscription_sequences(df)
    assert len(seqs) == 5536
    assert sum(len(s) for s in seqs) == 18065
    assert all("000" not in seq for seq in seqs)


def _gap_frame(direction="L/R", complete=True):
    return to_tidy([{
        "id": "one", "cisi": "catalogue-1", "site": "north",
        "artifact_id": "object-1", "direction": direction,
        "complete": complete, "symbols": ["A", "000", "B", "C"],
    }])


def test_analysis_split_preserves_gaps_and_identity():
    from arthanvesana.stats.ngrams import count_ngrams

    records = analysis_records(_gap_frame())
    assert [r["sequence"] for r in records] == [["A"], ["B", "C"]]
    assert count_ngrams([r["sequence"] for r in records], 2) == {("B", "C"): 1}
    assert {(r["inscription_id"], r["artifact_id"], r["site"]) for r in records} == {
        ("one", "object-1", "north")
    }
    assert [(r["start_complete"], r["end_complete"]) for r in records] == [
        (True, False), (False, True)
    ]
    assert [(r["span_start"], r["span_end"]) for r in records] == [(0, 1), (2, 4)]


def test_analysis_reverse_before_split_and_conservative_boundaries():
    records = analysis_records(_gap_frame("R/l"))
    assert [r["sequence"] for r in records] == [["C", "B"], ["A"]]
    assert [(r["start_complete"], r["end_complete"]) for r in records] == [
        (True, False), (False, True)
    ]
    assert all(not r["start_complete"] and not r["end_complete"]
               for r in analysis_records(_gap_frame(complete="False")))


def test_gap_policies_and_legacy_compatibility():
    frame = _gap_frame()
    assert inscription_sequences(frame) == [["A", "B", "C"]]
    assert inscription_sequences(frame, drop_missing=False) == [["A", "000", "B", "C"]]
    assert inscription_sequences(frame, gap_policy="split") == [["A"], ["B", "C"]]
    assert analysis_records(frame, gap_policy="drop")[0]["sequence"] == ["A", "B", "C"]
    assert analysis_records(frame, gap_policy="reject") == []
    with pytest.raises(ValueError, match="gap_policy"):
        analysis_records(frame, gap_policy="invalid")
    with pytest.raises(ValueError, match="drop_missing"):
        inscription_sequences(frame, drop_missing=False, gap_policy="split")


@pytest.mark.parametrize("direction", ["OTHER", "BUS", "SYM", None])
def test_analysis_excludes_unknown_directions(direction):
    frame = _gap_frame(direction)
    assert analysis_records(frame) == []
    records = analysis_records(frame, known_direction_only=False)
    assert len(records) == 2
    assert all(not r["start_complete"] and not r["end_complete"] for r in records)


def test_string_boolean_direction_filter():
    frame = _gap_frame()
    frame["reading_order_known"] = "False"
    assert analysis_records(frame) == []
    assert inscription_sequences(frame, known_direction_only=True) == []
    frame["reading_order_known"] = " true "
    assert len(analysis_records(frame)) == 2


@pytest.mark.parametrize("value", [False, "False", " false ", "0", 0, "no", "", None, pd.NA])
def test_parse_false_csv_booleans(value):
    assert parse_bool(value) is False


@pytest.mark.parametrize("value", [True, "True", " TRUE ", "1", 1, "yes"])
def test_parse_true_csv_booleans(value):
    assert parse_bool(value) is True


def test_parse_invalid_csv_boolean():
    with pytest.raises(ValueError, match="boolean"):
        parse_bool("maybe")


def test_validate_csv_string_booleans():
    frame = to_tidy(load_raw(RAW, verify=False))
    for column in ("complete", "reading_order_known", "is_missing"):
        frame[column] = frame[column].astype(str)
    assert validate_tidy(frame)["missing_tokens"] == 1911
    assert frame["is_missing"].iloc[0] == "False"
    frame.loc[0, "is_missing"] = "True"
    with pytest.raises(ValueError, match="missing-flags"):
        validate_tidy(frame)


def test_all_missing_and_edge_gaps():
    frame = to_tidy([
        {"id": "a", "direction": "L/R", "complete": True, "symbols": ["000", "000"]},
        {"id": "b", "direction": "L/R", "complete": True, "symbols": ["000", "A", "000", "000", "B", "000"]},
    ])
    records = analysis_records(frame)
    assert [r["sequence"] for r in records] == [["A"], ["B"]]
    assert all(not r["start_complete"] and not r["end_complete"] for r in records)
    assert len(analysis_records(frame, drop_empty=False)) == 3
    assert records == analysis_records(frame.sample(frac=1, random_state=3))


def test_missing_catalogue_identity_falls_back_per_inscription():
    frame = to_tidy([
        {"id": key, "cisi": "-", "direction": "L/R", "symbols": ["A"]}
        for key in ("a", "b")
    ])
    records = analysis_records(frame)
    assert [r["artifact_id"] for r in records] == ["a", "b"]
    assert all(r["artifact_source"] == "inscription" for r in records)
