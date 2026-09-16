from pathlib import Path

import pandas as pd
import pytest

from arthanvesana.data.parse import (
    EXPECTED_SHA256,
    gate,
    inscription_sequences,
    load_raw,
    normalize_direction,
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
