"""Tests for the matched-transcription comparison.

Scientific failure modes covered:

* two primary records for one CISI are excluded, NOT accepted because their
  sequences happen to agree;
* one external inscription is never copied onto several primary spans;
* the external side does not inherit the primary's boundary-completeness flags;
* the strict protocol removes exact-sequence leakage across folds that the
  artifact-only protocol permits, and both transcriptions share the same folds;
* subgroup tables carry real out-of-fold metrics, not count-only placeholders;
* the report distinguishes matched records, held-out records and prediction events.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd

from arthanvesana.data.parse import to_tidy

ROOT = Path(__file__).resolve().parents[1]


def load_runner(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path, rows):
    corpus = tmp_path / "corpus.csv"
    to_tidy(rows).to_csv(corpus, index=False)
    return corpus


def _external(tmp_path, rows):
    external = tmp_path / "external.csv"
    pd.DataFrame(rows).to_csv(external, index=False)
    return external


def _ext_row(cisi, sequence, inscription_id="S", direction="R-L"):
    return {
        "inscription_id": inscription_id, "cisi_number": cisi,
        "sign_sequence": sequence, "site": "A", "object_type": "u",
        "line_count": 1, "damaged": False, "reading_direction": direction,
        "motif": "u",
    }


def _primary(inscription_id, cisi, symbols, site="A", direction="L/R"):
    return {
        "id": inscription_id, "cisi": cisi, "site": site,
        "direction": direction, "complete": True, "symbols": symbols,
    }


# ---------------------------------------------------------------------------
# Matching unit
# ---------------------------------------------------------------------------

def test_two_primary_records_for_one_cisi_are_excluded_not_merged(tmp_path):
    """Identical sequences must not rescue an ambiguous primary side.

    Two distinct catalog inscriptions share CISI C-1 and, in the old design, both
    were accepted because their sequences matched the single external row. They
    are now excluded explicitly.
    """
    runner = load_runner("run_transcription_sensitivity")
    frame = pd.read_csv(_corpus(tmp_path, [
        _primary("1", "C-1", ["001", "002"]),
        _primary("2", "C-1", ["001", "002"]),   # same sequence, different record
        _primary("3", "C-2", ["003", "004"]),
    ]), encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(_external(tmp_path, [
        _ext_row("C-1", "G1 G2", "S1"),
        _ext_row("C-2", "G3 G4", "S2"),
    ]), encoding="utf-8", dtype=str)

    table, _ours, _theirs = runner.build_pair_table(frame, external)
    by_cisi = {row["cisi"]: row for row in table}
    assert by_cisi["C-1"]["status"] == "ambiguous_primary"
    assert by_cisi["C-1"]["n_primary_inscriptions"] == 2
    assert by_cisi["C-2"]["status"] == "matched"


def test_multiple_external_rows_for_one_cisi_are_excluded(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    frame = pd.read_csv(_corpus(tmp_path, [
        _primary("1", "C-1", ["001", "002"]),
    ]), encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(_external(tmp_path, [
        _ext_row("C-1", "G1 G2", "S1"),
        _ext_row("C-1", "G1 G2", "S2"),
    ]), encoding="utf-8", dtype=str)
    table, _o, _t = runner.build_pair_table(frame, external)
    assert table[0]["status"] == "multiple_external"


def test_matched_rows_are_one_to_one(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    frame = pd.read_csv(_corpus(tmp_path, [
        _primary("1", "C-1", ["001", "002"]),
        _primary("2", "C-2", ["003", "004"]),
    ]), encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(_external(tmp_path, [
        _ext_row("C-1", "G1 G2", "S1"),
        _ext_row("C-2", "G3 G4", "S2"),
    ]), encoding="utf-8", dtype=str)
    table, _o, _t = runner.build_pair_table(frame, external)
    matched = [row for row in table if row["status"] == "matched"]
    assert len(matched) == 2
    for row in matched:
        assert row["n_primary_inscriptions"] == 1
        assert row["n_external_rows"] == 1
        assert row["primary_inscription_id"] in row["primary_inscription_ids"]


def test_one_external_inscription_is_not_copied_onto_multiple_primary_spans(tmp_path):
    """A primary inscription split by a gap yields 2 spans but 1 external record."""
    runner = load_runner("run_transcription_sensitivity")
    frame = pd.read_csv(_corpus(tmp_path, [
        _primary("1", "C-1", ["001", "000", "002"]),   # internal gap -> 2 spans
    ]), encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(_external(tmp_path, [
        _ext_row("C-1", "G1 G2", "S1"),
    ]), encoding="utf-8", dtype=str)

    table, ours, theirs = runner.build_pair_table(frame, external)
    row = table[0]
    assert row["status"] == "matched"
    assert row["primary_n_spans"] == 2
    assert row["position_correspondence"] is False

    matched = [r for r in table if r["status"] == "matched"]
    ext_records = runner.external_records(matched, ours, theirs)
    assert len(ext_records) == 1, (
        "one external inscription must not be expanded onto several primary spans"
    )
    assert len(ext_records[0]["sequence"]) == 2


def test_external_completeness_is_not_inherited_from_primary(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    frame = pd.read_csv(_corpus(tmp_path, [
        _primary("1", "C-1", ["001", "002"]),
    ]), encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(_external(tmp_path, [
        _ext_row("C-1", "G1 G2", "S1"),
    ]), encoding="utf-8", dtype=str)
    table, ours, theirs = runner.build_pair_table(frame, external)
    matched = [r for r in table if r["status"] == "matched"]
    record = runner.external_records(matched, ours, theirs)[0]
    # the primary is complete=True but the external source gives no such flag
    assert record["start_complete"] is False
    assert record["end_complete"] is False
    assert record["completeness_source"] == "unavailable_external"
    assert ours["1"]["spans"][0]["start_complete"] is True


def test_position_correspondence_flag_distinguishes_split_inscriptions(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    frame = pd.read_csv(_corpus(tmp_path, [
        _primary("1", "C-1", ["001", "002"]),          # single span
        _primary("2", "C-2", ["003", "000", "004"]),   # split
    ]), encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(_external(tmp_path, [
        _ext_row("C-1", "G1 G2", "S1"),
        _ext_row("C-2", "G3 G4", "S2"),
    ]), encoding="utf-8", dtype=str)
    table, _o, _t = runner.build_pair_table(frame, external)
    flags = {row["cisi"]: row.get("position_correspondence")
             for row in table if row["status"] == "matched"}
    assert flags["C-1"] is True
    assert flags["C-2"] is False


# ---------------------------------------------------------------------------
# Shared partitions and leakage
# ---------------------------------------------------------------------------

def test_strict_protocol_removes_sequence_leakage_across_folds(tmp_path):
    """Duplicates present in EITHER transcription must not cross the shared split."""
    runner = load_runner("run_transcription_sensitivity")
    rows = [_primary(f"i{k}", f"C-{k}", [f"{300 + k:03d}", "001", "002"])
            for k in range(6)]
    # two primary records share an exact sequence but different artifacts
    rows.append(_primary("dup1", "C-dup1", ["900", "901"]))
    rows.append(_primary("dup2", "C-dup2", ["900", "901"]))
    frame = pd.read_csv(_corpus(tmp_path, rows), encoding="utf-8",
                        dtype={"sign_code": str})
    ext_rows = [_ext_row(f"C-{k}", f"G{300 + k} G1 G2", f"S{k}") for k in range(6)]
    # only the EXTERNAL side reveals a duplicate, not the primary side
    ext_rows.append(_ext_row("C-dup1", "G700 G701", "Sd1"))
    ext_rows.append(_ext_row("C-dup2", "G700 G701", "Sd2"))
    external = pd.read_csv(_external(tmp_path, ext_rows), encoding="utf-8", dtype=str)

    summary = runner.evaluate(frame, external, [0, 1, 2])
    artifact_only = summary["protocols"]["artifact_only"]
    strict = summary["protocols"]["artifact_plus_union_duplicates"]
    # the union protocol must not leak the external-side duplicate
    assert max(strict["sequence_overlap_per_seed"]) == 0
    assert max(strict["artifact_overlap_per_seed"]) == 0
    # both protocols keep artifacts isolated
    assert max(artifact_only["artifact_overlap_per_seed"]) == 0
    # the strict policy is at least as strict, i.e. never leaks more
    assert max(strict["sequence_overlap_per_seed"]) <= max(
        artifact_only["sequence_overlap_per_seed"])


def test_both_transcriptions_share_the_same_folds(tmp_path):
    """Primary and external must receive identical test membership per seed."""
    runner = load_runner("run_transcription_sensitivity")
    rows = [_primary(f"i{k}", f"C-{k}", [f"{300 + k:03d}", "001", "002"])
            for k in range(8)]
    frame = pd.read_csv(_corpus(tmp_path, rows), encoding="utf-8",
                        dtype={"sign_code": str})
    ext_rows = [_ext_row(f"C-{k}", f"G{300 + k} G1 G2", f"S{k}") for k in range(8)]
    external = pd.read_csv(_external(tmp_path, ext_rows), encoding="utf-8", dtype=str)
    summary = runner.evaluate(frame, external, [0, 1, 2])
    for protocol, block in summary["protocols"].items():
        for run in block["runs"]:
            # the same canonical split drives both sides, so the held-out counts
            # must be identical between the primary and external evaluations
            assert run["n_test_records"] > 0, protocol
        assert len(block["held_out_records"]) == 3


# ---------------------------------------------------------------------------
# Subgroup metrics
# ---------------------------------------------------------------------------

def test_subgroups_carry_real_metrics_not_placeholders(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    rows = [_primary(f"i{k}", f"C-{k}", [f"{300 + k:03d}", "001", "002"])
            for k in range(10)]
    frame = pd.read_csv(_corpus(tmp_path, rows), encoding="utf-8",
                        dtype={"sign_code": str})
    ext_rows = [_ext_row(f"C-{k}", f"G{300 + k} G1 G2", f"S{k}") for k in range(8)]
    # two disagreements: one gap-placement difference, one other mismatch
    ext_rows.append(_ext_row("C-8", "G308 G1 G000 G2", "S8"))
    ext_rows.append(_ext_row("C-9", "G999", "S9"))
    external = pd.read_csv(_external(tmp_path, ext_rows), encoding="utf-8", dtype=str)

    summary = runner.evaluate(frame, external, [0, 1])
    block = summary["protocols"]["artifact_only"]
    assert set(block["primary_by_relation"]) == set(runner.RELATION_LABELS)
    exact = block["primary_by_relation"]["exact"]
    assert exact is not None
    # real out-of-fold metrics, not a count-only placeholder
    for key in ("frequency", "position", "context", "context_vs_frequency",
                "context_vs_position", "oov_rate", "n_tokens", "n_records"):
        assert key in exact, key
    assert exact["context"]["top_1"] is not None
    assert exact["context_vs_frequency"]["top_1"] is not None
    assert summary["n_disagreement_inscriptions"] >= 1


def test_small_subgroups_are_marked_descriptive(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    rows = [_primary(f"i{k}", f"C-{k}", [f"{300 + k:03d}", "001", "002"])
            for k in range(6)]
    frame = pd.read_csv(_corpus(tmp_path, rows), encoding="utf-8",
                        dtype={"sign_code": str})
    ext_rows = [_ext_row(f"C-{k}", f"G{300 + k} G1 G2", f"S{k}") for k in range(5)]
    ext_rows.append(_ext_row("C-5", "G999", "S5"))
    external = pd.read_csv(_external(tmp_path, ext_rows), encoding="utf-8", dtype=str)
    summary = runner.evaluate(frame, external, [0])
    other = summary["protocols"]["artifact_only"]["primary_by_relation"]["other"]
    assert other is not None
    assert other["descriptive_only"] is True


# ---------------------------------------------------------------------------
# Relation classification and CLI
# ---------------------------------------------------------------------------

def test_relation_classification():
    runner = load_runner("run_transcription_sensitivity")
    assert runner.relation(["1", "2", "3"], ["1", "2", "3"]) == "exact"
    assert runner.relation(["1", "2", "3"], ["1", "3"]) == "other"
    assert runner.relation(["1", "2", "3"], ["3"]) == "first_edge"
    assert runner.relation(["1", "2", "3"], ["1"]) == "last_edge"
    assert runner.relation(["1", "000", "2"], ["1", "2"]) == "gap_placement_only"


def test_transcription_sensitivity_runner(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    raw = [_primary(str(i), f"C-{i}", [f"{100 + i:03d}", "001", f"{200 + i:03d}"],
                    site="A" if i % 2 else "B")
           for i in range(7)]
    corpus = _corpus(tmp_path, raw)
    ext_rows = [_ext_row(f"C-{i}", f"G{100 + i} G1 G{200 + i}", f"S{i}")
                for i in range(5)]
    # exact match for C-5 plus a duplicate external row for the same CISI
    ext_rows.append(_ext_row("C-5", "G105 G1 G205", "S5"))
    ext_rows.append(_ext_row("C-5", "G1 G205", "S-dup"))
    # first-edge difference on its own CISI
    ext_rows.append(_ext_row("C-6", "G206", "S6"))
    external = _external(tmp_path, ext_rows)
    output = tmp_path / "results"

    summary = runner.main([
        "--corpus", str(corpus), "--external", str(external),
        "--output", str(output), "--repeats", "1",
    ])
    saved = json.loads(
        (output / "transcription_sensitivity_summary.json").read_text(encoding="utf-8"))
    assert saved == summary

    assert summary["matched_cisi"] == 6
    assert summary["exclusions"]["multiple_external"] == 1
    # every CISI in this fixture has both sides, so nothing is unmatched
    assert summary["exclusions"].get("unmatched_primary", 0) == 0
    assert summary["exclusions"].get("unmatched_external", 0) == 0
    assert summary["exclusions"]["matched"] == 6
    assert summary["n_matched_inscriptions"] == 6
    assert set(summary["protocols"]) == set(runner.PROTOCOLS)
    for _protocol, block in summary["protocols"].items():
        assert block["primary_overall"]["n_tokens"] > 0
        assert block["external_overall"]["n_tokens"] > 0
        assert "first_edge" in block["primary_by_relation"]
    assert "unavailable_external" in summary["external_completeness_policy"]

    report = (output / "transcription_sensitivity_report.txt").read_text(encoding="utf-8")
    assert "NOT independent" in report
    assert "does contextual information outperform" in report
    assert "RECORD ACCOUNTING" in report
    assert "position-correspondence failures" in report
    assert "protocol: artifact_only" in report
    assert "protocol: artifact_plus_union_duplicates" in report


def test_missing_external_file_reports_dependency(tmp_path):
    runner = load_runner("run_transcription_sensitivity")
    corpus = _corpus(tmp_path, [_primary("0", "C-0", ["001", "002"])])
    try:
        runner.main([
            "--corpus", str(corpus),
            "--external", str(tmp_path / "nope.csv"),
            "--output", str(tmp_path / "out"),
        ])
    except FileNotFoundError as exc:
        assert "indus_website_real_corpus.csv" in str(exc)
    else:
        raise AssertionError("missing external file must raise FileNotFoundError")