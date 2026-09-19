import pandas as pd

from scripts.build_metadata import attach_metadata, build_sidecar


def _frame(rows):
    return pd.DataFrame(rows)


def test_build_sidecar_drops_unknown_and_reports_coverage():
    external = _frame([
        {"inscription_id": "S1", "cisi_number": "Agr-1", "motif": "Bull1",
         "reading_direction": "R-L", "line_count": "1", "object_type": "unknown",
         "sign_sequence": "G1", "site": "s", "damaged": False},
        {"inscription_id": "S2", "cisi_number": "unknown", "motif": "unknown",
         "reading_direction": "R-L", "line_count": "1", "object_type": "unknown",
         "sign_sequence": "G1", "site": "s", "damaged": False},
    ])
    corpus = _frame([
        {"inscription_id": "INDUS-1", "cisi": "Agr-1"},
        {"inscription_id": "INDUS-2", "cisi": "Other-9"},
    ])
    sidecar, report = build_sidecar(external, corpus)
    assert len(sidecar) == 1
    assert sidecar.iloc[0]["cisi"] == "Agr-1"
    assert report["matched_inscriptions"] == 1
    assert report["external_unknown_cisi"] == 1
    assert report["motif_known"] == 1


def test_attach_metadata_marks_missing():
    sidecar = _frame([{
        "cisi": "Agr-1", "ext_id": "S1", "motif": "Bull1",
        "ext_direction": "R-L", "ext_line_count": 1,
        "ext_object_type": "unknown", "source": "test",
    }])
    records = [
        {"inscription_id": "a", "cisi": "Agr-1", "sequence": ["x"]},
        {"inscription_id": "b", "cisi": "Nope-0", "sequence": ["y"]},
    ]
    enriched = attach_metadata(records, sidecar)
    assert enriched[0]["ext_motif"] == "Bull1"
    assert enriched[0]["has_ext_metadata"] is True
    assert enriched[1]["ext_motif"] is None
    assert enriched[1]["has_ext_metadata"] is False
