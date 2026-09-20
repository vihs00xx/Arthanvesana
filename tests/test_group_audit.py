import json

from arthanvesana.stats.sampling import connected_groups, split_records
from scripts import run_group_audit


def _rec(iid, seq, artifact=None, site="S", span_index=0):
    return {
        "inscription_id": iid,
        "artifact_id": artifact,
        "artifact_group": (None, "explicit", artifact) if artifact else None,
        "site": site,
        "sequence": list(seq),
        "span_index": span_index,
        "span_start": 0,
        "start_complete": True,
        "end_complete": True,
    }


def test_connected_groups_transitive_connectivity():
    # A shares artifact with B; B shares exact sequence with C -> one component.
    records = [
        _rec("a", ["1", "2"], artifact="X"),
        _rec("b", ["3", "4"], artifact="X"),
        _rec("c", ["3", "4"], artifact="Y"),
        _rec("d", ["9"], artifact="Z"),
    ]
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    assert sum(len(g["indices"]) for g in groups) == len(records)
    # transitive: a, b, c in one group; d separate
    sizes = sorted(len(g["indices"]) for g in groups)
    assert sizes == [1, 3]
    # every record in exactly one component
    all_idx = [i for g in groups for i in g["indices"]]
    assert sorted(all_idx) == [0, 1, 2, 3]


def test_connected_groups_deterministic_ids():
    records = [
        _rec("a", ["1", "2"], artifact="X"),
        _rec("b", ["1", "2"], artifact="Y"),
        _rec("c", ["7"], artifact="Z"),
    ]
    g1 = connected_groups(records, track="artifact", group_duplicates=True)
    g2 = connected_groups(list(reversed(records)), track="artifact", group_duplicates=True)
    ids1 = sorted(g["group_id"] for g in g1)
    ids2 = sorted(g["group_id"] for g in g2)
    assert ids1 == ids2  # stable under input reordering


def test_connected_groups_match_split_records_grouping():
    records = [
        _rec("a", ["1", "2"], artifact="X"),
        _rec("b", ["3", "4"], artifact="X"),
        _rec("c", ["3", "4"], artifact="Y"),
        _rec("d", ["9"], artifact="Z"),
    ]
    groups = connected_groups(records, track="artifact", group_duplicates=True)
    gid = {}
    for g in groups:
        for i in g["indices"]:
            gid[i] = g["group_id"]
    train, test, _ = split_records(records, 0.5, 0, track="artifact", group_duplicates=True)
    # split_group is the same stable component id; assert no group crosses sides
    train_groups = {r["split_group"] for r in train}
    test_groups = {r["split_group"] for r in test}
    assert not (train_groups & test_groups)
    # and split_group agrees with connected_groups ids
    for r in train + test:
        assert r["split_group"] in set(gid.values())


def test_group_audit_runner(tmp_path):
    raw = [
        {"id": "1", "cisi": "C-1", "site": "A", "direction": "L/R",
         "complete": True, "symbols": ["001", "002"], "artifact_id": "X"},
        {"id": "2", "cisi": "C-2", "site": "A", "direction": "L/R",
         "complete": True, "symbols": ["003", "004"], "artifact_id": "X"},
        {"id": "3", "cisi": "C-3", "site": "B", "direction": "L/R",
         "complete": True, "symbols": ["003", "004"], "artifact_id": "Y"},
        {"id": "4", "cisi": "C-4", "site": "B", "direction": "L/R",
         "complete": True, "symbols": ["009"], "artifact_id": "Z"},
    ]
    from arthanvesana.data.parse import to_tidy
    corpus = tmp_path / "corpus.csv"
    to_tidy(raw).to_csv(corpus, index=False)
    out = tmp_path / "out"
    summary = run_group_audit.main(["--corpus", str(corpus), "--output", str(out)])
    saved = json.loads((out / "group_audit_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
    assert summary["n_records"] == 4
    assert summary["n_components"] == 2  # {1,2,3} via artifact+sequence, {4}
    assert "component_token_percentiles" in summary
    assert len(summary["largest_components"]) >= 1
    report = (out / "group_audit_report.txt").read_text(encoding="utf-8")
    assert "Connected components" in report
    assert "WARNING" in report
