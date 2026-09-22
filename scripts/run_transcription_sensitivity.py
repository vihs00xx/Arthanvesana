"""Same-family transcription sensitivity, with an explicit pair table.

Scientific question: does contextual information outperform frequency and
position under BOTH same-family ICIT transcriptions?

Matching unit
-------------
Pairs are formed at the **artifact/inscription level, before gap splitting**, in
an explicit pair table. The primary comparison uses only CISI where there is
**exactly one unambiguous record on each side**. Multiple primary records are
never accepted merely because their sequences happen to agree, and one full
external inscription is never copied onto several primary spans.

Source-specific information is preserved: the external side keeps its own
(missing) completeness information rather than inheriting the primary's boundary
flags, because the external source does not supply them. When a primary
inscription splits into several spans, position correspondence between the two
transcriptions cannot be established, so those records are counted, excluded from
position-aligned subgroup metrics, and their differing token counts are reported.

Shared partitions
-----------------
Two grouping protocols are reported, both assigning the SAME folds to the same
matched artifacts in both transcriptions:

* ``artifact_only`` — groups by artifact identity.
* ``artifact_plus_union_duplicates`` — additionally groups the union of
  exact-sequence equivalence relations found in EITHER transcription, so neither
  source can leak an equivalent sequence across the shared split.

Agreement between these ICIT-derived sources is NOT independent inter-annotator
agreement. Raw external sequences are not redistributed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from arthanvesana.data.parse import analysis_records, sha256_file
from arthanvesana.replicate.robustness import (
    _metric_block,
    _paired_block,
    evaluate_split_records,
)
from arthanvesana.stats.sampling import split_records
from scripts.run_audit import normalize_gcode

EXT_DIRECTION_MAP = {"R-L": "L/R", "L-R": "R/L"}

PROTOCOLS = ("artifact_only", "artifact_plus_union_duplicates")

#: Subgroup labels reported separately.
RELATION_LABELS = ("exact", "gap_placement_only", "first_edge", "last_edge", "other")
#: Below this many evaluated tokens a subgroup is descriptive only.
MIN_SUBGROUP_TOKENS = 50


def external_sequences(external):
    """Parse the external CSV into per-CISI rows in reading order."""
    out = {}
    for _, row in external.iterrows():
        cisi = row["cisi_number"]
        if cisi == "unknown":
            continue
        codes = [normalize_gcode(c) for c in str(row["sign_sequence"]).split()]
        direction = EXT_DIRECTION_MAP.get(str(row["reading_direction"]), "OTHER")
        if direction == "R/L":
            codes = list(reversed(codes))
        out.setdefault(cisi, []).append({
            "cisi": cisi, "sequence": codes, "direction": direction,
            "site": str(row.get("site", "unknown")),
            "object_type": str(row.get("object_type", "unknown")),
            "damaged_raw": str(row.get("damaged", "unknown")),
            "reading_direction_raw": str(row.get("reading_direction", "unknown")),
            "external_inscription_id": str(row.get("inscription_id", "unknown")),
        })
    return out


def relation(mine, theirs):
    """Audit-relation classification for one matched pair (raw comparison)."""
    if theirs == mine:
        return "exact"
    if [x for x in theirs if x != "000"] == [x for x in mine if x != "000"]:
        return "gap_placement_only"
    if theirs == mine[2:]:
        return "first_edge"
    if theirs == mine[:-2]:
        return "last_edge"
    return "other"


def primary_inscriptions(frame):
    """One entry per primary inscription: its spans and its gated full sequence."""
    spans = analysis_records(frame, gap_policy="split", known_direction_only=False)
    by_inscription = {}
    for record in spans:
        by_inscription.setdefault(record["inscription_id"], []).append(record)
    out = {}
    for inscription_id, records in by_inscription.items():
        records = sorted(records, key=lambda r: r.get("span_index", 0))
        out[inscription_id] = {
            "inscription_id": inscription_id,
            "spans": records,
            "sequence": [s for r in records for s in r["sequence"]],
            "artifact_id": records[0].get("artifact_id"),
            "artifact_group": records[0].get("artifact_group"),
            "cisi": records[0].get("cisi"),
            "site": records[0].get("site"),
            "n_spans": len(records),
            "has_gap": any(r.get("has_gap") for r in records),
        }
    return out


def build_pair_table(frame, external):
    """Explicit artifact/inscription-level pair table, built before gap splitting.

    Each row records both sides' counts and the reason for inclusion or
    exclusion. Only ``status == "matched"`` rows enter the primary comparison, and
    a matched row has exactly one primary inscription and exactly one external
    row.
    """
    ours = primary_inscriptions(frame)
    theirs = external_sequences(external)

    by_cisi = {}
    for inscription_id, entry in ours.items():
        if entry["cisi"]:
            by_cisi.setdefault(entry["cisi"], []).append(inscription_id)

    table = []
    for cisi in sorted(set(by_cisi) | set(theirs)):
        mine = sorted(by_cisi.get(cisi, []))
        ext_rows = theirs.get(cisi, [])
        row = {
            "cisi": cisi,
            "n_primary_inscriptions": len(mine),
            "n_external_rows": len(ext_rows),
            "primary_inscription_ids": mine,
            "external_inscription_ids": sorted(
                r["external_inscription_id"] for r in ext_rows),
        }
        if not mine and ext_rows:
            status = "unmatched_external"
        elif mine and not ext_rows:
            status = "unmatched_primary"
        elif len(mine) > 1:
            # repeated catalog records / fragments: never resolved by matching
            # sequences, because that is exactly the error this guards against
            status = "ambiguous_primary"
        elif len(ext_rows) > 1:
            status = "multiple_external"
        else:
            entry = ours[mine[0]]
            ext_seq = ext_rows[0]["sequence"]
            row.update({
                "primary_inscription_id": mine[0],
                "primary_n_spans": entry["n_spans"],
                "primary_n_tokens": len(entry["sequence"]),
                "external_n_tokens": len(ext_seq),
                # position-by-position correspondence is only definable when the
                # primary inscription was not split at an internal gap
                "position_correspondence": entry["n_spans"] == 1,
                "token_counts_differ": len(entry["sequence"]) != len(ext_seq),
            })
            if not ext_seq or not entry["sequence"]:
                status = "empty_sequence"
            else:
                status = "matched"
                row["audit_relation"] = relation(entry["sequence"], ext_seq)
        row["status"] = status
        table.append(row)
    return table, ours, theirs


def union_duplicate_labels(primary_seq, external_seq):
    """Union of exact-sequence equivalence relations across BOTH transcriptions.

    Inscriptions are linked when their sequences agree in either transcription,
    so the resulting components are at least as strict as either source alone.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for seq_by_inscription in (primary_seq, external_seq):
        find_all = list(seq_by_inscription)
        for inscription_id in find_all:
            find(inscription_id)
        by_sequence = {}
        for inscription_id, seq in seq_by_inscription.items():
            by_sequence.setdefault(tuple(seq), []).append(inscription_id)
        for inscription_ids in by_sequence.values():
            for other in inscription_ids[1:]:
                union(inscription_ids[0], other)
    return {inscription_id: find(inscription_id) for inscription_id in parent}


def _canonical_records(matched, primary_inscriptions_map):
    """One canonical record per matched inscription, used only for splitting."""
    records = []
    for row in matched:
        entry = primary_inscriptions_map[row["primary_inscription_id"]]
        records.append({
            "inscription_id": row["primary_inscription_id"],
            "artifact_id": entry["artifact_id"],
            "artifact_group": entry["artifact_group"],
            "site": entry["site"],
            "sequence": entry["sequence"],
            "span_index": 0,
            "span_start": 0,
            "start_complete": True,
            "end_complete": True,
        })
    return records


def external_records(matched, primary_inscriptions_map, theirs):
    """One external record per matched inscription.

    The external side does NOT inherit the primary's ``start_complete`` /
    ``end_complete`` flags: the external source does not supply completeness, so
    those are recorded as unknown (``False``) rather than silently transferred.
    """
    records = []
    for row in matched:
        entry = primary_inscriptions_map[row["primary_inscription_id"]]
        ext = theirs[row["cisi"]][0]
        records.append({
            "inscription_id": row["primary_inscription_id"],
            "artifact_id": entry["artifact_id"],
            "artifact_group": entry["artifact_group"],
            "site": entry["site"],
            "sequence": list(ext["sequence"]),
            "span_index": 0,
            "span_start": 0,
            "start_complete": False,
            "end_complete": False,
            "completeness_source": "unavailable_external",
            "damaged_raw": ext["damaged_raw"],
            "reading_direction_raw": ext["reading_direction_raw"],
        })
    return records


def _split_canonical(canonical, protocol, union_labels, seed, train_frac=0.8):
    if protocol == "artifact_only":
        train, test, diagnostics = split_records(
            canonical, train_frac, seed, track="artifact", group_duplicates=False)
    else:
        keys = [union_labels.get(r["inscription_id"], r["inscription_id"])
                for r in canonical]
        train, test, diagnostics = split_records(
            canonical, train_frac, seed, track="artifact",
            group_duplicates=False, group_keys=keys)
    return ({r["inscription_id"] for r in train},
            {r["inscription_id"] for r in test},
            diagnostics)


def _empty_bucket():
    return {"frequency": [], "position": [], "context": [], "n_records": 0, "n_oov": 0}


def _accumulate(store, per_record, label_of):
    for row in per_record["rows"]:
        inscription_id = row["record"]["inscription_id"]
        label = label_of.get(inscription_id, "unmatched")
        bucket = store.setdefault(label, _empty_bucket())
        bucket["n_records"] += 1
        for token in row["tokens"]:
            bucket["frequency"].append(token["rank_frequency"])
            bucket["position"].append(token["rank_position"])
            bucket["context"].append(token["rank_context"])
            if token["rank_frequency"] is None:
                bucket["n_oov"] += 1


def _block(bucket):
    """Out-of-fold metrics for one subgroup, with sample sizes and an OOV rate."""
    if not bucket["frequency"]:
        return None
    n_tokens = len(bucket["frequency"])
    return {
        "n_records": bucket["n_records"],
        "n_tokens": n_tokens,
        "oov_rate": bucket["n_oov"] / n_tokens,
        "frequency": _metric_block(bucket["frequency"], bucket["n_oov"]),
        "position": _metric_block(bucket["position"], bucket["n_oov"]),
        "context": _metric_block(bucket["context"], bucket["n_oov"]),
        "context_vs_frequency": _paired_block(bucket["context"], bucket["frequency"]),
        "context_vs_position": _paired_block(bucket["context"], bucket["position"]),
        "descriptive_only": n_tokens < MIN_SUBGROUP_TOKENS,
    }


def _merge_blocks(blocks):
    if not blocks:
        return None
    merged = _empty_bucket()
    for block in blocks:
        for key in ("frequency", "position", "context"):
            merged[key].extend(block[key])
        merged["n_records"] += block["n_records"]
        merged["n_oov"] += block["n_oov"]
    return _block(merged)


def evaluate(frame, external, seeds):
    table, ours, theirs = build_pair_table(frame, external)
    matched = [row for row in table if row["status"] == "matched"]
    matched_ids = [row["primary_inscription_id"] for row in matched]
    canonical = _canonical_records(matched, ours)
    ext_records = external_records(matched, ours, theirs)

    primary_seq = {row["primary_inscription_id"]:
                   ours[row["primary_inscription_id"]]["sequence"] for row in matched}
    external_seq = {row["primary_inscription_id"]:
                    list(theirs[row["cisi"]][0]["sequence"]) for row in matched}
    union_labels = union_duplicate_labels(primary_seq, external_seq)

    position_ok = {row["primary_inscription_id"]: row["position_correspondence"]
                   for row in matched}
    relation_of = {row["primary_inscription_id"]: row.get("audit_relation", "other")
                   for row in matched}

    exclusions = Counter(row["status"] for row in table)

    protocols = {}
    for protocol in PROTOCOLS:
        primary_store, external_store = {}, {}
        overlap = []
        held_out = []
        for seed in seeds:
            train_ids, test_ids, diagnostics = _split_canonical(
                canonical, protocol, union_labels, seed)
            overlap.append({
                "seed": seed,
                "sequence_overlap": diagnostics["overlap"]["sequence"],
                "artifact_overlap": diagnostics["overlap"]["artifact"],
                "n_train_records": diagnostics["n_train"],
                "n_test_records": diagnostics["n_test"],
                "largest_group": diagnostics["largest_group"],
            })
            held_out.append(len(test_ids))
            primary_train = [r for row in ours.values() for r in row["spans"]
                             if row["inscription_id"] in train_ids]
            primary_test = [r for row in ours.values() for r in row["spans"]
                            if row["inscription_id"] in test_ids]
            ext_train = [r for r in ext_records if r["inscription_id"] in train_ids]
            ext_test = [r for r in ext_records if r["inscription_id"] in test_ids]
            if not primary_train or not primary_test:
                continue
            _accumulate(primary_store,
                        evaluate_split_records(primary_train, primary_test),
                        relation_of)
            if ext_train and ext_test:
                _accumulate(external_store,
                            evaluate_split_records(ext_train, ext_test),
                            relation_of)
        # component sizes under this protocol
        if protocol == "artifact_only":
            labels = [str(canonical[i]["artifact_group"] or canonical[i]["artifact_id"])
                      for i in range(len(canonical))]
        else:
            labels = [union_labels.get(r["inscription_id"], r["inscription_id"])
                      for r in canonical]
        sizes = Counter(labels)
        ordered = sorted(sizes.values())
        group_block = {
            "n_groups": len(sizes),
            "largest_group_records": ordered[-1] if ordered else 0,
            "median_group_records": ordered[len(ordered) // 2] if ordered else 0,
        }
        protocols[protocol] = {
            "grouping_policy": (
                "artifact identity only" if protocol == "artifact_only"
                else "artifact identity plus the union of exact-sequence "
                     "equivalence relations found in either transcription"),
            **group_block,
            "runs": overlap,
            "sequence_overlap_per_seed": [o["sequence_overlap"] for o in overlap],
            "artifact_overlap_per_seed": [o["artifact_overlap"] for o in overlap],
            "held_out_records": held_out,
            "primary_overall": _block({
                "frequency": [v for b in primary_store.values() for v in b["frequency"]],
                "position": [v for b in primary_store.values() for v in b["position"]],
                "context": [v for b in primary_store.values() for v in b["context"]],
                "n_records": sum(b["n_records"] for b in primary_store.values()),
                "n_oov": sum(b["n_oov"] for b in primary_store.values()),
            }),
            "external_overall": _block({
                "frequency": [v for b in external_store.values() for v in b["frequency"]],
                "position": [v for b in external_store.values() for v in b["position"]],
                "context": [v for b in external_store.values() for v in b["context"]],
                "n_records": sum(b["n_records"] for b in external_store.values()),
                "n_oov": sum(b["n_oov"] for b in external_store.values()),
            }) if external_store else None,
            "primary_by_relation": {
                label: _block(primary_store.get(label, _empty_bucket()))
                for label in RELATION_LABELS
            },
            "external_by_relation": {
                label: _block(external_store.get(label, _empty_bucket()))
                for label in RELATION_LABELS
            } if external_store else {},
        }

    # "disagreement" = every matched pair that is not an exact match
    disagree = [i for i in matched_ids if relation_of.get(i) != "exact"]
    return {
        "pair_table": table,
        "matched_cisi": len(matched),
        "exclusions": dict(exclusions),
        "n_matched_inscriptions": len(matched),
        "n_matched_primary_spans": sum(
            ours[i]["n_spans"] for i in matched_ids),
        "n_position_aligned_inscriptions": sum(
            1 for i in matched_ids if position_ok[i]),
        "n_position_mismatch_inscriptions": sum(
            1 for i in matched_ids if not position_ok[i]),
        "n_token_count_mismatch_inscriptions": sum(
            1 for row in matched if row["token_counts_differ"]),
        "n_disagreement_inscriptions": len(disagree),
        "seeds": list(seeds),
        "protocols": protocols,
        "external_completeness_policy": (
            "the external source supplies no boundary completeness, so the "
            "primary's start_complete/end_complete flags are NOT transferred; "
            "external records are marked unknown (False) and labelled "
            "completeness_source='unavailable_external'"),
    }


def render_report(summary):
    lines = [
        "Same-family transcription sensitivity (explicit artifact-level pair table)",
        "",
        "Scientific question: does contextual information outperform frequency and",
        "position under BOTH same-family transcriptions?",
        "",
        "Matching unit: pairs are formed at the artifact/inscription level BEFORE",
        "gap splitting. Only CISI with exactly one unambiguous record on each side",
        "enter the primary comparison. Multiple primary records are never merged",
        "because their sequences agree, and one external inscription is never",
        "copied onto several primary spans.",
        f"External completeness: {summary['external_completeness_policy']}",
        "",
        "Agreement between these ICIT-derived sources is NOT independent",
        "inter-annotator agreement. Raw external sequences are not redistributed.",
        "",
        "RECORD ACCOUNTING (these three numbers are different):",
        f"  total matched records (inscriptions): {summary['n_matched_inscriptions']}",
        f"  primary spans underlying them:       {summary['n_matched_primary_spans']}",
        f"  held-out records per seed:           "
        f"{summary['protocols'][PROTOCOLS[0]]['held_out_records']}",
        f"  position-aligned inscriptions:       "
        f"{summary['n_position_aligned_inscriptions']}",
        f"  position-correspondence failures:    "
        f"{summary['n_position_mismatch_inscriptions']} "
        "(primary split at an internal gap; excluded from position-aligned strata)",
        f"  token-count mismatches:              "
        f"{summary['n_token_count_mismatch_inscriptions']}",
        f"  disagreement pairs (not exact):      "
        f"{summary['n_disagreement_inscriptions']}",
        "",
        "Pair-table exclusions: " + "; ".join(
            f"{k}={v}" for k, v in sorted(summary["exclusions"].items())),
        "",
    ]
    for protocol in PROTOCOLS:
        block = summary["protocols"][protocol]
        lines.append(f"=== protocol: {protocol} ===")
        lines.append(f"  grouping: {block['grouping_policy']}")
        lines.append(f"  sequence overlap across folds per seed: "
                     f"{block['sequence_overlap_per_seed']}")
        lines.append(f"  artifact overlap across folds per seed: "
                     f"{block['artifact_overlap_per_seed']}")
        for side in ("primary_overall", "external_overall"):
            metrics = block[side]
            if not metrics:
                lines.append(f"  {side}: no data")
                continue
            lines.append(
                f"  {side}: records {metrics['n_records']}, tokens "
                f"{metrics['n_tokens']}, OOV {metrics['oov_rate']:.4f}, "
                f"context top-1 {metrics['context']['top_1']:.4f}, "
                f"frequency {metrics['frequency']['top_1']:.4f}, "
                f"position {metrics['position']['top_1']:.4f}, "
                f"ctx-freq {metrics['context_vs_frequency']['top_1']:+.4f}, "
                f"ctx-pos {metrics['context_vs_position']['top_1']:+.4f}")
        lines.append("  by relation (out-of-fold, descriptive where small):")
        lines.append("    relation | side | records | tokens | OOV | context | "
                     "freq | pos | ctx-freq | ctx-pos")
        for label in RELATION_LABELS:
            for side, table in (("primary", block["primary_by_relation"]),
                                ("external", block["external_by_relation"])):
                metrics = table.get(label)
                if not metrics:
                    lines.append(f"    {label} | {side} | 0 | 0 | - | - | - | - | - | -")
                    continue
                lines.append(
                    f"    {label} | {side} | {metrics['n_records']} | "
                    f"{metrics['n_tokens']} | {metrics['oov_rate']:.4f} | "
                    f"{metrics['context']['top_1']:.4f} | "
                    f"{metrics['frequency']['top_1']:.4f} | "
                    f"{metrics['position']['top_1']:.4f} | "
                    f"{metrics['context_vs_frequency']['top_1']:+.4f} | "
                    f"{metrics['context_vs_position']['top_1']:+.4f}"
                    + ("  [descriptive: small]" if metrics["descriptive_only"] else ""))
        lines.append("")
    lines.extend([
        "These are artificial single-sign masks, not verified restorations.",
        f"Subgroups below {MIN_SUBGROUP_TOKENS} evaluated tokens are marked "
        "descriptive and carry no claim.",
        "SD/intervals describe split variability, NOT confidence intervals.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Same-family transcription sensitivity")
    parser.add_argument("--corpus", type=Path, default=ROOT / "data" / "processed" / "corpus.csv")
    parser.add_argument("--external", type=Path,
                        default=ROOT / "data" / "external" / "indus_website_real_corpus.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "transcription_sensitivity")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.external.exists():
        raise FileNotFoundError(
            f"External transcription file missing: {args.external}. "
            "Expected data/external/indus_website_real_corpus.csv "
            "(joyboseroy/indus_decipher, config indus_website). "
            "Provide it with --external; nothing is downloaded."
        )
    frame = pd.read_csv(args.corpus, encoding="utf-8", dtype={"sign_code": str})
    external = pd.read_csv(args.external, encoding="utf-8", dtype=str)
    seeds = list(range(args.seed, args.seed + args.repeats))

    result = evaluate(frame, external, seeds)
    summary = {
        "manifest": {
            "corpus_sha256": sha256_file(args.corpus),
            "external_sha256": sha256_file(args.external),
            "source_sha256": {
                "scripts/run_transcription_sensitivity.py": sha256_file(
                    Path(__file__).resolve()),
                "src/arthanvesana/replicate/robustness.py": sha256_file(
                    ROOT / "src" / "arthanvesana" / "replicate" / "robustness.py"),
            },
            "python": platform.python_version(),
            "dependencies": {n: version(n) for n in ("numpy", "pandas", "scipy")},
            "seeds": seeds,
            "protocols": list(PROTOCOLS),
            "matching_unit": ("artifact/inscription level, built before gap "
                              "splitting; one unambiguous record per side"),
            "task": "masked single-sign restoration; OOV targets count as failures",
            "grouping_policy": ("both protocols assign identical folds to the same "
                                "matched artifacts in both transcriptions"),
            "note": "same-family comparison; not inter-annotator agreement",
        },
        **result,
    }
    report = render_report(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "transcription_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (args.output / "transcription_sensitivity_report.txt").write_text(
        report, encoding="utf-8")
    print(report)
    return summary


if __name__ == "__main__":
    main()