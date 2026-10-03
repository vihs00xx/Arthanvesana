"""Normalise CDLI comparison corpora into the Indus tidy token schema.

Reads the pinned downloads from ``fetch_comparison_corpora.py`` and writes one
``corpus.csv`` per comparison corpus under ``data/external/comparison/<name>/``
with the columns of ``data/processed/corpus.csv`` (plus ``artifact_id``), so
``analysis_records`` and every statistic in the pipeline read them unchanged.

Corpora (selected with the CDLI catalogue's period, language and object type):

* ``proto_cuneiform``: Uruk IV and Uruk III tablets (ATF lang qpc). One
  sequence per numbered ATF line (a case of an account).
* ``proto_elamite``: Proto-Elamite period tablets (ATF lang qpc). One
  sequence per numbered line (an entry).
* ``ur3_admin``: Ur III Sumerian tablets that are not seals; seal impressions
  rolled on tablets are excluded. One sequence per numbered line.
* ``seal_legends``: Sumerian seal legends (CDLI seal objects, almost all
  Ur III). One sequence per seal, lines joined in order, because an Indus seal
  inscription is also read as one whole.

Every sequence is written left to right (``direction`` L/R). Its artifact is
the CDLI P-number, so grouped splits keep lines of one tablet together. A
sequence is ``complete`` when it has no missing sign and its line label is not
primed (a primed label counts from a break, so the line's place is uncertain).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arthanvesana.data.atf import (  # noqa: E402
    MISSING,
    iter_texts,
    load_sign_map,
    proto_tokens,
    sumerian_tokens,
)

RAW = ROOT / "data" / "external" / "comparison" / "raw"
OUT = ROOT / "data" / "external" / "comparison"

COLUMNS = [
    "inscription_id", "cisi", "site", "artefact_type", "direction_raw", "direction",
    "reading_order_known", "complete", "stored_length", "position", "sign_code",
    "is_missing", "artifact_id",
]


def load_catalogue(path: Path) -> dict[int, dict]:
    keep = ("period", "language", "object_type", "provenience", "genre")
    catalogue: dict[int, dict] = {}
    csv.field_size_limit(1 << 30)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                pnum = int(row["id_text"])
            except (TypeError, ValueError):
                continue
            catalogue[pnum] = {k: (row.get(k) or "").strip() for k in keep}
    return catalogue


def corpus_of(meta: dict, lang: str | None) -> str | None:
    period = meta.get("period", "")
    obj = meta.get("object_type", "").lower()
    if lang == "qpc" and period.startswith(("Uruk IV ", "Uruk III ")):
        return "proto_cuneiform"
    if lang == "qpc" and period.startswith("Proto-Elamite"):
        return "proto_elamite"
    if lang == "sux" and obj.startswith("seal"):
        return "seal_legends"
    if lang == "sux" and period.startswith("Ur III") and obj in {"tablet", "tablet & envelope"}:
        return "ur3_admin"
    return None


def site_of(meta: dict) -> str:
    site = meta.get("provenience", "")
    return site.split(" (")[0] if site else ""


def build(raw: Path, out: Path) -> dict:
    catalogue = load_catalogue(raw / "cdli_cat.csv")
    map_lines = []
    for name in ("token.tsv", "cuneiform_vocab.txt"):  # token.tsv wins, as in CuneiML
        map_lines += (raw / name).read_text(encoding="utf-8", errors="replace").splitlines()
    sign_map = load_sign_map(map_lines)

    sequences: dict[str, list[dict]] = {k: [] for k in
                                         ("proto_cuneiform", "proto_elamite", "ur3_admin", "seal_legends")}
    unmapped: Counter = Counter()
    mapped_tokens = Counter()
    atf = (raw / "cdliatf_unblocked.atf").read_text(encoding="utf-8", errors="replace")
    for text in iter_texts(atf):
        meta = catalogue.get(text.pnum)
        if meta is None:
            continue
        name = corpus_of(meta, text.lang)
        if name is None:
            continue
        sumerian = name in {"ur3_admin", "seal_legends"}
        if name == "seal_legends":
            signs: list[str] = []
            complete = True
            for line in text.lines:
                tokens = sumerian_tokens(line.content, sign_map, unmapped)
                complete &= "'" not in line.label
                signs += tokens
                mapped_tokens[name] += sum(t != MISSING for t in tokens)
            if signs:
                sequences[name].append(_record(text, meta, "", signs, complete))
            continue
        for line in text.lines:
            if line.in_seal:
                continue
            tokens = (sumerian_tokens(line.content, sign_map, unmapped) if sumerian
                      else proto_tokens(line.content))
            if not tokens:
                continue
            if sumerian:
                mapped_tokens[name] += sum(t != MISSING for t in tokens)
            label = "/".join(x for x in (line.surface, line.column, line.label) if x)
            sequences[name].append(_record(text, meta, label, tokens, "'" not in line.label))

    summary = {"unmapped_readings_top": unmapped.most_common(40),
               "unmapped_reading_tokens": sum(unmapped.values()), "corpora": {}}
    for name, records in sequences.items():
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        with open(target / "corpus.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            for rec in records:
                for pos, sign in enumerate(rec["signs"], start=1):
                    writer.writerow({
                        "inscription_id": rec["id"], "cisi": rec["designation"],
                        "site": rec["site"], "artefact_type": rec["object_type"],
                        "direction_raw": "L/R", "direction": "L/R",
                        "reading_order_known": True,
                        "complete": rec["complete"] and MISSING not in rec["signs"],
                        "stored_length": len(rec["signs"]), "position": pos,
                        "sign_code": sign, "is_missing": sign == MISSING,
                        "artifact_id": rec["artifact"],
                    })
        tokens = [s for rec in records for s in rec["signs"]]
        summary["corpora"][name] = {
            "sequences": len(records),
            "artifacts": len({rec["artifact"] for rec in records}),
            "token_rows": len(tokens),
            "missing_tokens": sum(s == MISSING for s in tokens),
            "distinct_signs": len(set(tokens) - {MISSING}),
        }
    total_sumerian = sum(mapped_tokens.values()) + summary["unmapped_reading_tokens"]
    summary["unmapped_reading_rate"] = (
        summary["unmapped_reading_tokens"] / total_sumerian if total_sumerian else 0.0)
    with open(out / "build_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    return summary


def _record(text, meta, label, signs, complete) -> dict:
    return {
        "id": f"P{text.pnum:06d}" + (f":{label}" if label else ""),
        "artifact": f"P{text.pnum:06d}",
        "designation": text.designation,
        "site": site_of(meta),
        "object_type": meta.get("object_type", ""),
        "signs": signs,
        "complete": bool(complete),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    summary = build(args.raw, args.out)
    print(json.dumps({k: v for k, v in summary.items() if k != "unmapped_readings_top"},
                     indent=2))


if __name__ == "__main__":
    main()
