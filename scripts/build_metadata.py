from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXTERNAL = ROOT / "data" / "external" / "indus_website_real_corpus.csv"
CSV = ROOT / "data" / "processed" / "corpus.csv"
OUT = ROOT / "data" / "processed" / "inscription_metadata.csv"

COLUMNS = [
    "cisi", "ext_id", "motif", "ext_direction",
    "ext_line_count", "ext_object_type", "source",
]


def load_external(path: str | Path = EXTERNAL) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", dtype=str)
    df["line_count"] = pd.to_numeric(df["line_count"], errors="raise")
    return df


def build_sidecar(external: pd.DataFrame, corpus: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    external = external.copy()
    external["line_count"] = pd.to_numeric(external["line_count"], errors="raise")
    known = external[external["cisi_number"] != "unknown"].copy()
    if known["cisi_number"].duplicated().any():
        raise ValueError("duplicate cisi_number in external metadata")
    sidecar = pd.DataFrame({
        "cisi": known["cisi_number"],
        "ext_id": known["inscription_id"],
        "motif": known["motif"],
        "ext_direction": known["reading_direction"],
        "ext_line_count": known["line_count"],
        "ext_object_type": known["object_type"],
        "source": "joyboseroy/indus_decipher:indus_website",
    })[COLUMNS]
    our_cisi = set(corpus["cisi"].dropna())
    matched = set(sidecar["cisi"]) & our_cisi
    report = {
        "external_rows": len(external),
        "external_unknown_cisi": int((external["cisi_number"] == "unknown").sum()),
        "sidecar_rows": len(sidecar),
        "matched_inscriptions": len(matched),
        "corpus_inscriptions": int(corpus["inscription_id"].nunique()),
        "motif_known": int((sidecar["motif"] != "unknown").sum()),
        "line_count_gt1": int((sidecar["ext_line_count"] > 1).sum()),
        "all_undamaged_note": "external damaged flag is uniformly False; unusable",
    }
    return sidecar, report


def attach_metadata(records: list[dict], sidecar: pd.DataFrame) -> list[dict]:
    lookup = sidecar.set_index("cisi").to_dict(orient="index")
    out = []
    for record in records:
        meta = lookup.get(record.get("cisi"))
        enriched = dict(record)
        enriched["ext_motif"] = meta["motif"] if meta else None
        enriched["ext_direction"] = meta["ext_direction"] if meta else None
        enriched["has_ext_metadata"] = meta is not None
        out.append(enriched)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build inscription metadata sidecar")
    parser.add_argument("--external", type=Path, default=EXTERNAL)
    parser.add_argument("--corpus", type=Path, default=CSV)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    external = load_external(args.external)
    corpus = pd.read_csv(args.corpus, encoding="utf-8", dtype=str)
    sidecar, report = build_sidecar(external, corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(args.output, index=False, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output.name} ({len(sidecar)} rows)")


if __name__ == "__main__":
    main()
