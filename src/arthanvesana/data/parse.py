"""Tidy parsing of the raw Indus corpus JSON.

Symbols are stored physical left-to-right; reading order is as-stored for
direction 'L/R' and reversed for 'R/L'. ICIT '000' marks an illegible sign
(missing data) and is gated out of sequences. Directions outside L/R and R/L
are kept in stored order with reading_order_known=False. Conventions follow
the upstream deposit's chr_lib; see data/PROVENANCE.md for the one deviation.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

EXPECTED_SHA256 = "345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68"
MISSING = "000"
_CODE_RE = re.compile(r"^\d{3}$")

REQUIRED_KEYS = (
    "id",
    "cisi",
    "site",
    "artefact_type",
    "direction",
    "complete",
    "symbols",
    "source",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path: str | Path, verify: bool = True) -> list[dict]:
    path = Path(path)
    if verify:
        got = sha256_file(path)
        if got.lower() != EXPECTED_SHA256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {path}: got {got}, "
                f"expected {EXPECTED_SHA256}."
            )
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError("Expected top-level JSON list of inscription records.")
    return records


def parse_bool(value, *, default: bool = False) -> bool:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "1.0"}:
        return True
    if text in {"false", "f", "no", "n", "0", "0.0"}:
        return False
    if text in {"", "na", "n/a", "nan", "none", "null", "<na>"}:
        return default
    raise ValueError(f"Invalid boolean value: {value!r}")


def identity_value(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "-", "?", "unknown", "na", "n/a", "nan", "none", "null"}:
        return None
    return text


def normalize_direction(raw: str | None) -> str:
    s = identity_value(raw) or ""
    if s == "L/R":
        return "L/R"
    if s in ("R/L", "R/l"):
        return "R/L"
    return "OTHER"


def reading_order_known(direction: str) -> bool:
    return direction in ("L/R", "R/L")


def sequence_in_reading_order(symbols: list[str], direction: str) -> list[str]:
    if direction == "R/L":
        return list(reversed(symbols))
    return list(symbols)


def gate(sequence: list[str]) -> list[str]:
    return [s for s in sequence if s != MISSING]


def to_tidy(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        direction = normalize_direction(rec.get("direction"))
        symbols = rec.get("symbols", [])
        for pos, code in enumerate(symbols, start=1):
            rows.append(
                {
                    "inscription_id": rec.get("id"),
                    "cisi": rec.get("cisi"),
                    "site": rec.get("site"),
                    "artefact_type": rec.get("artefact_type"),
                    "direction_raw": rec.get("direction"),
                    "direction": direction,
                    "reading_order_known": reading_order_known(direction),
                    "complete": rec.get("complete"),
                    "stored_length": len(symbols),
                    "position": pos,
                    "sign_code": code,
                    "is_missing": code == MISSING,
                    "artifact_id": rec.get("artifact_id", rec.get("artefact_id")),
                }
            )
    df = pd.DataFrame(
        rows,
        columns=[
            "inscription_id",
            "cisi",
            "site",
            "artefact_type",
            "direction_raw",
            "direction",
            "reading_order_known",
            "complete",
            "stored_length",
            "position",
            "sign_code",
            "is_missing",
            "artifact_id",
        ],
    )
    return df


def inscription_sequences(
    df: pd.DataFrame,
    *,
    reading_order: bool = True,
    drop_missing: bool = True,
    drop_empty: bool = True,
    known_direction_only: bool = False,
    gap_policy: str | None = None,
) -> list[list[str]]:
    if gap_policy is not None:
        if not drop_missing:
            raise ValueError("gap_policy requires drop_missing=True")
        return [r["sequence"] for r in analysis_records(
            df, gap_policy=gap_policy, reading_order=reading_order,
            known_direction_only=known_direction_only, drop_empty=drop_empty,
        )]
    frame = df
    if known_direction_only:
        frame = frame[frame["reading_order_known"].map(parse_bool)]
    seqs: list[list[str]] = []
    for _, group in frame.groupby("inscription_id", sort=False):
        group = group.sort_values("position")
        symbols = group["sign_code"].tolist()
        direction = group["direction"].iloc[0]
        if reading_order:
            symbols = sequence_in_reading_order(symbols, direction)
        if drop_missing:
            symbols = gate(symbols)
        if symbols or not drop_empty:
            seqs.append(symbols)
    return seqs


def analysis_records(
    df: pd.DataFrame,
    *,
    gap_policy: str = "split",
    reading_order: bool = True,
    known_direction_only: bool = True,
    drop_empty: bool = True,
) -> list[dict]:
    if gap_policy not in {"split", "drop", "reject"}:
        raise ValueError("gap_policy must be split, drop, or reject")
    records = []
    for inscription_id, group in df.groupby("inscription_id", sort=True):
        group = group.sort_values("position", kind="stable")
        row = group.iloc[0]
        direction = normalize_direction(row["direction"])
        known = reading_order_known(direction) and parse_bool(
            row.get("reading_order_known", True)
        )
        if known_direction_only and not known:
            continue
        symbols = group["sign_code"].tolist()
        if reading_order:
            symbols = sequence_in_reading_order(symbols, direction)
        has_gap = MISSING in symbols
        if gap_policy == "reject" and has_gap:
            continue
        spans = []
        if gap_policy == "drop":
            kept = [i for i, sign in enumerate(symbols) if sign != MISSING]
            if kept:
                spans.append((kept[0], kept[-1] + 1, [symbols[i] for i in kept]))
        else:
            start = 0
            for end in range(len(symbols) + 1):
                if end == len(symbols) or symbols[end] == MISSING:
                    if end > start:
                        spans.append((start, end, symbols[start:end]))
                    start = end + 1
        if not spans and not drop_empty:
            spans = [(0, 0, [])]
        site = identity_value(row.get("site"))
        artifact = identity_value(row.get("artifact_id"))
        if artifact is None:
            artifact = identity_value(row.get("artefact_id"))
        cisi = identity_value(row.get("cisi"))
        artifact_source = "explicit" if artifact else "cisi" if cisi else "inscription"
        artifact = artifact or cisi or str(inscription_id)
        complete = parse_bool(row.get("complete"))
        for span_index, (start, end, sequence) in enumerate(spans):
            records.append({
                "inscription_id": str(inscription_id),
                "artifact_id": artifact,
                "artifact_group": (None if artifact_source == "explicit" else site, artifact_source, artifact),
                "artifact_source": artifact_source,
                "cisi": cisi,
                "site": site,
                "direction": direction,
                "reading_order_known": known,
                "complete": complete,
                "span_index": span_index,
                "span_start": start,
                "span_end": end,
                "start_complete": bool(sequence) and known and complete and start == 0,
                "end_complete": bool(sequence) and known and complete and end == len(symbols),
                "has_gap": has_gap,
                "gap_policy": gap_policy,
                "sequence": sequence,
            })
    return records
