"""Fetch sanitized_corpus.json from the pinned upstream commit. The file is
discarded unless its SHA-256 matches the published value.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .parse import EXPECTED_SHA256

RAW_URL = (
    "https://raw.githubusercontent.com/ShaktiOSindia/indus-sign-regimes-deposit/"
    "e48b3ec1e90f368079f5126790613cded6f6f56c/sanitized_corpus.json"
)

CHUNK = 1 << 20


def download(dest: str | Path = "data/raw/sanitized_corpus.json") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    digest = hashlib.sha256()
    with urllib.request.urlopen(RAW_URL, timeout=120) as resp, open(tmp, "wb") as fh:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            fh.write(chunk)
    got = digest.hexdigest()
    if got.lower() != EXPECTED_SHA256.lower():
        tmp.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for downloaded corpus: got {got}, "
            f"expected {EXPECTED_SHA256}. File discarded."
        )
    tmp.replace(dest)
    return dest


if __name__ == "__main__":
    path = download()
    print(f"downloaded and verified: {path}")
