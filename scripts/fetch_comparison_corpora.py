"""Download the comparison corpora at pinned versions and verify SHA-256.

Files land in ``data/external/comparison/raw/`` (gitignored); the repository
holds only this list. Sources and terms:

* CDLI full ATF dump and catalogue, ``cdli-gh/data`` at a pinned commit. CDLI
  data are free for academic use with credit to the Cuneiform Digital Library
  Initiative; there is no formal licence.
* CuneiML reading-to-sign tables (Chen et al. 2023, JOHD 10.5334/johd.151),
  ``taineleau/CuneiML`` at a pinned commit, CC0 1.0.

Run ``build_comparison_corpora.py`` afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "comparison" / "raw"

CDLI = "d66b12b065af39a57d640576b4c7e098db5dac7f"
CUNEIML = "407b46cf147a5b07364797c6b32135c39d8df23b"

FILES = {
    "cdliatf_unblocked.atf": (
        f"https://media.githubusercontent.com/media/cdli-gh/data/{CDLI}/cdliatf_unblocked.atf",
        "2896ec253767fa07fcaa5424af6fc25d6a047dc30b99c95f99d57ce75384d836",
    ),
    "cdli_cat.csv": (
        f"https://media.githubusercontent.com/media/cdli-gh/data/{CDLI}/cdli_cat.csv",
        "2e3232f75325b61c4d1e788d4d8c074c6230a947aed422110f9f35a6e353d09c",
    ),
    "cuneiform_vocab.txt": (
        f"https://raw.githubusercontent.com/taineleau/CuneiML/{CUNEIML}/cuneiform_unicode/cuneiform_vocab.txt",
        "e26b6593dd748d8d6e40bd6958ff09f200624ab1153fcdadf788ddbc558fc3ac",
    ),
    "token.tsv": (
        f"https://raw.githubusercontent.com/taineleau/CuneiML/{CUNEIML}/cuneiform_unicode/token.tsv",
        "fc68f4e2a64ae59c50e13efd5711a6bd2ca39e22b8b815787f8376822ddae5ee",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, (url, expected) in FILES.items():
        path = args.raw / name
        if not path.exists() or sha256(path) != expected:
            print(f"downloading {name}")
            urllib.request.urlretrieve(url, path)
        got = sha256(path)
        status = "ok" if got == expected else f"MISMATCH (got {got})"
        failed |= got != expected
        print(f"{name}: {status}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
