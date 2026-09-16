from __future__ import annotations

import pandas as pd

from arthanvesana.data.parse import inscription_sequences
from arthanvesana.stats.ngrams import NGramModel
from arthanvesana.stats.sampling import train_test_split


def site_sequences(df: pd.DataFrame, min_inscriptions: int = 100) -> dict[str, list]:
    sites = {}
    for site, group in df.groupby("site"):
        seqs = inscription_sequences(group)
        if len(seqs) >= min_inscriptions:
            sites[site] = seqs
    return sites


def cross_perplexity(
    sites: dict[str, list], method: str = "wittenbell", seed: int = 0
) -> dict:
    names = sorted(sites)
    trains = {}
    tests = {}
    for name in names:
        tr, te = train_test_split(sites[name], train_frac=0.8, seed=seed)
        trains[name] = tr
        tests[name] = te
    matrix = {}
    for train_site in names:
        model = NGramModel(trains[train_site], 2, method=method)
        row = {}
        for test_site in names:
            row[test_site] = model.perplexity(tests[test_site])
        matrix[train_site] = row
    return matrix
