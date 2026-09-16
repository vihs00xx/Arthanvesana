"""Morfessor unsupervised segmentation applied to sign sequences. Each
inscription is treated as a 'word' of sign 'letters'; Morfessor learns a
morph lexicon by minimum description length. Boundary sets allow direct
agreement comparison with the greedy LLR segmenter.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path


def train(seqs: list[list[str]]):
    import morfessor

    counts = Counter(tuple(s) for s in seqs)
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        with open(corpus, "w", encoding="utf-8") as fh:
            for seq, c in counts.items():
                fh.write(f"{c} {' '.join(seq)}\n")
        io = morfessor.MorfessorIO()
        data = list(io.read_corpus_file(str(corpus)))
    model = morfessor.BaselineModel()
    model.load_data(data)
    model.train_batch()
    return model


def segment(model, seq: list[str]) -> list[tuple]:
    parts = model.viterbi_segment(tuple(seq))[0]
    out = []
    for p in parts:
        if isinstance(p, str):
            out.append((p,))
        else:
            out.append(tuple(p))
    return out


def boundaries(segmentation: list[tuple]) -> set:
    cuts = set()
    pos = 0
    for morph in segmentation[:-1]:
        pos += len(morph)
        cuts.add(pos)
    return cuts


def boundary_f1(a: list[tuple], b: list[tuple]) -> float:
    ba, bb = boundaries(a), boundaries(b)
    if not ba and not bb:
        return 1.0
    if not ba or not bb:
        return 0.0
    inter = len(ba & bb)
    prec = inter / len(ba)
    rec = inter / len(bb)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
