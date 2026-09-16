# Arthānveṣaṇa

**AI-assisted structural analysis of the Sindhu-Sarasvatī script using statistical baselines, sign embeddings, and self-supervised learning.**


Inspired by the Sanskrit idea of searching and inquiry, **Arthanvesana** ("search for meaning") reflects this project's purpose: to investigate ancient signs through evidence, uncover structural patterns, and generate testable hypotheses.

## Status

Early scaffold. Planned phases:

1. **Corpus** — ingestion and validation of a machine-readable Indus sign corpus
2. **Statistics** — frequencies, n-grams, transitions, positional analysis, shuffled controls
3. **Embeddings** — sign vector representations (PPMI/SVD, skip-gram), clustering, visualization

## Source corpus

- Upstream: [ShaktiOSindia/indus-sign-regimes-deposit](https://github.com/ShaktiOSindia/indus-sign-regimes-deposit)
- File: `sanitized_corpus.json` (5,704 catalogued inscriptions; ~18k sign tokens; 713 distinct signs; `000` marks an illegible sign, not a sign)
- Upstream commit: `e48b3ec1e90f368079f5126790613cded6f6f56c` (2026-08-12)
- SHA-256: `345241b13fedada87b4783c24cd241123491bbd7edaf5bf636f9cdb36c01da68`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## License

Apache 2.0
