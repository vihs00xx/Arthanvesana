# Comparison corpora: first pass (extension 1)

This note records the first run of the Indus statistics on other early sign
systems. It is descriptive. None of the comparison corpora below is a
non-linguistic control, so nothing here tests whether the Indus script
encodes language.

## Sources

| Corpus | Selection | Unit of sequence | Terms |
|---|---|---|---|
| Proto-cuneiform | CDLI, ATF lang `qpc`, period Uruk IV or Uruk III | one numbered ATF line (a case) | CDLI: free academic use with credit |
| Proto-Elamite | CDLI, ATF lang `qpc`, period Proto-Elamite | one numbered line (an entry) | CDLI |
| Ur III admin | CDLI, ATF lang `sux`, period Ur III, tablets; seal impressions on tablets excluded | one numbered line | CDLI |
| Sumerian seals | CDLI seal objects, ATF lang `sux` (almost all Ur III) | one whole legend | CDLI |

CDLI files come from `cdli-gh/data` at commit `d66b12b` (catalogue and ATF
dump, last updated August 2022). Sumerian readings are mapped to signs with
the CuneiML tables (Chen et al. 2023, CC0) at commit `407b46c`, so that two
readings of one sign count as one sign; 0.5% of reading tokens have no
mapping and are treated as missing. `scripts/fetch_comparison_corpora.py`
downloads every file at those commits and checks SHA-256;
`scripts/build_comparison_corpora.py` writes one `corpus.csv` per corpus in
the Indus tidy schema.

Not yet included: TLA Earlier Egyptian (Hugging Face) and the Linear B
Knossos D series (Zenodo) are open but their hosts are blocked by the cloud
environment's network policy. Sproat's non-linguistic corpora and Linear A
wait on licence answers.

## Normalisation choices

* Illegible signs, `x`, `[...]` and signs wholly restored inside `[ ]` become
  the Indus missing marker `000`, so spans split there as they do for Indus.
* A numeral group (`3(N01)`, `2(gesz2)`) is one token, as ICIT gives each
  stroke-count group one code. A second run drops numeral tokens entirely,
  because numerals dominate the early accounting texts.
* All sequences are read left to right. A sequence is complete when it has
  no missing sign and its line label is not primed.
* The artifact is the CDLI P-number, so the grouped split keeps a tablet's
  lines together.

## Size matching

Most measures depend on sample size, so each comparison corpus is subsampled
to the Indus analysable token count (16,469 known-direction tokens):

* token-matched: whole artifacts drawn until the token total is reached;
* length-matched: spans drawn to reproduce the Indus span-length histogram,
  with the achieved coverage reported where a corpus is short of a length.

Ten replicates per corpus; replicate r uses seed r for the subsample and the
artifact-grouped 80/20 split. Indus is used whole, so its spread comes only
from the split.

## Results (length-matched, numerals dropped; mean of 10)

| Measure | Indus | Proto-cuneiform | Proto-Elamite | Ur III admin | Sumerian seals |
|---|---|---|---|---|---|
| Distinct signs | 651 | 1043 | 1171 | 305 | 250 |
| H(s2\|s1), Miller-Madow (bits) | 3.53 | 4.62 | 3.78 | 3.98 | 3.62 |
| Held-out bigram gain over unigram, MKN (bits/token) | 1.04 | 0.21 | 0.32 | 1.44 | 1.41 |
| Restoration top-1, bigram context | 29.2% | 8.5% | 13.4% | 31.6% | 41.8% |
| Complete spans ending in the commonest end sign | 28.1% | 2.1% | 11.4% | 7.1% | 6.5% |
| Spans with an adjacent repeated sign | 0.6% | 2.2% | 2.9% | 5.5% | 12.8% |

Length-histogram coverage: proto-cuneiform 74%, proto-Elamite 56%, Ur III
100%, seals 61%. Full tables for both numeral treatments and both matching
schemes are written to `outputs/comparison/comparison_report.md` (pass `--drop-numerals` for the second run).

What the first pass shows:

1. **Sequential predictability.** The Indus held-out bigram gain (1.04 ± 0.09
   bits) and context restoration (29%) sit with Ur III Sumerian (1.1 to 1.4
   bits, 26 to 32%) and far above the proto-cuneiform and proto-Elamite
   accounting signs once numerals are removed (−0.1 to 0.3 bits, 3 to 13%).
   Sumerian seal legends are more predictable still (41 to 56%) because they
   repeat a few formulas.
2. **Terminal sign.** 28% of complete Indus inscriptions end in one sign
   (740). No comparison corpus comes close without numerals (at most 13%).
   With numerals kept, proto-Elamite reaches 21 to 23%, and that ending is a
   numeral.
3. **Repetition.** Indus has the lowest rate of adjacent repeated signs
   (0.6%) of any corpus here.

Caveats: the comparison set is all Mesopotamian or Iranian administrative
and seal material; Sumerian is a full writing of language, and how far
proto-cuneiform records language is itself debated. The shuffle-drop measure
is not comparable across these corpora: in proto-cuneiform lines a few
numerals precede many signs, so within-span shuffling lowers the plug-in
H(s2|s1) and the drop comes out negative.
