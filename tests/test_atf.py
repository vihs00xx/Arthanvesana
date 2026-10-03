from collections import Counter

from arthanvesana.data.atf import (
    MISSING,
    iter_texts,
    load_sign_map,
    normalise_reading,
    proto_tokens,
    sumerian_tokens,
)

SIGN_MAP = load_sign_map([
    "ki\t𒆠", "lugal\t𒈗", "d\t𒀭", "szu\t𒋗", "suen\t𒂗𒍪", "uri5\t𒋀𒀊",
    "ma\t𒈠", "e2\t𒂍", "ra\t𒊏", "DU\t𒁺", "SZE\t𒊺", "KIN\t𒆥", "gid₂\t𒁍",
])

ATF = """&P000001 = Test 1
#atf: lang qpc
@tablet
@obverse
@column 1
$ beginning broken
1'. 1(N01) , [...]
2. 3(N14) , GAL~a# |GA~a.ZATU753|
#note: skipped
>>Q000002 015
&P000002 = Test 2
#atf: lang sux
@tablet
@obverse
1. ki lugal
@seal 1
1. {d}szu-{d}suen
"""


def test_iter_texts_reads_labels_lang_and_seal_sections():
    texts = list(iter_texts(ATF))
    assert [t.pnum for t in texts] == [1, 2]
    assert texts[0].lang == "qpc"
    assert [line.label for line in texts[0].lines] == ["1'", "2"]
    assert texts[0].lines[0].column == "1"
    assert [line.in_seal for line in texts[1].lines] == [False, True]


def test_proto_tokens_keep_signs_and_numeral_groups_and_gate_damage():
    assert proto_tokens("3(N14) , GAL~a# |GA~a.ZATU753|") == ["3(N14)", "GAL~a", "|GA~a.ZATU753|"]
    assert proto_tokens("1(N01)# , [...]") == ["1(N01)", MISSING]
    assert proto_tokens("x x x , [...]") == [MISSING]
    assert proto_tokens("[GAL] UMUN2") == [MISSING, "UMUN2"]
    assert proto_tokens("N(N01) , SZE") == [MISSING, "SZE"]


def test_sumerian_readings_map_to_signs_and_restorations_are_missing():
    assert sumerian_tokens("lugal uri5{ki}-ma", SIGN_MAP) == ["𒈗", "𒋀", "𒀊", "𒆠", "𒈠"]
    assert sumerian_tokens("{d}szu-{d}suen", SIGN_MAP) == ["𒀭", "𒋗", "𒀭", "𒂗", "𒍪"]
    assert sumerian_tokens("[...] e2 [{d}szu]-ra", SIGN_MAP) == [MISSING, "𒂍", MISSING, "𒊏"]
    assert sumerian_tokens("2(gesz2) ki", SIGN_MAP) == ["2(gesz2)", "𒆠"]


def test_sumerian_qualified_readings_and_compounds_use_the_sign_name():
    assert sumerian_tokens("kux(DU)", SIGN_MAP) == ["𒁺"]
    assert sumerian_tokens("gurx(|SZE.KIN|)", SIGN_MAP) == ["𒊺", "𒆥"]


def test_unmapped_readings_are_counted_and_gated():
    unmapped = Counter()
    assert sumerian_tokens("ki zzz", SIGN_MAP, unmapped) == ["𒆠", MISSING]
    assert unmapped == Counter({"zzz": 1})


def test_normalise_reading_unifies_unicode_and_ascii_spellings():
    assert normalise_reading("gid₂") == "gid2"
    assert normalise_reading("šu") == "szu"
    assert normalise_reading("gáb") == "gab2"
