"""Parse CDLI ATF transliterations into sign sequences for comparison corpora.

The unit written out is one sign token per row in the same tidy schema as
``data/processed/corpus.csv``, so the Indus statistics run unchanged. Signs
that are illegible, broken away or fully restored by the editor become the
missing-sign placeholder ``000``; the analysis code splits spans there exactly
as it does for Indus ``000``.

Two tokenisers are provided:

* ``proto_tokens`` for proto-cuneiform and proto-Elamite, whose ATF already
  names signs (``GAL~a``, ``M388``, ``|GA~a.ZATU753|``) and numeral groups
  (``3(N01)``).
* ``sumerian_tokens`` for Sumerian, whose ATF gives readings (``ka``, ``dug4``)
  that are mapped to the underlying cuneiform sign so that two readings of the
  same sign count as one sign. Readings without a mapping become ``000`` and
  are counted, so their rate can be reported.

A numeral group such as ``2(gesz2)`` or ``3(N01)`` is one token, matching the
ICIT practice of giving each stroke-count group its own sign code.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

MISSING = "000"

_TEXT_START = re.compile(r"^&P(\d+)\s*=?\s*(.*)$")
_LINE_LABEL = re.compile(r"^([0-9]+[a-z0-9.']*)\.\s+(.*)$")
_NUMERAL = re.compile(r"^(?:n|N|[0-9]+(?:/[0-9]+)?)\(([^()]+)\)$")
_DAMAGE = str.maketrans("", "", "#?!*⸢⸣")
_ACCENTS = {"á": ("a", "2"), "à": ("a", "3"), "é": ("e", "2"), "è": ("e", "3"),
            "í": ("i", "2"), "ì": ("i", "3"), "ú": ("u", "2"), "ù": ("u", "3")}
_SUBSCRIPT = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


@dataclass
class AtfLine:
    surface: str
    column: str
    label: str
    content: str
    in_seal: bool


@dataclass
class AtfText:
    pnum: int
    designation: str
    lang: str | None
    lines: list[AtfLine] = field(default_factory=list)


def iter_texts(atf: str) -> Iterator[AtfText]:
    """Yield each ``&P`` text of a CDLI ATF dump with its numbered lines."""
    current: AtfText | None = None
    surface = column = ""
    in_seal = False
    for raw in atf.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = _TEXT_START.match(line)
        if head:
            if current is not None:
                yield current
            current = AtfText(int(head.group(1)), head.group(2).strip(), None)
            surface, column, in_seal = "", "", False
            continue
        if current is None:
            continue
        if line.startswith("#atf:"):
            found = re.search(r"lang\s+(\S+)", line)
            current.lang = found.group(1).rstrip(",") if found else None
        elif line.startswith("@"):
            key = line[1:].strip()
            word = key.split()[0].lower() if key else ""
            if word == "column":
                column = key.split()[1] if len(key.split()) > 1 else ""
            elif word == "seal":
                in_seal, surface, column = True, key, ""
            elif word in {"obverse", "reverse", "left", "right", "top", "bottom",
                          "edge", "surface", "face", "envelope", "tablet",
                          "object", "fragment", "bulla", "prism"}:
                if word not in {"tablet", "object", "envelope"}:
                    surface = key
                column = ""
                if word in {"tablet", "envelope"}:
                    in_seal = False
        elif line[0] in "#$>":
            continue
        else:
            labelled = _LINE_LABEL.match(line)
            if labelled:
                current.lines.append(AtfLine(surface, column, labelled.group(1),
                                             labelled.group(2), in_seal))
    if current is not None:
        yield current


def _clean_inline(content: str) -> str:
    content = re.sub(r"\(\$.*?\$\)", " ", content)  # inline $ comments
    content = re.sub(r"<<.*?>>", " ", content)  # editorial deletions
    content = content.replace("<", "").replace(">", "")  # editorial omissions
    return content


def _bracket_after(word: str, inside: bool) -> bool:
    for ch in word:
        if ch == "[":
            inside = True
        elif ch == "]":
            inside = False
    return inside


def _restored_parts(word: str, inside: bool) -> list[tuple[str, bool]]:
    """Return the sign pieces of ``word`` with a restored flag each."""
    parts: list[tuple[str, bool]] = []
    buf = ""
    state = inside
    for ch in word:
        if ch in "[]":
            if buf:
                parts.append((buf, state))
                buf = ""
            state = ch == "["
        else:
            buf += ch
    if buf:
        parts.append((buf, state))
    return parts


def proto_tokens(content: str) -> list[str]:
    """Sign tokens for a proto-cuneiform or proto-Elamite ATF line."""
    tokens: list[str] = []
    inside = False
    for word in _clean_inline(content).split():
        if word == ",":
            continue
        pieces = _restored_parts(word, inside)
        inside = _bracket_after(word, inside)
        text = "".join(p for p, _ in pieces).translate(_DAMAGE).strip(",")
        restored = any(r for _, r in pieces) and not any(
            not r and p.translate(_DAMAGE).strip(",") for p, r in pieces
        )
        if not text:
            if restored:
                _append_missing(tokens)
            continue
        if restored or text in {"...", "x", "X", "N"} or text.startswith("N("):
            _append_missing(tokens)
            continue
        if _NUMERAL.match(text) and text[0] in "nN":
            _append_missing(tokens)
            continue
        tokens.append(text)
    return tokens


def _append_missing(tokens: list[str]) -> None:
    # Consecutive unreadable signs collapse to one gap marker: the analysis
    # only uses gaps to split spans, and the count of lost signs is unknown.
    if not tokens or tokens[-1] != MISSING:
        tokens.append(MISSING)


def normalise_reading(text: str) -> str:
    """Normalise a reading or vocabulary key to one ASCII ATF spelling."""
    text = text.translate(_SUBSCRIPT).translate(_DAMAGE)
    for src, dst in (("š", "sz"), ("Š", "SZ"), ("ṣ", "s,"), ("Ṣ", "S,"),
                     ("ṭ", "t,"), ("Ṭ", "T,"), ("ŋ", "j"), ("Ŋ", "J"),
                     ("ḫ", "h"), ("Ḫ", "H")):
        text = text.replace(src, dst)
    for accent, (plain, index) in _ACCENTS.items():
        if accent in text:
            text = text.replace(accent, plain) + index
    if text.endswith(("@c", "@t")):
        text = text[:-2]
    return text.strip("_")


def load_sign_map(lines: list[str]) -> dict[str, tuple[str, ...]]:
    """Build reading -> sign codepoints from CuneiML ``reading<TAB>unicode``
    lines. Each codepoint is one written sign."""
    table: dict[str, tuple[str, ...]] = {}
    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            continue
        key = normalise_reading(parts[0])
        signs = tuple(ch for ch in parts[1] if not ch.isspace())
        if key and signs and key not in table:
            table[key] = signs
        lower = key.lower()
        if lower and signs and lower not in table:
            table[lower] = signs
    return table


def _split_word(word: str) -> list[str]:
    """Split a Sumerian ATF word into sign pieces at ``- . : +`` and around
    ``{...}`` determinatives, but not inside ``|compound|`` or ``(SIGN)``."""
    pieces: list[str] = []
    buf = ""
    depth_paren = 0
    in_pipe = False
    i = 0
    while i < len(word):
        ch = word[i]
        if ch == "{" and depth_paren == 0 and not in_pipe:
            if buf:
                pieces.append(buf)
            end = word.find("}", i)
            end = len(word) - 1 if end < 0 else end
            pieces.append(word[i : end + 1])
            buf = ""
            i = end + 1
            continue
        if ch == "|":
            in_pipe = not in_pipe
        elif ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(depth_paren - 1, 0)
        if ch in "-.:+" and depth_paren == 0 and not in_pipe:
            if buf:
                pieces.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    if buf:
        pieces.append(buf)
    return pieces


def _lookup(text: str, sign_map: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    """Map a reading, a ``reading(SIGN)`` qualified reading, a sign name or a
    ``|compound|`` to sign codepoints; ``None`` when no mapping is known."""
    key = normalise_reading(text)
    signs = sign_map.get(key) or sign_map.get(key.lower())
    if signs:
        return signs
    qualified = re.match(r"^[^()]+\((.+)\)$", key)
    if qualified:
        return _lookup(qualified.group(1), sign_map)
    if key.startswith("|") and key.endswith("|"):
        inner = key[1:-1]
        signs = sign_map.get(inner) or sign_map.get(inner.lower())
        if signs:
            return signs
        if re.fullmatch(r"[^.×x%&@~]+(\.[^.×x%&@~]+)+", inner):
            parts = [_lookup(part, sign_map) for part in inner.split(".")]
            if all(parts):
                return tuple(s for part in parts for s in part)
        return (key.upper(),)  # one ligature sign, identified by its name
    if "@" in key or "~" in key:
        return _lookup(re.split(r"[@~]", key)[0], sign_map)
    return None


def sumerian_tokens(
    content: str, sign_map: dict[str, tuple[str, ...]], unmapped: Counter | None = None
) -> list[str]:
    """Sign tokens for a Sumerian ATF line, with readings mapped to signs."""
    tokens: list[str] = []
    inside = False
    for word in _clean_inline(content).replace("_", " ").split():
        word = word.replace("...", "x")  # a lost stretch, not three separators
        start_inside = inside
        inside = _bracket_after(word, inside)
        for piece in _split_word(word):
            if piece.startswith("{+"):
                start_inside = _bracket_after(piece, start_inside)
                continue  # phonetic gloss, not a written sign of the text
            restored_parts = _restored_parts(piece, start_inside)
            start_inside = _bracket_after(piece, start_inside)
            text = "".join(p for p, _ in restored_parts)
            restored = bool(restored_parts) and all(r for _, r in restored_parts)
            text = text.strip("{}").translate(_DAMAGE)
            if not text:
                if restored:
                    _append_missing(tokens)
                continue
            if restored or text in {"...", "x", "X", "n"} or text.startswith("x("):
                _append_missing(tokens)
                continue
            if _NUMERAL.match(text):
                if text[0] in "nN":
                    _append_missing(tokens)
                else:
                    tokens.append(text.lower())
                continue
            signs = _lookup(text, sign_map)
            if signs is None:
                if unmapped is not None:
                    unmapped[normalise_reading(text)] += 1
                _append_missing(tokens)
            else:
                tokens.extend(signs)
    return tokens
