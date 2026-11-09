"""Stage-2 quality gates: cheap checks on a backend's output that decide whether
a page is accepted or escalated one tier.

- ``word_recall``: share of the raw text layer's words that survive in the
  structured output. Guards against layout analysis dropping blocks
  (pymupdf4llm's documented "formula box drops its text" failure).
- ``table_issues``: symptoms of a mangled Markdown table -- ragged rows,
  orphan punctuation cells, split words, two side-by-side tables merged into
  one wide table (BERT Tables 2/3 in the fixture pack).
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_SEPARATOR_ROW = re.compile(r"^\|?[\s:\-|]+\|?$")
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"}


def _dehyphenate(text: str) -> str:
    # "sen-\ntence" -> "sentence"; the raw text layer keeps line-break hyphens
    # that structured output joins, so compare on joined words.
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def word_set(text: str) -> set[str]:
    for glyph, ascii_form in _LIGATURES.items():
        text = text.replace(glyph, ascii_form)
    return {w.lower() for w in _WORD.findall(_dehyphenate(text))}


def word_recall(reference_text: str, candidate_text: str) -> float:
    """1.0 when every >=3-letter word of the reference appears in the candidate."""
    ref = word_set(reference_text)
    if not ref:
        return 1.0
    cand = word_set(candidate_text)
    return len(ref & cand) / len(ref)


def markdown_tables(markdown: str) -> list[list[list[str]]]:
    """Parse pipe tables into rows of cells (separator rows removed)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if _SEPARATOR_ROW.match(stripped):
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return [t for t in tables if len(t) >= 2]


def table_issues(markdown: str) -> list[str]:
    issues: list[str] = []
    for index, rows in enumerate(markdown_tables(markdown)):
        header = rows[0]
        width = len(header)
        label = f"table {index + 1}"
        ragged = sum(1 for r in rows if len(r) != width)
        if width >= 2 and ragged / len(rows) > 0.2:
            issues.append(f"{label}: ragged rows ({ragged}/{len(rows)})")
        if any(cell in {".", ",", "..", ":"} for r in rows for cell in r):
            issues.append(f"{label}: orphan punctuation cell")
        # "D | ev | Te | st" is a word cut into pieces; "EM | F1 | A | B" are
        # legitimate abbreviations, so only count pieces that contain lowercase.
        fragments = sum(1 for c in header if _HEADER_FRAGMENT.match(c))
        if fragments >= 2:
            issues.append(f"{label}: fragmented header")
        labels = [c for c in header if c]
        if width >= 6 and len(set(labels)) < len(labels):
            issues.append(f"{label}: repeated header labels (merged side-by-side tables?)")
        if any(_has_split_word_in_cell(cell) for r in rows for cell in r):
            issues.append(f"{label}: split word inside a cell")
        if any(_text_split_across_cells(r) for r in rows):
            issues.append(f"{label}: text split across cells")
    return issues


def _has_split_word_in_cell(cell: str) -> bool:
    if "<br>" not in cell:
        return False
    parts = [p.strip() for p in cell.split("<br>")]
    return any(p.isalpha() and len(p) <= 2 for p in parts)


_HEADER_FRAGMENT = re.compile(r"^(?=.*[a-z])[A-Za-z]{1,2}$")
_SHORT_LOWER_TAIL = re.compile(r"\S+\s+[a-z]{1,3}$")
_LOWER_HEAD = re.compile(r"^[a-z]{2,}")


def _text_split_across_cells(row: list[str]) -> bool:
    """'Single-task si' | 'ngle models': a phrase cut mid-word by a cell
    boundary. Requires the left cell to hold more than one token so that
    legitimate short lowercase headers ("data" | "bsz") do not match."""
    for left, right in zip(row, row[1:]):
        # <br> marks line breaks inside a cell; line i of the left cell sits
        # beside line i of the right cell, so compare them pairwise.
        left_lines = [p.strip() for p in left.split("<br>")]
        right_lines = [p.strip() for p in right.split("<br>")]
        for left_line, right_line in zip(left_lines, right_lines):
            if _SHORT_LOWER_TAIL.search(left_line) and _LOWER_HEAD.match(right_line):
                return True
    return False


def looks_empty(markdown: str, min_chars: int = 20) -> bool:
    return len(re.sub(r"\s+", "", markdown)) < min_chars
