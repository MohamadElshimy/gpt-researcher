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
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")
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
            cells = _split_row(stripped)
            if cells and all(_SEPARATOR_CELL.match(c) for c in cells if c) and any(cells):
                continue  # |---|:--:| separator; a data row of lone "-" cells is kept
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return [t for t in tables if len(t) >= 2]


def _split_row(stripped: str) -> list[str]:
    """Split one pipe row into cells, keeping empty first/last cells
    ("||Dev|Test|" has an empty corner cell, a common header shape)."""
    inner = stripped[1:] if stripped.startswith("|") else stripped
    if inner.endswith("|"):
        inner = inner[:-1]
    return [c.strip() for c in inner.split("|")]


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
        if any(_text_split_across_cells(r, is_header=(index_row == 0)) for index_row, r in enumerate(rows)):
            issues.append(f"{label}: text split across cells")
    return issues


_LOWER_FRAGMENT = re.compile(r"^[a-z]{1,2}$")


def _has_split_word_in_cell(cell: str) -> bool:
    """'Publishe<br>d': a word broken across a line break inside one cell.
    'Yes<br>No' or 'EM<br>F1' are legitimate two-line cells and must not match."""
    if "<br>" not in cell:
        return False
    parts = [p.strip() for p in cell.split("<br>")]
    for before, after in zip(parts, parts[1:], strict=False):
        if _LOWER_FRAGMENT.match(after) and before[-1:].islower():
            return True
        if _LOWER_FRAGMENT.match(before) and after[:1].islower():
            return True
    return False


_HEADER_FRAGMENT = re.compile(r"^(?=.*[a-z])[A-Za-z]{1,2}$")
_SHORT_LOWER_TAIL = re.compile(r"\S+\s+[a-z]{1,3}$")
_LOWER_HEAD = re.compile(r"^[a-z]{2,}")
_LOWER_HEAD_FRAGMENT = re.compile(r"^[a-z]{1,2}(?!\w)")


def _text_split_across_cells(row: list[str], is_header: bool = False) -> bool:
    """'Single-task si' | 'ngle models': a phrase cut mid-word by a cell
    boundary. Requires the left cell to hold more than one token so that
    legitimate short lowercase headers ("data" | "bsz") do not match. The
    single-fragment form ('Publishe' | 'd', '(without NSP los' | 's):') is
    only checked on data rows: header rows legitimately hold short lowercase
    abbreviations such as "lr" or "bs"."""
    for left, right in zip(row, row[1:], strict=False):
        # <br> marks line breaks inside a cell; line i of the left cell sits
        # beside line i of the right cell, so compare them pairwise.
        left_lines = [p.strip() for p in left.split("<br>")]
        right_lines = [p.strip() for p in right.split("<br>")]
        for left_line, right_line in zip(left_lines, right_lines, strict=False):
            if _SHORT_LOWER_TAIL.search(left_line) and _LOWER_HEAD.match(right_line):
                return True
            # 'Publishe' | 'd': the right cell starts with a 1-2 letter lowercase fragment
            if not is_header and left_line[-1:].islower() and _LOWER_HEAD_FRAGMENT.match(right_line):
                return True
    return False


def looks_empty(markdown: str, min_chars: int = 20) -> bool:
    return len(re.sub(r"\s+", "", markdown)) < min_chars
