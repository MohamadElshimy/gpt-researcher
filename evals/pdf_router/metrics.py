"""Text-quality metrics for PDF extraction, against ar5iv HTML ground truth.

- sentence recall: share of ground-truth body sentences (>= 8 words, math
  stripped) found verbatim in the extraction after normalisation. Column
  interleaving, dropped blocks and word-per-line flattening all break
  sentences. Sanity check: ``get_text(sort=True)`` on the BERT paper (which
  interleaves its columns) scores 1.4% against 86% for stream order.
- 8-gram recall: lenient variant, tolerant to small wording differences.
- table-cell recall: share of ground-truth table cells found inside Markdown
  table rows (lines starting with ``|``) of the extraction. The current
  extractor flattens tables, so it scores 0 even though the numbers are
  present in the text ("cells anywhere").

Normalisation joins hyphenated line breaks and in-word hyphens, expands
ligatures and drops punctuation, so "pre-train", "pre-\\ntrain" and
"pretrain" compare equal; sentences that contain formulas can never match
because ar5iv strips the math, and that loss is identical for every extractor.
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\u2019": "'",
    "\u2018": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
}
_HTML_TAG = re.compile(r"</?(br|sup|sub|b|i|u|em|strong|span|td|tr|th|table|tbody|thead|p|div)\b[^<>]{0,40}>", re.I)


def normalise(text: str) -> str:
    for glyph, ascii_form in LIGATURES.items():
        text = text.replace(glyph, ascii_form)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\s*\n\s*", "", text)  # hyphenated line break -> join
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)  # in-word hyphen -> join
    text = _HTML_TAG.sub(" ", text)
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


class _Ar5ivParser(HTMLParser):
    """Collect body paragraphs and table cells; skip math, bibliography,
    footnotes, author blocks, scripts."""

    def __init__(self):
        super().__init__()
        self.paragraphs: list[str] = []
        self.cells: list[str] = []
        self._skip = 0
        self._in_p = 0
        self._in_cell = 0
        self._p_buf: list[str] = []
        self._cell_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if tag in ("math", "script", "style") or any(
            marker in cls for marker in ("ltx_bibliography", "ltx_note", "ltx_page_footer", "ltx_authors")
        ):
            self._skip += 1
        if self._skip:
            return
        if tag == "p" and "ltx_p" in cls:
            self._in_p += 1
            self._p_buf = []
        if tag in ("td", "th"):
            self._in_cell += 1
            self._cell_buf = []

    def handle_endtag(self, tag):
        if tag in ("math", "script", "style"):
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            if tag in ("section", "div", "footer", "span", "sup"):
                self._skip = max(0, self._skip - 1)
            return
        if tag == "p" and self._in_p:
            self._in_p -= 1
            self.paragraphs.append("".join(self._p_buf))
        if tag in ("td", "th") and self._in_cell:
            self._in_cell -= 1
            self.cells.append("".join(self._cell_buf))

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_p:
            self._p_buf.append(data)
        if self._in_cell:
            self._cell_buf.append(data)


def sentences_from_paragraphs(paragraphs: list[str], min_words: int = 8) -> list[str]:
    out: list[str] = []
    for para in paragraphs:
        for raw in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", para):
            sentence = normalise(raw)
            if len(sentence.split()) >= min_words and "http" not in sentence:
                out.append(sentence)
    return out


def ground_truth_from_ar5iv(html: str) -> tuple[list[str], list[str]]:
    parser = _Ar5ivParser()
    parser.feed(html)
    sentences = sentences_from_paragraphs(parser.paragraphs)
    cells = [normalise(c) for c in parser.cells]
    cells = [c for c in cells if len(c) >= 2 and not re.fullmatch(r"[-\s]*", c)]
    return sentences, cells


def ground_truth_from_text(text: str) -> list[str]:
    return sentences_from_paragraphs([p for p in re.split(r"\n\s*\n", text) if p.strip()])


def sentence_recall(gt_sentences: list[str], extracted: str) -> float:
    if not gt_sentences:
        return float("nan")
    text = normalise(extracted)
    return sum(1 for s in gt_sentences if s in text) / len(gt_sentences)


def ngram_recall(gt_sentences: list[str], extracted: str, n: int = 8) -> float:
    grams: set[tuple[str, ...]] = set()
    for s in gt_sentences:
        grams |= ngrams(s.split(), n)
    if not grams:
        return float("nan")
    return len(grams & ngrams(normalise(extracted).split(), n)) / len(grams)


def table_cell_recall(gt_cells: list[str], extracted: str) -> tuple[float, float]:
    """(cells found inside Markdown table rows, cells found anywhere)."""
    if not gt_cells:
        return float("nan"), float("nan")
    table_text = " " + " ".join(normalise(l) for l in extracted.splitlines() if l.strip().startswith("|")) + " "
    full_text = " " + normalise(extracted) + " "
    in_tables = sum(1 for c in gt_cells if f" {c} " in table_text or c in table_text)
    anywhere = sum(1 for c in gt_cells if c in full_text)
    return in_tables / len(gt_cells), anywhere / len(gt_cells)


def markdown_table_count(extracted: str) -> int:
    return sum(1 for l in extracted.splitlines() if re.match(r"^\|[\s:\-|]+\|$", l.strip()))


def heading_count(extracted: str) -> int:
    return sum(1 for l in extracted.splitlines() if re.match(r"^#{1,6} ", l))
