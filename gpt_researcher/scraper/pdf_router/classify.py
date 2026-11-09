"""Stage-1 page triage: cheap geometric signals, no ML.

One ``get_text("dict")`` (images excluded), one ``get_image_info()`` and one
``get_cdrawings()`` per page: 10-20 ms on this hardware. ``find_tables()`` is
deliberately not used -- it costs 100-300 ms per page and, since pymupdf 2.x,
runs the layout model by default.

Threshold provenance (see DECISIONS.md):
- scan / blank: Artifex "triage your documents" post (chars < 30 and image
  cover >= 25% -> OCR; chars < 10 and cover < 2% -> skip); MinerU uses
  < 50 chars/page; Tika ``ocrStrategy=auto`` uses < 10 chars.
- broken text layer: Artifex ``BAD_CHAR_THRESHOLD = 0.10`` (share of U+FFFD);
  Tika "> 10 unmapped glyphs"; Kreuzberg >= 5 U+FFFD.
- column jump rate and numeric-line count: measured on the fixture pack, see
  the table in DECISIONS.md (synthetic row-major page 0.92 vs <= 0.06 on every
  real page; table pages 12-73 numeric lines vs <= 8 on prose).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum

import pymupdf

from .settings import RouterSettings

_NUMERIC_LINE = re.compile(r"^[\d.,%±\-–—/()\s]+$")


class PageClass(str, Enum):
    BLANK = "blank"
    PLAIN = "plain"  # single column or column-major stream; no structure to recover
    MULTI_COLUMN = "multi_column"  # content stream is row-major across columns
    TABLE = "table"
    FIGURE = "figure"  # image-heavy page with text (captions, reading order)
    SCAN = "scan"  # no text layer, page is a bitmap
    BROKEN_TEXT = "broken_text"  # text layer present but unmapped glyphs
    CHART = "chart"  # reserved for a vision tier; never emitted by triage


# Higher value = more important to keep on an ML tier when the page budget binds.
SEVERITY = {
    PageClass.SCAN: 6,
    PageClass.BROKEN_TEXT: 5,
    PageClass.TABLE: 4,
    PageClass.MULTI_COLUMN: 3,
    PageClass.FIGURE: 2,
    PageClass.PLAIN: 0,
    PageClass.BLANK: -1,
    PageClass.CHART: 1,
}


@dataclass(frozen=True)
class PageSignals:
    chars: int
    bad_ratio: float
    image_cover: float
    line_count: int
    numeric_lines: int
    column_jump_rate: float
    rule_lines: int
    millis: float


@dataclass(frozen=True)
class PageTriage:
    page_no: int  # 0-based
    page_class: PageClass
    signals: PageSignals


def compute_signals(page: pymupdf.Page) -> PageSignals:
    t0 = time.perf_counter()
    rect = page.rect
    width, height = rect.width, rect.height
    area = max(width * height, 1.0)

    flags = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES
    blocks = page.get_text("dict", flags=flags)["blocks"]

    chars = 0
    bad = 0
    lines: list[pymupdf.Rect] = []
    numeric_lines = 0
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line.get("spans", [])).strip()
            if not text:
                continue
            chars += len(text)
            bad += text.count("�")
            lines.append(pymupdf.Rect(line["bbox"]))
            if any(ch.isdigit() for ch in text) and _NUMERIC_LINE.match(text):
                numeric_lines += 1

    # Consecutive lines in content-stream order that jump horizontally by more
    # than 30% of the page width while overlapping vertically: the signature of
    # a row-major stream over a multi-column layout (Word/InDesign exports,
    # OCR'd PDFs). Column-major streams only jump once per column, and those
    # two lines never share a row.
    jumps = 0
    for prev, cur in zip(lines, lines[1:]):
        if abs(cur.x0 - prev.x0) > 0.3 * width and min(prev.y1, cur.y1) - max(prev.y0, cur.y0) > 0:
            jumps += 1
    jump_rate = jumps / (len(lines) - 1) if len(lines) > 1 else 0.0

    image_area = 0.0
    try:
        for info in page.get_image_info():
            image_area += abs(pymupdf.Rect(info["bbox"]) & rect)
    except Exception:  # pragma: no cover - defensive: malformed image dictionaries
        pass

    rule_lines = 0
    try:
        for path in page.get_cdrawings():
            r = pymupdf.Rect(path["rect"])
            if r.height <= 2.0 and r.width >= 0.2 * width:
                rule_lines += 1
    except Exception:  # pragma: no cover - defensive: pathological vector content
        pass

    return PageSignals(
        chars=chars,
        bad_ratio=(bad / chars) if chars else 0.0,
        image_cover=min(image_area / area, 1.0),
        line_count=len(lines),
        numeric_lines=numeric_lines,
        column_jump_rate=jump_rate,
        rule_lines=rule_lines,
        millis=(time.perf_counter() - t0) * 1000.0,
    )


def classify_signals(signals: PageSignals, settings: RouterSettings) -> PageClass:
    s = signals
    if s.chars < settings.blank_max_chars and s.image_cover < settings.blank_max_image_cover:
        return PageClass.BLANK
    if s.chars < settings.scan_max_chars and s.image_cover >= settings.scan_min_image_cover:
        return PageClass.SCAN
    if s.chars >= settings.scan_max_chars and s.bad_ratio >= settings.broken_text_bad_ratio:
        return PageClass.BROKEN_TEXT
    if s.line_count >= 10 and s.column_jump_rate >= settings.column_jump_rate:
        return PageClass.MULTI_COLUMN
    if s.numeric_lines >= settings.table_min_numeric_lines or s.rule_lines >= settings.table_min_rule_lines:
        return PageClass.TABLE
    if s.image_cover >= settings.figure_min_image_cover:
        return PageClass.FIGURE
    return PageClass.PLAIN


def triage_page(page: pymupdf.Page, settings: RouterSettings) -> PageTriage:
    signals = compute_signals(page)
    return PageTriage(page_no=page.number, page_class=classify_signals(signals, settings), signals=signals)
