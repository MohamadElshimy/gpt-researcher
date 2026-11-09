"""Build ``rowmajor-two-column.pdf``: a two-column PDF whose content stream is
row-major (line 1 of the left column, line 1 of the right column, ...).

The four fixtures in the pack are LaTeX PDFs whose content stream already
follows reading order, so plain ``get_text()`` gets their columns right. PDFs
exported by Word, InDesign or produced by OCR often do not, and that is the
case the brief describes ("columns interleave mid-sentence"). This fixture
reproduces it with text taken from the fixture pack itself (first two pages
of the BERT paper, read in correct order), so it is reproducible and carries
no new licensing. The reading-order text is written next to it as ground
truth for the evaluation script.

Usage: python -m tests.fixtures.pdf_router.make_rowmajor [source.pdf] [out.pdf]
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent
DEFAULT_SOURCE = HERE / "two-column-paper.pdf"
DEFAULT_OUT = HERE / "rowmajor-two-column.pdf"


def reading_order_text(source_pdf: Path, pages=(0, 1)) -> str:
    """Plain text of the given pages in content-stream order, which for these
    LaTeX fixtures is the true reading order; hyphenated line breaks joined."""
    with pymupdf.open(source_pdf) as doc:
        text = "\n".join(doc[p].get_text("text") for p in pages)
    for glyph, ascii_form in {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}.items():
        text = text.replace(glyph, ascii_form)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    # Keep real prose only: long paragraphs, no author/affiliation lines.
    return "\n\n".join(p for p in paragraphs if len(p.split()) >= 25)


def write_rowmajor_pdf(text: str, out_path: Path, chars_per_line: int = 52) -> int:
    width, height = 595, 842
    margin, gutter = 50, 24
    col_width = (width - 2 * margin - gutter) / 2
    fontsize, leading = 9, 11.5
    lines: list[str] = []
    for para in text.split("\n\n"):
        lines += textwrap.wrap(para, chars_per_line) + [""]
    per_col = int((height - 2 * margin - 20) // leading)

    doc = pymupdf.open()
    i = 0
    while i < len(lines):
        page = doc.new_page(width=width, height=height)
        left = lines[i : i + per_col]
        right = lines[i + per_col : i + 2 * per_col]
        for row in range(per_col):
            y = margin + 10 + row * leading
            # Left line, then right line of the same row: a row-major content stream.
            if row < len(left):
                page.insert_text((margin, y), left[row], fontsize=fontsize, fontname="helv")
            if row < len(right):
                page.insert_text((margin + col_width + gutter, y), right[row], fontsize=fontsize, fontname="helv")
        i += 2 * per_col
    doc.set_metadata({"title": "Synthetic row-major two-column fixture"})
    doc.save(out_path, garbage=4, deflate=True)
    pages = doc.page_count
    doc.close()
    return pages


def main(argv: list[str]) -> None:
    source = Path(argv[1]) if len(argv) > 1 else DEFAULT_SOURCE
    out = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUT
    text = reading_order_text(source)
    pages = write_rowmajor_pdf(text, out)
    out.with_suffix(".gt.txt").write_text(text, encoding="utf-8")
    print(f"wrote {out} ({pages} pages) and {out.with_suffix('.gt.txt')}")


if __name__ == "__main__":
    main(sys.argv)
