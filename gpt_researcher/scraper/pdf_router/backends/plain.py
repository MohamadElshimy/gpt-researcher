"""Tier 0: plain PyMuPDF text extraction. What the scraper did before, minus LangChain."""

from __future__ import annotations

from collections.abc import Sequence

import pymupdf

from ..classify import PageClass
from .base import PageResult

# Unambiguous typographic ligatures that embed as single glyphs. Left as-is
# they break search and embeddings ("signiﬁcant" != "significant").
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "st",
    "ﬆ": "st",
}


def normalise_text(text: str) -> str:
    for glyph, ascii_form in _LIGATURES.items():
        text = text.replace(glyph, ascii_form)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


class PlainPyMuPDFBackend:
    name = "plain"
    tier = 0
    handles = frozenset({PageClass.PLAIN, PageClass.BLANK})

    def available(self) -> bool:
        return True

    def parse(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> dict[int, PageResult]:
        results: dict[int, PageResult] = {}
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for p in page_numbers:
                text = doc[p].get_text("text")
                results[p] = PageResult(page_no=p, markdown=normalise_text(text), backend=self.name)
        return results
