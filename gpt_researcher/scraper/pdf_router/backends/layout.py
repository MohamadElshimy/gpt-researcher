"""Tier 1: PyMuPDF layout mode.

Since pymupdf 2.0 the former ``pymupdf4llm`` package and Artifex's layout
analyser (a 1.8M-parameter graph neural network over MuPDF geometry, ONNX,
CPU) ship inside the ``pymupdf`` wheel itself, so this tier adds no
dependency. It recovers reading order across columns, headings, and tables
in Markdown at roughly 0.2-0.45 s/page on an M3 Pro.

OCR is switched off here on purpose: the OCR engine pymupdf4llm picks up is
opportunistic (RapidOCR is only present because Docling installs it) and it
degrades to a warning when absent. Pages that need OCR go to the Docling tier.

Licensing note: the standalone ``pymupdf-layout`` 1.x wheels were "Polyform
Noncommercial or Artifex Commercial"; in 2.x the same code ships under the
pymupdf wheel's "AGPL-3.0 or Artifex Commercial" metadata with its source
absent from the public repo. ``PDF_ROUTER_LAYOUT_TIER=false`` removes this
tier (see DECISIONS-DETAILED.md).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

import pymupdf

from ..classify import PageClass
from .base import BackendError, PageResult


class PyMuPDFLayoutBackend:
    name = "layout"
    tier = 1
    handles = frozenset({PageClass.PLAIN, PageClass.MULTI_COLUMN, PageClass.TABLE, PageClass.FIGURE})

    def available(self) -> bool:
        return importlib.util.find_spec("pymupdf4llm") is not None and importlib.util.find_spec("pymupdf.layout") is not None

    def parse(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> dict[int, PageResult]:
        import pymupdf4llm  # heavy: activates the ONNX layout model on first use
        from pymupdf4llm.ocr import OCRMode

        pages = sorted(set(page_numbers))
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            chunks = pymupdf4llm.to_markdown(
                doc,
                pages=pages,
                page_chunks=True,
                use_ocr=OCRMode.NEVER,
                header=False,  # drop running page headers/footers (arXiv stamp, page numbers)
                footer=False,
                show_progress=False,
            )
        if not isinstance(chunks, list):
            raise BackendError("pymupdf4llm.to_markdown did not return page chunks")

        results: dict[int, PageResult] = {}
        for chunk in chunks:
            meta = chunk.get("metadata") or {}
            page_number = meta.get("page_number")  # 1-based in pymupdf4llm
            if page_number is None:
                continue
            p = int(page_number) - 1
            results[p] = PageResult(page_no=p, markdown=(chunk.get("text") or "").strip(), backend=self.name)
        missing = [p for p in pages if p not in results]
        if missing:
            raise BackendError(f"layout backend returned no chunk for pages {missing}")
        return results
