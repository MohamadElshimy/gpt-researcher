"""Tier 2: Docling (MIT). Layout model + TableFormer + OCR.

One pipeline configuration only, with OCR enabled: measured, OCR adds nothing
on pure-text pages (0.21 s either way) and only runs on bitmap regions, while
a second pipeline configuration would cost another ~1.3 GB of resident
memory. The converter is created once per process and reused; Docling
documents its converter as thread-safe with per-run isolation (>= 2.60), and
2.139 fixed the per-conversion thread/memory leak, hence the version floor.

``document_timeout`` is Docling's cooperative timeout (checked between
pipeline stages). The hard stop lives in the worker pool, which kills the
process when the task timeout expires.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import threading
from collections.abc import Sequence

from ..classify import PageClass
from ..settings import RouterSettings
from .base import BackendError, PageResult, contiguous_runs


class DoclingBackend:
    name = "docling"
    tier = 2
    handles = frozenset(
        {
            PageClass.PLAIN,
            PageClass.MULTI_COLUMN,
            PageClass.TABLE,
            PageClass.FIGURE,
            PageClass.SCAN,
            PageClass.BROKEN_TEXT,
        }
    )

    _converter = None
    _lock = threading.Lock()

    def available(self) -> bool:
        return importlib.util.find_spec("docling") is not None

    @classmethod
    def _get_converter(cls):
        if cls._converter is not None:
            return cls._converter
        with cls._lock:
            if cls._converter is None:
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import (
                    DocumentConverter,
                    PdfFormatOption,
                )

                settings = RouterSettings.from_env()
                options = PdfPipelineOptions()
                options.do_ocr = True  # engine auto-selected: OcrMac on macOS, RapidOCR elsewhere
                options.do_table_structure = True
                options.document_timeout = max(10.0, settings.heavy_timeout * 0.9)
                converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
                )
                converter.initialize_pipeline(InputFormat.PDF)  # load weights now, not on the first page
                cls._converter = converter
        return cls._converter

    def parse(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> dict[int, PageResult]:
        from docling.datamodel.base_models import ConversionStatus, DocumentStream

        converter = self._get_converter()
        results: dict[int, PageResult] = {}
        for run in contiguous_runs(page_numbers):
            first, last = run[0] + 1, run[-1] + 1  # Docling page ranges are 1-based, inclusive
            try:
                result = converter.convert(
                    DocumentStream(name="document.pdf", stream=io.BytesIO(pdf_bytes)),
                    page_range=(first, last),
                    raises_on_error=False,
                )
            except Exception as exc:  # ConversionError and friends
                raise BackendError(f"docling convert failed for pages {first}-{last}: {exc!r}") from exc
            if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                raise BackendError(f"docling status {result.status} for pages {first}-{last}")

            notes = ("partial_success",) if result.status == ConversionStatus.PARTIAL_SUCCESS else ()
            confidence_pages = getattr(getattr(result, "confidence", None), "pages", None) or {}
            for p in run:
                markdown = result.document.export_to_markdown(page_no=p + 1, image_placeholder="")
                conf = confidence_pages.get(p + 1)
                score = getattr(conf, "mean_score", None) if conf is not None else None
                if score is not None and (not isinstance(score, (int, float)) or math.isnan(score)):
                    score = None
                results[p] = PageResult(
                    page_no=p,
                    markdown=markdown.strip(),
                    backend=self.name,
                    confidence=float(score) if score is not None else None,
                    notes=notes,
                )
            with contextlib.suppress(Exception):
                result.input.release()  # free parser buffers early (docling >= 2.145)
        return results
