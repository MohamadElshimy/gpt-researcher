"""Regression test: PyMuPDFScraper must not leave files behind.

Historically the scraper downloaded remote PDFs to a
``NamedTemporaryFile(delete=False, suffix=".pdf")`` and could leak it when
parsing failed. The scraper now keeps the download in memory and hands the
bytes to the page router, so no temp file exists to leak -- this test pins
that, together with the empty-result contract on parse failure.
"""

import glob
import os
import tempfile
from unittest.mock import patch

from gpt_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper


class _FakeResponse:
    headers = {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        yield b"%PDF-1.4 not-a-real-pdf"


def _temp_pdfs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))


def test_no_tempfile_and_empty_result_when_parsing_fails():
    scraper = PyMuPDFScraper("https://example.com/broken.pdf")

    before = _temp_pdfs()

    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
        return_value=_FakeResponse(),
    ), patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.route_pdf_detailed",
        side_effect=RuntimeError("corrupt PDF"),
    ):
        content, images, title = scraper.scrape()

    assert (content, images, title) == ("", [], "")
    leaked = _temp_pdfs() - before
    assert not leaked, f"PyMuPDFScraper leaked temp file(s): {leaked}"
