"""PyMuPDFScraper keeps its ``(content, image_urls, title)`` contract and hands
the downloaded bytes to the router; nothing is written to disk."""

from __future__ import annotations

import glob
import os
import tempfile
from unittest.mock import patch

import pytest

from gpt_researcher.scraper.pdf_router.router import RoutedDocument
from gpt_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper

MODULE = "gpt_researcher.scraper.pymupdf.pymupdf"


class _FakeResponse:
    def __init__(self, data: bytes, headers=None):
        self._data = data
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]


def routed(markdown="# Title\n\nbody " * 30, title="Doc Title", notes=None) -> RoutedDocument:
    return RoutedDocument(
        source_url="",
        title=title,
        pages=[],
        markdown=markdown,
        elapsed=0.01,
        backend_counts={"plain": 1},
        class_counts={"plain": 1},
        notes=notes or [],
    )


def _temp_pdfs() -> set:
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "*.pdf")))


def test_url_download_is_passed_to_router_as_bytes(fixtures_dir):
    data = (fixtures_dir / "scanned-page.pdf").read_bytes()
    before = _temp_pdfs()
    with patch(f"{MODULE}.requests.get", return_value=_FakeResponse(data)) as get, patch(
        f"{MODULE}.route_pdf_detailed", return_value=routed()
    ) as route:
        content, images, title = PyMuPDFScraper("https://example.com/paper.pdf").scrape()

    get.assert_called_once()
    assert get.call_args.kwargs["stream"] is True
    passed_bytes, passed_url = route.call_args.args
    assert passed_bytes == data and passed_url == "https://example.com/paper.pdf"
    assert content.startswith("# Title") and images == [] and title == "Doc Title"
    assert _temp_pdfs() == before  # no temp files any more


def test_local_path_is_read_and_routed(fixtures_dir):
    path = str(fixtures_dir / "scanned-page.pdf")
    with patch(f"{MODULE}.route_pdf_detailed", return_value=routed(title="")) as route:
        content, images, title = PyMuPDFScraper(path).scrape()
    assert route.call_args.args[0] == open(path, "rb").read()
    assert content and images == [] and title == ""


def test_router_failure_keeps_empty_result_contract():
    with patch(f"{MODULE}.requests.get", return_value=_FakeResponse(b"%PDF-1.4 broken")), patch(
        f"{MODULE}.route_pdf_detailed", side_effect=RuntimeError("corrupt PDF")
    ):
        scraper = PyMuPDFScraper("https://example.com/broken.pdf")
        assert scraper.scrape() == ("", [], "")
    assert "corrupt PDF" in scraper.failure_reason


def test_download_timeout_keeps_empty_result_contract():
    import requests

    with patch(f"{MODULE}.requests.get", side_effect=requests.exceptions.Timeout()):
        scraper = PyMuPDFScraper("https://example.com/slow.pdf")
        assert scraper.scrape() == ("", [], "")
    assert "timed out" in scraper.failure_reason


def test_oversized_pdf_is_rejected_before_parsing(monkeypatch):
    monkeypatch.setenv("PDF_ROUTER_MAX_PDF_BYTES", "1000")
    with patch(f"{MODULE}.requests.get", return_value=_FakeResponse(b"x" * 5000)), patch(
        f"{MODULE}.route_pdf_detailed"
    ) as route:
        scraper = PyMuPDFScraper("https://example.com/huge.pdf")
        assert scraper.scrape() == ("", [], "")
    route.assert_not_called()
    assert "larger than" in scraper.failure_reason


def test_declared_content_length_short_circuits(monkeypatch):
    monkeypatch.setenv("PDF_ROUTER_MAX_PDF_BYTES", "1000")
    response = _FakeResponse(b"x" * 10, headers={"Content-Length": "999999"})
    with patch(f"{MODULE}.requests.get", return_value=response), patch(f"{MODULE}.route_pdf_detailed") as route:
        assert PyMuPDFScraper("https://example.com/huge.pdf").scrape() == ("", [], "")
    route.assert_not_called()


def test_empty_router_output_sets_failure_reason():
    with patch(f"{MODULE}.requests.get", return_value=_FakeResponse(b"%PDF-1.4")), patch(
        f"{MODULE}.route_pdf_detailed", return_value=routed(markdown="", title="", notes=["encrypted PDF, no text extracted"])
    ):
        scraper = PyMuPDFScraper("https://example.com/enc.pdf")
        assert scraper.scrape() == ("", [], "")
    assert "encrypted" in scraper.failure_reason


def test_is_url():
    assert PyMuPDFScraper("https://example.com/a.pdf").is_url()
    assert not PyMuPDFScraper("/tmp/a.pdf").is_url()
    assert not PyMuPDFScraper("a.pdf").is_url()


@pytest.mark.skipif(os.environ.get("PDF_ROUTER_RUN_SLOW") != "1", reason="set PDF_ROUTER_RUN_SLOW=1 to run the real ladder")
def test_end_to_end_inline_on_scanned_fixture(fixtures_dir, monkeypatch):
    monkeypatch.setenv("PDF_ROUTER_INLINE", "1")
    content, images, title = PyMuPDFScraper(str(fixtures_dir / "scanned-page.pdf")).scrape()
    assert "facsimile" in content.lower()
    assert images == []
