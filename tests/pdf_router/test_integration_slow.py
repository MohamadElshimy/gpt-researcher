"""Real ladder on the fixture pack. Needs pymupdf 2.x and Docling with its
weights; opt in with ``PDF_ROUTER_RUN_SLOW=1``. Runs inline so it exercises the
backends, not the process pool (test_workers covers that)."""

from __future__ import annotations

import importlib.util
import os

import pytest

from gpt_researcher.scraper.pdf_router import PageClass, PageRouter, RouterSettings

pytestmark = [
    pytest.mark.skipif(os.environ.get("PDF_ROUTER_RUN_SLOW") != "1", reason="set PDF_ROUTER_RUN_SLOW=1"),
    pytest.mark.skipif(importlib.util.find_spec("docling") is None, reason="docling not installed"),
    pytest.mark.timeout(600),
]


@pytest.fixture(scope="module")
def router():
    return PageRouter(settings=RouterSettings(inline=True))


def test_mixed_document_routes_every_class(router, fixtures_dir):
    doc = router.route((fixtures_dir / "mixed-document.pdf").read_bytes(), "mixed-document.pdf")
    classes = [p.page_class for p in doc.pages]
    assert classes == [PageClass.PLAIN, PageClass.PLAIN, PageClass.SCAN, PageClass.TABLE, PageClass.TABLE]
    assert doc.pages[2].backend == "docling"
    assert "facsimile" in doc.pages[2].markdown.lower()
    assert doc.pages[0].backend == "plain"
    for page in doc.pages[3:]:
        assert page.backend in ("layout", "docling")
        assert any(line.startswith("|") for line in page.markdown.splitlines()), page.attempts


def test_rowmajor_fixture_reads_in_order(router, fixtures_dir):
    doc = router.route((fixtures_dir / "rowmajor-two-column.pdf").read_bytes())
    assert doc.pages[0].backend in ("layout", "docling")
    text = " ".join(doc.markdown.split()).lower()
    assert "we introduce a new language representation model called bert" in text


def test_side_by_side_tables_escalate_to_docling(router, fixtures_dir):
    doc = router.route((fixtures_dir / "two-column-paper.pdf").read_bytes())
    page7 = doc.pages[6]
    assert page7.page_class == PageClass.TABLE
    assert page7.backend == "docling", page7.attempts
    assert any("gate failed" in a for a in page7.attempts)
