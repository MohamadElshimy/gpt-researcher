"""PageRouter behaviour with controllable backends: routing, gates,
escalation, fallbacks, budgets, deadlines, output assembly."""

from __future__ import annotations

import pymupdf
import pytest

from gpt_researcher.scraper.pdf_router.backends import (
    BackendError,
    BackendRegistry,
    BackendTimeout,
    BackendUnavailable,
)
from gpt_researcher.scraper.pdf_router.backends.plain import PlainPyMuPDFBackend
from gpt_researcher.scraper.pdf_router.classify import PageClass
from gpt_researcher.scraper.pdf_router.router import InlineExecutor, PageRouter
from gpt_researcher.scraper.pdf_router.settings import RouterSettings

from .fake_backends import FakeBackend

LAYOUT_HANDLES = {PageClass.PLAIN, PageClass.MULTI_COLUMN, PageClass.TABLE, PageClass.FIGURE}
DOCLING_HANDLES = LAYOUT_HANDLES | {PageClass.SCAN, PageClass.BROKEN_TEXT}

GOOD_TABLE = "\n\n|Model|A|B|\n|---|---|---|\n|x|1.0|2.0|\n|y|3.0|4.0|\n"
BAD_TABLE = "\n\n|System|D|ev|Te|st|System|Dev|Test|\n|---|---|---|---|---|---|---|---|\n|a|-|-|845|905|b|81.6|-|\n||||.|.|c|1|2|\n"
OCR_TEXT = "Dear Pete, permit me to introduce you to the facility of facsimile transmission."


def plain_text(pdf_bytes: bytes, page: int) -> str:
    return PlainPyMuPDFBackend().parse(pdf_bytes, [page])[page].markdown


def make_router(pdf_bytes, layout: FakeBackend | None = None, docling: FakeBackend | None = None, **settings):
    backends = [PlainPyMuPDFBackend()]
    if layout is not None:
        backends.append(layout)
    if docling is not None:
        backends.append(docling)
    registry = BackendRegistry(backends)
    defaults = dict(inline=True, light_workers=2, heavy_workers=1, doc_deadline=30.0)
    defaults.update(settings)
    return PageRouter(settings=RouterSettings(**defaults), executor=InlineExecutor(registry), registry=registry)


def keep_text(pdf_bytes, extra=""):
    """Fake output that keeps the page's prose (passes the word-recall gate) plus ``extra``."""
    return lambda p: plain_text(pdf_bytes, p) + extra


# --------------------------------------------------------------------------- routing
def test_each_class_goes_to_the_cheapest_capable_backend(pdf_builder):
    pdf = pdf_builder("pts")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=OCR_TEXT)
    doc = make_router(pdf, layout, docling).route(pdf, "x.pdf")

    assert [p.page_class for p in doc.pages] == [PageClass.PLAIN, PageClass.TABLE, PageClass.SCAN]
    assert [p.backend for p in doc.pages] == ["plain", "layout", "docling"]
    assert layout.calls == [[1]]
    assert docling.calls == [[2]]
    assert doc.backend_counts == {"plain": 1, "layout": 1, "docling": 1}
    # Output keeps page order: prose, then the table, then the OCR text.
    assert doc.markdown.index("Section 0") < doc.markdown.index("|Model|A|B|") < doc.markdown.index("Dear Pete")


def test_plain_pages_never_touch_ml_backends(pdf_builder):
    pdf = pdf_builder("ppp")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default="never")
    doc = make_router(pdf, layout).route(pdf)
    assert layout.calls == []
    assert all(p.backend == "plain" for p in doc.pages)


def test_blank_pages_are_dropped_silently(pdf_builder):
    pdf = pdf_builder("pbp")
    doc = make_router(pdf).route(pdf)
    assert doc.pages[1].page_class == PageClass.BLANK
    assert "<!--" not in doc.markdown
    assert doc.markdown.count("Section") == 2


# --------------------------------------------------------------------------- gates / escalation
def test_garbled_table_escalates_to_next_tier(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, BAD_TABLE))
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    doc = make_router(pdf, layout, docling).route(pdf)

    page = doc.pages[0]
    assert page.backend == "docling"
    assert docling.calls == [[0]]
    assert any("gate failed" in a and "fragmented header" in a for a in page.attempts)
    assert "|Model|A|B|" in doc.markdown and "845" not in doc.markdown


def test_table_issues_without_higher_tier_are_accepted_with_note(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, BAD_TABLE))
    doc = make_router(pdf, layout).route(pdf)
    page = doc.pages[0]
    assert page.backend == "layout"  # structure with a dodgy table still beats flattened cells
    assert any("accepted with table issues" in a for a in page.attempts)


def test_dropped_text_falls_back_to_plain_text(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default="## Heading only, the body text is gone" + GOOD_TABLE)
    doc = make_router(pdf, layout).route(pdf)
    page = doc.pages[0]
    assert page.backend == "plain"
    assert any("word recall" in a for a in page.attempts)
    assert any("kept plain output" in a for a in page.attempts)
    assert "Model 3" in doc.markdown  # the raw text layer survived


def test_dropped_text_escalates_when_a_higher_tier_exists(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default="## Heading only" + GOOD_TABLE)
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    doc = make_router(pdf, layout, docling).route(pdf)
    assert doc.pages[0].backend == "docling"


def test_empty_ocr_output_is_not_accepted(pdf_builder):
    pdf = pdf_builder("ps")
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default="")
    doc = make_router(pdf, None, docling).route(pdf)
    scan = doc.pages[1]
    assert scan.backend == "plain"
    assert any("empty output" in a for a in scan.attempts)
    assert "<!-- page 2: no text extracted" in doc.markdown


# --------------------------------------------------------------------------- failures
def test_backend_error_falls_back_down_the_ladder(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, error=BackendError("boom"))
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    doc = make_router(pdf, layout, docling).route(pdf)
    assert doc.pages[0].backend == "docling"
    assert any("layout: BackendError: boom" in a for a in doc.pages[0].attempts)


def test_all_ml_backends_failing_keeps_plain_text(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, error=BackendError("boom"))
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, error=BackendUnavailable("breaker open"))
    doc = make_router(pdf, layout, docling).route(pdf)
    assert doc.pages[0].backend == "plain"
    assert "Model 3" in doc.markdown


def test_unexpected_exception_in_backend_does_not_kill_the_document(pdf_builder):
    pdf = pdf_builder("tp")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, error=RuntimeError("bug"))
    doc = make_router(pdf, layout).route(pdf)
    assert doc.pages[0].backend == "plain"
    assert any("unexpected RuntimeError" in a for a in doc.pages[0].attempts)
    assert "Section 1" in doc.markdown


def test_timeout_from_executor_is_a_fallback(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    registry = BackendRegistry([PlainPyMuPDFBackend(), layout])

    class TimingOutExecutor:
        max_concurrency = 1

        def run(self, name, pdf_bytes, pages, timeout):
            raise BackendTimeout(f"{name}: no result within {timeout:.0f}s")

    router = PageRouter(settings=RouterSettings(inline=True), executor=TimingOutExecutor(), registry=registry)
    doc = router.route(pdf)
    assert doc.pages[0].backend == "plain"
    assert any("BackendTimeout" in a for a in doc.pages[0].attempts)


def test_task_timeout_is_clamped_to_the_document_deadline(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    registry = BackendRegistry([PlainPyMuPDFBackend(), layout])
    seen: list[float] = []

    class RecordingExecutor:
        max_concurrency = 1

        def run(self, name, pdf_bytes, pages, timeout):
            seen.append(timeout)
            return registry.get(name).parse(pdf_bytes, pages)

    router = PageRouter(
        settings=RouterSettings(inline=True, light_timeout=30.0, doc_deadline=5.0),
        executor=RecordingExecutor(),
        registry=registry,
    )
    router.route(pdf)
    assert seen and all(t <= 5.0 for t in seen), seen  # sidecar kills the worker at the document deadline


def test_no_new_ml_tasks_start_when_the_deadline_is_nearly_spent(pdf_builder):
    pdf = pdf_builder("t")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    doc = make_router(pdf, layout, doc_deadline=0.5).route(pdf)
    assert layout.calls == []
    assert any("deadline" in a for a in doc.pages[0].attempts)


def test_document_deadline_stops_waiting(pdf_builder):
    pdf = pdf_builder("tt")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE), delay=3.0)
    doc = make_router(pdf, layout, doc_deadline=1.2).route(pdf)
    assert all(p.backend == "plain" for p in doc.pages)
    assert any("deadline" in a for p in doc.pages for a in p.attempts)
    assert doc.elapsed < 2.5


# --------------------------------------------------------------------------- budgets and switches
def test_ml_page_budget_prefers_scans_over_tables(pdf_builder):
    pdf = pdf_builder("tst")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=OCR_TEXT)
    doc = make_router(pdf, layout, docling, max_ml_pages=1).route(pdf)
    assert [p.backend for p in doc.pages] == ["plain", "docling", "plain"]
    assert any("budget" in n for n in doc.notes)
    assert any("budget exhausted" in a for a in doc.pages[0].attempts)


def test_pages_beyond_triage_cap_stay_plain(pdf_builder):
    pdf = pdf_builder("pt")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    doc = make_router(pdf, layout, max_triage_pages=1).route(pdf)
    assert doc.pages[1].backend == "plain"
    assert layout.calls == []
    assert any("triage cap" in n for n in doc.notes)


def test_docling_runs_in_contiguous_batches(pdf_builder):
    pdf = pdf_builder("sspss")
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=OCR_TEXT, batch_pages=8)
    make_router(pdf, None, docling).route(pdf)
    assert sorted(docling.calls) == [[0, 1], [3, 4]]


def test_disabled_tier_is_skipped(pdf_builder):
    pdf = pdf_builder("s")
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=OCR_TEXT)
    doc = make_router(pdf, None, docling, docling_tier=False).route(pdf)
    assert docling.calls == []
    assert any("no enabled backend handles scan" in a for a in doc.pages[0].attempts)
    assert doc.markdown.startswith("<!-- page 1: no text extracted")


def test_unavailable_backend_is_not_in_the_ladder(pdf_builder):
    pdf = pdf_builder("s")
    docling = FakeBackend("docling", 2, DOCLING_HANDLES, default=OCR_TEXT, available=False)
    router = make_router(pdf, None, docling)
    assert [b.name for b in router.ladder] == ["plain"]


def test_router_disabled_returns_plain_everything(pdf_builder):
    pdf = pdf_builder("ts")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    doc = make_router(pdf, layout, enabled=False).route(pdf)
    assert layout.calls == []
    assert "router disabled" in doc.notes


# --------------------------------------------------------------------------- document level
def test_encrypted_pdf_yields_empty_result_with_note(tmp_path):
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "secret text " * 20)
    path = tmp_path / "enc.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    routed = make_router(b"").route(path.read_bytes())
    assert routed.markdown == ""
    assert any("encrypted" in n for n in routed.notes)


def test_non_pdf_bytes_raise(pdf_builder):
    with pytest.raises(Exception):
        make_router(b"").route(b"this is not a pdf")


def test_title_from_metadata_then_first_page_heading(pdf_builder):
    pdf = pdf_builder("t", title="Metadata Title")
    layout = FakeBackend("layout", 1, LAYOUT_HANDLES, default=keep_text(pdf, GOOD_TABLE))
    assert make_router(pdf, layout).route(pdf).title == "Metadata Title"

    pdf = pdf_builder("tt")
    layout = FakeBackend(
        "layout", 1, LAYOUT_HANDLES, outputs={0: lambda p: "# **First Page Heading**\n\n" + plain_text(pdf, p) + GOOD_TABLE, 1: lambda p: "# Later Heading\n\n" + plain_text(pdf, p) + GOOD_TABLE}
    )
    assert make_router(pdf, layout).route(pdf).title == "First Page Heading"

    pdf = pdf_builder("p")
    assert make_router(pdf).route(pdf).title == ""


def test_route_text_returns_markdown_string(pdf_builder):
    pdf = pdf_builder("p")
    router = make_router(pdf)
    assert router.route_text(pdf) == router.route(pdf).markdown
