"""Stage-1 triage on the fixture pack and on synthetic pages. No ML involved."""

from __future__ import annotations

import pymupdf
import pytest

from gpt_researcher.scraper.pdf_router.classify import (
    PageClass,
    PageSignals,
    classify_signals,
    compute_signals,
    triage_page,
)
from gpt_researcher.scraper.pdf_router.settings import RouterSettings

S = RouterSettings()


def _classes(path):
    with pymupdf.open(path) as doc:
        return [triage_page(p, S).page_class for p in doc]


def test_scanned_page_is_scan(fixtures_dir):
    assert _classes(fixtures_dir / "scanned-page.pdf") == [PageClass.SCAN]


def test_mixed_document_classes(fixtures_dir):
    classes = _classes(fixtures_dir / "mixed-document.pdf")
    assert classes[0] == PageClass.PLAIN
    assert classes[1] == PageClass.PLAIN
    assert classes[2] == PageClass.SCAN
    assert classes[3] == PageClass.TABLE
    assert classes[4] == PageClass.TABLE


def test_roberta_table_pages_detected(fixtures_dir):
    classes = _classes(fixtures_dir / "borderless-table.pdf")
    # Pages 4-9 (1-based) carry Tables 1-8 of the paper; the front matter does not.
    assert classes[:3] == [PageClass.PLAIN] * 3
    assert all(c == PageClass.TABLE for c in classes[3:9]), classes


def test_bert_prose_pages_stay_plain(fixtures_dir):
    classes = _classes(fixtures_dir / "two-column-paper.pdf")
    assert classes[0] == PageClass.PLAIN  # two-column, column-major stream: plain is enough
    assert classes[1] == PageClass.PLAIN
    assert PageClass.TABLE in classes[5:9]


def test_rowmajor_fixture_is_multi_column(fixtures_dir):
    path = fixtures_dir / "rowmajor-two-column.pdf"
    with pymupdf.open(path) as doc:
        tri = triage_page(doc[0], S)
    assert tri.page_class == PageClass.MULTI_COLUMN
    assert tri.signals.column_jump_rate > 0.5


def test_real_pages_have_low_jump_rate(fixtures_dir):
    with pymupdf.open(fixtures_dir / "two-column-paper.pdf") as doc:
        rates = [compute_signals(p).column_jump_rate for p in doc]
    assert max(rates) < 0.15, rates


def test_triage_is_cheap(fixtures_dir):
    with pymupdf.open(fixtures_dir / "two-column-paper.pdf") as doc:
        millis = [compute_signals(p).millis for p in doc]
    assert sum(millis) / len(millis) < 150, millis  # measured 10-20 ms/page; generous bound for CI


def _sig(**kw) -> PageSignals:
    base = dict(chars=3000, bad_ratio=0.0, image_cover=0.0, line_count=100, numeric_lines=0, column_jump_rate=0.0, rule_lines=0, millis=1.0)
    base.update(kw)
    return PageSignals(**base)


@pytest.mark.parametrize(
    "signals, expected",
    [
        (_sig(chars=0, image_cover=0.0, line_count=0), PageClass.BLANK),
        (_sig(chars=5, image_cover=0.01, line_count=1), PageClass.BLANK),
        (_sig(chars=0, image_cover=0.98, line_count=0), PageClass.SCAN),
        (_sig(chars=20, image_cover=0.5, line_count=2), PageClass.SCAN),
        (_sig(bad_ratio=0.3), PageClass.BROKEN_TEXT),
        (_sig(column_jump_rate=0.9), PageClass.MULTI_COLUMN),
        (_sig(numeric_lines=25), PageClass.TABLE),
        (_sig(rule_lines=4), PageClass.TABLE),
        (_sig(image_cover=0.4), PageClass.FIGURE),
        (_sig(), PageClass.PLAIN),
        (_sig(numeric_lines=3, image_cover=0.05, column_jump_rate=0.05), PageClass.PLAIN),
    ],
)
def test_classify_signals(signals, expected):
    assert classify_signals(signals, S) == expected


def test_scan_with_text_stamp_is_still_a_scan(pdf_builder):
    pdf = pdf_builder("S")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        tri = triage_page(doc[0], S)
    assert tri.page_class == PageClass.SCAN
    assert 30 <= tri.signals.chars < 300 and tri.signals.image_cover > 0.5


def test_thresholds_come_from_settings():
    strict = RouterSettings(table_min_numeric_lines=2)
    assert classify_signals(_sig(numeric_lines=3), strict) == PageClass.TABLE
    assert classify_signals(_sig(numeric_lines=3), S) == PageClass.PLAIN
