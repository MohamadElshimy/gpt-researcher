from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "pdf_router"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


def _numeric_table_page(page: pymupdf.Page) -> None:
    page.insert_text((60, 60), "Table 1: Results on the benchmark", fontsize=10)
    y = 90
    for row in range(14):
        page.insert_text((60, y), f"Model {row}", fontsize=9)
        for col, x in enumerate((200, 260, 320, 380)):
            page.insert_text((x, y), f"{80 + row + col * 0.7:.1f}", fontsize=9)
        y += 14


def _prose_page(page: pymupdf.Page, seed: str) -> None:
    text = (
        f"{seed} language model pretraining has led to significant performance gains but careful "
        "comparison between different approaches is challenging. Training is computationally "
        "expensive, often done on private datasets of different sizes, and hyperparameter choices "
        "have significant impact on the final results. We present a replication study that carefully "
        "measures the impact of many key hyperparameters and training data size."
    )
    page.insert_textbox(pymupdf.Rect(50, 50, 545, 400), text, fontsize=10)


def _scan_page(page: pymupdf.Page, stamp: str | None = None) -> None:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 160), False)
    pix.clear_with(230)
    page.insert_image(page.rect, pixmap=pix)
    if stamp:
        page.insert_text((40, 820), stamp, fontsize=7)


def build_pdf(spec: str, title: str | None = None) -> bytes:
    """Build a PDF from a spec string: one char per page, p=prose, t=table,
    s=scan, S=scan with a download stamp (~100 chars of text), b=blank."""
    doc = pymupdf.open()
    for index, kind in enumerate(spec):
        page = doc.new_page(width=595, height=842)
        if kind == "p":
            _prose_page(page, f"Section {index}:")
        elif kind == "t":
            _numeric_table_page(page)
        elif kind == "s":
            _scan_page(page)
        elif kind == "S":
            _scan_page(page, "Downloaded from https://repository.example.edu on 2026-11-09. Authorized licensed use limited to: Example University.")
        elif kind == "b":
            pass
        else:
            raise ValueError(kind)
    if title is not None:
        doc.set_metadata({"title": title})
    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    return data


@pytest.fixture
def pdf_builder():
    return build_pdf
