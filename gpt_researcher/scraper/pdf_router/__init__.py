"""Per-page PDF routing for the scraper.

``route_pdf(pdf_bytes, source_url)`` inspects every page, sends each to the
cheapest backend that can handle it (plain PyMuPDF -> PyMuPDF layout mode ->
Docling), gates the results and returns one Markdown string in page order.
See DECISIONS.md at the repository root for the design and its evidence.
"""

from __future__ import annotations

import threading

from .classify import PageClass, PageSignals, PageTriage, triage_page
from .router import (
    BackendExecutor,
    InlineExecutor,
    PageRouter,
    RoutedDocument,
    RoutedPage,
)
from .settings import RouterSettings

__all__ = [
    "BackendExecutor",
    "InlineExecutor",
    "PageClass",
    "PageRouter",
    "PageSignals",
    "PageTriage",
    "RoutedDocument",
    "RoutedPage",
    "RouterSettings",
    "get_router",
    "route_pdf",
    "route_pdf_detailed",
    "triage_page",
]

_router: PageRouter | None = None
_router_lock = threading.Lock()


def get_router(settings: RouterSettings | None = None) -> PageRouter:
    """Process-wide router built from the environment on first use. Sidecar
    pools are shared by every scraper thread so the worker counts act as
    global caps on ML work. Passing ``settings`` builds a separate router with
    those settings (tests, evaluation); it still shares the global pools unless
    ``settings.inline`` is set."""
    global _router
    if settings is not None:
        return _build_router(settings)
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = _build_router(RouterSettings.from_env())
    return _router


def _build_router(settings: RouterSettings) -> PageRouter:
    if settings.inline:
        return PageRouter(settings=settings)
    from .workers import get_pooled_executor

    return PageRouter(settings=settings, executor=get_pooled_executor(settings))


def route_pdf(pdf_bytes: bytes, source_url: str = "", settings: RouterSettings | None = None) -> str:
    return get_router(settings).route_text(pdf_bytes, source_url)


def route_pdf_detailed(pdf_bytes: bytes, source_url: str = "", settings: RouterSettings | None = None) -> RoutedDocument:
    return get_router(settings).route(pdf_bytes, source_url)
