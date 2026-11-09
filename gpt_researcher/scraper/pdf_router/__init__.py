"""Per-page PDF routing for the scraper.

``route_pdf(pdf_bytes, source_url)`` inspects every page, sends each to the
cheapest backend that can handle it (plain PyMuPDF -> PyMuPDF layout mode ->
Docling), gates the results and returns one Markdown string in page order.
See DECISIONS.md at the repository root for the design and its evidence.
"""

from __future__ import annotations

import threading

from .classify import PageClass, PageSignals, PageTriage, triage_page
from .router import BackendExecutor, InlineExecutor, PageRouter, RoutedDocument, RoutedPage
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
    """Process-wide router. Sidecar pools are shared by every scraper thread so
    the worker counts act as global caps on ML work."""
    global _router
    if _router is None or settings is not None:
        with _router_lock:
            if _router is None or settings is not None:
                settings = settings or RouterSettings.from_env()
                if settings.inline:
                    router = PageRouter(settings=settings)
                else:
                    from .workers import get_pooled_executor

                    router = PageRouter(settings=settings, executor=get_pooled_executor(settings))
                if _router is None:
                    _router = router
                return router
    return _router


def route_pdf(pdf_bytes: bytes, source_url: str = "", settings: RouterSettings | None = None) -> str:
    return get_router(settings).route_text(pdf_bytes, source_url)


def route_pdf_detailed(pdf_bytes: bytes, source_url: str = "", settings: RouterSettings | None = None) -> RoutedDocument:
    return get_router(settings).route(pdf_bytes, source_url)
