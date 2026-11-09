"""PageRouter: triage every page, send each to the cheapest backend that can
handle it, gate the result, escalate one tier on failure, recombine in order.

Execution is delegated to a ``BackendExecutor`` so the same router runs with
backends in-process (tests, evaluation) or in sidecar worker processes
(production, see workers.py).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import pymupdf

from .backends import (
    BackendError,
    BackendRegistry,
    PageBackend,
    PageResult,
    contiguous_runs,
    default_registry,
)
from .classify import SEVERITY, PageClass, PageSignals, triage_page
from .gates import looks_empty, table_issues, word_recall
from .settings import RouterSettings

logger = logging.getLogger(__name__)


class BackendExecutor(Protocol):
    max_concurrency: int

    def run(
        self, backend_name: str, pdf_bytes: bytes, page_numbers: Sequence[int], timeout: float
    ) -> dict[int, PageResult]:
        """Run one backend task; raise BackendError on failure."""


class InlineExecutor:
    """Runs backends in the calling thread. PyMuPDF is not thread-safe, so
    inline execution is serialised (max_concurrency = 1)."""

    max_concurrency = 1

    def __init__(self, registry: BackendRegistry):
        self.registry = registry

    def run(self, backend_name, pdf_bytes, page_numbers, timeout):
        return self.registry.get(backend_name).parse(pdf_bytes, list(page_numbers))


@dataclass
class RoutedPage:
    page_no: int  # 0-based
    page_class: PageClass
    signals: PageSignals | None
    backend: str
    markdown: str
    confidence: float | None = None
    attempts: list[str] = field(default_factory=list)


@dataclass
class RoutedDocument:
    source_url: str
    title: str
    pages: list[RoutedPage]
    markdown: str
    elapsed: float
    backend_counts: dict[str, int]
    class_counts: dict[str, int]
    notes: list[str]


class PageRouter:
    MIN_TASK_SECONDS = 1.0  # below this much remaining budget, do not start new ML tasks

    def __init__(
        self,
        settings: RouterSettings | None = None,
        executor: BackendExecutor | None = None,
        registry: BackendRegistry | None = None,
    ):
        self.settings = settings or RouterSettings.from_env()
        self.registry = registry or default_registry()
        self.executor = executor or InlineExecutor(self.registry)
        self.ladder = self._build_ladder()
        self.base = self.ladder[0]

    # ------------------------------------------------------------------ setup
    def _build_ladder(self) -> list[PageBackend]:
        s = self.settings
        switches = {"layout": s.layout_tier, "docling": s.docling_tier}
        ladder: list[PageBackend] = []
        for backend in self.registry.all():
            if not switches.get(backend.name, True):
                logger.info("pdf_router: tier %r disabled by settings", backend.name)
                continue
            if not backend.available():
                logger.warning(
                    "pdf_router: tier %r is not installed; pages needing it fall to a cheaper tier "
                    "(install with: pip install 'gpt-researcher[pdf-rich]')",
                    backend.name,
                )
                continue
            ladder.append(backend)
        if not ladder:
            raise RuntimeError("pdf_router: no usable backend")
        return ladder

    def _pick(self, page_class: PageClass, above_tier: int) -> PageBackend | None:
        for backend in self.ladder:
            if backend.tier > above_tier and page_class in backend.handles:
                return backend
        return None

    def _timeout_for(self, backend: PageBackend) -> float:
        return self.settings.heavy_timeout if backend.tier >= 2 else self.settings.light_timeout

    # ------------------------------------------------------------------ main
    def route_text(self, pdf_bytes: bytes, source_url: str = "") -> str:
        return self.route(pdf_bytes, source_url).markdown

    def route(self, pdf_bytes: bytes, source_url: str = "") -> RoutedDocument:
        s = self.settings
        t0 = time.monotonic()
        deadline = t0 + s.doc_deadline
        notes: list[str] = []

        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass:
                return self._finish(source_url, "", {}, t0, ["encrypted PDF, no text extracted"])
            n = doc.page_count
            title = ((doc.metadata or {}).get("title") or "").strip()
            triage = {p: triage_page(doc[p], s) for p in range(min(n, s.max_triage_pages))}

        base_results = self.base.parse(pdf_bytes, range(n))
        pages: dict[int, RoutedPage] = {}
        for p in range(n):
            tri = triage.get(p)
            page = RoutedPage(
                page_no=p,
                page_class=tri.page_class if tri else PageClass.PLAIN,
                signals=tri.signals if tri else None,
                backend=self.base.name,
                markdown=base_results[p].markdown,
            )
            if tri is None:
                page.attempts.append("beyond triage page cap")
            pages[p] = page
        if n > s.max_triage_pages:
            notes.append(f"{n - s.max_triage_pages} pages beyond the triage cap kept on {self.base.name}")

        if not s.enabled:
            notes.append("router disabled")
            return self._finish(source_url, title, pages, t0, notes)
        if len(self.ladder) == 1:
            notes.append("no ML tiers available")

        # ---- plan: first backend per page, then apply the ML page budget
        targets: dict[int, PageBackend] = {}
        for p, page in pages.items():
            if page.page_class in (PageClass.PLAIN, PageClass.BLANK):
                continue
            backend = self._pick(page.page_class, above_tier=self.base.tier)
            if backend is None:
                page.attempts.append(f"no enabled backend handles {page.page_class.value}")
                continue
            targets[p] = backend
        if len(targets) > s.max_ml_pages:
            keep = sorted(targets, key=lambda p: (-SEVERITY[pages[p].page_class], p))[: s.max_ml_pages]
            dropped = sorted(set(targets) - set(keep))
            for p in dropped:
                pages[p].attempts.append("ML page budget exhausted")
                del targets[p]
            notes.append(f"ML page budget: {len(dropped)} pages kept on {self.base.name}")

        # ---- execute rounds: run, gate, escalate one tier, repeat
        pending = targets
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                for p in pending:
                    pages[p].attempts.append("document deadline reached before scheduling")
                notes.append(f"document deadline ({s.doc_deadline:.0f}s) reached")
                break
            results, failures = self._run_round(pdf_bytes, pending, deadline)
            next_round: dict[int, PageBackend] = {}

            for p, result in results.items():
                page = pages[p]
                backend = self.registry.get(result.backend)
                ok, reasons = self._gate(page, result, base_results[p].markdown)
                if ok:
                    self._accept(page, result)
                    continue
                page.attempts.append(f"{result.backend}: gate failed ({'; '.join(reasons)})")
                higher = self._pick(page.page_class, above_tier=backend.tier)
                if higher is not None:
                    next_round[p] = higher
                    self._accept_provisional(page, result, reasons)
                else:
                    self._accept_provisional(page, result, reasons)

            for p, (backend_name, reason) in failures.items():
                page = pages[p]
                page.attempts.append(f"{backend_name}: {reason}")
                backend = self.registry.get(backend_name)
                higher = self._pick(page.page_class, above_tier=backend.tier)
                if higher is not None and time.monotonic() < deadline:
                    next_round[p] = higher
            pending = next_round

        return self._finish(source_url, title, pages, t0, notes)

    # ------------------------------------------------------------------ rounds
    def _run_round(
        self, pdf_bytes: bytes, pending: dict[int, PageBackend], deadline: float
    ) -> tuple[dict[int, PageResult], dict[int, tuple[str, str]]]:
        by_backend: dict[str, list[int]] = {}
        for p, backend in pending.items():
            by_backend.setdefault(backend.name, []).append(p)

        tasks: list[tuple[PageBackend, list[int]]] = []
        for name, page_list in by_backend.items():
            backend = self.registry.get(name)
            batch = max(1, int(getattr(backend, "batch_pages", 1)))
            for run in contiguous_runs(page_list, max_len=batch):
                tasks.append((backend, run))

        results: dict[int, PageResult] = {}
        failures: dict[int, tuple[str, str]] = {}
        remaining = deadline - time.monotonic()
        if remaining < self.MIN_TASK_SECONDS:
            for backend, run in tasks:
                for p in run:
                    failures[p] = (backend.name, "document deadline reached")
            return results, failures
        workers = max(1, min(len(tasks), self.executor.max_concurrency))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pdf-router")
        try:
            # Clamp every task timeout to the document deadline so a sidecar
            # task cannot outlive the document it serves: when the deadline
            # passes, the worker is killed, not left running to its own limit.
            futures = {
                pool.submit(
                    self.executor.run, backend.name, pdf_bytes, run, min(self._timeout_for(backend), remaining)
                ): (backend, run)
                for backend, run in tasks
            }
            done, not_done = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
            for fut in done:
                backend, run = futures[fut]
                try:
                    out = fut.result()
                except BackendError as exc:
                    for p in run:
                        failures[p] = (backend.name, f"{type(exc).__name__}: {exc}")
                    continue
                except Exception as exc:  # a backend bug must never take the document down
                    logger.exception("pdf_router: backend %s raised", backend.name)
                    for p in run:
                        failures[p] = (backend.name, f"unexpected {type(exc).__name__}: {exc}")
                    continue
                for p in run:
                    if p in out:
                        results[p] = out[p]
                    else:
                        failures[p] = (backend.name, "no result for page")
            for fut in not_done:
                backend, run = futures[fut]
                for p in run:
                    failures[p] = (backend.name, "document deadline reached")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return results, failures

    # ------------------------------------------------------------------ gates
    def _gate(self, page: RoutedPage, result: PageResult, base_text: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if looks_empty(result.markdown):
            if page.page_class in (PageClass.SCAN, PageClass.BROKEN_TEXT) or not looks_empty(base_text):
                reasons.append("empty output")
            return (not reasons, reasons)
        if not looks_empty(base_text) and page.page_class != PageClass.BROKEN_TEXT:
            recall = word_recall(base_text, result.markdown)
            if recall < self.settings.min_word_recall:
                reasons.append(f"word recall {recall:.2f} < {self.settings.min_word_recall:.2f}")
        for issue in table_issues(result.markdown):
            reasons.append(issue)
        return (not reasons, reasons)

    @staticmethod
    def _accept(page: RoutedPage, result: PageResult) -> None:
        page.backend = result.backend
        page.markdown = result.markdown
        page.confidence = result.confidence
        page.attempts.append(f"{result.backend}: ok")

    def _accept_provisional(self, page: RoutedPage, result: PageResult, reasons: list[str]) -> None:
        """Keep a gate-failing result only when its prose is complete: structure
        with a slightly broken table beats flattened cells, but dropped text
        never beats the raw text layer."""
        text_problem = any(r.startswith("word recall") or r == "empty output" for r in reasons)
        if text_problem:
            page.attempts.append(f"{result.backend}: kept {page.backend} output instead")
            return
        page.backend = result.backend
        page.markdown = result.markdown
        page.confidence = result.confidence
        page.attempts.append(f"{result.backend}: accepted with table issues")

    # ------------------------------------------------------------------ output
    def _finish(self, source_url, title, pages: dict[int, RoutedPage], t0, notes) -> RoutedDocument:
        ordered = [pages[p] for p in sorted(pages)]
        parts: list[str] = []
        for page in ordered:
            if page.page_class == PageClass.BLANK:
                continue
            if looks_empty(page.markdown):
                # A page we know carries text (scan, broken layer) or tried to
                # parse must not vanish silently: leave a marker downstream can see.
                if page.attempts or page.page_class in (PageClass.SCAN, PageClass.BROKEN_TEXT):
                    reason = page.attempts[-1] if page.attempts else f"{page.page_class.value} page, no OCR tier"
                    parts.append(f"<!-- page {page.page_no + 1}: no text extracted ({reason}) -->")
                continue
            parts.append(page.markdown)
        markdown = "\n\n".join(parts).strip()
        if not title and ordered:
            # No metadata title: take a top-level heading from the first page only,
            # never from a later page (a section header is not the document title).
            for line in ordered[0].markdown.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip().strip("*").strip()
                    break
        backend_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        for page in ordered:
            backend_counts[page.backend] = backend_counts.get(page.backend, 0) + 1
            class_counts[page.page_class.value] = class_counts.get(page.page_class.value, 0) + 1
        elapsed = time.monotonic() - t0
        logger.info(
            "pdf_router: %s pages=%d elapsed=%.1fs backends=%s classes=%s%s",
            source_url or "<bytes>",
            len(ordered),
            elapsed,
            backend_counts,
            class_counts,
            f" notes={notes}" if notes else "",
        )
        return RoutedDocument(
            source_url=source_url,
            title=title,
            pages=ordered,
            markdown=markdown,
            elapsed=elapsed,
            backend_counts=backend_counts,
            class_counts=class_counts,
            notes=notes,
        )
