"""Backend contract.

A backend is a pure function of ``(pdf_bytes, page_numbers)`` returning
Markdown per page. Arguments and results are plain picklable data, so the
same class runs unchanged inside the calling thread or inside a sidecar
worker process. Adding a parser means one class plus one registry entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

from ..classify import PageClass


@dataclass(frozen=True)
class PageResult:
    page_no: int  # 0-based
    markdown: str
    backend: str
    confidence: float | None = None  # backend-reported quality, 0..1, when available
    notes: tuple[str, ...] = ()


class BackendError(Exception):
    """A backend could not produce a result for the requested pages."""


class BackendUnavailable(BackendError):
    """The backend is disabled, not installed, or its circuit breaker is open."""


class BackendTimeout(BackendError):
    """The backend exceeded its per-task timeout and was stopped."""


class BackendCrashed(BackendError):
    """The backend's worker died (segfault, OOM kill, abnormal exit)."""


@runtime_checkable
class PageBackend(Protocol):
    name: str
    tier: int  # cost rank; the router reaches for the lowest tier that handles a page class
    handles: frozenset[PageClass]

    def available(self) -> bool:
        """Cheap check (no heavy imports) that the backend can run in this environment."""

    def parse(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> dict[int, PageResult]:
        """Parse the given 0-based pages. May raise BackendError."""


def contiguous_runs(page_numbers: Sequence[int], max_len: int | None = None) -> list[list[int]]:
    """Group sorted page numbers into contiguous runs, optionally capped in length."""
    runs: list[list[int]] = []
    for p in sorted(set(page_numbers)):
        if runs and p == runs[-1][-1] + 1 and (max_len is None or len(runs[-1]) < max_len):
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs
