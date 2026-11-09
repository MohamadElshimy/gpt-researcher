"""Backend registry: name -> backend instance, built lazily per process."""

from __future__ import annotations

from collections.abc import Iterable

from .base import (
    BackendCrashed,
    BackendError,
    BackendTimeout,
    BackendUnavailable,
    PageBackend,
    PageResult,
    contiguous_runs,
)
from .docling_backend import DoclingBackend
from .layout import PyMuPDFLayoutBackend
from .plain import PlainPyMuPDFBackend

__all__ = [
    "BackendCrashed",
    "BackendError",
    "BackendRegistry",
    "BackendTimeout",
    "BackendUnavailable",
    "DEFAULT_BACKENDS",
    "PageBackend",
    "PageResult",
    "contiguous_runs",
    "default_registry",
]

DEFAULT_BACKENDS: tuple[type, ...] = (PlainPyMuPDFBackend, PyMuPDFLayoutBackend, DoclingBackend)


class BackendRegistry:
    def __init__(self, backends: Iterable[PageBackend | type]):
        self._backends: dict[str, PageBackend] = {}
        for b in backends:
            instance = b() if isinstance(b, type) else b
            self._backends[instance.name] = instance

    def get(self, name: str) -> PageBackend:
        try:
            return self._backends[name]
        except KeyError:
            raise BackendUnavailable(f"unknown backend {name!r}") from None

    def all(self) -> list[PageBackend]:
        return sorted(self._backends.values(), key=lambda b: b.tier)

    def names(self) -> list[str]:
        return [b.name for b in self.all()]


_default: BackendRegistry | None = None


def default_registry() -> BackendRegistry:
    global _default
    if _default is None:
        _default = BackendRegistry(DEFAULT_BACKENDS)
    return _default
