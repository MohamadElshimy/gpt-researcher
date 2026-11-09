"""Controllable backends for router and worker-pool tests.

The worker pool imports this module inside spawned child processes through the
``registry_factory`` hook, so everything here must be importable without the
test session's fixtures.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Sequence

from gpt_researcher.scraper.pdf_router.backends import BackendRegistry, PageResult
from gpt_researcher.scraper.pdf_router.backends.plain import PlainPyMuPDFBackend
from gpt_researcher.scraper.pdf_router.classify import PageClass

ALL_CLASSES = frozenset(PageClass)


class FakeBackend:
    """Returns ``outputs[page]`` (string or callable) or raises ``error``."""

    def __init__(
        self,
        name: str,
        tier: int,
        handles=ALL_CLASSES,
        outputs: dict[int, str | Callable[[int], str]] | None = None,
        default: str | None = None,
        error: Exception | None = None,
        available: bool = True,
        delay: float = 0.0,
        batch_pages: int = 1,
    ):
        self.name = name
        self.tier = tier
        self.handles = frozenset(handles)
        self.outputs = outputs or {}
        self.default = default
        self.error = error
        self._available = available
        self.delay = delay
        self.batch_pages = batch_pages
        self.calls: list[list[int]] = []

    def available(self) -> bool:
        return self._available

    def parse(self, pdf_bytes: bytes, page_numbers: Sequence[int]) -> dict[int, PageResult]:
        self.calls.append(list(page_numbers))
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        results = {}
        for p in page_numbers:
            out = self.outputs.get(p, self.default)
            if out is None:
                continue
            text = out(p) if callable(out) else out
            results[p] = PageResult(page_no=p, markdown=text, backend=self.name, confidence=0.5)
        return results


# ---- backends for the spawned-worker tests --------------------------------
class EchoBackend:
    name = "echo"
    tier = 1
    handles = ALL_CLASSES

    def available(self):
        return True

    def parse(self, pdf_bytes, page_numbers):
        return {p: PageResult(p, f"echo page {p} pid {os.getpid()}", self.name) for p in page_numbers}


class SleepBackend:
    """Sleeps for ``page_numbers[0]`` seconds, then echoes. Used to trigger timeouts."""

    name = "sleep"
    tier = 1
    handles = ALL_CLASSES

    def available(self):
        return True

    def parse(self, pdf_bytes, page_numbers):
        time.sleep(float(page_numbers[0]))
        return {p: PageResult(p, f"slept {p}", self.name) for p in page_numbers}


class CrashBackend:
    name = "crash"
    tier = 2
    handles = ALL_CLASSES

    def available(self):
        return True

    def parse(self, pdf_bytes, page_numbers):
        os._exit(3)  # simulate a native crash: no exception, the process just dies


class RaisingBackend:
    name = "raising"
    tier = 2
    handles = ALL_CLASSES

    def available(self):
        return True

    def parse(self, pdf_bytes, page_numbers):
        raise ValueError("bad pdf for test")


def registry() -> BackendRegistry:
    return BackendRegistry([PlainPyMuPDFBackend(), EchoBackend(), SleepBackend(), CrashBackend(), RaisingBackend()])


def broken_registry() -> BackendRegistry:
    raise RuntimeError("registry factory exploded on purpose")


def dying_registry() -> BackendRegistry:
    os._exit(7)  # the child dies before the handshake, like an OOM kill during model import
