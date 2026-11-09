"""Sidecar worker pools for the ML tiers.

Why processes and not threads:
- PyMuPDF documents itself as not thread-safe and never releases the GIL
  ("use multiprocessing"); its own batch API is a process pool.
- Docling's failure reports are native (glibc heap corruption, C++
  ``bad_alloc``, SIGABRT); an in-process thread cannot survive or stop them,
  and Python threads cannot be killed at all.
- Model memory (~1.8 GB for Docling) is only reliably returned on process exit.

Why ``spawn``: PyTorch's OpenMP runtime deadlocks after ``fork`` and Python
3.12+ warns about forking a threaded process (the scraper already runs 15
threads). macOS defaults to spawn; Linux defaulted to fork until 3.14, so the
context is passed explicitly.

Each worker is one process with one duplex pipe. The parent polls with a
per-task timeout and SIGKILLs the worker when it expires; the next task spawns
a fresh one (~7 s with Docling, ~1 s with the layout model). A per-backend
circuit breaker stops a failing backend from being retried on every page.
"""

from __future__ import annotations

import atexit
import importlib
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import Iterable, Sequence

from .backends import (
    BackendCrashed,
    BackendError,
    BackendRegistry,
    BackendTimeout,
    BackendUnavailable,
    PageResult,
    default_registry,
)
from .settings import RouterSettings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- child
DEFAULT_REGISTRY_FACTORY = "gpt_researcher.scraper.pdf_router.backends:default_registry"


def _load_registry(factory: str) -> BackendRegistry:
    module_name, _, attr = factory.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr or "default_registry")()


def _worker_main(conn, backend_names: tuple[str, ...], registry_factory: str) -> None:  # pragma: no cover - child process
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent decides when we die
    try:
        registry = _load_registry(registry_factory)
    except Exception as exc:
        conn.send(("startup-error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", os.getpid()))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message is None:
            return
        backend_name, pdf_bytes, page_numbers = message
        try:
            if backend_name not in backend_names:
                raise BackendUnavailable(f"backend {backend_name!r} not served by this worker")
            result = registry.get(backend_name).parse(pdf_bytes, list(page_numbers))
            conn.send(("ok", result))
        except Exception as exc:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))


class WorkerStartupError(BackendCrashed):
    """The sidecar process could not be started or died before saying hello.

    ``deterministic`` is True when retrying cannot help: the child reported
    that the backend registry does not import, or the interpreter's
    ``__main__`` has no importable file (stdin, some notebooks) so every
    ``spawn`` will fail the same way. Everything else (resource exhaustion,
    a slow host) is treated as transient."""

    def __init__(self, message: str, deterministic: bool = False):
        super().__init__(message)
        self.deterministic = deterministic


def spawn_possible() -> bool:
    """False when multiprocessing's spawn bootstrap would fail to re-import
    ``__main__`` in every child (``python -`` sets ``__file__`` to ``<stdin>``)."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path:
        return True
    return os.path.exists(path)


class _Worker:
    STARTUP_TIMEOUT = 60.0  # spawn + interpreter start + pymupdf import; generous for slow hosts

    def __init__(self, ctx, pool_name: str, backend_names: tuple[str, ...], index: int, registry_factory: str):
        self.conn, child_conn = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(
            target=_worker_main,
            args=(child_conn, backend_names, registry_factory),
            name=f"{pool_name}-{index}",
            daemon=True,
        )
        self.proc.start()
        child_conn.close()
        self.tasks = 0
        self.last_used = time.monotonic()
        self._handshake()

    def _handshake(self) -> None:
        try:
            if not self.conn.poll(self.STARTUP_TIMEOUT):
                raise WorkerStartupError(f"{self.proc.name}: no handshake within {self.STARTUP_TIMEOUT:.0f}s")
            status, detail = self.conn.recv()
        except (EOFError, OSError) as exc:
            code = self.proc.exitcode
            self.kill()
            raise WorkerStartupError(
                f"{self.proc.name}: died during startup (exit code {code})",
                deterministic=not spawn_possible(),
            ) from exc
        except WorkerStartupError:
            self.kill()
            raise
        if status != "ready":
            self.kill()
            # The child itself reported that the registry does not import: no retry will fix it.
            raise WorkerStartupError(f"{self.proc.name}: {status} {detail}", deterministic=(status == "startup-error"))

    def alive(self) -> bool:
        return self.proc.is_alive()

    def run(self, backend_name: str, pdf_bytes: bytes, page_numbers: Sequence[int], timeout: float) -> dict[int, PageResult]:
        try:
            self.conn.send((backend_name, pdf_bytes, list(page_numbers)))
        except (OSError, ValueError) as exc:
            self.kill()
            raise BackendCrashed(f"{backend_name}: worker pipe closed ({exc})") from exc
        if not self.conn.poll(timeout):
            self.kill()
            raise BackendTimeout(f"{backend_name}: no result within {timeout:.0f}s, worker killed")
        try:
            status, payload = self.conn.recv()
        except (EOFError, OSError) as exc:
            code = self.proc.exitcode
            self.kill()
            raise BackendCrashed(f"{backend_name}: worker died (exit code {code})") from exc
        self.tasks += 1
        self.last_used = time.monotonic()
        if status == "ok":
            return payload
        raise BackendError(payload)

    def kill(self) -> None:
        try:
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join(2.0)
        except Exception:  # pragma: no cover - best effort
            pass
        try:
            self.conn.close()
        except Exception:  # pragma: no cover
            pass

    def stop(self) -> None:
        try:
            self.conn.send(None)
            self.proc.join(1.0)
        except Exception:
            pass
        self.kill()


# --------------------------------------------------------------------------- breaker
class CircuitBreaker:
    """Closed -> open after N consecutive timeouts/crashes -> half-open probe after reset."""

    def __init__(self, failures: int, reset_seconds: float):
        self.threshold = max(1, failures)
        self.reset_seconds = reset_seconds
        self.consecutive = 0
        self.opened_at: float | None = None
        self._probing = False
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            if time.monotonic() - self.opened_at >= self.reset_seconds and not self._probing:
                self._probing = True  # one half-open probe
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self.consecutive = 0
            self.opened_at = None
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive += 1
            self._probing = False
            if self.consecutive >= self.threshold:
                self.opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self.opened_at is not None


# --------------------------------------------------------------------------- pool
class MLWorkerPool:
    def __init__(
        self,
        name: str,
        backend_names: Iterable[str],
        size: int,
        settings: RouterSettings,
        start_method: str = "spawn",
        registry_factory: str = DEFAULT_REGISTRY_FACTORY,
    ):
        self.name = name
        self.backend_names = tuple(backend_names)
        self.registry_factory = registry_factory
        self.size = max(1, size)
        self.settings = settings
        self._ctx = mp.get_context(start_method)
        self._idle: list[_Worker] = []
        self._live = 0
        self._spawned = 0
        self._cond = threading.Condition()
        self._breakers = {b: CircuitBreaker(settings.breaker_failures, settings.breaker_reset) for b in self.backend_names}
        self._reaper: threading.Thread | None = None
        self._closed = False

    # -- public
    def run(self, backend_name: str, pdf_bytes: bytes, page_numbers: Sequence[int], timeout: float) -> dict[int, PageResult]:
        if backend_name not in self._breakers:
            raise BackendUnavailable(f"{backend_name}: not served by pool {self.name}")
        breaker = self._breakers[backend_name]
        if not breaker.allow():
            raise BackendUnavailable(f"{backend_name}: circuit breaker open after repeated failures")
        worker = self._acquire(timeout)  # raises WorkerStartupError if the sidecar cannot start
        try:
            result = worker.run(backend_name, pdf_bytes, page_numbers, timeout)
        except (BackendTimeout, BackendCrashed):
            breaker.record_failure()
            raise
        except BackendError:
            breaker.record_success()  # the backend answered; a bad PDF is not a health problem
            raise
        finally:
            self._release(worker)
        breaker.record_success()
        return result

    def breaker(self, backend_name: str) -> CircuitBreaker:
        return self._breakers[backend_name]

    def shutdown(self) -> None:
        with self._cond:
            self._closed = True
            workers = list(self._idle)
            self._idle.clear()
            self._live = 0
            self._cond.notify_all()
        for w in workers:
            w.stop()

    @property
    def live_workers(self) -> int:
        with self._cond:
            return self._live

    # -- internals
    def _acquire(self, timeout: float) -> _Worker:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                if self._closed:
                    raise BackendUnavailable(f"pool {self.name} is shut down")
                while self._idle:
                    worker = self._idle.pop()
                    if worker.alive():
                        return worker
                    self._live -= 1  # died while idle
                if self._live < self.size:
                    self._live += 1
                    self._spawned += 1
                    index = self._spawned
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendUnavailable(f"pool {self.name}: all {self.size} workers busy for {timeout:.0f}s")
                self._cond.wait(remaining)
        try:
            worker = _Worker(self._ctx, self.name, self.backend_names, index, self.registry_factory)
        except WorkerStartupError:
            with self._cond:
                self._live -= 1
                self._cond.notify()
            raise
        except Exception as exc:
            with self._cond:
                self._live -= 1
                self._cond.notify()
            raise WorkerStartupError(f"pool {self.name}: could not start worker ({exc})") from exc
        self._ensure_reaper()
        return worker

    def _release(self, worker: _Worker) -> None:
        recycle = (not worker.alive()) or worker.tasks >= self.settings.worker_max_tasks
        with self._cond:
            if recycle or self._closed:
                self._live -= 1
            else:
                self._idle.append(worker)
            self._cond.notify()
        if recycle or self._closed:
            worker.kill()

    def _ensure_reaper(self) -> None:
        if self._reaper is not None or self.settings.worker_idle_secs <= 0:
            return
        self._reaper = threading.Thread(target=self._reap_loop, name=f"{self.name}-reaper", daemon=True)
        self._reaper.start()

    def _reap_loop(self) -> None:  # pragma: no cover - timing dependent
        interval = max(5.0, min(30.0, self.settings.worker_idle_secs / 4))
        while not self._closed:
            time.sleep(interval)
            cutoff = time.monotonic() - self.settings.worker_idle_secs
            stale: list[_Worker] = []
            with self._cond:
                keep = []
                for w in self._idle:
                    (stale if w.last_used < cutoff else keep).append(w)
                self._idle = keep
                self._live -= len(stale)
            for w in stale:
                logger.info("pdf_router: recycling idle worker %s", w.proc.name)
                w.stop()


# --------------------------------------------------------------------------- executor
class PooledExecutor:
    """Routes layout-tier tasks to a light pool and Docling tasks to a heavy pool.

    Fail-closed when sidecars cannot start: the heavy tier (Docling, untrusted
    PDFs through torch/ONNX) is reported unavailable so those pages fall back
    to plain text; the light tier (pymupdf, the same library already running
    in the scraper threads) runs in-process, serialised. A deterministic
    startup failure (``__main__`` not importable, registry does not import)
    latches that mode; a transient one is retried after a growing cool-down
    (15 s doubling to 5 min). ``PDF_ROUTER_INLINE=1`` is the explicit opt-in
    for running every tier in-process."""

    COOLDOWN_BASE = 15.0
    COOLDOWN_MAX = 300.0

    def __init__(
        self,
        settings: RouterSettings,
        registry: BackendRegistry | None = None,
        registry_factory: str = DEFAULT_REGISTRY_FACTORY,
    ):
        registry = registry or default_registry()
        self._registry = registry
        light = [b.name for b in registry.all() if 0 < b.tier < 2]
        heavy = [b.name for b in registry.all() if b.tier >= 2]
        self._pools: dict[str, MLWorkerPool] = {}
        self.pools: list[MLWorkerPool] = []
        if light:
            pool = MLWorkerPool("pdf-light", light, settings.light_workers, settings, registry_factory=registry_factory)
            self.pools.append(pool)
            self._pools.update({name: pool for name in light})
        if heavy:
            pool = MLWorkerPool("pdf-heavy", heavy, settings.heavy_workers, settings, registry_factory=registry_factory)
            self.pools.append(pool)
            self._pools.update({name: pool for name in heavy})
        self.max_concurrency = sum(p.size for p in self.pools) or 1
        self.degraded = False  # deterministic: sidecars can never start in this process
        self.retry_after = 0.0  # transient: monotonic time before spawning is attempted again
        self._startup_failures = 0
        self._inline_lock = threading.Lock()

    def run(self, backend_name, pdf_bytes, page_numbers, timeout):
        pool = self._pools.get(backend_name)
        if pool is None:
            raise BackendUnavailable(f"{backend_name}: no worker pool serves this backend")
        if self.degraded or time.monotonic() < self.retry_after:
            return self._run_degraded(backend_name, pdf_bytes, page_numbers)
        try:
            result = pool.run(backend_name, pdf_bytes, page_numbers, timeout)
        except WorkerStartupError as exc:
            self._note_startup_failure(exc)
            return self._run_degraded(backend_name, pdf_bytes, page_numbers)
        self._startup_failures = 0
        return result

    def _note_startup_failure(self, exc: WorkerStartupError) -> None:
        self._startup_failures += 1
        if exc.deterministic:
            if not self.degraded:
                self.degraded = True
                self.max_concurrency = 1
                logger.warning(
                    "pdf_router: sidecar workers cannot start in this process (%s); light tier runs "
                    "in-process, heavy tier disabled. Set PDF_ROUTER_INLINE=1 to run every tier in-process.",
                    exc,
                )
            return
        cooldown = min(self.COOLDOWN_MAX, self.COOLDOWN_BASE * 2 ** min(self._startup_failures - 1, 6))
        self.retry_after = time.monotonic() + cooldown
        logger.warning(
            "pdf_router: sidecar worker failed to start (%s); retrying in %.0fs, heavy tier unavailable meanwhile",
            exc,
            cooldown,
        )

    def _run_degraded(self, backend_name, pdf_bytes, page_numbers):
        backend = self._registry.get(backend_name)
        if backend.tier >= 2:
            raise BackendUnavailable(
                f"{backend_name}: sidecar workers unavailable and heavy tiers never run in-process "
                "automatically (set PDF_ROUTER_INLINE=1 to allow it)"
            )
        with self._inline_lock:  # PyMuPDF is not thread-safe
            return backend.parse(pdf_bytes, list(page_numbers))

    def shutdown(self) -> None:
        for pool in self.pools:
            pool.shutdown()


_executor: PooledExecutor | None = None
_executor_lock = threading.Lock()


def get_pooled_executor(settings: RouterSettings, registry: BackendRegistry | None = None) -> PooledExecutor:
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                _executor = PooledExecutor(settings, registry)
                atexit.register(_executor.shutdown)
    return _executor


def worker_count() -> int:
    """Live sidecar processes (for diagnostics)."""
    if _executor is None:
        return 0
    return sum(p.live_workers for p in _executor.pools)


