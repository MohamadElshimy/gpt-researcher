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
context is passed explicitly. Spawned children re-import the parent's
``__main__`` module: run the server through a console script (``uvicorn
main:app``, as the Dockerfile does) rather than ``python main.py`` so that
module-level setup does not run again in every worker.

Each worker is one process with one duplex pipe. The parent sends a task under
a watchdog (a child that stopped reading cannot block the caller), polls for
the reply with the remaining budget, and SIGKILLs the worker when it expires;
the next task spawns a fresh one (~7 s with Docling, ~1 s with the layout
model). A per-backend circuit breaker stops a failing backend from being
retried on every page.

Fail-closed when sidecars cannot start: the heavy tier (Docling, untrusted PDFs
through torch/ONNX) is reported unavailable so those pages fall back to plain
text; the light tier (pymupdf plus its bundled ONNX layout model, no torch)
runs in-process, serialised. A deterministic startup failure latches that
mode for the pool; a transient one is retried after a cool-down that doubles
from 15 s to 5 min. ``PDF_ROUTER_INLINE=1`` is the explicit opt-in for running
every tier in-process.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from collections.abc import Iterable, Sequence

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

DEFAULT_REGISTRY_FACTORY = "gpt_researcher.scraper.pdf_router.backends:default_registry"

# Exceptions a child may raise that keep their type when crossing the pipe.
_ERROR_TYPES: dict[str, type[BackendError]] = {
    "BackendTimeout": BackendTimeout,
    "BackendCrashed": BackendCrashed,
    "BackendUnavailable": BackendUnavailable,
}


def _load_registry(factory: str) -> BackendRegistry:
    module_name, _, attr = factory.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr or "default_registry")()


# --------------------------------------------------------------------------- child
def _worker_main(conn, backend_names: tuple[str, ...], registry_factory: str) -> None:  # pragma: no cover - child process
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent decides when we die
    try:
        registry = _load_registry(registry_factory)
    except Exception as exc:
        conn.send(("startup-error", type(exc).__name__, f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", os.getpid(), None))
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
            conn.send(("ok", result, None))
        except MemoryError:
            # Report, then exit: a process that just ran out of memory is not
            # trusted with another task; the parent recycles it.
            with contextlib.suppress(Exception):
                conn.send(("error", "MemoryError", f"{backend_name}: worker ran out of memory"))
            return
        except Exception as exc:
            conn.send(("error", type(exc).__name__, f"{type(exc).__name__}: {exc}"))


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
        self.dead = False
        self._handshake()

    def _handshake(self) -> None:
        try:
            if not self.conn.poll(self.STARTUP_TIMEOUT):
                raise WorkerStartupError(f"{self.proc.name}: no handshake within {self.STARTUP_TIMEOUT:.0f}s")
            status, payload, detail = self.conn.recv()
        except (EOFError, OSError) as exc:
            self.proc.join(1.0)
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
        return not self.dead and self.proc.is_alive()

    def run(self, backend_name: str, pdf_bytes: bytes, page_numbers: Sequence[int], timeout: float) -> dict[int, PageResult]:
        deadline = time.monotonic() + max(0.0, timeout)
        # The send can block once the payload exceeds the pipe buffer if the
        # child stopped reading (wedged in native code, SIGSTOP, cgroup freeze);
        # a watchdog kills the child so the caller is released.
        send_timed_out = threading.Event()

        def _watchdog() -> None:
            send_timed_out.set()
            self.kill()

        timer = threading.Timer(max(0.1, timeout), _watchdog)
        timer.daemon = True
        timer.start()
        try:
            self.conn.send((backend_name, pdf_bytes, list(page_numbers)))
        except (OSError, ValueError) as exc:
            self.kill()
            if send_timed_out.is_set():
                raise BackendTimeout(f"{backend_name}: worker stopped reading; killed after {timeout:.0f}s") from exc
            raise BackendCrashed(f"{backend_name}: worker pipe closed ({exc})") from exc
        finally:
            timer.cancel()
        if send_timed_out.is_set():
            raise BackendTimeout(f"{backend_name}: worker stopped reading; killed after {timeout:.0f}s")

        if not self.conn.poll(max(0.0, deadline - time.monotonic())):
            self.kill()
            raise BackendTimeout(f"{backend_name}: no result within {timeout:.0f}s, worker killed")
        try:
            status, payload, detail = self.conn.recv()
        except (EOFError, OSError) as exc:
            self.proc.join(1.0)  # reap so the exit code (or -signal) is available for the message
            code = self.proc.exitcode
            self.kill()
            raise BackendCrashed(f"{backend_name}: worker died (exit code {code})") from exc
        self.tasks += 1
        self.last_used = time.monotonic()
        if status == "ok":
            return payload
        if status == "error":
            if payload == "MemoryError":
                self.kill()
                raise BackendCrashed(detail)
            raise _ERROR_TYPES.get(payload, BackendError)(detail)
        self.kill()
        raise BackendCrashed(f"{backend_name}: unexpected message {status!r} from worker")

    def kill(self) -> None:
        self.dead = True
        try:
            if self.proc.is_alive():
                self.proc.kill()
            self.proc.join(2.0)
        except Exception:  # pragma: no cover - best effort
            pass
        with contextlib.suppress(Exception):
            self.conn.close()

    def stop(self) -> None:
        # Ask nicely first: a worker that exits on its own releases its native
        # resources (torch/ONNX teardown takes a moment); kill only if it lingers.
        with contextlib.suppress(Exception):
            self.conn.send(None)
            self.proc.join(3.0)
        self.kill()


# --------------------------------------------------------------------------- breaker
class CircuitBreaker:
    """Closed -> open after N consecutive timeouts/crashes -> one half-open
    probe after ``reset_seconds``. The probe is a lease: a caller that cannot
    use it (no worker could be acquired) must ``cancel_probe()``, and a probe
    nobody reports on expires after another ``reset_seconds`` so the breaker
    can never get stuck half-open."""

    def __init__(self, failures: int, reset_seconds: float):
        self.threshold = max(1, failures)
        self.reset_seconds = reset_seconds
        self.consecutive = 0
        self.opened_at: float | None = None
        self._probe_started: float | None = None
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            if self.opened_at is None:
                return True
            now = time.monotonic()
            if now - self.opened_at < self.reset_seconds:
                return False
            if self._probe_started is None or now - self._probe_started >= self.reset_seconds:
                self._probe_started = now
                return True
            return False

    def cancel_probe(self) -> None:
        with self._lock:
            self._probe_started = None

    def record_success(self) -> None:
        with self._lock:
            self.consecutive = 0
            self.opened_at = None
            self._probe_started = None

    def record_failure(self) -> None:
        with self._lock:
            self.consecutive += 1
            self._probe_started = None
            if self.consecutive >= self.threshold:
                self.opened_at = time.monotonic()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self.opened_at is not None


# --------------------------------------------------------------------------- pool
class MLWorkerPool:
    COOLDOWN_BASE = 15.0
    COOLDOWN_MAX = 300.0

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
        self.size = max(1, size)
        self.settings = settings
        self.registry_factory = registry_factory
        self._ctx = mp.get_context(start_method)
        self._idle: list[_Worker] = []
        self._busy: set[_Worker] = set()
        self._live = 0
        self._spawned = 0
        self._cond = threading.Condition()
        self._breakers = {b: CircuitBreaker(settings.breaker_failures, settings.breaker_reset) for b in self.backend_names}
        self._reaper: threading.Thread | None = None
        self._closed = False
        # Startup health of this pool's sidecars (see module docstring).
        self.degraded = False
        self.retry_after = 0.0
        self._startup_failures = 0

    # -- public
    def run(self, backend_name: str, pdf_bytes: bytes, page_numbers: Sequence[int], timeout: float) -> dict[int, PageResult]:
        """Run one task within ``timeout`` seconds counted from now, including
        the wait for a free worker. Raises BackendError subclasses."""
        if backend_name not in self._breakers:
            raise BackendUnavailable(f"{backend_name}: not served by pool {self.name}")
        breaker = self._breakers[backend_name]
        if not breaker.allow():
            raise BackendUnavailable(f"{backend_name}: circuit breaker open after repeated failures")
        deadline = time.monotonic() + max(0.0, timeout)
        try:
            worker = self._acquire(deadline)
        except WorkerStartupError as exc:
            breaker.cancel_probe()
            self._note_startup_failure(exc)
            raise
        except BackendError:
            breaker.cancel_probe()
            raise
        self._startup_failures = 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._release(worker)
            breaker.cancel_probe()
            raise BackendTimeout(f"{backend_name}: budget spent waiting for a worker")
        try:
            result = worker.run(backend_name, pdf_bytes, page_numbers, remaining)
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

    def in_cooldown(self) -> bool:
        return self.degraded or time.monotonic() < self.retry_after

    def breaker(self, backend_name: str) -> CircuitBreaker:
        return self._breakers[backend_name]

    def shutdown(self) -> None:
        with self._cond:
            self._closed = True
            idle = list(self._idle)
            busy = list(self._busy)
            self._idle.clear()
            self._busy.clear()
            self._live = 0
            self._cond.notify_all()
        for w in idle:
            w.stop()
        for w in busy:
            w.kill()  # mid-task at shutdown: nothing to wait for

    @property
    def live_workers(self) -> int:
        with self._cond:
            return self._live

    # -- internals
    def _note_startup_failure(self, exc: WorkerStartupError) -> None:
        self._startup_failures += 1
        if exc.deterministic:
            if not self.degraded:
                self.degraded = True
                logger.warning(
                    "pdf_router: %s sidecar workers cannot start in this process (%s); its backends run "
                    "degraded (light tier in-process, heavy tier disabled). Set PDF_ROUTER_INLINE=1 to run "
                    "every tier in-process on purpose.",
                    self.name,
                    exc,
                )
            return
        cooldown = min(self.COOLDOWN_MAX, self.COOLDOWN_BASE * 2 ** min(self._startup_failures - 1, 6))
        self.retry_after = time.monotonic() + cooldown
        logger.warning("pdf_router: %s sidecar worker failed to start (%s); retrying in %.0fs", self.name, exc, cooldown)

    def _acquire(self, deadline: float) -> _Worker:
        with self._cond:
            while True:
                if self._closed:
                    raise BackendUnavailable(f"pool {self.name} is shut down")
                while self._idle:
                    worker = self._idle.pop()
                    if worker.alive():
                        self._busy.add(worker)
                        return worker
                    self._live -= 1  # died while idle
                if self._live < self.size:
                    self._live += 1
                    self._spawned += 1
                    index = self._spawned
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BackendUnavailable(f"pool {self.name}: all {self.size} workers busy within the task budget")
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
        with self._cond:
            if self._closed:
                worker.kill()
                self._live = max(0, self._live - 1)
                raise BackendUnavailable(f"pool {self.name} is shut down")
            self._busy.add(worker)
            self._ensure_reaper()
        return worker

    def _release(self, worker: _Worker) -> None:
        recycle = worker.dead or not worker.alive() or worker.tasks >= self.settings.worker_max_tasks
        with self._cond:
            was_tracked = worker in self._busy
            self._busy.discard(worker)
            if not was_tracked:
                kill = True  # shutdown already took it over
            elif recycle or self._closed:
                self._live = max(0, self._live - 1)
                kill = True
            else:
                self._idle.append(worker)
                kill = False
            self._cond.notify()
        if kill:
            worker.kill()

    def _ensure_reaper(self) -> None:  # called with self._cond held
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
    """Routes layout-tier tasks to a light pool and Docling tasks to a heavy
    pool. Each pool tracks its own startup health, so a Docling spawn failure
    never pushes the healthy layout pool in-process."""

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
            self._pools.update(dict.fromkeys(light, pool))
        if heavy:
            pool = MLWorkerPool("pdf-heavy", heavy, settings.heavy_workers, settings, registry_factory=registry_factory)
            self.pools.append(pool)
            self._pools.update(dict.fromkeys(heavy, pool))
        self._inline_lock = threading.Lock()

    @property
    def max_concurrency(self) -> int:
        return max(1, sum(p.size for p in self.pools if not p.in_cooldown()))

    @property
    def degraded(self) -> bool:
        return any(p.degraded for p in self.pools)

    @property
    def retry_after(self) -> float:
        return max((p.retry_after for p in self.pools), default=0.0)

    def run(self, backend_name, pdf_bytes, page_numbers, timeout):
        pool = self._pools.get(backend_name)
        if pool is None:
            raise BackendUnavailable(f"{backend_name}: no worker pool serves this backend")
        backend = self._registry.get(backend_name)
        if pool.in_cooldown():
            return self._run_degraded(backend, pdf_bytes, page_numbers)
        try:
            return pool.run(backend_name, pdf_bytes, page_numbers, timeout)
        except WorkerStartupError:
            return self._run_degraded(backend, pdf_bytes, page_numbers)  # the pool recorded the failure

    def _run_degraded(self, backend, pdf_bytes, page_numbers):
        if backend.tier >= 2:
            raise BackendUnavailable(
                f"{backend.name}: sidecar workers unavailable and heavy tiers never run in-process "
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
    """Process-wide executor: the pool sizes are the global caps on ML work
    across every scraper thread, so the first settings used win."""
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
