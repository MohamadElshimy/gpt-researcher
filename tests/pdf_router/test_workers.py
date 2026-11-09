"""Sidecar worker pool: reuse, timeout kill + respawn, crash isolation,
circuit breaker, recycling, graceful degradation when workers cannot start.
These spawn real processes, so they are slower (a few seconds each)."""

from __future__ import annotations

import os
import time

import pytest

from gpt_researcher.scraper.pdf_router.backends import (
    BackendCrashed,
    BackendError,
    BackendTimeout,
    BackendUnavailable,
)
from gpt_researcher.scraper.pdf_router.settings import RouterSettings
from gpt_researcher.scraper.pdf_router.workers import (
    CircuitBreaker,
    MLWorkerPool,
    PooledExecutor,
    WorkerStartupError,
)

from .fake_backends import registry as fake_registry

pytestmark = pytest.mark.timeout(180)

FACTORY = "tests.pdf_router.fake_backends:registry"
BROKEN_FACTORY = "tests.pdf_router.fake_backends:broken_registry"
DYING_FACTORY = "tests.pdf_router.fake_backends:dying_registry"
BACKENDS = ("echo", "sleep", "crash", "raising", "childtimeout")


def make_pool(size=1, **settings) -> MLWorkerPool:
    defaults = dict(breaker_failures=2, breaker_reset=0.5, worker_idle_secs=0, worker_max_tasks=1000)
    defaults.update(settings)
    return MLWorkerPool("test-pool", BACKENDS, size, RouterSettings(**defaults), registry_factory=FACTORY)


def pid_of(result) -> int:
    return int(result[0].markdown.rsplit("pid", 1)[1])


def test_worker_is_reused_between_tasks():
    pool = make_pool()
    try:
        first = pool.run("echo", b"", [0], timeout=30)
        second = pool.run("echo", b"", [0, 1], timeout=30)
        assert first[0].markdown.startswith("echo page 0")
        assert set(second) == {0, 1}
        assert pid_of(first) == pid_of(second) != os.getpid()
        assert pool.live_workers == 1
    finally:
        pool.shutdown()


def test_timeout_kills_worker_and_next_task_gets_a_fresh_one():
    pool = make_pool()
    try:
        before = pid_of(pool.run("echo", b"", [0], timeout=30))
        t0 = time.monotonic()
        with pytest.raises(BackendTimeout):
            pool.run("sleep", b"", [5], timeout=0.5)  # sleeps 5 s, allowed 0.5 s
        assert time.monotonic() - t0 < 4.0
        after = pid_of(pool.run("echo", b"", [0], timeout=30))
        assert after != before
        assert pool.live_workers == 1
    finally:
        pool.shutdown()


def test_crash_is_isolated_and_reported():
    pool = make_pool(breaker_failures=10)
    try:
        with pytest.raises(BackendCrashed):
            pool.run("crash", b"", [0], timeout=30)
        # The parent is fine and a new worker serves the next task.
        assert pool.run("echo", b"", [1], timeout=30)[1].markdown.startswith("echo page 1")
    finally:
        pool.shutdown()


def test_backend_exceptions_cross_the_pipe_without_tripping_the_breaker():
    pool = make_pool(breaker_failures=1)
    try:
        with pytest.raises(BackendError, match="bad pdf for test"):
            pool.run("raising", b"", [0], timeout=30)
        assert not pool.breaker("raising").is_open
        assert pool.run("echo", b"", [0], timeout=30)
    finally:
        pool.shutdown()


def test_circuit_breaker_opens_after_consecutive_failures_and_probes_after_reset():
    pool = make_pool(breaker_failures=2, breaker_reset=0.5)
    try:
        for _ in range(2):
            with pytest.raises(BackendCrashed):
                pool.run("crash", b"", [0], timeout=30)
        assert pool.breaker("crash").is_open
        with pytest.raises(BackendUnavailable, match="circuit breaker open"):
            pool.run("crash", b"", [0], timeout=30)
        time.sleep(0.6)
        with pytest.raises(BackendCrashed):  # half-open probe is allowed through, fails, reopens
            pool.run("crash", b"", [0], timeout=30)
        assert pool.breaker("crash").is_open
        # Other backends in the same pool are unaffected.
        assert pool.run("echo", b"", [0], timeout=30)
    finally:
        pool.shutdown()


def test_child_side_backend_timeout_keeps_its_type_and_counts_as_failure():
    pool = make_pool(breaker_failures=5)
    try:
        with pytest.raises(BackendTimeout, match="backend-level timeout"):
            pool.run("childtimeout", b"", [0], timeout=30)
        assert pool.breaker("childtimeout").consecutive == 1
    finally:
        pool.shutdown()


def test_shutdown_kills_busy_workers_and_never_goes_negative():
    import threading

    pool = make_pool()
    outcome: list = []

    def long_task():
        try:
            pool.run("sleep", b"", [30], timeout=60)
        except BackendError as exc:
            outcome.append(type(exc).__name__)

    t = threading.Thread(target=long_task)
    t.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not pool._busy:  # wait until the worker is really mid-task
        time.sleep(0.05)
    assert pool.live_workers == 1
    pool.shutdown()
    t.join(10)
    assert not t.is_alive()
    # Killed mid-task (crash/timeout) or, if shutdown raced the spawn, refused: never a hang, never a result.
    assert outcome and outcome[0] in ("BackendCrashed", "BackendTimeout", "BackendUnavailable")
    assert pool.live_workers == 0


def test_breaker_probe_is_a_lease():
    breaker = CircuitBreaker(failures=1, reset_seconds=0.2)
    breaker.record_failure()
    assert breaker.is_open and not breaker.allow()
    time.sleep(0.25)
    assert breaker.allow()  # the half-open probe
    assert not breaker.allow()  # only one probe at a time
    breaker.cancel_probe()  # the prober could not get a worker
    assert breaker.allow()  # the lease is back
    time.sleep(0.25)
    assert breaker.allow()  # an unreported probe expires after reset_seconds


def test_breaker_not_stuck_when_acquire_fails_during_probe():
    pool = make_pool(breaker_failures=1, breaker_reset=0.2)
    try:
        with pytest.raises(BackendCrashed):
            pool.run("crash", b"", [0], timeout=30)
        assert pool.breaker("crash").is_open
        time.sleep(0.25)
        pool.shutdown()  # acquiring a worker for the probe now fails
        with pytest.raises(BackendUnavailable, match="shut down"):
            pool.run("crash", b"", [0], timeout=30)
        assert pool.breaker("crash").allow()  # probe lease was returned, breaker not wedged
    finally:
        pool.shutdown()


def test_workers_are_recycled_after_max_tasks():
    pool = make_pool(worker_max_tasks=1)
    try:
        a = pid_of(pool.run("echo", b"", [0], timeout=30))
        b = pid_of(pool.run("echo", b"", [0], timeout=30))
        assert a != b
    finally:
        pool.shutdown()


def test_shutdown_stops_workers():
    pool = make_pool()
    pool.run("echo", b"", [0], timeout=30)
    assert pool.live_workers == 1
    pool.shutdown()
    assert pool.live_workers == 0
    with pytest.raises(BackendUnavailable):
        pool.run("echo", b"", [0], timeout=5)


def test_startup_failure_is_reported_as_worker_startup_error():
    pool = MLWorkerPool("broken", BACKENDS, 1, RouterSettings(worker_idle_secs=0), registry_factory=BROKEN_FACTORY)
    try:
        with pytest.raises(WorkerStartupError, match="exploded on purpose"):
            pool.run("echo", b"", [0], timeout=30)
        assert pool.live_workers == 0
    finally:
        pool.shutdown()


def test_deterministic_startup_failure_degrades_light_tier_inline_and_disables_heavy():
    settings = RouterSettings(light_workers=1, heavy_workers=1, worker_idle_secs=0)
    executor = PooledExecutor(settings, registry=fake_registry(), registry_factory=BROKEN_FACTORY)
    try:
        result = executor.run("echo", b"", [0], timeout=30)
        assert executor.degraded is True
        assert executor.max_concurrency == 1
        assert pid_of(result) == os.getpid()  # light tier ran in this process
        # Heavy tiers never run in-process automatically: pages fall back to plain text instead.
        with pytest.raises(BackendUnavailable, match="PDF_ROUTER_INLINE"):
            executor.run("raising", b"", [0], timeout=30)
        # Subsequent light calls stay inline without trying to spawn again.
        assert pid_of(executor.run("echo", b"", [0], timeout=30)) == os.getpid()
    finally:
        executor.shutdown()


def test_transient_startup_failure_sets_a_cooldown_instead_of_latching(monkeypatch):
    monkeypatch.setattr("gpt_researcher.scraper.pdf_router.workers.spawn_possible", lambda: True)
    settings = RouterSettings(light_workers=1, heavy_workers=1, worker_idle_secs=0)
    executor = PooledExecutor(settings, registry=fake_registry(), registry_factory=DYING_FACTORY)
    try:
        result = executor.run("echo", b"", [0], timeout=30)
        assert executor.degraded is False
        assert executor.retry_after > time.monotonic()
        assert pid_of(result) == os.getpid()
        with pytest.raises(BackendUnavailable):
            executor.run("raising", b"", [0], timeout=30)
        # Within the cool-down no spawn is attempted (would take a handshake round-trip otherwise).
        t0 = time.monotonic()
        executor.run("echo", b"", [0], timeout=30)
        assert time.monotonic() - t0 < 0.5
    finally:
        executor.shutdown()


def test_startup_error_flags_determinism():
    pool = MLWorkerPool("broken", BACKENDS, 1, RouterSettings(worker_idle_secs=0), registry_factory=BROKEN_FACTORY)
    try:
        with pytest.raises(WorkerStartupError) as info:
            pool.run("echo", b"", [0], timeout=30)
        assert info.value.deterministic is True
    finally:
        pool.shutdown()


def test_pooled_executor_routes_tiers_to_pools():
    settings = RouterSettings(light_workers=1, heavy_workers=1, worker_idle_secs=0, breaker_failures=5)
    executor = PooledExecutor(settings, registry=fake_registry(), registry_factory=FACTORY)
    try:
        light = pid_of(executor.run("echo", b"", [0], timeout=30))
        with pytest.raises(BackendError):
            executor.run("raising", b"", [0], timeout=30)  # heavy pool (tier 2)
        assert light != os.getpid()
        assert executor.degraded is False
        with pytest.raises(BackendUnavailable):
            executor.run("plain", b"", [0], timeout=30)  # tier 0 is never pooled
    finally:
        executor.shutdown()
