"""Runtime settings for the PDF page router.

Scrapers are constructed as ``Scraper(link, session)`` and never see the
``Config`` object, so the router reads its knobs from the environment. Every
knob has a default that is safe inside the research loop (15 concurrent
scraper threads, ~15 sources per query).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping

_PREFIX = "PDF_ROUTER_"


def _default_light_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) - 1))


@dataclass(frozen=True)
class RouterSettings:
    # --- tiers -------------------------------------------------------------
    enabled: bool = True
    layout_tier: bool = True  # PyMuPDF layout mode (GNN bundled in pymupdf>=2.0)
    docling_tier: bool = True  # Docling; auto-disabled when not importable
    inline: bool = False  # run ML backends in-process (tests, debugging)

    # --- sidecar worker pools --------------------------------------------
    light_workers: int = _default_light_workers()  # layout tier, ~150 MB each
    heavy_workers: int = 1  # docling, ~1.8 GB each; also the global Docling cap
    light_timeout: float = 30.0  # seconds per task
    heavy_timeout: float = 120.0  # seconds per task (Docling recommends 90-120)
    worker_idle_secs: float = 600.0  # recycle idle workers to give memory back
    worker_max_tasks: int = 200  # recycle a worker after this many tasks
    breaker_failures: int = 3  # consecutive failures that open the breaker
    breaker_reset: float = 60.0  # seconds before a half-open probe

    # --- budgets per document ----------------------------------------------
    max_pdf_bytes: int = 50_000_000  # downloads above this are rejected before parsing
    max_ml_pages: int = 40  # pages allowed on ML tiers; the rest stay plain
    max_triage_pages: int = 150  # pages inspected; the rest stay plain
    doc_deadline: float = 180.0  # wall-clock budget for one PDF
    docling_run_size: int = 8  # contiguous pages per Docling call

    # --- triage thresholds (see classify.py for provenance) ---------------
    blank_max_chars: int = 10
    blank_max_image_cover: float = 0.02
    scan_max_chars: int = 30
    scan_min_image_cover: float = 0.25
    broken_text_bad_ratio: float = 0.10
    column_jump_rate: float = 0.20
    table_min_numeric_lines: int = 10
    table_min_rule_lines: int = 3
    figure_min_image_cover: float = 0.10

    # --- post-parse quality gates ------------------------------------------
    min_word_recall: float = 0.90

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "RouterSettings":
        env = os.environ if env is None else env
        values = {}
        for f in fields(cls):
            raw = env.get(_PREFIX + f.name.upper())
            if raw is None or raw == "":
                continue
            values[f.name] = _coerce(f.type, raw, f.name)
        return cls(**values)


def _coerce(type_name: object, raw: str, name: str):
    kind = type_name if isinstance(type_name, str) else getattr(type_name, "__name__", str(type_name))
    raw = raw.strip()
    try:
        if kind == "bool":
            return raw.lower() in ("1", "true", "yes", "on")
        if kind == "int":
            return int(raw)
        if kind == "float":
            return float(raw)
    except ValueError as exc:
        raise ValueError(f"{_PREFIX}{name.upper()}={raw!r} is not a valid {kind}") from exc
    return raw
