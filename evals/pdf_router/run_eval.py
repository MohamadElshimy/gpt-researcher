"""Compare the previous extractor (plain PyMuPDF text, as PyMuPDFLoader produced
it) with the page router on the fixture pack, and score both against ground
truth where one exists.

    python -m evals.pdf_router.run_eval            # inline backends, fetches ar5iv ground truth once
    python -m evals.pdf_router.run_eval --pooled   # through the sidecar worker pools
    python -m evals.pdf_router.run_eval --offline  # cached ground truth only
    python -m evals.pdf_router.run_eval --sanity   # also show get_text(sort=True) to prove the metric sees interleaving

Ground truth: ar5iv's HTML rendering of the two arXiv papers (fetched into
``--gt-cache``, not committed) and the reading-order text written next to the
synthetic row-major fixture by ``tests/fixtures/pdf_router/make_rowmajor.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import pymupdf

from gpt_researcher.scraper.pdf_router import PageRouter, RouterSettings
from gpt_researcher.scraper.pdf_router.workers import get_pooled_executor

from . import metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "pdf_router"
ARXIV_GT = {"two-column-paper.pdf": "1810.04805", "borderless-table.pdf": "1907.11692"}
FIXTURE_ORDER = [
    "two-column-paper.pdf",
    "borderless-table.pdf",
    "scanned-page.pdf",
    "mixed-document.pdf",
    "rowmajor-two-column.pdf",
]


def fetch_ground_truth(arxiv_id: str, cache: Path, offline: bool) -> str | None:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{arxiv_id}.ar5iv.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    if offline:
        return None
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "gpt-researcher pdf-router eval"})
    with urllib.request.urlopen(request, timeout=60) as response:
        html = response.read().decode("utf-8", errors="ignore")
    path.write_text(html, encoding="utf-8")
    return html


def previous_extractor(pdf_bytes: bytes) -> str:
    """What PyMuPDFScraper returned before: PyMuPDFLoader page texts joined by newlines."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def sorted_extractor(pdf_bytes: bytes) -> str:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text(sort=True) for page in doc)


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def evaluate(name: str, text: str, gt_sentences, gt_cells) -> dict:
    row = {
        "chars": len(text),
        "md_tables": metrics.markdown_table_count(text),
        "headings": metrics.heading_count(text),
    }
    if gt_sentences:
        row["sentence_recall"] = metrics.sentence_recall(gt_sentences, text)
        row["ngram_recall"] = metrics.ngram_recall(gt_sentences, text)
    if gt_cells:
        row["cells_in_tables"], row["cells_anywhere"] = metrics.table_cell_recall(gt_cells, text)
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixtures-dir", type=Path, default=FIXTURES)
    parser.add_argument("--gt-cache", type=Path, default=HERE / ".gt_cache")
    parser.add_argument("--offline", action="store_true", help="never fetch ground truth")
    parser.add_argument("--pooled", action="store_true", help="run ML tiers through the sidecar worker pools")
    parser.add_argument("--sanity", action="store_true", help="also score get_text(sort=True)")
    parser.add_argument("--out-dir", type=Path, default=None, help="write extracted outputs here for inspection")
    parser.add_argument("--json", type=Path, default=None, help="write all numbers to this JSON file")
    args = parser.parse_args(argv)

    settings = RouterSettings.from_env()
    if args.pooled:
        router = PageRouter(settings=settings, executor=get_pooled_executor(settings))
    else:
        router = PageRouter(settings=RouterSettings.from_env(), executor=None)
    print(f"ladder: {' -> '.join(b.name for b in router.ladder)}  ({'pooled' if args.pooled else 'inline'})")

    fixtures = [args.fixtures_dir / n for n in FIXTURE_ORDER if (args.fixtures_dir / n).exists()]
    # Warm the ML tiers once so per-fixture timings reflect steady state; the
    # cold start (model load) is reported separately.
    t0 = time.perf_counter()
    router.route((args.fixtures_dir / "scanned-page.pdf").read_bytes(), "warm-up")
    warm_up = time.perf_counter() - t0

    report: dict[str, dict] = {"_warm_up_seconds": round(warm_up, 1)}
    for path in fixtures:
        name = path.name
        pdf_bytes = path.read_bytes()
        gt_sentences: list[str] = []
        gt_cells: list[str] = []
        if name in ARXIV_GT:
            html = fetch_ground_truth(ARXIV_GT[name], args.gt_cache, args.offline)
            if html:
                gt_sentences, gt_cells = metrics.ground_truth_from_ar5iv(html)
        elif name == "rowmajor-two-column.pdf" and path.with_suffix(".gt.txt").exists():
            gt_sentences = metrics.ground_truth_from_text(path.with_suffix(".gt.txt").read_text(encoding="utf-8"))

        t0 = time.perf_counter()
        before = previous_extractor(pdf_bytes)
        t_before = time.perf_counter() - t0
        t0 = time.perf_counter()
        routed = router.route(pdf_bytes, name)
        t_after = time.perf_counter() - t0

        entry = {
            "pages": len(routed.pages),
            "ground_truth": {"sentences": len(gt_sentences), "cells": len(gt_cells)},
            "previous": {"seconds": round(t_before, 3), **evaluate(name, before, gt_sentences, gt_cells)},
            "router": {
                "seconds": round(t_after, 2),
                "backends": routed.backend_counts,
                "classes": routed.class_counts,
                "notes": routed.notes,
                "escalations": sum(1 for p in routed.pages if any("gate failed" in a for a in p.attempts)),
                **evaluate(name, routed.markdown, gt_sentences, gt_cells),
            },
        }
        if args.sanity and name == "two-column-paper.pdf":
            entry["sorted_text_sanity"] = evaluate(name, sorted_extractor(pdf_bytes), gt_sentences, gt_cells)
        report[name] = entry

        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / f"{name}.previous.txt").write_text(before, encoding="utf-8")
            (args.out_dir / f"{name}.router.md").write_text(routed.markdown, encoding="utf-8")
            (args.out_dir / f"{name}.router.pages.json").write_text(
                json.dumps(
                    [
                        {
                            "page": p.page_no + 1,
                            "class": p.page_class.value,
                            "backend": p.backend,
                            "confidence": p.confidence,
                            "attempts": p.attempts,
                            "chars": len(p.markdown),
                        }
                        for p in routed.pages
                    ],
                    indent=1,
                ),
                encoding="utf-8",
            )

    print(f"\nwarm-up (model load, first scanned page): {warm_up:.1f}s\n")
    header = "| fixture | pages | extractor | time | chars | sentence recall | 8-gram recall | cells in md tables | md tables | headings | backends |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for name, entry in report.items():
        if name.startswith("_"):
            continue
        rows = [("previous", entry["previous"]), ("router", entry["router"])]
        if "sorted_text_sanity" in entry:
            rows.append(("sort=True (sanity)", entry["sorted_text_sanity"]))
        for label, row in rows:
            backends = row.get("backends", "")
            if backends:
                backends = ", ".join(f"{k}:{v}" for k, v in backends.items())
                if row.get("escalations"):
                    backends += f" (escalated {row['escalations']})"
            print(
                f"| {name} | {entry['pages']} | {label} | {row.get('seconds', 0):.2f}s | {row['chars']} | "
                f"{pct(row.get('sentence_recall'))} | {pct(row.get('ngram_recall'))} | {pct(row.get('cells_in_tables'))} | "
                f"{row['md_tables']} | {row['headings']} | {backends} |"
            )
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
