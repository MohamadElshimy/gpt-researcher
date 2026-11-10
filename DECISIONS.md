# DECISIONS — PDF Page Router

Short version. [DECISIONS-DETAILED.md](DECISIONS-DETAILED.md) has every measurement, source and rejected alternative, under the same section names.

**What changed.** `PyMuPDFScraper` ran plain `page.get_text()` on every PDF. It now hands the downloaded bytes to `gpt_researcher/scraper/pdf_router/`, which triages each page, sends it to the cheapest backend that can handle it, gates the result, escalates one tier on failure and returns one Markdown string. The scraper contract `(content, image_urls, title)` is unchanged; `Scraper._retry_as_pdf` gets routing for free. Timings are from an M3 Pro laptop; expect Linux CPU to be 3–5× slower on the ML tiers.

## Wiring

- Wired: `scraper/pymupdf/pymupdf.py` → `route_pdf_detailed(bytes, url)`. Download code kept, temp file gone, 50 MB cap.
- Not wired, on purpose: the local document loaders return one `Document` per page and feed chunking and citations, so changing them changes downstream behaviour. `scrape_skills.py::scrape_pdf_with_pymupdf` is unreachable through `Scraper.get_scraper()`.
- Used: the scraper already runs 15 threads (15 PDFs in parallel); `Scraper` rejects content under 100 chars, so an empty result is a clean failure.

## Classification

Stage 1, geometry only, 10–13 ms/page from one `get_text("dict")`, `get_image_info()`, `get_cdrawings()`:

| class | rule | threshold from |
|---|---|---|
| SCAN | < 30 chars and image cover ≥ 0.25; or cover ≥ 0.5 and < 300 chars (bitmap plus a stamp) | Artifex, MinerU, Tika |
| BLANK | < 10 chars and cover < 0.02 | Tika |
| BROKEN_TEXT | U+FFFD share ≥ 10% | Artifex, Tika |
| MULTI_COLUMN | ≥ 20% of consecutive lines jump > 30% of page width while overlapping vertically | measured: 0.92 on a row-major page, ≤ 0.06 on real pages |
| TABLE | ≥ 10 numeric-only lines or ≥ 3 horizontal rules | measured: 12–73 on table pages, ≤ 8 on prose |
| FIGURE | image cover ≥ 0.10 with text | — |

Everything else is PLAIN on tier 0. Stage 2 gates the ML output and escalates one tier when it fails: word recall of the raw text layer < 0.90 (a dropped block), table sanity (ragged rows, orphan `.` cells, headers cut into `D|ev|Te|st`, merged tables, words split across cells), empty output, Docling's status. Dropped text never beats the raw layer; a slightly broken table does beat flattened cells, so it is accepted with a note when no higher tier exists.

Rejected: `find_tables()` as a signal (100–300 ms/page; it runs the layout model in pymupdf 2.x, so it costs what it would gate); the layout model as the classifier (0.3 s on every page vs 15 ms); a trained classifier (five fixtures). Shipped routers are all two-way text-vs-OCR; a third layout tier has little precedent, hence the gates.

## Backends

| tier | backend | cost | closes | does not |
|---|---|---|---|---|
| 0 | `plain`: `get_text()` + ligature fix | 3–6 ms/page | single-column and column-major streams (all LaTeX output) | tables, headings, scans, row-major streams |
| 1 | `layout`: PyMuPDF layout mode (GNN bundled in pymupdf ≥ 2.0), OCR off | 0.2–0.45 s/page, ~150 MB | row-major columns (32% → 97%), headings, most tables (95–98% cells) | side-by-side tables, scans, formulas |
| 2 | `docling`: layout + TableFormer + OCR | 0.2–0.3 s/page digital, 3.3 s scan, ~1.8 GB | scans, broken text, complex tables (96–99.5% cells), best prose (+1.5 pts) | formulas, charts |

- pymupdf4llm is no longer the cheap heuristic tier the brief describes: pymupdf 2.0 (Oct 2026) absorbed it and the layout model, and this repo's `pymupdf>=1.26` already resolves to 2.1. The old heuristic mode measured 100–150 ms/page with no table or row-major gain. Rejected.
- Licence: `pymupdf-layout` 1.x was Polyform Noncommercial; in 2.x it ships inside the AGPL/commercial wheel without public source. Unresolved; `PDF_ROUTER_LAYOUT_TIER=false` drops the tier. Docling and its models are MIT / Apache / CDLA.
- Not added: pdfplumber (slow, no column logic), Unstructured fast (no tables), marker/MinerU (RAIL/AGPL, GPU-oriented), Tesseract (binary dependency, silent degradation), a VLM tier (reads charts, $0.01–0.06 and 5–30 s per page; `PageClass.CHART` reserved for it).
- Interface: `PageBackend` = `name`, `tier`, `handles`, `available()`, `parse(pdf_bytes, page_numbers) -> dict[int, PageResult]`. Picklable in and out, so the same class runs inline or in a sidecar. A new parser is one class and one registry line.

## Parallelism

- Across documents: the existing 15 scraper threads. Inside one: triage and tier 0 inline; tiers 1–2 in sidecar processes, a light pool for layout (`min(4, cpus-1)` workers, runs of ≤ 4 pages per task) and a heavy pool for Docling (1 worker, runs of ≤ 8). Pools are process-global, so worker counts cap ML work across all threads.
- Processes, not threads: PyMuPDF is not thread-safe and never releases the GIL (8 threads → 1.66×). `spawn`, not fork: torch's OpenMP deadlocks after fork. Cost: ~1 s layout, ~7 s Docling per fresh worker; children re-import `__main__`, so start via `uvicorn main:app` (as the Dockerfile does).
- Constraints: memory = workers × RSS; pool state is per API process.
- If sidecars cannot start, fail closed per pool: the heavy tier is reported unavailable (untrusted PDFs never hit torch in the API process unless `PDF_ROUTER_INLINE=1`) and its pages keep plain text; the light tier runs in-process, serialised. Deterministic failures latch; transient ones retry after a 15 s → 5 min cool-down.

## Failure behaviour: fails, throws, hangs

- Throws: any exception marks those pages failed; they fall to the next tier or keep the lower-tier result. A backend bug cannot take the document down.
- Hangs: per-task timeout (30 s light, 120 s heavy) covering queue wait, send and reply; expiry SIGKILLs the worker, the next task spawns a fresh one. A watchdog on the pipe send kills a child that stopped reading. Docling's cooperative `document_timeout` cannot interrupt a stuck native call; the kill can. Threads cannot be killed, which is why processes.
- Crashes: segfault / OOM ends only the worker (`BackendCrashed` with exit code); a child reporting `MemoryError` is recycled. `shutdown()` kills busy workers so a server stop never waits.
- Repeated failure: per-backend circuit breaker, 3 consecutive timeouts/crashes → open 60 s → one half-open probe held as an expiring lease, so it cannot wedge. A bad PDF does not count.
- Budgets: 40 ML pages per document (by class severity), 150 triaged pages, 180 s deadline. Task timeouts are computed at start and clamped to the remaining deadline, so at the deadline in-flight workers are killed; late pages keep their best lower-tier result. Idle workers recycled after 10 min / 200 tasks.
- Never silent, never fake: a page known to carry text that ends empty gets `<!-- page N: no text extracted (reason) -->`; a document of only markers returns `""` so `Scraper`'s 100-char check rejects it as before. Every page records its attempts; one INFO line per document.

## Verification

Metrics (`evals/pdf_router/`): sentence and 8-gram recall against ar5iv HTML of the same arXiv papers; share of ground-truth table cells found inside Markdown table rows; chars, headings, seconds. Sanity: `get_text(sort=True)` on BERT scores 1.4% recall vs 86% for stream order, so the metric sees interleaving. Fifth fixture `rowmajor-two-column.pdf` has a generated row-major stream (what Word and OCR produce) because all four LaTeX fixtures already read in order.

`python -m evals.pdf_router.run_eval --pooled --sanity --all-docling`, warm-up excluded; "docling everywhere" is the no-router upper bound:

| fixture | extractor | time | sentence recall | GT cells in md tables | headings | backends |
|---|---|---|---|---|---|---|
| BERT, 16 p | previous | 0.07 s | 86.0% | 0% | 0 | — |
| | **router** | 7.6 s | 86.0% | **96.0%** | 11 | plain 11, layout 3, docling 2 (2 escalated) |
| | docling everywhere | 20.1 s | 87.4% | 96.3% | 34 | docling 16 |
| RoBERTa, 13 p | previous | 0.05 s | 84.6% | 0% | 0 | — |
| | **router** | 9.5 s | 85.5% | **99.5%** | 10 | plain 6, layout 4, docling 3 (3 escalated) |
| | docling everywhere | 14.5 s | 87.4% | 99.5% | 29 | docling 13 |
| scanned page | previous | 0.00 s | **0 chars**, no error | — | 0 | — |
| | **router** | 3.9 s | **1 095 chars**, `ocr_score` 0.98 | — | 1 | docling 1 |
| mixed, 5 p | previous | 0.02 s | — | 0 tables | 0 | — |
| | **router** | 5.8 s | — | 2 tables | 4 | plain 2, layout 1, docling 2 |
| row-major, 2 p | previous | 0.01 s | **32.4%** | — | 0 | — |
| | **router** | 0.3 s | **97.3%** | — | 0 | layout 1, plain 1 |

- Improved: tables 0% → 95–99.5% of cells; scanned page 0 → 1.1 k chars; headings; row-major columns 32% → 97%; side-by-side tables (layout garbles, gate catches, Docling renders). Every escalation to Docling was a real defect in the layout tier's table, checked by hand.
- Did not improve: prose recall on the LaTeX fixtures. Their streams were already in reading order and the old loader did not sort, so "columns interleave" does not occur on these files; the ~13% unmatched sentences contain formulas, identical for every extractor.
- Why not Docling everywhere: +1.4–1.9 points prose recall and more headings, at 1.5–2.6× the wall-clock, every page through a 1.8 GB tier, and a mandatory model dependency. The router keeps 17 of 35 fixture pages on a 4 ms tier. On a GPU box the argument shrinks and `PDF_ROUTER_LAYOUT_TIER=false` plus Docling for every non-plain page is defensible.
- Latency: 70 ms → 7.6 s on BERT here, likely 20–40 s on Linux CPU. Sources fetch concurrently, so a run pays roughly the slowest PDF plus a one-off 7–15 s model load, bounded by the deadline and the ML budget.
- Tests: `tests/pdf_router/` covers triage on fixture pages, gates on real garbled output, the router with fake backends, the spawned pool (timeout kill, crash, breaker, degraded mode) and the scraper wiring; `PDF_ROUTER_RUN_SLOW=1` runs the real ladder.

## Trade-offs

- Docling is optional for library installs (`[pdf-rich]`), required in `requirements.txt` / Dockerfile (weights prefetched at build). Without it: two tiers, logged once.
- OCR lives only in Docling; pymupdf4llm's OCR exists only because Docling installed RapidOCR, and alone it warns and returns nothing.
- Thresholds are env-overridable constants, not `Config` fields: `Scraper(link, session)` is the interface to keep.
- No de-hyphenation on tier 0 (ambiguous); ligatures only. The layout tier drops running heads and page numbers; the recall gate catches a misclassified body block.

## Another week

- VLM tier for `CHART` pages behind a flag with a per-page cost budget.
- Route on Docling's per-page confidence; expose router events through `emit_research_event`.
- Warm the heavy pool at server start; cache routed output by URL hash.
- Tune thresholds on DocLayNet / OmniDocBench pages; measure Docling on x86 CPU.
- Formula handling (mark math glyph soup, or CodeFormula on demand); adopt the router in the local loaders behind a page-aware return type.

## Unsure / weak

- Thresholds are tuned on five fixtures; a false escalation costs ~1 s, a false negative costs quality silently.
- Layout-model licence inside pymupdf 2.x is unresolved; confirm with Artifex before shipping that tier commercially.
- Docling Linux latency not re-measured (its paper: x86 median 0.79 s/page). One unexplained 168 s first warm-up seen once.
- Breaker and pools are per process; nothing shared across uvicorn workers.
- Table-sanity heuristics are written against the garbled outputs I saw. Two review passes found real defects in the first version (markers counted as content, empty corner cells read as ragged, a stuck half-open breaker, a blocking pipe send); fixed and tested, but this layer needs a broader corpus.

## Running it

```bash
pip install -e ".[test,pdf-rich]"          # or: pip install -r requirements.txt
docling-tools models download layout tableformer rapidocr
pytest tests/pdf_router                    # fast suite (spawns worker processes)
PDF_ROUTER_RUN_SLOW=1 pytest tests/pdf_router/test_integration_slow.py
python -m evals.pdf_router.run_eval --pooled --sanity --all-docling --out-dir evals/pdf_router/out
```

Knobs: `PDF_ROUTER_*` in `settings.py` (tiers, inline mode, workers, timeouts, ML budget, deadline, size cap).
