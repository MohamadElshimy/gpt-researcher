# DECISIONS — PDF Page Router

Scope: `PyMuPDFScraper` used to run `PyMuPDFLoader` (plain `page.get_text()`) over every PDF. It now hands the downloaded bytes to `gpt_researcher/scraper/pdf_router/`, which triages each page, sends it to the cheapest backend that can handle it, gates the result, escalates one tier on failure, and recombines one Markdown string. The scraper contract `(content, image_urls, title)` is unchanged; `Scraper._retry_as_pdf` (extensionless PDFs) gets routing for free. Measurements below are from an M3 Pro (11 cores, 18 GB, MPS) unless stated; expect Linux CPU to be 3–5× slower on the ML tiers.

## Where it is wired, and where it is not

- Wired: `gpt_researcher/scraper/pymupdf/pymupdf.py` → `route_pdf_detailed(bytes, url)`. Download code kept (same timeouts, same SSL fallback); the temp file is gone, bytes stay in memory with a 50 MB cap (`PDF_ROUTER_MAX_PDF_BYTES`).
- Not wired, on purpose: `document/document.py` and `online_document.py` also call `PyMuPDFLoader`, but they return one `Document` **per page** with page metadata for local/hybrid research; swapping that for one Markdown string changes chunking and page-level citations downstream, which the brief says must not happen. Adoption is one line (`route_pdf`) when wanted. `browser/processing/scrape_skills.py::scrape_pdf_with_pymupdf` is unreachable through `Scraper.get_scraper()` (`.pdf` is dispatched to `PyMuPDFScraper` before the configured scraper is consulted) and returns `str(doc)` — dead code, left alone.
- Context that shaped the design: the scraper already runs in a `ThreadPoolExecutor(MAX_SCRAPER_WORKERS=15)`, so up to 15 PDFs parse concurrently; defaults give ~15 sources per research run; `Scraper` rejects content shorter than 100 chars, so an empty result is a clean fetch failure rather than garbage.

## How pages are classified

Two stages. Stage 1 is geometry only, 10–20 ms/page, one `get_text("dict")` without images, `get_image_info()`, `get_cdrawings()`:

| signal | rule | provenance |
|---|---|---|
| chars, image cover | `< 30` chars and cover `≥ 0.25` → SCAN; `< 10` chars and cover `< 0.02` → BLANK | Artifex triage post; MinerU (< 50 chars/page); Tika `ocrStrategy=auto` (< 10 chars) |
| U+FFFD share | `≥ 10%` → BROKEN_TEXT | Artifex `BAD_CHAR_THRESHOLD`; Tika ">10 unmapped glyphs" |
| column jump rate: consecutive lines in stream order that jump > 30% page width while overlapping vertically | `≥ 0.20` → MULTI_COLUMN | measured: 0.92 on the row-major fixture, ≤ 0.06 on every real page |
| numeric-only lines; horizontal rules | `≥ 10` lines or `≥ 3` rules → TABLE | measured: 12–73 on every table page, ≤ 8 on prose |
| image cover with text | `≥ 0.10` → FIGURE | — |

Everything else is PLAIN and stays on tier 0. Stage 2 gates the ML output and escalates one tier when it fails: word recall of the raw text layer `< 0.90` (layout analysis dropping blocks — pymupdf4llm has a documented "formula box drops its text" failure), table sanity (ragged rows, orphan `.` cells, headers cut into fragments like `D|ev|Te|st`, repeated header labels = two tables merged, text split across cells), empty output, Docling status. Dropped text never beats the raw text layer; a slightly broken table does beat flattened cells, so such a result is accepted with a note when no higher tier exists.

Alternatives considered: `find_tables()` as a signal — 100–300 ms/page and in pymupdf 2.x it runs the layout model by default, so it costs as much as the tier it would gate. Running the layout model on every page as the classifier — ~0.3 s/page on 100% of pages versus 15 ms; the GNN is the middle tier, not the triage. An ML page classifier — nothing to train on; five fixtures. Precedents: Unstructured's platform Auto (fast if embedded text and no images/tables), marker's per-page pdftext-vs-OCR check, Docling's per-cluster OCR, MinerU's txt/ocr classifier, Tika, Artifex's four buckets. Nearly all shipped routers are two-way (text layer vs OCR); routing into a third layout tier has little published precedent, which is why the gates exist.

## Backends and the gap each closes

| tier | backend | cost (here) | closes | does not |
|---|---|---|---|---|
| 0 | `plain` — `page.get_text()` + ligature fix | 3–6 ms/page | single-column and column-major streams (all LaTeX output) | tables (flattened), headings, scans, row-major streams |
| 1 | `layout` — PyMuPDF layout mode (`pymupdf4llm.to_markdown`, the GNN bundled in pymupdf ≥ 2.0), OCR off | 0.2–0.45 s/page, ~150 MB | row-major columns (30% → 98% sentence recall), headings, most tables (95–98% cells) | side-by-side tables, scans, formulas |
| 2 | `docling` — layout model + TableFormer + OCR (auto engine: OcrMac/RapidOCR) | 0.2–0.3 s/page digital, 3.3 s/page scan, ~1.8 GB | scans, broken text layers, complex tables (96–99.5% cells), best prose recall (+2 pts) | formulas (placeholder unless the 631 MB CodeFormula model is on), charts |

Why these and not others, with evidence:
- pymupdf4llm is **not** the cheap heuristic tier the brief describes any more. pymupdf 2.0 (2026-10-19) absorbed it and the layout analyser; the PyPI packages are empty shims; the project's `pymupdf>=1.26` already resolved to 2.1. The old heuristic mode (`pymupdf.use_layout(False)`) was measured: 100–150 ms/page, no table gain (2%/0% cells), worse prose recall, and it does **not** fix row-major streams (30.2%, same as plain). Rejected.
- Licensing: the standalone `pymupdf-layout` 1.x wheel was "Polyform Noncommercial or Artifex Commercial", closed source. In 2.x it ships inside the pymupdf wheel whose metadata says "AGPL-3.0 or Artifex Commercial", but the layout source is absent from the public repo. Unresolved; `PDF_ROUTER_LAYOUT_TIER=false` removes the tier and the router becomes plain → Docling. Docling and its default models are MIT / Apache-2.0 / CDLA-Permissive.
- Docling with OCR on adds nothing on pure-text pages (0.21 s either way; it only OCRs bitmap regions), while a second pipeline configuration costs another ~1.3 GB resident. One configuration, OCR on. TableFormer accurate vs fast measured within noise here; default kept.
- Not added: pdfplumber (MIT, but 5.6 s/doc at 74% text quality in py-pdf's benchmark, no column or heading logic), Unstructured fast (table score 0.000 on opendataloader-bench), marker/MinerU (RAIL $5 M cap / AGPL, GPU-oriented), Tesseract (binary dependency, silent degradation), a VLM tier (only thing that reads charts, ≈ $0.01–0.06 per page and 5–30 s inside a loop that fetches ~15 sources; the `PageClass.CHART` slot exists, tier 3 would be one class).

Interface: `PageBackend` = `name`, `tier`, `handles: frozenset[PageClass]`, `available()`, `parse(pdf_bytes, page_numbers) -> dict[int, PageResult]`. Pure function of picklable data, so the same class runs inline or in a sidecar process. Adding a parser is one class and one registry entry; the ladder is derived from `tier`.

## Parallelism

- Document-level concurrency already exists (15 scraper threads). Inside a document: triage and tier 0 run inline (ms). Tier 1 and 2 tasks go to **sidecar worker processes**: a light pool for the layout tier (`PDF_ROUTER_LIGHT_WORKERS`, default `min(4, cpus-1)`, one page per task) and a heavy pool for Docling (`PDF_ROUTER_HEAVY_WORKERS`, default 1, contiguous runs of ≤ 8 pages per task). The pools are process-global, so the worker counts are also the global caps on ML work across all 15 scraper threads.
- Why processes: PyMuPDF documents itself as not thread-safe and never releasing the GIL (measured: 8 threads → 1.66× on the layout tier); Artifex's own batch API is a process pool. Docling's threaded pipeline parallelises inside one `convert` call (torch releases the GIL), so per-run tasks are enough there.
- Why `spawn` explicitly: PyTorch's OpenMP runtime deadlocks after `fork` (issue #17199), Python 3.12 warns about forking a threaded process, Linux defaulted to fork until 3.14; Docling in Celery's fork pool (#1277) is this failure. Cost: a fresh worker re-imports and reloads models — ~1 s layout, ~7 s Docling.
- Constraints: each worker holds its own models (memory = workers × RSS); results cross a pipe (PDF bytes ≤ 1 MB per task, negligible against 0.3 s of work); pool state is per API process (multiple uvicorn workers each get their own pools). If sidecars cannot start at all (`__main__` not importable, subprocesses forbidden) the executor degrades once to serialised in-process execution with a warning, losing timeouts and crash isolation but not pages.

## Failure behaviour: fails, throws, hangs

- Throws: a `BackendError` (or any exception) in a backend marks those pages failed, the page falls to the next tier that handles its class or keeps the lower-tier result; a backend bug cannot take the document down.
- Hangs: per-task timeout (`PDF_ROUTER_LIGHT_TIMEOUT` 30 s, `PDF_ROUTER_HEAVY_TIMEOUT` 120 s) on the pipe; expiry SIGKILLs the worker and the next task spawns a new one. Docling's own cooperative `document_timeout` is set too, but it is checked between stages and cannot interrupt a stuck native call — the kill can. Threads cannot be killed; that is the reason for processes.
- Crashes: segfault / `bad_alloc` / OOM-kill ends only the worker (`BackendCrashed`); every Docling crash report found was native and would have killed the host process in-thread.
- Repeated failure: per-backend circuit breaker, 3 consecutive timeouts/crashes → open 60 s → one half-open probe. A bad PDF (`BackendError`) does not count.
- Budgets: `PDF_ROUTER_MAX_ML_PAGES` (40; kept pages chosen by class severity scan > broken text > table > multi-column > figure), `PDF_ROUTER_MAX_TRIAGE_PAGES` (150), `PDF_ROUTER_DOC_DEADLINE` (180 s; late pages keep their best lower-tier result). Idle workers are recycled after 10 min, and after 200 tasks, to give memory back.
- Never silent: a page that ends empty after we know it carries text gets `<!-- page N: no text extracted (reason) -->`; every page records its attempts; one INFO line per document summarises backends and classes. Encrypted PDFs return empty with a note; `Scraper` then rejects them as too short, as before.

## Verification

Metrics (`evals/pdf_router/`): sentence recall and 8-gram recall of body sentences from ar5iv's HTML of the same arXiv paper (math stripped, punctuation/hyphen/ligature-insensitive), share of ground-truth table cells found **inside Markdown table rows**, OCR characters, headings, seconds. Sanity check of the metric: `get_text(sort=True)` on the BERT paper (which interleaves its columns) scores 1.4% sentence recall against 86.0% for stream order, so the metric sees interleaving. Fifth fixture: `rowmajor-two-column.pdf`, generated by `tests/fixtures/pdf_router/make_rowmajor.py` from the BERT fixture's own text with a row-major content stream (what Word/InDesign/OCR produce), because the four LaTeX fixtures all have content streams in reading order.

Results (`python -m evals.pdf_router.run_eval --sanity`, backends inline and serialised, so router times are upper bounds; pooled runs were 7.5 s / 11.7 s for the two papers including worker cold starts; model warm-up 14 s once per process):

| fixture | pages | extractor | time | chars | sentence recall | 8-gram recall | GT cells in md tables | md tables | headings | backends used |
|---|---|---|---|---|---|---|---|---|---|---|
| two-column-paper.pdf (BERT) | 16 | previous | 0.07 s | 64 132 | 86.0% | 94.3% | 0.0% | 0 | 0 | — |
| | | **router** | 8.7 s | 70 232 | 86.0% | 94.2% | **96.0%** | 8 | 11 | plain 11, layout 3, docling 2 (2 escalated by gates) |
| | | `sort=True` (metric sanity) | — | 89 569 | 1.4% | 15.5% | 0.0% | 0 | 0 | — |
| borderless-table.pdf (RoBERTa) | 13 | previous | 0.04 s | 48 727 | 84.6% | 93.7% | 0.0% | 0 | 0 | — |
| | | **router** | 17.3 s | 58 445 | 85.0% | 94.0% | **99.0%** | 10 | 10 | plain 6, layout 4, docling 3 (3 escalated) |
| scanned-page.pdf | 1 | previous | 0.00 s | **0** (silent) | — | — | — | 0 | 0 | — |
| | | **router** | 5.0 s | **1 095** | — | — | — | 0 | 1 | docling 1 (`ocr_score` 0.98) |
| mixed-document.pdf | 5 | previous | 0.02 s | 16 666 | — | — | — | 0 | 0 | — |
| | | **router** | 5.3 s | 17 859 | — | — | — | 2 | 4 | plain 2, layout 2, docling 1 |
| rowmajor-two-column.pdf (synthetic) | 2 | previous | 0.01 s | 8 403 | **32.4%** | 36.5% | — | 0 | 0 | — |
| | | **router** | 1.1 s | 8 392 | **97.3%** | 99.0% | — | 0 | 0 | layout 1, plain 1 |

Per-page classes on the pack: BERT 11 plain / 5 table, RoBERTa 6 plain / 7 table, mixed 2 plain / 1 scan / 2 table, scan 1 scan, synthetic 1 multi-column / 1 plain (its second page holds one column). Gate escalations were all real: BERT p7 and p9 (two tables side by side rendered as one 8-column table with split digits), RoBERTa p8/p9/p13 (spanning header rows cut into cells such as `Single-task si | ngle models | on dev`). Triage cost 10–13 ms/page.

What improved: tables (0% → 95–99.5% of cells in Markdown tables), the scanned page (0 chars, no error → ~1.1 k chars with `ocr_score` 0.98), headings (0 → ~30), row-major columns (30% → ≥ 98% sentence recall), side-by-side tables (layout tier garbles them, the gate catches it, Docling renders them). What did not: prose recall on the LaTeX fixtures is unchanged within a point — their content streams were already in reading order and the old loader did not sort, so "columns interleave" does not occur on these files; the remaining ~13% of unmatched sentences contain formulas, identical for every extractor. How I know: the numbers above, per-page attempts in `evals/pdf_router/out/*.pages.json`, and 96 tests (`tests/pdf_router/`: triage on fixture pages, gates on real garbled output, router with controllable backends, spawned worker pool incl. timeout kill, crash, breaker, degraded mode, scraper wiring; `PDF_ROUTER_RUN_SLOW=1` runs the real ladder).

## Trade-offs

- Docling is an optional extra for library installs (`gpt-researcher[pdf-rich]`), a hard requirement in `requirements.txt`/Dockerfile (weights prefetched at image build). Without it the router logs once and runs two tiers.
- OCR lives in one place (Docling). pymupdf4llm would OCR via RapidOCR for ~1.5 s/page, but only because Docling installed RapidOCR; alone it prints a warning and returns nothing — the brief's "degrades quietly".
- Thresholds are constants in `RouterSettings`, env-overridable; no `Config` plumbing because scrapers are built as `Scraper(link, session)` and changing that signature is the interface the brief says to keep.
- No de-hyphenation on tier 0 ("pre-train" vs "pretrain" is ambiguous); ligatures only.
- The layout tier runs with `header=False, footer=False` (drops running heads, page numbers, the arXiv stamp); a misclassified body block would be caught by the word-recall gate.

## Another week

- VLM tier for `CHART` pages behind a flag, with a per-page cost budget; Docling's `do_picture_description` is the cheap local variant.
- Route on Docling's per-page confidence (`layout_score`, `ocr_score`) and expose router events through `emit_research_event` so fetch events carry backend/class counts.
- Warm the heavy pool at server start (`lifespan`) instead of on the first scanned PDF; cache routed output by URL hash across research runs.
- Adopt the router in the local document loaders behind a page-aware return type.
- Tune thresholds on a real corpus (DocLayNet / OmniDocBench pages) rather than five fixtures; measure Docling on x86 CPU.
- Formula handling: detect math-font glyph soup and at least mark it, or enable CodeFormula on demand.

## Unsure / weak

- Thresholds (`numlines ≥ 10`, `xjump ≥ 0.20`, word recall `0.90`) are tuned on five fixtures; false escalations cost ~1 s each, false negatives cost quality silently.
- Licensing of the layout model inside pymupdf 2.x is unresolved; I would confirm with Artifex before shipping the tier commercially.
- Docling production latency is from the 2.5.2 paper (x86 median 0.79 s/page, mean 3.1 s); not re-measured on Linux. One unexplained 168 s first-ever warm-up was observed once in a benchmark process; later cold starts were 5–7 s.
- Breaker and pools are per process; no shared state across uvicorn workers.
- The table-sanity heuristics are hand-written against the garbled outputs I saw; a different renderer's failure mode may pass them.

## Running it

```bash
pip install -e ".[test,pdf-rich]"          # or: pip install -r requirements.txt
docling-tools models download layout tableformer rapidocr
pytest tests/pdf_router                    # fast suite (spawns worker processes)
PDF_ROUTER_RUN_SLOW=1 pytest tests/pdf_router/test_integration_slow.py
python -m evals.pdf_router.run_eval --sanity --out-dir evals/pdf_router/out   # fetches ar5iv ground truth once
```
Knobs: `PDF_ROUTER_ENABLED`, `PDF_ROUTER_LAYOUT_TIER`, `PDF_ROUTER_DOCLING_TIER`, `PDF_ROUTER_INLINE`, `PDF_ROUTER_LIGHT_WORKERS`, `PDF_ROUTER_HEAVY_WORKERS`, `PDF_ROUTER_LIGHT_TIMEOUT`, `PDF_ROUTER_HEAVY_TIMEOUT`, `PDF_ROUTER_MAX_ML_PAGES`, `PDF_ROUTER_DOC_DEADLINE`, `PDF_ROUTER_MAX_PDF_BYTES` (see `settings.py`).
