# Bugs & Iterations

## 2026-05-09: BUG-001 — LanceDB distance type implicitly L2 with normalized vectors

**Problem:** `memory/writer.py:_find_related_queries` reads `_distance` from LanceDB and computes `1 - distance` as cosine similarity. With normalized 384-dim MiniLM embeddings, LanceDB defaults to L2, which is NOT `1 - cosine_sim`; similarities are systematically off, shifting τ_link's effective cutoff.

**Root cause:** LanceDB's default distance metric is implementation-dependent and changes between versions. Code never set it explicitly.

**Fix:** New `gmemory/query_graph.py:nearest_neighbors` calls `.distance_type("cosine")` explicitly. Documented in `docs/research/PHASE_2_GMEMORY.md` §10. One-shot migration `scripts/migrate-distance-type.py` planned for v0.2 release to recompute `query_edges.similarity` for existing rows.

**Found in:** v0.1.0–v0.1.1.
**Fix lands in:** v0.2 Phase 2 (G-Memory Tier 2 build-out).
**Commit:** TBD (`gmemory/query_graph.py`).

## 2026-05-09: ITER-001 — tests/bench/__init__.py shadowed root bench/ package

**Problem:** Pytest collection failed with `ModuleNotFoundError: No module named 'bench.detectors'`. Cause: `tests/bench/__init__.py` made `tests/bench/` a package named `bench`, shadowing the real `bench/` package at the repo root during test imports.

**Root cause:** Created the test directory with `__init__.py` out of habit. Pytest doesn't need it for test discovery, and `__init__.py` in tests turns the test directory into an importable package that shadows real modules.

**Fix:** Deleted `tests/bench/__init__.py`. Pytest auto-discovery finds the tests without it.

**Lesson:** never create `__init__.py` in test directories that share a name with a real package.

**Commit:** v0.2.0-alpha.0 follow-up (Phase 0 foundation commit).

## 2026-05-09: ITER-002 — Zero-variance paired-diff edge case in stats tests

**Problem:** `test_paired_t_test_positive_difference` and `test_cohens_d_positive` failed because the original `a` and `b` arrays had constant pairwise differences (e.g., `[2,2,2,2,2]`). The stats module returns `t=0` and `d=0` when stdev of diffs is 0 (intentional div-by-zero guard).

**Root cause:** Test fixtures didn't reflect realistic input distributions. Constant diffs are a degenerate edge case that the stats code handles by returning 0; the tests assumed strictly-positive output.

**Fix:** Updated test fixtures to have non-constant positive diffs (e.g., `[1.5, 1.5, 2.0, 2.5, 2.0]`). Both tests now pass.

**Lesson:** when writing tests for paired statistics, vary the diffs even when modeling a "consistent positive shift" — real data never has zero variance.

**Commit:** v0.2.0-alpha.0 follow-up (Phase 0 foundation commit).

<!-- Format:
## YYYY-MM-DD: Short Title

**Problem:** What went wrong or needed changing
**Root cause:** Why it happened
**Fix:** What was done to resolve it
-->
