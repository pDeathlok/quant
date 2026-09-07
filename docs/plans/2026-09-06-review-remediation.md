# Architecture Review Remediation

## Preconditions

- Work in `/Users/didi/Project/quant`; preserve the existing dirty working tree.
- Test with `env PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest`.
- Use isolated fixtures for failures, deletion and SQL behavior; do not refresh or delete production data during development.
- Keep the current single-script routine entrypoint unchanged for users.
- Integrate independent edits before broad regression tests. No commit or deployment is part of this turn.

## Work Items

### Verified Status (2026-09-07)

- [x] F1 implementation and isolated publication/SQL/final-gate failure tests.
- [x] F2 strict canonical reads and outage/empty-source regression coverage.
- [x] F3 response ordering, generation pinning, bounded 409 recovery; desktop/mobile regressions.
- [x] F4 reference/lease-aware retention and any-weekday integrity repair.
- [ ] F5 full production closure: shared scheduling is wired; strict core checkpoints remain opt-in and require a real canonical pinned reader. Do not enable by inventing snapshot evidence.
- [x] F6 bounded finite rule windows, shared price work, causal append state correction, historical invalidation on contract changes.
- [x] F7 per-symbol vector identity, append reuse, crash-pending rejection/repair and compiled-generation retention.
- [x] F8 packaged production research implementations and static/dynamic import boundary tests.
- [x] Observed CI runtime constraints for Python 3.9 and 3.13; these are not portable hashed production locks.
- [x] Interim full suite: 1,643 passed in 126.71 seconds; final integration rerun is recorded separately.

The detailed acceptance descriptions below remain the scope, not a claim that
live deployment, production data repair, or all F5 migration has completed.

- [x] F1: Introduce committed publication generations in `/Users/didi/Project/quant/src/quant/infrastructure/`; integrate staged workspace/selector/long outputs and strict validation with `/Users/didi/Project/quant/src/quant/webapp/services.py`. Failed generations must not become visible. Verify interruption, failed SQL persistence, and concurrent reads with temporary stores.
- [x] F2: Make canonical SQL reads in `/Users/didi/Project/quant/src/quant/data/market_data_store.py` distinguish empty results from failures and forbid unverified mirror fallback. Verify same-date stale revisions, SQL outages, and empty responses in focused data-store tests.
- [x] F3: Protect all date-sensitive loaders in `/Users/didi/Project/quant/web/app.js` against obsolete responses, errors, and completion callbacks. Verify actual loader functions with delayed promises and a browser regression where available.
- [x] F4 and retention: Protect referenced/in-use production vector configurations in `/Users/didi/Project/quant/src/quant/routine/cache_retention.py`; separate missing/corrupt cache repair from the weekly schedule. Add artifact ownership and dry-run storage inventory. Verify exclusively against temporary directories with `tests/test_cache_retention.py`.
- [ ] F5: Complete identity/change propagation, safe checkpoint injection, registration validation and resource ownership in `/Users/didi/Project/quant/src/quant/routine/operation_contracts.py`, `operation_adapters.py`, `production_dag.py`, `dag_executor.py` and related tests. Production adapters must never claim a cache hit without complete input/output identities.
- [x] F6: Reduce redundant history/rule computation in `/Users/didi/Project/quant/scripts/research/rebuild_strategy_signal_cache.py` and its production rule modules. Preserve a full-rebuild oracle; verify append, corrected history, changed rule contracts and insufficient history with equivalent outputs.
- [x] F7: Replace global vector invalidation with source-specific identities and append-aware historical reuse in `/Users/didi/Project/quant/src/quant/research/similar_patterns.py`. Verify unchanged reruns, single-symbol revisions, append/full equality and source changes during construction in `tests/research/test_similar_patterns.py`.
- [x] F8: Move production-used research implementations into `/Users/didi/Project/quant/src/quant/research/`, retaining thin compatible script entrypoints; extend `/Users/didi/Project/quant/tests/test_architecture_boundaries.py` to dynamic imports. Verify existing long/signal tests and installed-package imports.
- [x] Integration: current implemented changes passed 1,653 full-suite tests; the final child-worker cap change passed 144 focused tests. Frontend desktop/mobile and three real-symbol incremental/full comparisons passed. F5 production cutover remains open, as documented in the September 7 results and rollout plan.

## Rollback and Publication

No destructive data migration is planned. New publication state must be versioned and preserve the prior committed generation. Abort retires only the new run's staging files for later eligible cleanup, never current production artifacts. Compatibility readers must not allow an aborted generation to reappear through legacy fallback. Retention defaults to protected references and conservative handling of unknown ownership.

## Acceptance

All reproduced F1-F4 paths must have regression coverage. Caching and incremental execution must use explicit source/contract identity and remain equivalent to full computation. Resource and retention decisions must be expressed through shared contracts rather than repeated workspace-specific exceptions. Record exact tests and measured performance; do not infer a full-market speedup from a single-symbol sample.
