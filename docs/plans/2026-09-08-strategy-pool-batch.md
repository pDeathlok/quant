# Strategy Pool Batch Reuse

## Scope and Sequence

1. In `/Users/didi/Project/quant/src/quant/webapp/services.py`, prepare exact-date
   feature rows once for the union of candidate symbols used by strategy pools.
   Pass them explicitly into each pool's existing scoring path. Keep group flags,
   matched counts, quality statistics, missing-value checks and per-pool ordering.
2. In `/Users/didi/Project/quant/src/quant/application/selector_ranking.py`, expose
   an explicitly scoped batch API that loads validated left/right ranking inputs
   once, rejects date/config mismatches and cannot reuse values across batches.
   Preserve existing standalone callers. Integrate the API only at the pool batch.
3. Record shared-feature, pool-scoring and batch-write elapsed time in refresh
   results, without changing the existing strategy-count return contract.
4. Add `/Users/didi/Project/quant/tests/test_strategy_pool_batch_reuse.py` for
   equivalence, per-strategy scoring differences, call counts, failed inputs,
   repeated batches and empty pools. Extend ranking tests for batch boundaries.
   Keep existing snapshot SQL batch, publication and freshness tests.
5. Run `env PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q
   tests/test_strategy_pool_batch_reuse.py tests/test_selector_snapshot_batch.py
   tests/test_selector_ranking.py tests/test_webapp_api.py`; all must pass.
   Benchmark actual saved candidate payloads without production writes, compare
   old and new pool results excluding only generated timestamps.

## Safety

No new persistent cache, no global date-only reuse, no skipped freshness gate,
no live refresh or server restart. Preserve unrelated dirty changes. Do not copy
ALL-pool scores into individual strategies: their input groups can differ.
Keep scoring serial initially; eliminate duplicate I/O before adding parallelism.

## Completed Verification

All implementation steps are complete. The full production-Python regression
batch passed 261 tests; the Python 3.9 dependency/identity batch passed 34 tests.
Ruff fatal-name checks and `git diff --check` passed. All 14 real-data strategy
pools matched the standalone path exactly, excluding generated timestamps.
Measured local-MySQL build latency decreased from 32.398 to 7.669 seconds.
See `docs/reviews/2026-09-08-strategy-pool-batch.md` for measurement boundaries.
