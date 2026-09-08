# Daily Top List and Snapshot I/O

## Scope

1. In `/Users/didi/Project/quant/src/quant/application/daily_dependencies.py`
   and reference refresh code, ensure production top-list consumers activate
   daily polling. Persist actual request evidence, including valid empty-event
   days, and reject stale or failed sources before publication.
2. In `/Users/didi/Project/quant/src/quant/data/market_snapshot.py`, capture
   canonical supplemental Parquet bytes without decoding/re-encoding every
   value column. Verify the copied content and source stability; retain the
   normalization path for noncanonical schemas/dates or filtered exports.
   Bound concurrency and preserve deterministic manifest ordering.
3. In `/Users/didi/Project/quant/src/quant/routine/production_dag.py`, surface
   separate capture/verification timing and file counters in operation metrics,
   never in content identity. Do not remove freshness or integrity gates.
4. Add mutation, missing-date, empty-event, output parity and worker-bound
   regressions. Benchmark a fixed real-data sample using temporary output only.
   Do not change raw data during performance measurements.

## Verification

Run `env PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q
tests/test_market_snapshot.py tests/test_daily_dependencies.py
tests/test_daily_dependency_runtime.py tests/test_reference_data_refresh.py`;
all selected tests must pass, with new focused suites included as created.
Run equivalent supported tests under Python 3.9 and Ruff fatal-name checks.
Report measured sample latency, not an extrapolated whole-refresh promise.

## Safety

Existing unrelated dirty files remain untouched. No SQL schema changes and no
deletion of historical raw data. A failed capture removes only its own temporary
staging directory and never advances a publication. Whole-market SQL export and
full-rebuild signal policy are outside this narrow small-file optimization.

## Progress

- Complete: unconditional daily source dependency and canonical live reader path.
- Complete: independent-byte capture, bounded concurrency, shared distinct-date
  normalization and stage metrics outside content identity.
- Complete: full row parity on a fixed 600-file sample; 216.510 to 3.811 seconds.
- Complete: bounded historical catch-up and retry-persistent missing-date
  handling for top-list inputs. Final combined regressions passed 235 tests on
  Python 3.13 and 201 data-side tests on Python 3.9.
