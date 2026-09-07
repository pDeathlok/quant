# Live Feature Date-Scope Failure

## Incident

Run `all-20260907155824-385ee9b9` ran from 15:58:25 to 17:37:22
Asia/Shanghai and failed after 5937.724 seconds. The shared feature stage
reported `daily_basic` turnover coverage of 91.77%, below the unchanged 98%
threshold. The publication generation was aborted; its products were not
promoted to the current generation.

The source refresh had completed for 2026-09-07. Direct inspection of the
current B1, family, and Z gates found 1,018 unique target-date symbols, all
matching non-null turnover in the target-date raw daily_basic file.

## Root Cause

The strict operation adapter translated full cache invalidation into
`--incremental-start-date 19900101`, including for `--live-only` output.
The script selected all historical B1 candidate symbols and invoked the
training dataset builder with that start date. It did not restrict the B1
calculation or subsequent coverage check to the live output date.

The gate contained 189,170 historical B1 rows across 5,451 symbols, but only
132 B1 symbols on the target date. The training builder calculates B1 rows
from the requested start, not merely the exact keys in the gate. Thus live
refresh unnecessarily rebuilt historical samples and forward labels, and
applied a live quality gate to that expanded historical date domain.

The failed intermediate feature frame was not persisted, so the exact missing
key set behind 91.77% cannot be recovered from the manifest alone. A separate
read-only audit of the historical 2020-2026 gate found 1,491 missing turnover
matches, all in 2020-2022; that is not the same population as the failed
training-builder output. Historical source gaps must not be represented as
current-day completeness failures or filled with current values.

## Repair

- Production adapter passes the target trade date explicitly, even on a full
  rebuild. Invalidation controls reuse, not the output date domain.
- Live mode independently bounds all three gates to the canonical latest
  date and rejects a supplied target that differs from that source date.
- Latest-date discovery uses the store metadata API instead of materializing
  all market rows merely to compute a maximum date.
- One canonical factor build consumes the exact-date union across consumers.
  It preserves the existing six-year factor lookback, generates no training
  labels, and never rewrites the historical training cache.
- A lightweight raw-key preflight precedes heavy computation and validates
  each requested date separately. No historical row may substitute for a
  missing target-date row. Post-calculation coverage validation remains.
- Failure diagnostics distinguish absent source rows from null turnover and
  include bounded date counts and symbol/date samples. Joins enforce a
  many-to-one source key contract and do not retain stale daily_basic factors.

## Validation

- Expanded Python 3.13 feature/source/DAG/publication/Web regression selections:
  426 distinct tests passed, including 13 lifecycle regressions. The Web selection
  passed 153 tests; the final non-Web selection passed 301 (28 overlap).
- Python 3.9 snapshot and daily_basic selection: 75 passed.
- Ruff fatal/undefined-name checks passed for the changed source and tests.
- Read-only real-data parity: three B1 symbols, 110 market factor columns
  each, matched the existing training entry point with the same six-year
  window (`rtol=atol=1e-12`).
- Isolated real-data live refresh: success in 56.474 seconds with four workers.
  Source preflight: 1,018/1,018 matched. Output: 1,001 rows; 17 existing
  ST/delisting policy exclusions; zero unexplained missing candidates or
  calculation errors; all 145 required B1 feature columns present with no
  entirely null required column. Daily_basic enrichment read 33 of 3,322 files
  and matched 100%. Output SHA-256:
  `739cc1c0f942c8c51d36dd8e71a5561af8c39ef76644e67d1e7448c1d6de6850`.

This isolated measurement is not a production end-to-end benchmark. The
earlier failed shared stage took 2190.724 seconds under different system load
and with sealed inputs. Snapshot export (1247.411 seconds in that run) remains
a separate cost. A complete production refresh must still pass downstream
checks before being described as published successfully.

## Production Rerun And Lifecycle Repair

The single entry point `python3 scripts/run_daily_web_refresh.py` started
`all-20260907181127-69c4f9af` at 18:11:27. Its shared feature step succeeded
from 18:52:22 to 18:57:12 (290.373 seconds), without the previous coverage
failure. This is a real pipeline result, not the isolated benchmark above.

At 19:49:55 the Web status reader declared the still-owned job stale after
30 minutes without step progress. Its refresh thread and Chan child process
had not exited. Subsequent automatic retries collided with the publication
writer lock; the entry point exited 1 after three attempts, ending with
`all-20260907195248-62e50581` and `Another publication is already running`.
The affected run's manifest has no retained result payload, so its step
timings establish feature completion but not a detailed coverage count.

The lifecycle fix now:

- Tracks the owning thread across generation construction, computation,
  publication lock release, and retention cleanup.
- Does not let a status read expire an owner that is still alive. The former
  six-hour limit is surfaced as an execution warning while an owner remains
  active, rather than falsely claiming it has been cancelled.
- Emits an independent 30-second heartbeat without changing actual step
  progress timestamps or declaring a calculation successful.
- Blocks repeated queued submissions and refuses a new job while the old
  owner is unwinding, even if its internal status is already terminal.
- Makes the entry point wait for `job_active=false` before accepting a
  terminal result or retrying, and prevents an explicit restart while that
  owner remains active. Genuine process loss still follows interrupted-run
  recovery; a persisted heartbeat cannot prove an owner is alive.

At 20:19-20:21, read-only process/lock inspection still found Web PID 42543
holding the lock and waiting on Chan PID 2818, which was consuming CPU.
A native one-second stack sample was saved under
`/private/tmp/quant-refresh-owner-20260907.sample.txt`. Its generation remained
`staging`; `current.json` still referenced the legacy baseline. No lock was
deleted, no process was killed, and no extra refresh was submitted.

The lifecycle patch is tested on disk but is not loaded into this existing
Web process. Safely drain or explicitly recover the old owner before
restarting the Web service and rerunning the single entry point. End-to-end
successful publication is therefore still unverified. The Chan command's
1990 historical rebuild is an additional observed runtime cost, not a
validated optimization in this change. Thread liveness also does not prove
useful computational progress; cooperative cancellation or process-level
supervision would be needed for guaranteed recovery from a truly stuck
in-process computation.

## Follow-up: Chan Training Reference Destruction

The old owner eventually exited and released its lock. A fresh single-entry
run, `all-20260907203956-77c2cf99`, began at 20:39:56 with the lifecycle patch
loaded. It stayed correctly owned beyond the old 30-minute silence limit.
Right and left unified operations wrote successful checkpoints at 21:16 and
21:31. At 21:56:55 the run failed with a real Chan error, then the entry point
automatically retried after ownership release at 21:57:14.

The actual Chan error was `scored candidates must contain non-empty
pred_target_good` in `compute_chan_model_thresholds(reference_split="train")`.
It did not mean all live predictions were absent: the full 1990-start refresh
had overwritten the scored file with 157,235 rows all marked `live`, removing
every training row. The unchanged strategy output still contained 69,964
training rows. The independent `chan_model_dataset.parquet` remained intact
with 106,640 labeled train/test/OOT rows through 2026-05-25.

The follow-up script repair validates that independent reference before heavy
work, scores it with the current model bundles, and preserves its split/label
identity separately from live candidates. Live computation starts no earlier
than the day after the reference's final date. Full invalidation still
rebuilds every live date after that boundary, but cannot rewrite training rows
as live. A corrupted all-live scored cache is repaired from the independent
reference; no prior-day market values replace current-day values. Strategy
output validation now precedes overwriting the scored cache.

The repaired training thresholds exactly match the previous successful
strategy file: top50=0.30604133009910583, top80=0.39315930008888245,
top90=0.4561798572540283. Focused Chan/reference/research/F5 tests: 44 passed.
The script patch was saved and tested while the automatic retry was still
sealing inputs, before its next Chan subprocess launched. The training
reference is also included in the operation input identity. This registry-only
edit was applied after the active generation drained, so workers were not
invalidated mid-run. A regression verifies that modifying the reference
changes the checkpoint identity. Final expanded regression selection: 103 passed.

## Final Recovery Result

The single-entry runner exited 0. Final successful attempt:
`all-20260907215714-1a6614b7`, from 21:57:14 to 23:18:57 Asia/Shanghai,
4903.824 seconds, failed_count=0, error_summary=null. This timing is for the
successful attempt only, not the earlier failures and repair work.

All 19 steps succeeded. The strict postflight audited 36 dependency nodes,
reported zero failures and no pending refresh nodes, and committed generation
`all-20260907215714-1a6614b7-2fdd07d6` for 2026-09-07. The current publication
pointer was independently verified to reference that validated generation.

Market daily processed 5,558 symbols with no failures. Daily_basic was
official, through 20260907, and explicitly reported no missing data. Chan
completed in 375.130 seconds, preserving the 106,640-row training reference
and producing 2,673 live rows through 2026-09-07; its coverage status is valid.
The vector build staged all 5,511 symbols; the reference library refreshed at
23:04:48 and all 61 watchlist analyses reached 2026-09-07. Snapshot cleanup
and publication retention also reported success with no errors.

Automation `quant` was restored from temporary ten-minute follow-up to its
original Monday-Friday 15:55 schedule and original routine prompt. The
long-term daily automation remains active.

## Release Verification

The runner now checks for an existing listener before spawning a replacement
after a health timeout. A busy service retains its PID file and is monitored
without starting a duplicate process; this is not treated as proof of health.
The existing post-spawn race fallback and explicit restart path remain covered.

Only refresh-related changes were exported from the Git index into a clean
temporary directory, excluding concurrent model and research work. Python 3.13
passed 514 selected source, model, lifecycle, publication and checkpoint tests;
Python 3.9 passed 419 supported source/runner/contract tests. Two older Chan
tests now explicitly isolate MARKET_DATA_ROOT so CI environment defaults cannot
redirect their fixture reads. Syntax/undefined-name lint and Python 3.9 source
compilation passed. The whole research test suite was not run.

The new source coverage and identity regressions run in both CI versions.
Live model/reference and owner lifecycle regressions also run in the Python
3.13 job, which includes the supported strategy runtime.
