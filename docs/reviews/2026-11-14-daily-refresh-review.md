# Daily Refresh Coverage and Efficiency Review

## Scope and Baseline

Review of the current working tree, including its existing unpublished strategy changes.
No model thresholds, market-data quality thresholds, publication pointers, or historical
data were changed for this review. Verification uses isolated tests and bounded benchmarks;
it is not evidence of a new full production refresh.

The last successful production report is `data/routine/latest_refresh_status.json`,
target 2026-11-13, 22:03:45 through 23:06:10 Asia/Shanghai (62m25s).

| Measured item | Duration | Interpretation |
| --- | ---: | --- |
| Sealed source export | 996.795s | 16m37s, about 26.6% of total; 5,373,048,898 bytes |
| Canonical export within sealing | 789.130s | Included in the export duration, not additional |
| Output model DAG | 865.596s | Parallel stage wall time, not sum of model CPU time |
| Long workspace | 597.740s | Late workspace; overlaps other late work |
| Similarity workspace | 1096.581s | Previously started only after selector outputs |
| Final publication validation | 30.700s | Includes retention, validation and commit |

UI step durations can include waiting for sibling operations. They must not be summed
or reported as pure computation time. Per-operation execution durations and checkpoint
reuse metrics are now retained in the feature/output DAG summaries.

## Page Coverage

| Page | Daily refresh ownership |
| --- | --- |
| Short strategies | Selector, left/right rankers and enabled application portfolio |
| Chan | Daily live scores and workspace snapshot |
| Long strategies | Long stock pools and blood-chip daily plan |
| Convertible bonds | Daily inputs, references and grid plan |
| Bond allotment | Events, daily-basic inputs and allotment workspace |
| Market transition | Shared daily inputs and transition workspace |
| Watchlist/similarity | Daily target context; reference vectors follow the existing weekly policy |
| Operation plans | User-owned CRUD records; intentionally not overwritten by refresh |
| Reviews | User-owned CRUD records; intentionally not overwritten by refresh |

`test_refresh_workspace_coverage.py` parses actual page markup and checks that every
non-manual page has a registered scope whose dependency closure is included in `all`.
New pages must either satisfy this contract or explicitly be classified as manual records.
This checks wiring, not the correctness of every financial/model feature.

## Findings and Changes

1. **P1: the output stage duplicated the dependency registry with a manual model list.**
   `production_output_node_ids()` now derives executable nodes from the requested closure,
   excluding completed operation ownership groups. New registered executable models are
   included without another web coordinator allowlist. Missing owners fail validation.
   Composition-owned source/page callbacks remain explicit; they are not automatically
   declared identity-audited or safe to cache.
2. **P1: an empty node selection expanded into the full graph.**
   Explicit empty selections must execute nothing; only `None` means the default graph.
   A regression verifies this through the real executor, not just the selection helper.
3. **P2: short-only refresh unnecessarily computed Chan scores.**
   Scope-derived selection eliminates this unrelated operation while retaining it in `all`.
4. **P2: similarity work waited for selectors and could invoke other workspaces to build badges.**
   Vector analysis now starts after source acquisition/sealing, under the shared CPU/IO/DB
   budget. It does not build strategy badges. After all strategy snapshots are written,
   the coordinator computes watchlist buy/hold scores from the now-current features,
   rejects missing scores, incomplete feature quality, mismatched score/quality dates or
   changed watchlist membership, and
   attaches same-date badges using read-only snapshot access. Missing or
   old snapshots cannot trigger another strategy computation or become current badges.
   Failure still prevents commit, and running workers finish before sealed inputs are released.
   Page GETs also use date-bound read-only badges, so a later short-only publication cannot
   attach a newer strategy hit to an older similarity forecast.
   Single-scope and resume paths use the same score finalizer. Both result and progress
   remain unfinished until finalization succeeds; entering the snapshot stage cannot
   implicitly mark similarity complete. The exception handler also no longer auto-completes
   prior pending steps; score-finalization errors mark similarity failed rather than snapshot.
   A failed finalization is not a reusable success.
5. **P2: late joining overwrote allotment quality evidence.**
   The collected workspace result retains its `quality` evidence through postflight.
6. **P1: aggregate dates did not prove all required sub-strategies were current.**
   Long products now validate named `tea`, `tea_safe`, `v44` results and the blood-chip
   plan; transition validates both `return` and `win`. A missing required date, duplicate
   named variant, or contradictory exact-date evidence cannot pass. Both transition
   variants are checked before successful summary evidence is returned; all summary
   paths preserve the variants. `needs_account` remains a valid data result, not a
   fabricated trading account or permission to place orders.
7. **P2: optional ETF state was absent from the page coverage inventory.**
   A `page_refresh_coverage` section now records its status, source dates, consumers and
   existing permitted lag. Missing ETF context still means unavailable panel/base ranking
   according to the existing producer/model policy, not a new blanket hard failure.
8. **P1: portfolio presentation imported infrastructure directly.**
   The application function now requires a path resolver injected by the web composition
   root. Generation-bound reads, date/release checks and ranking hashes remain enforced;
   the architecture boundary test was not weakened.
9. **P2: ordered child-task draining blocked refill behind a slow first task.**
   Bounded child mapping refills on completion, then restores input order in the result.
   It preserves the pending-task bound and resource grants, cancels pending work on failure,
   and joins running workers before releasing the parent grant.

## Bounded Snapshot Benchmark

The existing dirty implementation was saved before this review's edit. A fixed sample of
64 real symbols (164,233 rows, 8,320 monthly blocks) was compared before/after extracting
date boundaries once per frame instead of constructing two Series per monthly block.
Sorting, Arrow encoding, monthly contents, SQL isolation and integrity checks are unchanged.

| Measurement | Before median | After median | Result |
| --- | ---: | ---: | --- |
| Monthly block fingerprints, 8 alternating pairs | 1.372s | 1.050s | 23.5% lower locally |
| Isolated SQLite sample export, 5 alternating pairs | 2.919s | 3.135s | No demonstrated whole-export improvement |

All block identities, full manifests, Parquet hashes and read-back frames were equal.
The sample export timings are noisy and their median increased 7.4%; no production export
speedup is claimed. Raw timing series and version hashes are retained in
`reports/research/daily_refresh_review_20261114/snapshot_comparison.json`.
Temporary samples and the reproduction script are in `/private/tmp/quant-snapshot-perf-nIM4jz`;
they are not production caches or new long-term data partitions.

## Architecture Boundaries

- Keep one dependency registry and executable operation registry. Scope selection, progress
  metadata and page coverage derive from these contracts; UI labels must not block execution.
- Keep the existing shared resource scheduler. Starting work earlier does not authorize
  exceeding CPU, memory, I/O, database or provider quotas.
- Cache only on complete input, code, model and output identity. Existing composition
  callbacks still report `callback_identity_audited=false`; this review does not pretend
  that every callback has become a reusable incremental node.
- Full canonical SQL export remains a candidate for a future revision-aware incremental
  snapshot design. It requires a trustworthy source revision/change log and cross-run
  consistency proof. Reusing files by age or skipping content checks is not an acceptable
  shortcut and was not introduced.
- Local benchmark gains and increased concurrency are not promises of an equal reduction
  in next trading day's wall time. Input changes, weekly vector rebuilding, cache identity
  invalidation and resource contention still affect total duration.

## Verification

Integration regressions cover early vector execution, late read-only badges, stale-date
exclusion, no unrelated short-scope Chan work, single execution of multi-output owners,
missing-owner rejection, empty selections and shared source-pin lifetime on failure.
Commands used for integration validation:

```sh
PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q --tb=short -rs
/opt/homebrew/bin/node --test tests/frontend/*.test.js
git diff --check
```

Final results: **2,769 passed, 7 skipped, 10 subtests passed** in 168.64 seconds.
The seven skipped modules require PyTorch, which is not installed in this interpreter.
There were 188 warnings from pandas fragmentation, the SQLite date adapter and physical
CPU discovery; these are not counted as failures or suppressed by this review.
The focused web service regression passed 150 tests, including failure/progress
consistency and single-scope finalization. Frontend regressions passed 72 tests, and
`git diff --check` passed. These are test-environment checks, not a production refresh,
live browser run or deployed release. No commit or remote push was made in this review.

The resource-map skew benchmark uses four synthetic sleep tasks, two workers and
`max_pending=2`, comparing the previous FIFO drain with completion-driven refill over
five runs. The integrated rerun's median runtime was 0.289s versus 0.165s (1.75x). This demonstrates removal of
head-of-line idle time, not a measured production trading-day speedup. Reproduce with
`PYTHONPATH=src /Users/didi/miniforge3/bin/python tests/test_resource_scheduler.py`.
