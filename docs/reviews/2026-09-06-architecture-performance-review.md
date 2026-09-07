# Quant Architecture and Performance Review

Reviewed: 2026-09-06, Asia/Shanghai.

## Scope and Confidence

- Reviewed the current working tree on `main`, based on commit `f4bb25731102c00a4ce6a281fc8cbf08016a60b5`, including existing uncommitted work. This is not a review of a clean release checkout.
- Focus: daily orchestration, canonical storage, feature and vector computation, snapshot publication, page loading, cache retention, and architecture enforcement.
- Evidence: production manifests and artifacts, source inspection, 37 focused passing tests, isolated fault/race probes, and a one-symbol incremental profile.
- No production refresh was started, no real caches were deleted, and no application code was changed. This document is the only repository addition from this review.
- These findings identify reproducible failure paths and optimization opportunities. They do not establish that the currently published September 4 data is incorrect. That run's recorded postflight freshness audit passed.
- This was not an exhaustive audit of every model's statistical validity, backtest leakage, security, live SQL query plans, or every current dataset row.

## Findings

### F1. P1: Global validation runs after live publication

Evidence: [refresh completion](/Users/didi/Project/quant/src/quant/webapp/services.py:10585), [selector writes](/Users/didi/Project/quant/src/quant/webapp/services.py:5220), [workspace writes](/Users/didi/Project/quant/src/quant/infrastructure/workspace_snapshots.py:197).

The refresh writes selector snapshots and runs post-snapshot cleanup before calling the strict global postflight dependency gate. Other workspaces publish independently earlier. A later failure marks the task failed but does not restore the previously published generation. Selector row-level date/quality checks are present, but they do not provide all-workspace atomicity. Filesystem snapshots are also written before the SQL transaction; the generic workspace repository suppresses SQL write errors.

Consequence: a failed run can leave readers seeing a partially updated collection of products, or different versions depending on which storage backend serves the read.

Fix: write run-scoped staging artifacts; validate the entire staged dependency closure; publish a committed generation pointer only after validation and canonical persistence succeed. Pin reads to a generation, and retain the previous generation until commit. A per-file atomic rename is not a transaction across files and SQL.

Acceptance: inject upstream staleness, SQL write failure, and interruption between writes. Concurrent readers must continue to see the previous complete generation, and no failed generation may become the default.

### F2. P1: SQL failures can silently become reads from an older mirror

Evidence: [exception conversion](/Users/didi/Project/quant/src/quant/data/market_data_store.py:529), [range fallback](/Users/didi/Project/quant/src/quant/data/market_data_store.py:175), [single-symbol fallback](/Users/didi/Project/quant/src/quant/data/market_data_store.py:152).

The SQL range reader catches query exceptions and returns an empty DataFrame. With mirroring enabled, callers then read Parquet without checking that it represents the same canonical revision. An unavailable source and a successful empty result are indistinguishable. The returned frame carries no enforced source/revision/degraded contract.

An isolated probe injected a SQL query exception and supplied a September 3 mirror. The ordinary range API returned the September 3 row without raising. Date checks downstream may catch an older day, but cannot establish equality of two different revisions on the same day.

Fix: distinguish successful empty results from source failures. Production canonical reads should fail closed unless a mirror is explicitly verified against the required revision. Propagate source identity and quality status through materialized outputs.

Acceptance: cover source outage, legitimately empty data, stale-date mirrors, same-date stale revisions, and verified matching mirrors.

### F3. P1: Date-switch response races can show an old result under a new date

Evidence: [long loader](/Users/didi/Project/quant/web/app.js:1759), [Chan loader](/Users/didi/Project/quant/web/app.js:1999), [existing selector request guard](/Users/didi/Project/quant/web/app.js:865).

The long loader checks only the variant before accepting a result; the Chan loader assigns the response unconditionally. Neither protects against two requests for different dates arriving out of order. Loading and error state have the same obsolete-request problem.

Executing the actual loader functions in an isolated Node context reproduced both cases: request September 3, switch to September 4, finish September 4 first, then September 3. Final state: selected date September 4, payload date September 3.

Fix: share a loader pattern with immutable query identity and a monotonically increasing request ID. Ignore outdated success/error/finally callbacks; abort superseded requests where practical. Apply this to other date-sensitive workspaces, including convertible bonds, rather than only these two functions.

Acceptance: delayed-response browser tests for repeated date switches and variant switches must never render an obsolete response as the current selection.

### F4. P1: Retention can delete the active vector configuration; repair then waits for Friday

Evidence: [vector deletion](/Users/didi/Project/quant/src/quant/routine/cache_retention.py:1012), [refresh decision](/Users/didi/Project/quant/src/quant/webapp/services.py:567).

Vector configuration directories are ordered by directory modification time and only the newest is retained. There is no check for the active production configuration or an in-progress reader/build. A newer experimental directory can therefore displace the production directory.

The refresh decision detects a missing cache, but the non-Friday branch takes priority and returns `waiting_for_friday_close`. Damage repair is effectively gated by the normal weekly schedule.

Both paths were reproduced using temporary directories and mocked metadata: the production directory was removed while the experimental directory survived; a Monday check with zero cached files returned `due=false`.

Fix: pin the active production configuration, last committed generation, and in-use builds/readers. Separate integrity repair from periodic refresh policy. Missing mandatory references must either be repaired or reported as unavailable, never accepted as a healthy scheduled skip.

Acceptance: experiments cannot evict production; interrupted builds are not selected as the retained configuration; missing/corrupt production caches trigger repair or an explicit blocking failure on every weekday.

### F5. P2: The DAG contract is not yet the single production execution authority

Evidence: [registered adapters](/Users/didi/Project/quant/src/quant/routine/default_operations.py:52), [executor facade](/Users/didi/Project/quant/src/quant/routine/production_dag.py:48), [output DAG construction](/Users/didi/Project/quant/src/quant/webapp/services.py:10255), [adapter results](/Users/didi/Project/quant/src/quant/routine/operation_adapters.py:25).

There are 51 dependency nodes and 44 operations, but only 5 operations have executable production adapters; the others use `shadow_only`. The live feature/output DAG constructors do not inject a checkpoint store. Adapters return node payloads and worker metrics without filling changed partitions/keys, dataset revisions, or output fingerprints. Some initial dirty dates are supplied to the feature stage, but end-to-end change propagation is incomplete.

Local caches and manual checkpoint reuse already exist. The gap is that adding an operation to the registry does not yet guarantee resumability, correct invalidation, or centralized resource control. Web orchestration still owns early/late thread pools, and some workspaces create nested workers outside the partial DAG budget.

Fix: migrate the active production closure to executable operations under one scheduler. Require explicit input identity, output identity, dirty partitions/keys, and resource claims. Wire checkpoint reuse only after these identities are populated; merely enabling a checkpoint store with incomplete upstream identity can create stale hits. Give downstream workers scheduler-granted capacity and bounded queues.

Acceptance: registry checks reject active shadow-only nodes and incomplete identities. A same-input rerun does no heavy work; revising one source partition invalidates exactly its declared dependents. Peak CPU, RSS, DB connections, and API concurrency stay within measured budgets, including child workers.

### F6. P2: Incremental base factors still feed broad historical rule recomputation

Evidence: [600-day input context](/Users/didi/Project/quant/scripts/research/rebuild_strategy_signal_cache.py:914), [market read](/Users/didi/Project/quant/scripts/research/rebuild_strategy_signal_cache.py:976), [rule evaluation before result filtering](/Users/didi/Project/quant/scripts/research/rebuild_strategy_signal_cache.py:743).

The base factor layer supports stateful append, but the rule stage still processes broad per-symbol history and trims output to the replacement interval afterwards. Market reads do not project only the necessary columns. History normalization, adjusted-price construction, DataFrame allocation, and family/extended rule evaluation remain substantial work even for a one-day append.

September 4: 5,570 symbols, 222 factor cache hits, 5,242 incremental factor updates, no symbol errors. The signal subprocess took 1,074.179 seconds: computation 951.855 seconds, market read 102.080 seconds, cache merge 4.026 seconds, cache write 2.657 seconds. Cache writes are not the main bottleneck.

Local sample: 407 rows for `000001.SZ`, seed 406 rows then append one day. The mode was `incremental`; a profiled run took about 0.20 seconds, including family rules 0.088 seconds, extended rules 0.049 seconds, and factor attachment 0.049 seconds. Six continuous-OHLC constructions were observed. This is diagnostic evidence, not a full-market speedup forecast.

Fix: define per-feature/rule lookback and state semantics; use source change journals and dirty symbols; share normalized price arrays; process bounded affected windows instead of repeatedly evaluating unaffected history. Preserve full rebuild as the correctness oracle.

Acceptance: incremental/full equivalence for append, historical corrections, deletions, corporate actions, suspensions, and newly listed stocks. Measure wall time, rows read, allocations, and evaluated symbol-days under identical input versions.

### F7. P2: Vector invalidation is market-wide and rebuilding is not append-aware

Evidence: [whole-source identity read](/Users/didi/Project/quant/src/quant/research/similar_patterns.py:583), [global fingerprint distribution](/Users/didi/Project/quant/src/quant/research/similar_patterns.py:895), [cache validity and reconstruction](/Users/didi/Project/quant/src/quant/research/similar_patterns.py:789), [engine construction](/Users/didi/Project/quant/src/quant/data/market_data_store.py:371).

One global daily-data fingerprint is passed to every per-stock vector cache. Any source change invalidates all those caches. A miss loads full stock history and rebuilds all historical candidate vectors. The fingerprint itself reads and hashes the full SQL source. Repeated per-symbol reads construct/dispose SQL engines rather than exploiting a process-owned reusable pool.

The September 4 analysis records 5,131 rebuilt caches, zero reused, zero errors. The similar-pattern workspace occupied 2,685.269 seconds on the critical tail. Its saved results already use compiled matrix chunks over 1,666,048 candidates, so changing nearest-neighbor technology should not be the first intervention.

Fix: use per-symbol/partition revisions; preserve unchanged historical vectors; append newly eligible windows and mature forward labels; invalidate affected context on historical corrections. Batch source reads with column projection and bounded chunks. Use process-owned connection pools, not engines shared across forks.

The reference library is intentionally weekly while target vectors use current daily data. An older historical reference is not itself stale current-market input. Keep these two freshness policies explicit, including the source revision used by a committed weekly build.

Acceptance: an unchanged rerun rebuilds zero symbols; a single-symbol correction does not invalidate unrelated symbols; appended windows and labels match a full rebuild; a source revision change during construction prevents mixed-version publication.

### F8. P2: Architecture checks do not enforce the actual production/research boundary

Evidence: [dynamic research module loading](/Users/didi/Project/quant/src/quant/webapp/services.py:1144), [AST guard](/Users/didi/Project/quant/tests/test_architecture_boundaries.py:12).

The Web service dynamically executes research scripts with `spec_from_file_location` and `exec_module`, including changes to `sys.path`. The architecture test only examines `Import` and `ImportFrom`, so it passes while these production dependencies remain. The Web service is 10,942 lines and owns orchestration, storage interaction, model integration, and payload assembly; this makes cross-layer guarantees harder to apply consistently.

Fix: keep the modular monolith, extract production use cases and computations into package modules, and make research scripts thin entrypoints. Extend architecture tests to dynamic imports and the executable production dependency closure. Avoid a microservice rewrite: it would add deployment and distributed-consistency costs before fixing current boundaries.

Acceptance: production execution works from the installed package without dynamically loading repository research scripts. New models use the same operation, feature, snapshot, and quality contracts.

## Additional Improvements

### Artifact ownership and storage budgets

`du -k -d 1 data reports` measured about 30.32 GiB under `data` and 9.04 GiB under `reports`, approximately 39.36 GiB allocated combined, excluding MySQL and Python environments. `data/research` accounts for about 23.22 GiB. Sizes can change while other work is running.

The [retention inventory](/Users/didi/Project/quant/src/quant/routine/cache_retention.py:111) enumerates selected paths; it does not establish ownership/reachability for every research dataset and model artifact. The September 4 post-snapshot pass reduced measured managed allocation by about 19.37 MiB. This is evidence of limited cleanup scope, not proof that all other files are disposable.

Introduce an artifact registry with producer, input versions, active consumers, size, last access, rebuild cost, retention class, and a protected flag. Preview deletions against production/model/report reachability before applying quotas or TTLs. Do not indiscriminately purge raw history, active model inputs, or reproducibility evidence. Report logical and allocated bytes separately, including hardlink effects.

### Reproducibility and instrumentation

The [package definition](/Users/didi/Project/quant/pyproject.toml:8) largely declares lower dependency bounds. No repository dependency lock or GitHub Actions directory was found in this inspection. External CI may exist and was not inspected. Record the exact environment, code/model/feature contract versions, and input revisions per run; add a reproducible runtime and automated quality gates.

Record separate queue, source I/O, computation, persistence, validation, and publication times per operation, plus rows/bytes scanned, cache miss reasons, peak RSS, and child-worker usage. Current UI step durations can include waiting for a shared stage and are not interchangeable with subprocess CPU/computation time.

## Measured Baseline

Source: [September 4 successful run manifest](/Users/didi/Project/quant/data/routine/20260904_194133_all-20260904194133-e2d2f2dd/manifest.json), [saved similar analysis](/Users/didi/Project/quant/data/research/similar_patterns/web_watchlist_analysis.json).

Run: September 4, 19:41:33 to 21:05:10, approximately 83 minutes 37 seconds. Success with no recorded postflight freshness failures.

| Work | Recorded wall duration | Interpretation |
| --- | ---: | --- |
| Similar-pattern workspace | 44m 45s | Long critical tail; weekly reference rebuild on this Friday |
| Signal subprocess | 17m 54s | Of this, computation 15m 52s and market read 1m 42s |
| Signal UI step | 21m 05s | Includes shared-stage completion; not just signal computation |
| Shared project features | 3m 11s | Follows signal output in this stage |
| Convertible-bond allotment | 12m 18s | Overlaps the signal stage; not all additive to total |
| Long stock pools | 10m 23s | Overlaps the similar-pattern tail |
| Core selector | 6m 38s | Before the late workspace stage |
| Final snapshot step | 1m 59s | Includes more than a single file write |

Do not sum these durations: several steps overlap. One Friday run is not a weekday performance distribution, and it cannot establish whether factor-count growth caused a historical regression. That requires comparable manifests and identical-input benchmarks. The current review does establish remaining repeated computation despite incremental base factors.

## Recommended Sequence and Extension Contract

1. Close correctness gaps F1-F4 first: generation publication, strict source identity, response ordering, and safe retention/repair. Add fault-injection tests before optimization.
2. Complete production operation contracts and one scheduler (F5), with a small representative operation migrated end to end before broader rollout.
3. Optimize vector and signal work (F7/F6) against immutable input snapshots; use full rebuilds as equivalence oracles. Benchmark unchanged reruns, normal append, and historical corrections separately, on both weekly-rebuild and ordinary days.
4. Consolidate production/research boundaries (F8), add artifact reachability/budgets and reproducible runtime checks. Migrate incrementally, not through a big-bang rewrite.

Every new feature/model operation should declare: inputs and source revisions; code/schema/model/parameter identity; target-date and historical/PIT freshness policy; required lookback and append/correction behavior; partition/symbol change propagation; CPU/RSS/I/O/API claims; outputs and lineage; retry/idempotency policy; retention ownership; and full-versus-incremental equivalence tests. Missing declarations should fail registration or CI, not silently disable caching/incrementality.

## Validation Performed

Command (from project root):

```sh
env PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q \
  tests/test_architecture_boundaries.py tests/test_daily_dag_executor.py \
  tests/test_default_daily_operations.py tests/test_selector_snapshot_batch.py \
  tests/test_workspace_snapshots.py tests/test_cache_retention.py
```

Result: **37 passed in 3.71 seconds**.

Isolated probes: [Python storage/retention/profile probe](/private/tmp/quant_review_probes_20260906.py), [JavaScript request-race probe](/private/tmp/quant_review_frontend_race_20260906.cjs). Both completed successfully. They used temporary cache directories or mocked services; the profile read one symbol's local history. Temporary probe files are not production tests and may be removed by the operating system.

Not performed: full test suite, a new daily refresh, real database failure injection, comprehensive browser navigation, live peak-memory profiling, SQL EXPLAIN analysis, or benchmark verification of proposed speedups. Passing existing tests does not cover the reproduced failure paths; regression tests must be added when implementing fixes.
