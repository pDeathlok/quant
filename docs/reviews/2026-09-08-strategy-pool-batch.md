# Strategy Pool Batch Reuse

## Implementation

The strategy-pool batch prepares exact-date features once for its candidate
universe. Every strategy still builds its own group flags, matched count,
historical quality metrics and model scores. The ALL pool's scores are never
copied into a filtered pool.

`SelectorRankingBatch` lazily loads each required side through its existing
full validator. Reuse is bound to the target date, configuration, publication
context and file identity. Input correction during a batch fails closed;
a subsequent batch reloads the corrected ranking files. Failures are not cached.
Neither features nor ranking inputs are retained by the new batch after return.
Existing independent callers retain their original scoring path.
Publication context is injected by the composition root; the application layer
does not import the infrastructure layer.

Snapshot results now include feature preparation, each strategy's processing,
total build, batch-write and total elapsed seconds. Finalization separately
records staged pruning, dependency validation and publication commit time.
Measurement fields are excluded from dependency content fingerprints.

## Read-Only Real-Data Benchmark

Date: 2026-09-08. Published generation:
`all-20260908155626-5e22786c-fad43708`.
All 14 strategies, 1,074 candidates, production models and local MySQL reads.
The baseline executes each pool independently; the optimized batch shares
inputs. Both run in the same process, with models loaded before timing.

| Measurement | Independent pools | Shared batch |
| --- | ---: | ---: |
| Build seconds | 32.398 | 7.669 |
| Feature preparations | 13 | 1 |
| Market range reads | 26 | 2 |

CHANGAN was empty and needed no feature preparation. All per-strategy payload
fields, including stock membership, scores, ranking, quality and provenance,
matched exactly after excluding generated timestamps. No production snapshot
write, refresh or publication was performed by the benchmark.

This is a 76.3% reduction for the measured build path. The production run's
earlier 565-second pool build used a sealed-file input path, whereas this
benchmark used local MySQL. Do not report 565 seconds to 7.669 seconds as a
measured production improvement. A production rerun is still needed to measure
the complete finalization stage, including writes and final gates.

## Verification

Focused tests cover shared/standalone parity, distinct per-strategy scoring,
same-day corrections, missing/stale/null inputs, empty pools, no mutation of
caller input, SQL batch semantics, date/config/publication/file guards and
unchanged content identity under different timings.

Final production-Python regression batch: 261 passed. Python 3.9 dependency and
identity regression batch: 34 passed. Ruff fatal-name checks and
`git diff --check` passed. The broader batch includes page APIs, publication,
snapshot SQL writes and architecture boundary checks.

No persistent cache or additional parallel worker pool was introduced.
The measurements above preceded deployment. Isolated release-tree validation
is recorded in `2026-09-08-refresh-optimization-release.md`.
