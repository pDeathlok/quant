# Refresh Optimization Release

## Scope

Prepared on an isolated worktree based on main `1a328467`.
Includes daily top-list polling and freshness receipts, supplemental snapshot
I/O and date normalization, strategy-pool input reuse, finalization timings,
and transient refresh-monitor transport recovery.

Unpublished left-side V5, holding-model, overnight-strategy and unrelated UI
research changes in the working checkout are not part of this release.

## Release-Tree Validation

- Python 3.9: 515 source and refresh-contract tests passed.
- Python 3.13: all workflow test groups, 665 tests passed.
- Python 3.13: 327 additional focused/page API tests passed.
- Ruff fatal syntax/name checks and `git diff --check` passed.

These test batches overlap; the numbers are not unique-test counts.
Page tests now create temporary calibration artifacts and explicitly stub the
blood-chip plan dependency instead of requiring untracked production data.
The calibration test also verifies rejection after artifact content changes.

Measured read-only benchmarks are documented in the accompanying snapshot I/O
and strategy-pool reviews. No complete production refresh was rerun for release
validation. Full-run latency and the new finalization subtimings remain to be
measured by the next routine run.
