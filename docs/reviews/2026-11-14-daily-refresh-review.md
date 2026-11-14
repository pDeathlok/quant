# Daily Refresh Review: Isolated Release

## Release Boundary

This release is based on remote main `2e7670438b0d29f68dcb81fb1ef78133c8ec870f`.
It selectively ports the November 14 refresh review, rather than merging the local
branch or copying dirty files wholesale. The local unpublished commit `d24a634` and
unrelated working-tree changes are not part of this release.

Excluded: RankNet releases/configuration, application-portfolio implementation and
presentation, ETF-flow implementation, transition workspace, UI redesign, research
scripts, datasets, model artifacts and generated publication/cache files. The
portfolio path-resolver refactor and transition return/win validation remain with
their unpublished modules. Coverage in this release follows the existing remote
BYD workspace; it does not silently replace it with the local transition workspace.

## Changes

- Derive executable output nodes from the dependency and operation registries,
  rather than maintaining another model allowlist. Preserve the previous portfolio
  wiring fix when that node is registered. Validate missing operation owners.
- Treat an explicitly empty node selection as no work, not the entire graph.
- Do not compute Chan outputs for short-only refreshes.
- Start similarity vector analysis after sources are ready, using the existing
  shared resource budget. Join buy/hold scores only after dependent features and
  strategy snapshots finish. Require complete same-date scores/quality and stable
  watchlist membership before finalization.
- Keep similarity pending until finalization; failures cannot auto-complete prior
  unfinished progress steps. Single-scope and resume paths use the same finalizer.
- Read date-bound strategy badges from snapshots without recomputing missing
  workspaces or mixing dates during page GETs. A narrow `persisted_scores_only`
  display option also avoids rerunning historical score normalization or rankers.
- Preserve allotment quality evidence through the late join.
- Validate every named long-pool variant and blood-chip date. Reject missing
  required evidence dates, duplicate variants and contradictory exact dates.
- Record page coverage and per-operation execution metrics for diagnosis.
- Run the new coverage, scheduler and finalization regressions in Source Quality CI.
- Refill bounded child tasks when any task finishes, while preserving result order,
  resource limits, cancellation and join-before-release behavior.
- Read monthly snapshot date boundaries once per frame. Sorting, encoding,
  checksums and source consistency validation remain unchanged.

## Verification and Performance Limits

Validation is run inside the clean release worktree, without the primary workspace's
unpublished modules. The first unfiltered run had 2,016 passing tests and eight
asset-dependent failures/errors (three failures and five setup errors):

- Six good-company evaluation tests require untracked historical research inputs.
- One report-contract test requires the untracked `002595.SZ.md` report.
- One promoted-model test requires the untracked canonical v4 model/manifest.

All eight were reproduced on unchanged remote main `2e76704` in a separate clean
worktree. No test assertions were weakened, no source tests were marked skipped, and
no private datasets or models were added to the release. The remaining suite is run
with explicit deselection of only these baseline asset-dependent tests.

```sh
PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q --tb=short -rs
PYTHONPATH=src /Users/didi/miniforge3/bin/python -m pytest -q --tb=short -rs \
  --deselect=tests/research/test_good_company_deep_evaluation.py \
  --deselect=tests/research/test_individual_deep_report_contract.py::test_haomai_deep_report_only_has_expected_pending_archive_issues \
  --deselect=tests/test_left_side_unified_production.py::test_promoted_left_artifact_and_daily_registry_are_consistent
/opt/homebrew/bin/node --test tests/frontend/*.test.js
git diff --check
```

Final release-tree results: **2,061 passed, 8 deselected** in 125.94 seconds.
The new CI test group also passed with plugin autoload disabled: **274 passed**.
Frontend tests: **67 passed**. Python 3.9 compilation and `git diff --check` passed.
Ruff is not installed in the local verification interpreter; the existing CI lint
step remains enabled. Existing warnings were not suppressed (pandas fragmentation,
SQLite date-adapter deprecation and physical CPU discovery).

Prior bounded experiments found monthly fingerprint runtime decreased from 1.372s
to 1.050s on a fixed sample; isolated full-export median did not improve (2.919s to
3.135s). All compared manifests, hashes and frames were equal. A synthetic two-worker
skew benchmark decreased from 0.289s to 0.165s after completion-driven refill. Neither
measurement is a production-day speedup claim.

Full canonical SQL sealing remains a substantial bottleneck. Revision-aware
incremental export needs a trustworthy change journal and consistency proof, so it
is not introduced by this release. Existing callback cache-identity boundaries and
all data-quality/publication gates remain in place. No live refresh, data publication,
or service restart is performed as part of this Git release.
