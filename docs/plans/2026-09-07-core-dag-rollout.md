# Core DAG Rollout and F5 Integration

Date: 2026-09-07. Status: executable composition integrated; full identity/resource audit remains partial.

This supersedes the earlier shadow-only proposal. The F5 integration owner did
not run a production refresh or perform Git mutations. The main task owns
candidate-index validation and the user-authorized commit.

## Production Composition

- Unset `ROUTINE_DAG_EXECUTOR` selects `core` for `all`/`short`. Other scopes
  retain their existing orchestration. Explicit `shadow`/`legacy` are rollback
  paths; `enabled` rejects a claim of fully identity-audited cutover.
- The publication wrapper and strict postflight gate remain the publication
  boundary. One scheduler serves source callbacks, core, and early/late workspaces.
- `validate_composed_closure()` rejects unregistered active operation owners
  before source writes for `all`/`short`; CI exercises the same guard.
- The current graph has 35 active operation IDs for `all`, 16 for `short`.
  The earlier review's 44 also included inactive/research/shadow declarations.
- The core five use the central checkpoint store: signals, shared project
  features, right-side unified, left-side unified, and Chan live features/scores.
- Thirteen real callbacks execute through the DAG with explicit `CacheMode.NONE`:
  daily, daily-basic, reference refresh, daily plan, selector, strategy snapshots,
  long-factor snapshot, long variants, Chan workspace, convertible-bond grid,
  allotments, BYD, and similar-pattern analysis.
- Composite callbacks own internal graph nodes and invoke their existing routine
  once. They do not fabricate an independent cached success per internal node.
  Exceptions, explicit failed/shadow results, and missing results fail execution.
  Existing domain validation and postflight still validate actual products.
- `operation_execution_audit` reports executed core operations, actual callback
  groups, member coverage, and missing owners. Executable closure is distinct
  from hidden-input auditing: `callback_identity_audited=false` and
  `full_production_closure=false` remain explicit.

## Sealed Source Lifecycle

1. Finish source refresh and preflight before acquiring the full-history export.
   A caller-provided pre-refresh revision is not the acquisition protocol.
2. Export canonical `daily` under the data owner's consistent SQL snapshot or
   explicit file authority. Daily-basic/top-list are explicit supplemental file
   authorities, never fallback for missing SQL tables.
3. Enter `pinned_market_snapshot()` and the matching identity scope before early
   workspace submission, following source refresh/preflight. Parent market-store reads use the descriptor. Native workers
   receive the complete `pinned_market_environment()` factory result, including
   the manifest and the expected fingerprint supplied by the data owner.
   The composition checks both the manifest's content fingerprint and the acquired
   pin against the original export descriptor before permitting consumption.
4. Variable-library daily-basic features, market-sentiment top-list features,
   and Chan turnover history use the snapshot reader. Chan model daily-basic
   features use `read_pinned_parquet()` for sealed files.
   The data owner validates and reads with the same handle. Native per-date
   arguments are explicitly remapped; canonical daily is never disguised as
   the old monthly/per-symbol file layout.
5. Strict native factor caches use a fresh run-owned `derived_factors` directory.
   A global cache validated only by date coverage cannot establish sealed-source
   equivalence. Signal source identity uses the sealed daily entry in pinned mode.
6. The same pin covers early workspaces, core, selector, and late long/Chan/similar
   consumers through the final publication gate. Context-copying thread pools
   carry it into early/late tasks and resource-grant workers. Similar's spawned
   coordinator receives the complete environment factory result; its vector
   workers receive already-read frames. Its private SQL/file vector source paths
   explicitly use the sealed reader under a pin, even with a file backend.
7. Only the outer `finally` releases the pin, after early/late futures and native
   children join. Similar's queue/progress failures terminate and join the child
   before returning. Temporary row exports and derived factors are removed on
   normal/failure exit; no per-day multi-GB export retention is introduced.
8. Long and selector daily-basic and allotment share-capital reads use declared
   sealed files with `read_pinned_parquet()`. Long's date-only in-memory cache now
   includes the source fingerprint; its unverified historical feature cache is
   not reused in pinned mode. BYD, grid and allotment canonical source errors
   cannot fall back to live data under a pin.
   Long period caches use a stable `daily_basic:sealed` namespace and per-file
   manifest relative path/SHA identities, never the temporary export path or mtime.
   Daily monthly-feature caches use sealed daily content plus semantic parameters.
   Identical exports at different roots reuse both caches; a supplemental
   correction refreshes only its affected period.

The workspace audit found no public `MarketDataStore.write_frame` or
`write_market_batch` calls in these late/early workspaces. Grid's concrete writes
are `_merge_parquet()` under `data/convertible_bond/tushare`; allotment/BYD vendor
polling and workspace-snapshot repository writes remain independent callback
inputs/outputs, not canonical market datasets. Their underlying stock prices
still use sealed `daily`, and share capital uses sealed `daily_basic`. Reference
files (financials, stock metadata, indices) are refreshed before acquisition but
are not claimed to be part of the three-dataset sealed market contract.

## Identities and Incrementality

- Each dataset hashes its own sealed entry, not the whole export. A daily-basic
  correction does not invalidate daily-only strategy signals.
- After successful consumption (including late workspaces for `all`), only the small content manifest is written
  to `data/cache/routine_source_manifests/core.json` inside the publication
  generation. Uncommitted/failed postflight cannot advance the comparison baseline.
  Saving uses the active fingerprint-validated reader, not a fresh unverified path
  read; this validation also runs when all core handlers were checkpoint hits.
- Stable per-symbol SQL files carry monthly row-content hashes in
  `date_blocks_schema=1` metadata; this does not create one file per date/month.
  Journals compare blocks inside changed files, so a daily append repairs the
  changed month rather than the symbol's entire history. Historical corrections
  and deleted blocks retain their old/new earliest affected dates and symbols.
  Identical content gives empty changes; missing baseline, changed columns or
  coverage/universe, and incomplete chunk metadata request full rebuild.
- Journal comparison checks block row-count/date/symbol coverage and never rereads
  parquet. Missing, incompatible, or incomplete block evidence falls back to the
  complete old/new chunk range. Unchanged chunks do not dirty other symbols.
  Fixture journals do not prove full-market export/hash I/O efficiency.
- Forward target-date changes are not code/model contract changes. Rebuild
  decisions still require complete journals. Backward dates, missing/corrupt
  outputs, and changed code/config/model contracts remain conservative.
- Core manual feature-stage resume cannot bypass central identity checks.
  Verified output snapshots and change journals cross the feature/output boundary.
- Non-core callbacks intentionally do not use central reuse. Existing manual
  cache semantics are not promoted to a complete hidden-input contract.

## Cost and Remaining Limits

`market_snapshot.export_seconds` and `export_bytes` expose the full-history
transport cost. Even an unchanged run exports/hashes sources before a heavy
handler can be skipped. No measured full-market speedup is claimed.

Resource claims are cooperative CPU/IO/DB/API/memory reservations, not measured
process-tree RSS or OS limits. Numerical thread limits and worker forwarding do
not prove every descendant honors every envelope. Full F5 closure still needs
measured process/DB/API closure and independent hidden-input audits for non-core
callbacks. Manual/other scopes are not automatically identity-complete.

This integration did not run an end-to-end production refresh with production
models or publish production outputs. Preserve the last committed generation
on failure. Roll back by selecting `shadow` for subsequent runs, not by deleting
generations or weakening checkpoint validation.

## Evidence

Last-pin impacted-suite run: **272 passed in 83.88 seconds**, covering F5, API,
pipeline, BYD, convertible-bond grid/allotments, long loaders/caches and similar
pattern/vector incrementality. Added cross-export long-cache evidence separately
checks unchanged exports reuse both periods and price features, while a same-date
correction refreshes one of two periods and rebuilds changed price features.
Final source-journal/F5/long-cache regression after the namespace fix:
**42 passed in 7.23 seconds**. These runs overlap and are not additive test counts.

Final focused integration run: **199 passed in 62.63 seconds**, with two SQLite
date-adapter deprecation warnings. This covers the monthly source journal, real
SQL append/correction/deletion, native child reads, acquisition fingerprint races,
cached-core baseline validation, snapshot reader, operation identity, and API tests.
The main task's candidate-index/repository validation remains the commit evidence.

`tests/test_f5_composition.py` covers defaults/rollback, closure rejection,
actual callback execution/no reuse/failure/resource release, per-dataset identity,
append/correction/deletion journals, stable-symbol dirty propagation, temporary
lifetime, failed-publication baseline retention, realistic SQL plus supplemental
file reads, native descriptor/argument propagation, and first/unchanged/next-day
handler counts. API tests cover post-source pin timing, shared scheduler/store,
complete active callback ownership, and publication gates.

## Deployment Assets

The default release configurations exist in HEAD, but these nine deployment
assets are not tracked there. Provision them from the approved release artifact
store before enabling default core on a clean checkout; do not add ignored models
to this architecture commit:

```text
models/production/right_side_unified_canonical_v2/ranking.joblib
models/production/right_side_unified_canonical_v2/manifest.json
reports/research/right_side_unified_canonical_v5_rule113/production_rollout_approval.json
models/production/left_side_unified_canonical_v4_group4/ranking.joblib
models/production/left_side_unified_canonical_v4_group4/manifest.json
reports/research/left_side_unified_v3_group4_input_parity/ranking_replacement_decision.json
models/research/chan_daily/target_win10.joblib
models/research/chan_daily/target_big10.joblib
models/research/chan_daily/target_good.joblib
```

Release validation steps:

1. Resolve paths from the deployed HEAD release configurations, not an experimental
   local override. Require all nine regular nonempty files before any refresh.
2. Obtain a trusted SHA-256 checksum list covering exactly the nine relative paths
   above from the approved release publisher. From the project root run
   `shasum -a 256 -c /trusted-release/core-assets.sha256`; require every entry `OK`.
   Computing new local hashes alone is inventory, not authenticity verification.
3. Check that the model manifests, approval/decision JSON and configured model
   paths refer to the same approved release. Run candidate-index tests and the
   native adapter required-path/identity validation before production deployment.
   Missing files or mismatched hashes block deployment; never relax the identity
   gate or substitute a different locally available model.
