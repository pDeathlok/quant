# Remediation Results

## Scope

Implementation follows the September 6 architecture review in the existing dirty
working tree. No production refresh, deployment or production-cache deletion
was performed. These changes are scoped separately from the existing v5/Kronos
experiments. Already-published data has not been retroactively
repaired. Passing isolated tests does not establish current production data quality.

## Correctness

- Workspace output generations are staged and checked before a single visible
  pointer changes. Readers pin a generation. SQL persistence failure, failed final
  freshness checks, invalid state and interrupted writes cannot publish a new
  visible generation. Old generations remain available to active readers.
- Canonical SQL failures are errors, not empty results or permission to read an
  unverified Parquet mirror. Empty canonical results remain empty.
- All date-sensitive page loaders reject obsolete response/error/completion
  callbacks. Generation recovery is bounded when the server keeps returning 409.
- Vector configuration and compiled-cache cleanup uses registered ownership,
  references and leases. Missing/corrupt/interrupted libraries require repair
  independently of the Friday reference-library refresh schedule.

## Incremental Calculations

The realistic probe exposed a mixed price-basis defect: full factor construction
used causal forward-continuous OHLC, while appended rows still rescaled prior
state according to the legacy price basis. The append path now advances the same
causal chain and preserves historical state. Signal state schema is 2 and signal
factor cache version is `signal-v3-causal-state`.

Unknown or changed signal calculation contracts now rebuild the configured
historical range and do not merge old-contract historical signal rows. This
includes callers that omit an incremental date. The first refresh following the
migration may therefore be slower. Old published artifacts remain unchanged
until a new successful refresh validates and publishes replacements.

Verification used temporary factor stores with 400 observed rows each for
`000001.SZ`, `002862.SZ` and `600519.SH`: seed 399, append one, compare every
column with full calculation at `rtol=atol=1e-10`. All three passed. Synthetic
tests also cover prior corporate actions, actions at append, corrections,
deletions, suspensions and legacy state invalidation.

Finite rule windows and shared normalized prices reduce work and allocation.
Recursive indicators retain their required history. Do not infer a full-market
wall-clock speedup from these local tests.

Sealed SQL exports now use stable per-symbol files, with monthly Arrow-content
hashes in metadata. A single-symbol read prunes unrelated files; daily append
does not dirty that symbol's entire historical range. Corrections and deletions
still invalidate their affected months. Unverifiable journals remain conservative.

Verified signal factor state survives distinct sealed runs through
`SIGNAL_FACTOR_ROOT`, defaulting to the existing factor root. Every reuse checks
source rows, history coverage and the calculation implementation fingerprint.
Formula changes invalidate state without relying only on a manually bumped
version constant. Legacy base-factor caches remain run-isolated because date
coverage alone cannot prove source equivalence. A real-child three-run fixture
verified bootstrap, append-only advancement and historical-correction rebuild;
each matched fresh factor calculation.

Vector tests preserve unchanged vectors and append only newly eligible windows.
One three-symbol synthetic run measured 0.041 seconds unchanged, 0.201 seconds
append and 0.593 seconds full on the same final input. Append computed 30 vectors
and reused 807; full computed 837. Source reads and source-version validation
still have costs. This is not a production benchmark.

## F5 Integration

The five core feature/output operations have explicit identity contracts,
checkpoint integration and one shared cooperative resource scheduler with nested
grants. The ordinary `all`/`short` composition now acquires canonical sealed
inputs after source refresh and uses strict `core` execution. Explicit
`shadow`/`legacy` configuration remains available for rollback.

- SQL daily rows are exported under a consistent transaction. Daily-basic and
  top-list file authorities are explicitly captured, not SQL fallback mirrors.
  Child processes receive the pinned manifest and expected content identity.
- The right-side builder takes explicit staged shared-feature paths, with
  generation isolation and missing/corrupt staging tests.
- Active operation owners are validated before source writes. Composition-owned
  callbacks execute once through non-cacheable adapters instead of manufacturing
  individually cached results for internal operations.
- Only a successfully published source manifest advances the incremental
  comparison baseline. Changes to one dataset do not invalidate unrelated inputs.
- Model/configuration contracts are rechecked after computation. A runtime change
  fails execution before checkpoint persistence.

Remaining limits: composition callbacks have not all received independent hidden
input/cache-identity audits. Resource reservations are cooperative, not measured
native/process peak CPU, RSS or SQL/API concurrency. The new transport has not
been measured in a full-market production refresh. Executable coverage does not
justify declaring the full architecture/resource audit complete.

## Verification

- Interim full repository suite: **1,643 passed**, 186 warnings, 126.71 seconds.
- Factor/rule regression: **38 passed**; signal contract tests: **12 passed**.
- Frontend: **67 Node tests**, **50 focused pytest tests**, desktop/mobile browser
  regressions and JavaScript syntax checks passed.
- Vector/retention-focused agent suite: **178 passed** before final service hook.
- CI runtime tests: **9 passed** under each observed Python runtime.

Final integrated run after core-stage and vector activation hooks: **1,653
passed**, 186 warnings, 125.46 seconds. A subsequent isolated-worker grant
propagation fix passed **144** publication/API/resource tests in 44.06 seconds.
The spawned vector worker now receives its parent worker cap; this does not
claim measured native-library RSS or thread closure. Ruff E9/F63/F7/F82 checks
passed across `src/quant` and `scripts/ci`; `git diff --check` passed.

Warnings include existing fragmented-frame and physical-core detection warnings.
The remaining F5 rollout is described in
[the core DAG rollout plan](/Users/didi/Project/quant/docs/plans/2026-09-07-core-dag-rollout.md).

## Final Candidate Verification

The actual staged tree was exported to an isolated directory, excluding the
unrelated v5/Kronos work. Runtime assets used independent copy-on-write clones;
production credentials were not copied. Credential-dependent test fixtures were
replaced with explicit offline clients, and local-data fixtures select their
backend rather than inheriting the host's SQL configuration.

- Final Python 3.13 suite: **1,756 passed**, 188 warnings, **163.70 seconds**.
- Python 3.9 source-only snapshot/persistence/journal suite: **68 passed**.
  Broader strategy tests require the Python 3.10+ `akquant` runtime and were not
  counted as Python 3.9 validation. The CI matrix now includes these source tests.
- Candidate desktop 1440x1000 and mobile 390x844 request-race browser checks passed.
- Candidate Ruff E9/F63/F7/F82, JavaScript syntax and staged whitespace checks passed.

Final guards retain the source pin through early/core/late consumers and joined
children, reject incomplete executed-operation coverage before publication, and
bind parsed-manifest reuse to expected identity plus file state. Pinned signal
refresh reads full-market rows once; an unchanged rerun reads them zero times.
It no longer rehashes a second full-market materialization before publication.

No end-to-end production refresh or deployment was performed. Production wall
time, process-tree resource peaks and the remaining non-core identity audit are
not claimed complete by this isolated validation. Provision and verify the nine
external model/approval assets as documented before deploying a clean checkout.
