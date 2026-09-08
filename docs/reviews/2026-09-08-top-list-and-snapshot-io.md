# Daily Top List and Snapshot I/O Verification

## Findings and Changes

- The live Chan builder reads top-list inputs even when their effective model
  importance is zero. An explicit feature-projection edge removed the source
  from the daily active closure. The source is now an unconditional dependency.
- The independent Chan live command defaulted to `data/raw/moneyflow`; it now
  matches the daily writer at `data/raw/top_list`.
- Daily top-list polling bypasses cached provider wrappers. Poll receipts bind
  the requested sessions to checksummed output partitions. Failed requests and
  wrong-date responses cannot advance freshness using an existing file.
- The declared 120-calendar-day context bounds historical session checks, with
  a three-calendar-day repoll overlap before the last valid partition. Failed
  requests persist as pending dates across attempts. The provider calendar,
  validated partitions and fresh poll receipts must agree before freshness
  advances. Null, nonnumeric and nonfinite required numeric inputs fail closed.
  No-history bootstrap covers only the target; history older than the declared
  context is explicitly reported as unchecked, not silently certified complete.
- Canonical supplemental Parquet files are copied as independent bytes, not
  hard links. Only identity columns are decoded for metadata. Noncanonical or
  filtered inputs retain normalization and full encoding.
- Capture uses four workers by default, bounded to eight. Manifest ordering is
  deterministic. Initial hashes, verified open-file capture, output hashes,
  final source hashes, and source membership checks remain enforced.
- Date normalization parses and formats distinct values once, then maps them
  back to rows. Export and snapshot readers share the implementation. This
  removes repeated work for cross-sectional partitions without dropping date
  validation.
- Stage timings and file counters are exposed in `market_snapshot.capture_metrics`;
  these observations are excluded from content identity.

## Measured Sample

Local Python 3.13, same 600 source files copied into isolated temporary roots:
300 daily-basic and 300 top-list files, 1,662,771 rows in total.
Seven top-list files were empty; 593 output files were declared.

| Export implementation | Seconds |
| --- | ---: |
| Before this change | 216.510 |
| Independent copy, bounded workers, distinct-date normalization | 3.811 |

The completed baseline and final benchmark both passed full row comparison.
The final output copied all 593 nonempty files without re-encoding. Source
data was not modified. Sample export latency fell approximately 98.2%; this is
not a measured end-to-end refresh improvement. macOS sampling also observed
per-row date formatting repeatedly calling time-zone file routines.

Two superseded intermediate benchmark processes were stopped after the final
benchmark completed; their temporary source/output copies were removed.

## Limits

The Python 3.13 page/API and snapshot integration batch passed 185 tests.
Final combined regressions passed 235 tests on Python 3.13 and 201 data-side
tests on Python 3.9.
Six additional live-runtime integration cases cannot run in the local Python
3.9 environment because `akquant` is unavailable; those cases passed on Python
3.13, matching their existing CI runtime selection. Ruff fatal-name checks and
`git diff --check` passed.

No live provider refresh, production publication, server restart, commit or push
was performed for this change. Today's published snapshot therefore remains the
previous generation until the normal refresh entry runs successfully.
SQL full-history extraction and monthly row-content hashing still exist; the
previous 18-minute whole-snapshot duration has not yet been re-measured.
No persistent row-export cache was added, and raw history was not deleted.
