# Daily Basic Valuation Fallback

Tushare remains the primary source. Its `pe_ttm` convention is a positive
valuation for profitable issuers and null for loss-making issuers. A decline
in the profitable share of the market must not be mistaken for a late batch.

Whenever unresolved `pe_ttm` observations remain, the refresh
queries the historical Eastmoney endpoint used by AkShare `stock_value_em`.
It batches the exact trade date, validates all pages, and caches one small
cross-section per day under `data/cache/source_merge/eastmoney_valuation`.
The cache is reusable for ten minutes; it is not carried to another date.

Only missing Tushare values are eligible. The reference trade date must match,
symbols must be unique, and the reference total market capitalization must
agree with Tushare within 1% after conversion from yuan to ten-thousand yuan.
No previous-day PE, forward PE, mean, or current-day quote is substituted.
Passing the market-wide 70% gate does not suppress individual gap verification.
Previously validated same-date evidence is retained during supplemental queries.

- Positive verified reference values fill the standard `pe_ttm` column.
- Negative verified values leave `pe_ttm` null, following Tushare semantics.
- Signed evidence, source and exact date are stored in `pe_ttm_reference`,
  `pe_ttm_reference_source`, and `pe_ttm_reference_date` in the same parquet.
- Zero, non-finite, wrong-date, mismatched-cap and unavailable values remain
  unresolved. A reference source error does not bypass the original gate.

`feature_coverage` continues to report numeric non-null coverage.
`resolved_feature_coverage` additionally counts same-date verified losses as
resolved observations. The 70% quality threshold applies to the latter for
PE(TTM); other fields keep their existing rules. This is an explicit change
to the gate's semantics, not a claim that numeric coverage reached 70%.
The refresh and historical audit use the same resolution predicate. The audit
also reports how many loss observations were accepted and how many PE values
remain unresolved; publication is still subject to every other postflight gate.

Unresolved observations also use consolidated Tushare `income` reports when
complete as-of-date inputs exist. TTM parent earnings equal the previous annual
earnings plus current year-to-date earnings minus prior comparable year-to-date
earnings. For an annual observation, use annual earnings directly. Divide the
target day's Tushare total market cap (converted to yuan) by those earnings.
The cross-source 1% market-cap gate is unchanged; this independent derivation
does not substitute Eastmoney's market cap or scale yesterday's PE.

Only consolidated report types 1 and 4 are eligible. Reports must have been
announced by the target date, using actual announcement dates where present;
the latest available reporting period cannot be more than 200 days old. Missing
periods, conflicting same-version reports, non-finite or zero earnings are
unresolved. Negative earnings retain the canonical null and store signed PE
with source `tushare_income_ttm`. The `pe_ttm_derivation` JSON column records
the ticker, market cap and each report's period, availability, type and earnings.
Both refresh validation and historical audit recompute this evidence before
accepting a derived loss. Positive derivations fill only missing canonical PE.

Reference financial inputs are refreshed before daily-basic derivation in both
entry paths. Financial VIP polling reads every page for each quarter, rejects
repeated pages and never marks a partially fetched period complete. Its versioned
checkpoint invalidates successful legacy single-page polls once, so historical
report gaps are repaired by the regular refresh rather than manual parquet edits.
The default five-quarter overlap includes the prior comparable quarter needed
by TTM derivation, rather than relying on an already complete local history.

## Explicit-Date Retry

Run `python3 scripts/run_daily_web_refresh.py --trade-date YYYY-MM-DD` to retry
an unpublished date. The runner forwards that date through the web API to both
daily-price and daily-basic refreshes, and verifies the resulting source date
before downstream work. Without this option, the usual current-day behavior is
unchanged. A retry cannot roll the canonical daily watermark back from a newer
date; this entry point is not a historical simulation/replay engine.
