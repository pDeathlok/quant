"""Independent timing, sample-population and price-basis admission checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT
from quant.features.variable_library import PROJECT_FACTOR_COLUMNS


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    source = root / "right_t012_full_dataset_v4"
    output = root / "right_t012_full_dataset_audit_v1"
    check = json.loads((source / "complete.json").read_text())
    path = source / "dataset.parquet"
    if sha256(path) != check["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    keys = ["symbol", "date"]
    frame = pd.read_parquet(path, columns=keys + ["screen_date", "planned_exit_date", "entry_close", "entry_raw_pre_close", "screen_to_entry_return", *PROJECT_FACTOR_COLUMNS])
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate samples")
    all_candidates = pd.read_parquet(source / "all_candidates.parquet")
    expected = all_candidates.loc[all_candidates.eligible.eq(True) & all_candidates.planned_exit_date.notna(), keys]
    pd.testing.assert_frame_equal(frame[keys].sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True))
    bars = pd.read_parquet(root / "overnight_unfiltered_universe_v1/daily_bars.parquet", columns=keys + ["close", "pre_close"])
    raw = frame[keys].merge(bars, on=keys, how="left", validate="one_to_one")
    np.testing.assert_array_equal(frame.entry_close.to_numpy(), raw.close.to_numpy())
    np.testing.assert_array_equal(frame.entry_raw_pre_close.to_numpy(), raw.pre_close.to_numpy())
    np.testing.assert_allclose(frame.screen_to_entry_return, raw.close / raw.pre_close - 1, rtol=0, atol=0, equal_nan=True)
    calendar = pd.DatetimeIndex(bars.date.unique()).sort_values()
    links = pd.DataFrame(dict(screen_date=calendar, expected_entry=pd.Series(calendar).shift(-1), expected_exit=pd.Series(calendar).shift(-2)))
    dates = frame[["screen_date", "date", "planned_exit_date"]].merge(links, on="screen_date", validate="many_to_one")
    if not dates.date.eq(dates.expected_entry).all() or not dates.planned_exit_date.eq(dates.expected_exit).all():
        raise ValueError("Not consecutive global T/T+1/T+2 dates")
    market = check["strict_market_columns"]
    basic = [c for c in PROJECT_FACTOR_COLUMNS if c not in market]
    factors = pd.read_parquet(root / "overnight_unfiltered_full_factors_v1/dataset.parquet", columns=keys + list(PROJECT_FACTOR_COLUMNS))
    current = frame[keys].merge(factors[keys + market], on=keys, how="left", validate="one_to_one")
    prior = frame[["symbol", "screen_date"]].rename(columns={"screen_date": "date"}).merge(factors[keys + basic], on=keys, how="left", validate="one_to_one")
    np.testing.assert_array_equal(frame[market].to_numpy(np.float32), current[market].to_numpy(np.float32))
    np.testing.assert_array_equal(frame[basic].to_numpy(np.float32), prior[basic].to_numpy(np.float32))
    weekly_checks = []
    for path in sorted(source.glob("weekly_source_dates_*.parquet")):
        dates = pd.read_parquet(path)
        known = dates.source_date.notna()
        if not dates.loc[known, "source_date"].lt(dates.loc[known, "date"]).all():
            raise ValueError("Weekly source not prior to entry")
        pd.testing.assert_frame_equal(frame[keys], dates[keys])
        weekly_checks.append(dict(file=path.name, known=int(known.sum()), missing=int((~known).sum())))
    output.mkdir(exist_ok=False)
    result = dict(dataset_sha256=check["dataset_sha256"], rows=len(frame), raw_price_basis="exact",
                  candidate_population="exact eligible causal population; no outcome availability filter",
                  current_market_feature_cells=len(frame) * len(market), prior_basic_feature_cells=len(frame) * len(basic),
                  weekly_checks=weekly_checks,
                  limitations="Confirms join/timing/price basis, not vintage truth of legacy financial/analyst sources")
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
