"""Build exact T-1 moneyflow ratios, preserving missingness and source identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_overnight_moneyflow_coverage import AMOUNTS
from audit_overnight_right_provenance import sha256

from quant.core.paths import PROJECT_ROOT

ALIAS_DATE = pd.Timestamp("2025-02-17")
ALIAS_SOURCE = "https://disc.static.szse.cn/download/disc/disk03/finalpage/2025-02-15/cedb693a-f5ee-4463-9682-ea33d406b569.PDF"
FEATURES = [f"mf_lag1_{size}_net_share" for size in ("sm", "md", "lg", "elg")]
FEATURES += ["mf_lag1_large_activity_share", "mf_lag1_net_active_share"]


def ratios(raw: pd.DataFrame) -> pd.DataFrame:
    """Use the common eight-component gross amount, never zero-fill an absent row."""
    values = raw[[*AMOUNTS, "net_mf_amount"]].apply(pd.to_numeric, errors="coerce")
    gross = values[AMOUNTS].sum(axis=1, min_count=len(AMOUNTS))
    valid = np.isfinite(values).all(axis=1) & values[AMOUNTS].ge(0).all(axis=1) & gross.gt(0)
    result = pd.DataFrame(index=raw.index)
    for size in ("sm", "md", "lg", "elg"):
        result[f"mf_lag1_{size}_net_share"] = (values[f"buy_{size}_amount"] - values[f"sell_{size}_amount"]) / gross
    large = [f"{side}_{size}_amount" for size in ("lg", "elg") for side in ("buy", "sell")]
    result["mf_lag1_large_activity_share"] = values[large].sum(axis=1, min_count=4) / gross
    # Provider active net is an independent field, not a sum of size-bucket nets.
    result["mf_lag1_net_active_share"] = values.net_mf_amount / gross
    result.loc[~valid, :] = np.nan
    return result


def main() -> None:
    root = PROJECT_ROOT / "reports/research"
    output = root / "overnight_moneyflow_lag_v1"
    output.mkdir(exist_ok=False)
    source = root / "overnight_moneyflow_coverage_v1/key_availability.parquet"
    gap = root / "overnight_moneyflow_gap_fill_v1"
    keys = pd.read_parquet(source)
    if keys.duplicated(["symbol", "date"]).any() or not keys.flow_date.lt(keys.date).all():
        raise ValueError("Invalid candidate/date keys")
    hashes = {str(source): sha256(source)}
    requests = pd.read_csv(gap / "requests.csv")
    for row in requests.itertuples():
        if sha256(Path(row.path)) != row.sha256:
            raise ValueError("Gap cache hash mismatch")
    records = []
    for date, group in keys.groupby("flow_date", sort=True):
        path = PROJECT_ROOT / f"data/raw/moneyflow/tushare_moneyflow_{date:%Y%m%d}.parquet"
        if not path.exists():
            path = gap / f"day_{date:%Y%m%d}.parquet"
            if str(path) not in set(requests.path):
                raise ValueError("Unmanifested gap file")
        hashes[str(path)] = sha256(path)
        raw = pd.read_parquet(path, columns=["ts_code", "trade_date", *AMOUNTS, "net_mf_amount"])
        if raw.ts_code.isna().any() or raw.ts_code.duplicated().any() or not raw.trade_date.astype(str).eq(date.strftime("%Y%m%d")).all():
            raise ValueError("Invalid raw moneyflow keys")
        values = ratios(raw)
        values["source_symbol"] = raw.ts_code
        current = group[["symbol", "date", "flow_date", "status"]].rename(columns={"status": "original_status"}).copy()
        current["source_symbol"] = current.symbol
        alias = current.symbol.eq("302132.SZ") & (date < ALIAS_DATE)
        # Canonical entity join only; this does not expose a future code as a model feature.
        current.loc[alias, "source_symbol"] = "300114.SZ"
        current = current.merge(values, on="source_symbol", how="left", validate="many_to_one", indicator=True)
        current["status"] = np.select(
            [current._merge.eq("left_only"), current[FEATURES].isna().any(axis=1)],
            ["missing_stock_row", "invalid_fields"], default="ok")
        current["source_path"] = str(path)
        current["alias_applied"] = current.symbol.ne(current.source_symbol)
        records.append(current.drop(columns="_merge"))
    frame = pd.concat(records, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    expected = keys.sort_values(["date", "symbol"])[["symbol", "date", "flow_date"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[expected.columns], expected)
    if frame.loc[frame.status.ne("ok"), FEATURES].notna().any().any():
        raise ValueError("Missing flow became a numeric feature")
    for path, digest in hashes.items():
        if sha256(Path(path)) != digest:
            raise ValueError("Source changed while building")
    frame.to_parquet(output / "dataset.parquet", index=False)
    summary = {"rows": len(frame), "status": frame.status.value_counts().to_dict(),
               "alias_rows": int(frame.alias_applied.sum()), "features": FEATURES,
               "dataset_sha256": sha256(output / "dataset.parquet"),
               "alias": {"canonical": "302132.SZ", "previous": "300114.SZ", "effective": str(ALIAS_DATE.date()), "source": ALIAS_SOURCE},
               "timing": "Exact previous market day; no rolling or stale fill; current prices unchanged",
               "limitations": "Historical provider revision timestamps unverified; unresolved missing rows retained; no model gain claim",
               "script_sha256": sha256(Path(__file__))}
    (output / "sources.json").write_text(json.dumps(hashes, indent=2))
    (output / "complete.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
