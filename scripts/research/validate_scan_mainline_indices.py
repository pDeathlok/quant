"""Read-only industry momentum diagnostic; not a policy/theme backtest."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from quant.routine.paths import load_project_env

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/research/scan_a_shares_mainline_indices_20261007"


async def run() -> None:
    load_project_env()
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Existing market credential unavailable")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "frozen_before_fetch",
        "scope": "L1 industry price-trend proxy, not A/B/C policy-industry strategy",
        "taxonomy": "SW2014 through 2021; SW2021 from 2022; provider historical index vintages unverified",
        "signal": "Daily top2 ranked 0.65 percentile(ret20)+0.35 percentile(ret60), requiring ret20>0 and close>MA60",
        "truth": "Ex-post top3 forward20/60 close return within that day's available L1 universe; evaluation only",
        "optimization": "none; no post-result parameter search",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    audit = []
    async with httpx.AsyncClient(timeout=30) as client:
        async def query(api: str, params: dict, fields: str, path: Path) -> pd.DataFrame:
            if path.exists():
                return pd.read_parquet(path)
            response = await client.post("https://api.tushare.pro", json={"api_name": api, "token": token, "params": params, "fields": fields})
            response.raise_for_status()
            body = response.json()
            if body.get("code") != 0 or body.get("data") is None:
                audit.append({"api": api, "params": params, "status": "provider_error", "code": body.get("code")})
                (OUT / "fetch_audit.json").write_text(json.dumps(audit, indent=2))
                raise RuntimeError(f"Industry data unavailable; provider code {body.get('code')}")
            data = body["data"]
            frame = pd.DataFrame(data["items"], columns=data["fields"])
            if len(frame) >= 4000:
                raise ValueError("Possible API truncation")
            frame.to_parquet(path, index=False)
            audit.append({"api": api, "params": params, "status": "success", "rows": len(frame)})
            (OUT / "fetch_audit.json").write_text(json.dumps(audit, indent=2))
            await asyncio.sleep(.25)
            return frame

        classes = {}
        for src in ["SW2014", "SW2021"]:
            classes[src] = await query("index_classify", {"level": "L1", "src": src}, "index_code,industry_name,level,src", OUT / f"{src}.parquet")
        frames = []
        codes = sorted(set(pd.concat(classes.values()).index_code))
        for n, code in enumerate(codes):
            frame = await query("sw_daily", {"ts_code": code, "start_date": "20170101", "end_date": "20260930"}, "ts_code,trade_date,name,close", OUT / f"{code}.parquet")
            frames.append(frame)
            if (n+1) % 10 == 0:
                print(f"Industry indices fetched {n+1}/{len(codes)}", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw.trade_date, format="%Y%m%d")
    if raw.duplicated(["ts_code", "date"]).any():
        raise ValueError("Duplicate industry bars")
    price = raw.pivot(index="date", columns="ts_code", values="close").sort_index().astype(float)
    ret20 = price.pct_change(20, fill_method=None)
    ret60 = price.pct_change(60, fill_method=None)
    ma60 = price.rolling(60).mean()
    allowed = pd.DataFrame(False, index=price.index, columns=price.columns)
    allowed.loc[allowed.index < "2022-01-01", classes["SW2014"].index_code] = True
    allowed.loc[allowed.index >= "2022-01-01", classes["SW2021"].index_code] = True
    allowed &= price.notna() & ret60.notna()
    score = .65 * ret20.where(allowed).rank(axis=1, pct=True) + .35 * ret60.where(allowed).rank(axis=1, pct=True)
    eligible = allowed & (ret20 > 0) & (price > ma60)
    rank = score.where(eligible).rank(axis=1, method="first", ascending=False)
    selected = rank.le(2)
    # Verify causal features and selection are unchanged by truncating future bars.
    cut = "2021-12-31"
    pd.testing.assert_frame_equal(ret20.loc[:cut], price.loc[:cut].pct_change(20, fill_method=None))
    pd.testing.assert_frame_equal(ma60.loc[:cut], price.loc[:cut].rolling(60).mean())
    rows, summary = [], []
    for h in [20, 60]:
        future = price.shift(-h) / price - 1
        # Do not evaluate a forward window crossing the taxonomy switch.
        end_date = pd.Series(price.index, index=price.index).shift(-h)
        same_taxonomy = (price.index < pd.Timestamp("2022-01-01")) == (end_date < pd.Timestamp("2022-01-01"))
        mature = allowed & future.notna()
        mature.loc[~same_taxonomy] = False
        mature.loc[mature.index < "2018-01-01"] = False
        truth = future.where(mature).rank(axis=1, method="first", ascending=False).le(3)
        for period, lo, hi in [("full", "2018-01-01", "2026-09-30"), ("development", "2018-01-01", "2021-12-31"), ("chronological", "2022-01-01", "2024-12-31"), ("recent", "2025-01-01", "2026-09-30")]:
            m = mature.loc[lo:hi]
            p = (selected & mature).loc[lo:hi]
            t = truth.loc[lo:hi]
            f = future.loc[lo:hi]
            count = int(p.to_numpy().sum())
            expected = (t.sum(axis=1) / m.sum(axis=1).replace(0, np.nan)).mul(p.sum(axis=1)).sum() / max(count, 1)
            summary.append({"horizon": h, "period": period, "picked_industry_days": count, "top3_precision": float((p&t).to_numpy().sum()/max(count,1)), "top3_recall": float((p&t).to_numpy().sum()/max(t.to_numpy().sum(),1)), "matched_random_precision": float(expected), "selected_mean_return": float(f.where(p).stack().mean()), "all_industry_mean_return": float(f.where(m).stack().mean()), "positive_fraction": float((f.gt(0)&p).to_numpy().sum()/max(count,1))})
        long = pd.DataFrame({"forward_return": future.stack(), "selected": selected.stack(), "mature": mature.stack(), "top3": truth.stack()}).reset_index()
        long["horizon"] = h
        rows.append(long)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    pd.concat(rows).to_parquet(OUT / "details.parquet", index=False)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob("*.parquet")}
    manifest = {"status": "completed_industry_proxy_only", "start": "2018-01-01", "end": str(price.index.max().date()), "indices": len(codes), "rows": len(raw), "prefix_invariance": "passed", "sources_sha256": hashes, "limitations": ["Provider defaults to 2021-version history; 2014 historical index vintage and rebasing unverified", "Industry L1 is not fine-grained theme or causal policy/industry proof", "Overlapping daily observations are not independent samples", "No member breadth or point-in-time industry business exposure reconstructed", "No trading returns inferred from industry hit rate"]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    asyncio.run(run())
