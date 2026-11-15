"""Frozen causal market-scan module study; never represents full A/B/C validation.

Uses explicit local parquet mirrors. Keep its limits in every output: unknown
historical ST, estimated execution, approximate corporate-action accounting,
and limited historical mainline membership. No production mutations or fetching.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ASHARE = r"(?:60\d{4}|68[89]\d{3})\.SH|(?:00\d{4}|30\d{4})\.SZ|[489]\d{5}\.BJ"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def continuous(frame: pd.DataFrame) -> pd.DataFrame:
    """Forward-only scale: later splits must not rewrite earlier signal features."""
    f = frame.sort_values("date").copy()
    factor = (f.close.shift(1) / f.pre_close).fillna(1).cumprod()
    for col in ["open", "high", "low", "close"]:
        f[col] = f[col] * factor
    return f


def features(close: pd.DataFrame, amount: pd.DataFrame, observed: pd.DataFrame,
             high: pd.DataFrame, low: pd.DataFrame) -> dict[str, pd.DataFrame]:
    filled = close.ffill()
    age = observed.cumsum()
    ma = filled.rolling(60, min_periods=60).mean()
    liquidity = amount.fillna(0).rolling(20, min_periods=20).mean()
    previous = filled.shift(1)
    tr = np.maximum(high-low, np.maximum((high-previous).abs(), (low-previous).abs()))
    atr = pd.DataFrame(tr, index=close.index, columns=close.columns).rolling(14).mean()
    base_eligible = observed & (age >= 120) & (liquidity >= 50000)
    eligible = base_eligible & (filled > ma)
    return {"close": filled, "ma60": ma, "atr14": atr,
            "return20": filled.pct_change(20, fill_method=None),
            "return60": filled.pct_change(60, fill_method=None), "eligible": eligible,
            "base_eligible": base_eligible}


def choose_signals(info: dict[str, pd.DataFrame], days: pd.DatetimeIndex,
                   window: int) -> pd.DataFrame:
    rows = []
    score = info[f"return{window}"]
    for day in days:
        ok = info["eligible"].loc[day] & score.loc[day].gt(0)
        ranked = score.loc[day][ok].sort_index().sort_values(ascending=False, kind="stable")
        for rank, (symbol, value) in enumerate(ranked.head(20).items(), 1):
            rows.append({"signal_date": day, "symbol": symbol, "rank": rank,
                         "score": float(value), "window": window})
    return pd.DataFrame(rows, columns=["signal_date", "symbol", "rank", "score", "window"])


def forward_stats(entry: float, closes: np.ndarray, horizon: int) -> dict[str, float | bool]:
    """Unseen labels only. Incomplete windows never count as failures in precision."""
    mature = len(closes) >= horizon
    finite = closes[:horizon][np.isfinite(closes[:horizon])]
    return {"mature": mature, "peak_multiple": float(finite.max()/entry) if len(finite) else np.nan,
            "terminal_multiple": float(closes[horizon-1]/entry) if mature else np.nan}


def stamp_tax(day: pd.Timestamp) -> float:
    return .001 if day < pd.Timestamp("2023-08-28") else .0005


@dataclass(frozen=True)
class ExitRule:
    name: str
    ma_days: int = 60
    trailing: float = .25
    rotation: bool = False
    risk_sizing: bool = False


def simulate(close: pd.DataFrame, open_: pd.DataFrame, high: pd.DataFrame,
             low: pd.DataFrame, observed: pd.DataFrame, atr: pd.DataFrame,
             signals: pd.DataFrame, rule: ExitRule, delist: dict[str, pd.Timestamp],
             start: str, end: str, cost: float = .001,
             entry_allowed: pd.DataFrame | None = None,
             risk_warning: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close decisions -> next-open estimated fills; mark suspended holdings stale.

    Continuous-unit accounting is a research proxy, not a verified share ledger.
    Conservative absolute-gap rules cannot replace exact limit data.
    """
    dates = close.index
    codes = list(close.columns)
    mapping = {c: j for j, c in enumerate(codes)}
    c, o, hi, lo, obs, a = [x.to_numpy() for x in [close, open_, high, low, observed, atr]]
    allowed = entry_allowed.to_numpy() if entry_allowed is not None else obs
    warned = risk_warning.to_numpy() if risk_warning is not None else np.zeros_like(obs)
    ma = close.rolling(rule.ma_days, min_periods=rule.ma_days).mean().to_numpy()
    picks = {d: list(f.symbol) for d, f in signals.groupby("signal_date", sort=True)}
    cash, positions, pending_sell, pending_buy = 1., {}, {}, []
    records, trades = [], []
    last_marks: dict[int, float] = {}
    frozen, blocked_buy, blocked_sell, written_off = 0, 0, 0, 0
    for t, day in enumerate(dates):
        if day < pd.Timestamp(start) or day > pd.Timestamp(end):
            continue
        for j in list(positions):
            if codes[j] in delist and day >= delist[codes[j]]:
                p = positions.pop(j)
                trades.append({"symbol": codes[j], "entry_date": p["day"], "exit_date": day,
                               "entry": p["entry"], "exit": 0., "reason": "known_delist_zero_recovery",
                               "net_multiple": 0., "holding_days": t-p["t"]})
                pending_sell.pop(j, None)
                written_off += 1
        sold_today: set[int] = set()
        for j, reason in list(pending_sell.items()):
            if j not in positions:
                del pending_sell[j]
                continue
            p = positions[j]
            prev = last_marks.get(j, p["entry"])
            down_locked = np.isfinite(o[t,j]) and o[t,j]/prev-1 <= -.048
            if not obs[t,j] or not np.isfinite(o[t,j]) or down_locked or t <= p["t"]:
                blocked_sell += 1
                continue
            proceeds = p["units"]*o[t,j]*(1-cost-stamp_tax(day))
            cash += proceeds
            trades.append({"symbol": codes[j], "entry_date": p["day"], "exit_date": day,
                           "entry": p["entry"], "exit": float(o[t,j]), "reason": reason,
                           "net_multiple": float(o[t,j]/p["entry"]*(1-cost-stamp_tax(day))/(1+cost)),
                           "holding_days": t-p["t"]})
            del positions[j]
            del pending_sell[j]
            sold_today.add(j)
        nav_open = cash + sum(p["units"]*(o[t,j] if obs[t,j] and np.isfinite(o[t,j])
                                             else last_marks.get(j,p["entry"])) for j,p in positions.items())
        held_open = nav_open-cash
        for symbol in pending_buy:
            j = mapping[symbol]
            if j in positions or j in sold_today or len(positions) >= 20 or t == 0:
                continue
            prev = c[t-1,j]
            if not obs[t,j] or not allowed[t,j] or not np.isfinite(o[t,j]) or not np.isfinite(prev) or prev <= 0:
                blocked_buy += 1
                continue
            gap = o[t,j]/prev-1
            up_locked = gap >= .048
            if gap > .05 or up_locked:
                blocked_buy += 1
                continue
            distance = float(np.clip(3*a[t-1,j]/prev, .08, .20))
            if not np.isfinite(distance):
                blocked_buy += 1
                continue
            weight = min(.05,.0025/(distance+.01)) if rule.risk_sizing else .025
            budget = min(nav_open*weight, max(0,nav_open*.5-held_open), cash/(1+cost))
            if budget <= 1e-8:
                continue
            units = budget/o[t,j]
            cash -= budget*(1+cost)
            held_open += budget
            positions[j] = {"units": units, "entry": float(o[t,j]), "day": day, "t": t,
                            "peak": float(o[t,j]), "stop": float(o[t,j]*(1-distance))}
        pending_buy = []
        nav = cash
        for j,p in positions.items():
            mark = c[t,j]
            if np.isfinite(mark):
                last_marks[j] = float(mark)
            else:
                mark = last_marks.get(j,p["entry"])
            if not obs[t,j]:
                frozen += 1
            nav += p["units"]*mark
            p["peak"] = max(p["peak"], float(mark))
            if not obs[t,j]:
                continue
            # Orders placed after close, never filled using this same day's low.
            if warned[t,j]:
                pending_sell[j] = "historical_ST_risk_warning"
            elif mark <= p["stop"]:
                pending_sell[j] = "initial_3ATR_close_stop"
            elif not rule.rotation and mark/p["peak"]-1 <= -rule.trailing:
                pending_sell[j] = "trailing_close_drawdown"
            elif not rule.rotation and np.isfinite(ma[t,j]) and mark < ma[t,j]:
                pending_sell[j] = "close_below_MA"
            elif not rule.rotation and t-p["t"] >= 504:
                pending_sell[j] = "504_session_expiry"
        if day in picks:
            desired = picks[day]
            if rule.rotation:
                for j in positions:
                    if codes[j] not in desired:
                        pending_sell[j] = "weekly_rank_rotation"
            pending_buy = desired
        records.append({"date": day, "nav": float(nav), "cash": float(cash),
                        "positions": len(positions), "exposure": float((nav-cash)/nav),
                        "blocked_buy_cumulative": blocked_buy, "blocked_sell_cumulative": blocked_sell,
                        "stale_position_days_cumulative": frozen, "delist_writeoffs": written_off})
    for j,p in positions.items():
        trades.append({"symbol":codes[j],"entry_date":p["day"],"exit_date":pd.NaT,
                       "entry":p["entry"],"exit":np.nan,"reason":"open_at_end",
                       "net_multiple":np.nan,"holding_days":t-p["t"],
                       "unrealized_mark_multiple":last_marks.get(j,p["entry"])/p["entry"]})
    return pd.DataFrame(records), pd.DataFrame(trades)


def metrics(nav: pd.DataFrame) -> dict[str, float | int]:
    v = nav.nav.to_numpy(dtype=float)
    r = np.diff(np.r_[1.,v])/np.r_[1.,v[:-1]]
    peaks = np.maximum.accumulate(np.r_[1.,v])[1:]
    return {"sessions": len(v), "total_return": float(v[-1]-1),
            "annual_return": float(v[-1]**(252/len(v))-1),
            "max_drawdown": float(np.min(v/peaks-1)),
            "sharpe_zero_rf": float(np.mean(r)/np.std(r)*np.sqrt(252)) if np.std(r) else 0.,
            "average_exposure": float(nav.exposure.mean())}


def historical_status(names: pd.DataFrame, dates: pd.DatetimeIndex,
                      codes: list[str]) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Only intervals effective and announced by each day can enable entry."""
    known=np.zeros((len(dates),len(codes)),dtype=bool);st=np.zeros_like(known)
    mapping={c:j for j,c in enumerate(codes)}
    f=names.drop_duplicates().copy()
    for col in ["start_date","end_date","ann_date"]:
        f[col]=pd.to_datetime(f[col],format="%Y%m%d",errors="coerce")
    for row in f.itertuples(index=False):
        if row.ts_code not in mapping or pd.isna(row.start_date) or pd.isna(row.ann_date):
            continue
        start=max(row.start_date,row.ann_date)
        end=row.end_date if pd.notna(row.end_date) else dates[-1]
        lo=int(dates.searchsorted(start,side="left"));hi=int(dates.searchsorted(end,side="right"))
        if hi<=lo:
            continue
        j=mapping[row.ts_code];known[lo:hi,j]=True
        if "ST" in str(row.name).upper() or "退" in str(row.name):
            st[lo:hi,j]=True
    return (pd.DataFrame(known,index=dates,columns=codes),pd.DataFrame(st,index=dates,columns=codes))


def run(output: Path, supplement: Path | None = None) -> None:
    output=output.resolve()
    supplement=supplement.resolve() if supplement is not None else None
    if (output/"manifest.json").exists():
        raise ValueError("Completed run exists; create a new output/version")
    protocol = json.loads((output/"protocol.json").read_text())
    paths = [p for p in sorted((ROOT/"data/raw/daily_partitioned").glob("year_month=*/data.parquet"))
             if "201701" <= p.parent.name.split("=")[1] <= "202609"]
    sources, frames, coverage = {}, [], []
    cols = ["ts_code","trade_date","open","high","low","close","pre_close","amount","vol","volume","name","industry"]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = digest(path)
        f = pd.read_parquet(path, columns=cols)
        f["vol"]=f.vol.fillna(f.volume)
        f = f[f.ts_code.str.fullmatch(ASHARE,na=False)].copy()
        f["date"] = pd.to_datetime(f.trade_date.astype(str),format="%Y%m%d")
        numbers = f[["open","high","low","close","pre_close","amount","vol"]]
        f["valid"] = np.isfinite(numbers).all(axis=1) & numbers[["open","high","low","close","pre_close","vol"]].gt(0).all(axis=1)
        f["valid"] &= f.high.ge(f[["open","close","low"]].max(axis=1)) & f.low.le(f[["open","close","high"]].min(axis=1))
        audit = f.groupby("date").agg(observed=("ts_code","nunique"),valid=("valid","sum"),
                                        name_available=("name","count"),industry_available=("industry","count"))
        coverage.append(audit)
        f["name_available"]=f.name.notna();f["industry_available"]=f.industry.notna()
        frames.append(f.drop(columns=["trade_date","name","industry"]))
    if supplement is not None:
        for path in sorted((supplement/"delisted").glob("*.parquet")):
            sources[str(path.relative_to(ROOT))]=digest(path)
            f=pd.read_parquet(path)
            f["date"]=pd.to_datetime(f.trade_date.astype(str),format="%Y%m%d")
            n=f[["open","high","low","close","pre_close","amount","vol"]]
            f["valid"]=np.isfinite(n).all(axis=1)&n[["open","high","low","close","pre_close","vol"]].gt(0).all(axis=1)
            f["valid"] &= f.high.ge(f[["open","close","low"]].max(axis=1))&f.low.le(f[["open","close","high"]].min(axis=1))
            f["name_available"]=False;f["industry_available"]=False
            frames.append(f.drop(columns="trade_date"))
    raw = pd.concat(frames,ignore_index=True)
    del frames
    if raw.duplicated(["ts_code","date"]).any():
        dup=raw.loc[raw.duplicated(["ts_code","date"],keep=False)]
        for col in ["open","high","low","close","pre_close","vol","amount"]:
            g=dup.groupby(["ts_code","date"])[col]
            if not np.allclose(g.min(),g.max(),rtol=1e-8,atol=1e-6,equal_nan=True):
                raise ValueError("Supplement conflicts with original mirror: "+col)
        raw=raw.drop_duplicates(["ts_code","date"],keep="first")
    audit=raw.groupby("date").agg(observed=("ts_code","nunique"),valid=("valid","sum"),name_available=("name_available","sum"),industry_available=("industry_available","sum")).sort_index()
    print(f"Loaded {len(raw):,} rows; valid {raw.valid.sum():,}; symbols {raw.ts_code.nunique()}",flush=True)
    basic_path = ROOT/"data/raw/stock_basic_history.parquet"
    current_path = ROOT/"data/raw/stock_basic.parquet"
    sources[str(basic_path.relative_to(ROOT))] = digest(basic_path)
    sources[str(current_path.relative_to(ROOT))] = digest(current_path)
    basic = pd.read_parquet(basic_path)
    current = pd.read_parquet(current_path)
    basic = pd.concat([basic,current],ignore_index=True).drop_duplicates("ts_code",keep="first")
    basic = basic.loc[basic.ts_code.str.fullmatch(ASHARE,na=False)].copy()
    basic["listing"] = pd.to_datetime(basic.list_date,format="%Y%m%d",errors="coerce")
    basic["delisting"] = pd.to_datetime(basic.delist_date,format="%Y%m%d",errors="coerce")
    audit["expected_from_local_lifecycle"] = [int(((basic.listing <= d)&(basic.delisting.isna()|(basic.delisting > d))).sum()) for d in audit.index]
    audit["observed_over_expected"] = audit.observed/audit.expected_from_local_lifecycle
    audit.to_csv(output/"daily_coverage.csv")
    delist = dict(zip(basic.loc[basic.delisting.notna(),"ts_code"],basic.loc[basic.delisting.notna(),"delisting"]))
    name_map = dict(zip(basic.ts_code,basic.name))
    index_path = ROOT/"data/raw/index_000300.SH.parquet"
    sources[str(index_path.relative_to(ROOT))] = digest(index_path)
    bench = pd.read_parquet(index_path)
    bench["date"] = pd.to_datetime(bench.trade_date.astype(str),format="%Y%m%d")
    bench = bench.sort_values("date").set_index("date")
    dates = bench.loc["2017-01-01":min(raw.date.max(),pd.Timestamp("2026-10-07"))].index
    codes = sorted(raw.ts_code.unique())
    chunks = []
    for number,(symbol,f) in enumerate(raw.loc[raw.valid].groupby("ts_code",sort=True),1):
        chunks.append(continuous(f))
        if number%1500 == 0:
            print(f"Continuous adjustment {number}/{len(codes)}",flush=True)
    adjusted = pd.concat(chunks,ignore_index=True)
    del raw,chunks
    def matrix(frame: pd.DataFrame, col: str) -> pd.DataFrame:
        return frame.pivot(index="date",columns="ts_code",values=col).reindex(index=dates,columns=codes).astype("float32")
    close,open_,high,low,amount = [matrix(adjusted, col) for col in ["close","open","high","low","amount"]]
    observed = close.notna()
    del adjusted
    info = features(close,amount,observed,high,low)
    known=observed.copy();st=observed&False
    if supplement is not None:
        name_files=sorted((supplement/"names").glob("*.parquet"))
        for path in name_files:sources[str(path.relative_to(ROOT))]=digest(path)
        names=pd.concat([pd.read_parquet(path) for path in name_files],ignore_index=True)
        known,st=historical_status(names,dates,codes)
        info["eligible"] &= known & ~st
        pd.DataFrame({"date":dates,"observed":observed.sum(axis=1),"known_status_observed":(known&observed).sum(axis=1),"known_ST_observed":(known&st&observed).sum(axis=1),"unknown_status_observed":(~known&observed).sum(axis=1)}).to_csv(output/"historical_status_coverage.csv",index=False)
    price = info["close"]
    # Delisted prices must not be indefinitely marked at the last observation.
    for symbol,day in delist.items():
        if symbol in price:
            price.loc[price.index>=day,symbol] = 0.
    weekly = dates.to_series().groupby(dates.to_period("W-FRI")).last()
    weekly = pd.DatetimeIndex(weekly[weekly.ge("2018-01-01")])
    monthly = pd.DatetimeIndex(dates.to_series().groupby(dates.to_period("M")).last())
    monthly = monthly[monthly >= "2018-01-01"]
    selected = {w:choose_signals(info,weekly,w) for w in [20,60]}
    all_signals = pd.concat(selected.values(),ignore_index=True)
    all_signals["display_name_current_not_signal_input"] = all_signals.symbol.map(name_map)
    all_signals.to_parquet(output/"signals.parquet",index=False)
    print(f"Signals {len(all_signals):,}; data end {dates[-1].date()}",flush=True)
    # Prefix test with poisoned future, checks every score and selected code/rank.
    cut = pd.Timestamp("2021-12-31")
    small_codes = codes[:80]
    short = features(close.loc[:cut,small_codes],amount.loc[:cut,small_codes],observed.loc[:cut,small_codes],high.loc[:cut,small_codes],low.loc[:cut,small_codes])
    if supplement is not None:
        past_names=names.loc[pd.to_datetime(names.ann_date,format="%Y%m%d",errors="coerce")<=cut]
        short_known,short_st=historical_status(past_names,dates[dates<=cut],small_codes)
        short["eligible"] &= short_known&~short_st
        pd.testing.assert_frame_equal(short_known,known.loc[:cut,small_codes])
        pd.testing.assert_frame_equal(short_st,st.loc[:cut,small_codes])
    for key in ["return20","return60","ma60","atr14","eligible"]:
        pd.testing.assert_frame_equal(short[key],info[key].loc[:cut,small_codes],check_dtype=False)
    labels, opportunities = [], []
    symbol_ix = {c:j for j,c in enumerate(codes)}
    ix = {d:i for i,d in enumerate(dates)}
    arr = price.to_numpy()
    opens = open_.to_numpy()
    for row in all_signals.itertuples(index=False):
        t = ix[row.signal_date]+1
        j = symbol_ix[row.symbol]
        if t >= len(dates) or not np.isfinite(opens[t,j]) or opens[t,j] <= 0:
            continue
        for horizon in [252,504]:
            s = forward_stats(float(opens[t,j]),arr[t:t+horizon,j],horizon)
            labels.append({"signal_date":row.signal_date,"entry_date":dates[t],"symbol":row.symbol,
                           "window":row.window,"horizon":horizon,**s})
    outcomes = pd.DataFrame(labels)
    outcomes.to_parquet(output/"selected_forward_outcomes.parquet",index=False)
    for day in monthly:
        t = ix[day]+1
        if t >= len(dates):
            continue
        eligible = info["base_eligible"].loc[day].to_numpy()
        for j in np.flatnonzero(eligible & np.isfinite(opens[t]) & (opens[t]>0)):
            for horizon in [252,504]:
                if t+horizon > len(dates):
                    continue
                s = forward_stats(float(opens[t,j]),arr[t:t+horizon,j],horizon)
                if s["peak_multiple"] >= 3:
                    window = arr[t:t+horizon,j]
                    for multiple in [3,5]:
                        if s["peak_multiple"] < multiple:
                            continue
                        hits = np.flatnonzero(window >= opens[t,j]*multiple)
                        opportunities.append({"anchor":day,"entry_date":dates[t],"symbol":codes[j],
                                              "horizon":horizon,"multiple":multiple,
                                              "first_hit":dates[t+hits[0]],"known_non_ST_at_anchor":bool(known.iloc[t-1,j] and not st.iloc[t-1,j]),**s})
    bulls = pd.DataFrame(opportunities)
    # One opportunity per symbol/anchor-year/horizon/multiple, rather than counting
    # every monthly duplicate as a separately captured bull.
    bulls["year"] = bulls.anchor.dt.year
    scopes=[]
    for scope,subset in [("all",bulls),("known_non_ST",bulls.loc[bulls.known_non_ST_at_anchor])]:
        subset=subset.sort_values(["peak_multiple","anchor"],ascending=[False,True]).drop_duplicates(["symbol","year","horizon","multiple"]).copy()
        subset["scope"]=scope;scopes.append(subset)
    bulls=pd.concat(scopes,ignore_index=True)
    recalls = []
    for window,signal in selected.items():
        groups = {c:f.signal_date.to_numpy() for c,f in signal.groupby("symbol")}
        captured, early, remaining, firsts = [], [], [], []
        for row in bulls.itertuples(index=False):
            days = groups.get(row.symbol,np.array([],dtype="datetime64[ns]"))
            days = days[(days>=np.datetime64(row.anchor))&(days<np.datetime64(row.first_hit))]
            captured.append(bool(len(days)))
            if len(days):
                first = pd.Timestamp(days[0]);t=ix[first];j=symbol_ix[row.symbol]
                remaining.append(row.peak_multiple*opens[ix[row.entry_date],j]/price.iloc[t,j])
                early.append(price.iloc[t,j]/opens[ix[row.entry_date],j] <= 1.5)
                firsts.append(first)
            else:
                remaining.append(np.nan);early.append(False);firsts.append(pd.NaT)
        trial = bulls.copy();trial["window"]=window;trial["captured"]=captured
        trial["early_before_1_5x"]=early;trial["remaining_peak_multiple"]=remaining;trial["first_signal"]=firsts
        recalls.append(trial)
    recall = pd.concat(recalls,ignore_index=True)
    recall["display_name_current_not_signal_input"] = recall.symbol.map(name_map)
    recall.to_csv(output/"bull_coverage_details.csv",index=False)
    recall.groupby(["scope","window","horizon","multiple","year"]).agg(bulls=("symbol","size"),captured=("captured","sum"),early=("early_before_1_5x","sum"),remaining_multiple_median=("remaining_peak_multiple","median")).to_csv(output/"bull_coverage_by_year.csv")
    recall.loc[recall.scope.eq("known_non_ST")].groupby(["window","horizon","multiple","year"]).agg(bulls=("symbol","size"),captured=("captured","sum"),early=("early_before_1_5x","sum")).to_csv(output/"bull_coverage_non_ST_by_year.csv")
    precision = []
    for (w,h),part in outcomes.loc[outcomes.mature].groupby(["window","horizon"]):
        for multiple in [3,5]:
            precision.append({"window":int(w),"horizon":int(h),"multiple":multiple,"mature_signals":len(part),"unique_symbols":part.symbol.nunique(),"peak_hit_rate":float(part.peak_multiple.ge(multiple).mean()),"terminal_hit_rate":float(part.terminal_multiple.ge(multiple).mean())})
    pd.DataFrame(precision).to_csv(output/"selected_bull_precision.csv",index=False)
    print(f"Bull episodes {len(bulls):,}; causal prefix check passed",flush=True)
    # Frozen exit grid, optimize only development net CAGR, never future labels.
    rules = [ExitRule("weekly_rotation",rotation=True)] + [ExitRule(f"trend_MA{m}_trail{int(d*100)}",m,d) for m in [20,60,120] for d in [.15,.25,.35]]
    results = []
    for rule in rules:
        nav,trades = simulate(price,open_,high,low,observed,info["atr14"],selected[20],rule,delist,"2018-01-01",str(dates[-1].date()),entry_allowed=known&~st,risk_warning=st)
        nav.to_csv(output/f"nav_{rule.name}.csv",index=False)
        trades.to_csv(output/f"trades_{rule.name}.csv",index=False)
        for period,lo,hi in [("full","2018-01-01",str(dates[-1].date())),("development","2018-01-01","2021-12-31"),("chronological","2022-01-01","2024-12-31"),("recent","2025-01-01",str(dates[-1].date()))]:
            sample=nav.loc[nav.date.between(lo,hi)].copy()
            prior=nav.loc[nav.date<pd.Timestamp(lo),"nav"]
            sample["nav"] /= prior.iloc[-1] if len(prior) else 1.
            results.append({"strategy":rule.name,"period":period,**metrics(sample)})
        print(f"Simulated {rule.name}; completed trades {len(trades)}",flush=True)
    summary = pd.DataFrame(results)
    best = summary.loc[(summary.period=="development")&summary.strategy.ne("weekly_rotation")].sort_values("annual_return",ascending=False).iloc[0].strategy
    chosen = next(r for r in rules if r.name==best)
    for risk,cost,name in [(True,.001,best+"_risk_budget"),(False,.003,best+"_cost30bps"),(True,.003,best+"_risk_cost30bps")]:
        rule = ExitRule(name,chosen.ma_days,chosen.trailing,risk_sizing=risk)
        nav,trades = simulate(price,open_,high,low,observed,info["atr14"],selected[20],rule,delist,"2018-01-01",str(dates[-1].date()),cost=cost,entry_allowed=known&~st,risk_warning=st)
        nav.to_csv(output/f"nav_{name}.csv",index=False);trades.to_csv(output/f"trades_{name}.csv",index=False)
        for period,lo,hi in [("full","2018-01-01",str(dates[-1].date())),("development","2018-01-01","2021-12-31"),("chronological","2022-01-01","2024-12-31"),("recent","2025-01-01",str(dates[-1].date()))]:
            sample=nav.loc[nav.date.between(lo,hi)].copy();prior=nav.loc[nav.date<pd.Timestamp(lo),"nav"]
            sample["nav"] /= prior.iloc[-1] if len(prior) else 1.
            results.append({"strategy":name,"period":period,**metrics(sample)})
    benchmark=bench.close.reindex(dates)
    for period,lo,hi in [("full","2018-01-01",str(dates[-1].date())),("development","2018-01-01","2021-12-31"),("chronological","2022-01-01","2024-12-31"),("recent","2025-01-01",str(dates[-1].date()))]:
        ret=benchmark.pct_change(fill_method=None).loc[lo:hi].fillna(0)
        results.append({"strategy":"CSI300_price_index","period":period,**metrics(pd.DataFrame({"nav":(1+ret).cumprod(),"exposure":1.}))})
    pd.DataFrame(results).to_csv(output/"portfolio_summary.csv",index=False)
    mainline = mainline_study(price,amount,observed,bench,output,sources)
    for file,hash_ in sources.items():
        if digest(ROOT/file) != hash_:
            raise ValueError("Input changed during study: "+file)
    manifest={"status":"completed_module_diagnostic_not_full_strategy_validation","start":"2018-01-01","end":str(dates[-1].date()),"observed_symbols":len(codes),"observed_rows":int(audit.observed.sum()),"missing_historical_industry_fraction":float(1-audit.industry_available.sum()/audit.observed.sum()),"missing_historical_name_fraction":float(1-audit.name_available.sum()/audit.observed.sum()),"best_exit_development_only":best,"optimizations_tested":9,"prefix_invariance":"passed_80_symbols_all_features","sources_sha256":sources,"script_sha256":digest(Path(__file__)),"protocol_sha256":digest(output/"protocol.json"),"mainline":mainline,"limitations":protocol["limitations_declared_before_run"]+["canonical_mysql_unavailable; explicit_read_only_parquet_mirror","lifecycle_metadata_not_independently_exhaustive","monthly_best_bull_episode_labels_are_hindsight_evaluation_only","display_names_current_not_signal_inputs","repeated_selected_signals_not_independent","censored_signals_reported_separately","no_fundamental_or_event_signals_reconstructed"]}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ["sources_sha256","limitations"]},ensure_ascii=False),flush=True)


def mainline_study(price: pd.DataFrame, amount: pd.DataFrame, observed: pd.DataFrame,
                   benchmark: pd.DataFrame, output: Path, sources: dict[str,str]) -> dict:
    """Limited market-trend mainline proxy. Not narrative/policy-mainline truth."""
    path=ROOT/"reports/research/b1_mainline_leaders_20261004/historical_members.csv"
    sources[str(path.relative_to(ROOT))]=digest(path)
    members=pd.read_csv(path);members["year"]=pd.to_datetime(members.snapshot_date).dt.year+1
    returns=price.pct_change(fill_method=None)
    above=price>price.rolling(20).mean()
    market_amount=amount.fillna(0).sum(axis=1)
    rows=[]
    for (year,code),f in members.groupby(["year","index_code"]):
        symbols=[s for s in f.symbol.unique() if s in price.columns]
        days=price.index[price.index.year==year]
        obs=observed.loc[days,symbols]
        panel=pd.DataFrame({"date":days,"code":code,"name":f.index_name.iloc[0],
            "daily_return":returns.loc[days,symbols].where(obs).mean(axis=1).to_numpy(),
            "share":amount.loc[days,symbols].sum(axis=1).div(market_amount.loc[days]).to_numpy(),
            "breadth":above.loc[days,symbols].where(obs).mean(axis=1).to_numpy(),
            "coverage":obs.sum(axis=1).div(len(f.symbol.unique())).to_numpy()})
        rows.append(panel)
    panel=pd.concat(rows,ignore_index=True).sort_values(["code","date"])
    parts=[]
    for code,f in panel.groupby("code"):
        f=f.copy();f["equity"]=(1+f.daily_return.fillna(0)).cumprod()
        f["ret20"]=f.equity.pct_change(20,fill_method=None);f["ret60"]=f.equity.pct_change(60,fill_method=None)
        f["activity"]=f.share.rolling(5).mean()/f.share.shift(5).rolling(60).mean()
        f["above_ma60"]=f.equity>f.equity.rolling(60).mean()
        for h in [20,60]:f[f"future{h}"]=f.equity.shift(-h)/f.equity-1
        parts.append(f)
    panel=pd.concat(parts,ignore_index=True)
    bench20=benchmark.close.pct_change(20,fill_method=None)
    panel["relative20"]=panel.ret20-panel.date.map(bench20)
    panel["score"]=sum(panel.groupby("date")[col].rank(pct=True)*weight for col,weight in [("relative20",.45),("ret60",.25),("activity",.2),("breadth",.1)])
    panel["selected"]=False
    ok=(panel.ret20>0)&(panel.relative20>0)&panel.above_ma60&(panel.breadth>=.5)&(panel.coverage>=.7)
    top=panel.loc[ok].sort_values(["date","score","code"],ascending=[True,False,True]).groupby("date").head(2)
    panel.loc[top.index,"selected"]=True
    panel.to_csv(output/"mainline_proxy_daily.csv",index=False)
    summary=[]
    for horizon in [20,60]:
        valid=panel.loc[panel[f"future{horizon}"].notna()].copy()
        valid["future_top3"]=valid.groupby("date")[f"future{horizon}"].rank(method="first",ascending=False)<=3
        picks=valid.loc[valid.selected]
        summary.append({"horizon":horizon,"picked_theme_days":len(picks),"positive_forward_fraction":float(picks[f"future{horizon}"].gt(0).mean()),"future_top3_precision":float(picks.future_top3.mean()),"future_top3_recall":float((valid.selected&valid.future_top3).sum()/valid.future_top3.sum()),"selected_mean_forward_return":float(picks[f"future{horizon}"].mean()),"all_theme_mean_forward_return":float(valid[f"future{horizon}"].mean())})
    pd.DataFrame(summary).to_csv(output/"mainline_proxy_summary.csv",index=False)
    return {"start":str(panel.date.min().date()),"end":str(panel.date.max().date()),"themes":int(panel.code.nunique()),"status":"limited_2024plus_market_trend_proxy_not_2018plus_policy_mainline"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--supplement",type=Path)
    args=parser.parse_args()
    run(args.output,args.supplement)
