"""Predeclared spot-trend overlay ablations with exact-equity rolling selection."""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

from evaluate_index_futures_reentry_cash import (
    ReentryReplay, FastInner, POLICIES, VARIANTS, PROFILES, ALL_PROFILES, CAPITAL,
    HORIZONS, DATE_TYPES, past_windows, start_specs, window_stats, trade_episodes,
    annual_stats, risk_replay, exit_callback, calibrated_callback, equity_scores,
    gated_select, REFERENCE, EQUITY, OUT as PREVIOUS,
)
from index_futures_regime import load_states, MODES

OUT = PREVIOUS.parent/"bull_bear_regime"
MA_WINDOWS = (200, 120, 250)
CONFIGS = {f"{style}_{mode}": (style, mode)
           for style in ("fixed", "rolling") for mode in MODES}


class RegimeReplay(ReentryReplay):
    def __init__(self, window=200, base=None):
        if base is None:
            super().__init__()
        else:
            self.__dict__.update(base.__dict__)
        self.window = window
        self.states = load_states(window)
        self.market_regime = {(r.trade_date, r.symbol): int(r.state)
                             for r in self.states.dropna(subset=["state"]).itertuples(index=False)}
        assert all((r.trade_date, r.symbol) in self.market_regime for g in self.groups for r in g)
        assert all(r.trade_date < r.entry_date for g in self.groups for r in g)

    def run(self, groups, fee, margin, variant, policy="base", decision=None,
            capital=CAPITAL, regime_mode="none"):
        simple = exit_callback(self.lookup, "mean5", fee) if variant == "mean5" else None
        stateful = calibrated_callback(self.lookup, variant, fee) if variant in ("calibrated5", "tail5") else None
        return risk_replay(groups, self.opens, self.closes, fee, margin, policy=policy, capital=capital,
            decision=decision, trace=True, variant=variant, exit_decider=simple,
            state_exit_decider=stateful, entry_calendar=self.calendar, regime_mode=regime_mode,
            market_regime=self.market_regime, continuation_lookup=self.lookup)


class Selector:
    def __init__(self, replay, out):
        self.fast, self.out, self.window = FastInner(replay), out, replay.window
        self.reference = pd.read_csv(REFERENCE, dtype=DATE_TYPES)
        self.audit = []
        if out is not None:
            (out/"validation").mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=None)
    def template(self, cutoff, fee, margin, variant):
        r = self.reference
        return past_windows(r[r.fee.eq(fee)&r.margin.eq(margin)&r.variant.eq(variant)], cutoff)

    @lru_cache(maxsize=4096)
    def score(self, cutoff, fee, margin, variant, capital, mode):
        key = [cutoff, fee, margin, variant, float(capital).hex(), mode, self.window]
        vid = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        template = self.template(cutoff, fee, margin, variant)
        if mode == "none" and capital == CAPITAL:
            windows = template.copy()
        else:
            windows = pd.DataFrame([self.fast.run(r.start, r.horizon, r.policy, variant, fee,
                margin, capital, regime_mode=mode)[0] for r in template.itertuples(index=False)])
        assert windows.end.max() <= cutoff
        if self.out is not None:
            windows.assign(validation_capital=capital, cutoff=cutoff, regime_mode=mode,
                           ma_window=self.window).to_parquet(self.out/"validation"/f"{vid}.parquet", index=False)
        scores = {}
        for policy in POLICIES:
            scores[policy] = equity_scores(windows[windows.policy.eq(policy)], cutoff, capital)
            self.audit.append(dict(validation_id=vid, cutoff=cutoff, fee=fee, margin=margin,
                variant=variant, validation_capital=capital, regime_mode=mode,
                ma_window=self.window, policy=policy, **scores[policy]))
        return scores, vid

    def choose(self, cutoff, fee, margin, variant, capital, mode, incumbent,
               initial=False, rows=None, positions=None):
        scores, vid = self.score(cutoff, fee, margin, variant, float(capital), mode)
        selected, reason, detail = gated_select(scores, incumbent, False, initial=initial,
            rows=rows, positions=positions, equity=capital, fee=fee, margin=margin)
        return selected, dict(validation_id=vid, validation_capital=capital, reason=reason, **detail)


def register():
    OUT.mkdir(exist_ok=True)
    protocol = dict(capital=CAPITAL, max_lots=1, end="20261106", primary_ma=200,
        sensitivity_ma=[120, 250], sensitivity_scope="same full start/profile grid, mean5 exit only; no parameter selection",
        states="per underlying spot index: bull close>MA+ATR20 AND MA>MA.shift20; bear close<MA-ATR20 AND MA<MA.shift20; otherwise transition. ATR arithmetic20d mean true range; missing warmup fails validation, no backfill",
        entry="bull rejects new shorts; bear rejects new longs; transition unchanged; highest-confidence remaining affordable candidate, can substitute product",
        exit="retain all old exits; additionally, state != position direction AND frozen raw5 predicted incremental net return<=0 => exit next open on one observation; no exit solely on MA crossing, no immediate reversal",
        design="fixed base vs quarterly reselect x none/entry/exit/both; MA200 all5 original exit controls, MA120/250 mean5 only",
        model="existing quarterly-trained forecasts frozen; no new LR interactions fitted; this isolates operational state gates, not the efficacy of a retrained regime model",
        history="past36m complete122/244d monthly starts,12m recency half-life, exact current equity and same regime overlay in historical replay; same original candidate policies and switching gates",
        state_maturity="signal-day close only, strictly before execution date; state can update daily, new entries only on unchanged global5session calendar; state history never resets at account startup",
        fixed="fixed base entry policy; existing quarterly-updated prediction panel still used, no new retraining",
        controls="old fixed_none and actual_equity rerun and compare every account; all older studies retained in place",
        evaluation="same58monthly starts,first5January session phase grid,annual4cost/marginprofiles; single_year,to_latest,122/244sessions,continuous annual marks; 2025/2026 and worst starts; no2022-prefix-to-each-year-end tables",
        friction="original3/6bps each charged leg and15/25% margin scenarios, original20% reserve; no new cash gate/reentry/stop/topup; cash earns0",
        opportunity="matched-baseline positive daily pnl missed and negative pnl avoided while new account is flat; other path changes and close fees separately reconcile; diagnostic, not realizable cherry-picked returns",
        limits="no fresh OOT; overlapping windows not independent; daily-bar fills cannot certify intraday solvency; no parameter fitting against known losses")
    path = OUT/"protocol.json"
    text = json.dumps(protocol, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text() != text:
        raise RuntimeError("Refuse changing an existing registered protocol")
    path.write_text(text)
    (OUT/"global_calendar.csv").write_bytes((PREVIOUS/"global_calendar.csv").read_bytes())
    sources = []
    for window in MA_WINDOWS:
        states = load_states(window)
        path = OUT/f"states_{window}.parquet"
        states.to_parquet(path, index=False)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            first=states.trade_date.min(), last=states.trade_date.max(), rows=len(states)))
    for path in (PREVIOUS/"protocol.json", PREVIOUS/"summary.csv",
                 PREVIOUS.parent/"adaptation_comparison/quarterly_predictions.parquet",
                 PREVIOUS.parent/"exit_calibration_starts/calibrated_forecasts.parquet"):
        if path.exists():
            sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT/"sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2))


def validate():
    replay = RegimeReplay()
    fast = FastInner(replay)
    ref = pd.read_csv(REFERENCE, dtype=DATE_TYPES)
    columns = ["total_return", "max_drawdown", "trades", "active_days", "forced",
               "signal_opportunities", "blocked_opportunities"]
    for r in ref.itertuples(index=False):
        got, _ = fast.run(r.start, r.horizon, r.policy, r.variant, r.fee, r.margin, CAPITAL)
        np.testing.assert_allclose([got[k] for k in columns], [getattr(r,k) for k in columns], atol=1e-9, rtol=0)
    count, error = 0, 0.
    for window in MA_WINDOWS:
        r = replay if window == 200 else RegimeReplay(window, replay)
        f = fast if window == 200 else FastInner(r)
        for capital in (265421.952, 500000., 735432.123):
            for start in ("20230103", "20240902"):
                i = f.index[start]
                for fee, margin in PROFILES:
                    for variant in (VARIANTS if window == 200 else ("mean5",)):
                        for policy in POLICIES:
                            for mode in MODES[1:]:
                                result, daily = f.run(i,244,policy,variant,fee,margin,capital,trace=True,regime_mode=mode)
                                ledger, _, trace = r.run(r.groups[i:i+244],fee,margin,variant,policy,capital=capital,regime_mode=mode)
                                expected = np.column_stack([ledger.pnl,ledger.equity,ledger.gross_lots,ledger.fees,
                                    ledger.forced_exit,ledger.settlement_breach,trace.raw_candidates,trace.affordable_candidates])
                                error = max(error,float(abs(daily[:,:8]-expected).max()))
                                np.testing.assert_allclose(daily[:,:8],expected,atol=1e-6,rtol=0)
                                positions=["" if c<0 else f"{f.codes[int(c)]}:{int(s):+d}" for c,s in daily[:,8:10]]
                                assert positions == ledger.positions.tolist()
                                assert result["trades"] == len(trade_episodes(ledger,r.opens,fee))
                                count += 1
            print("validated",window,capital,count,error,flush=True)
    (OUT/"engine_checks.json").write_text(json.dumps(dict(reference_windows=len(ref),daily_cases=count,max_cash_error=error),indent=2))


def run_year(year):
    base = RegimeReplay()
    out = OUT/str(year)
    for folder in ("ledgers", "traces"):
        (out/folder).mkdir(parents=True,exist_ok=True)
    summaries, annual, trades_all, decisions, audits = [], [], [], [], []
    controls = 0
    for window in MA_WINDOWS:
        replay = base if window == 200 else RegimeReplay(window,base)
        selector = Selector(replay,out)
        for spec in start_specs(replay.panel,year):
            start = spec["start"]
            groups = [g for g in replay.groups if g[0].entry_date>=start]
            profiles = ALL_PROFILES if spec["annual"] else PROFILES
            variants = (VARIANTS[:3] if spec["start_kind"]=="phase" else VARIANTS) if window==200 else ("mean5",)
            for fee,margin in profiles:
                for variant in variants:
                    for mode,(style,overlay) in CONFIGS.items():
                        if window!=200 and overlay=="none":
                            continue
                        fields = dict(**spec,fee=fee,margin=margin,variant=variant,startup_mode=mode,
                                      ma_window=window,first_entry_date=groups[0][0].entry_date)
                        first = groups[0][0]
                        initial = "base"
                        if style=="rolling":
                            initial,detail = selector.choose(first.trade_date,fee,margin,variant,CAPITAL,overlay,
                                "base",initial=True,rows=groups[0],positions={})
                            decisions.append(dict(**fields,stage="startup",date=first.entry_date,
                                cutoff=first.trade_date,equity=CAPITAL,incumbent="base",selected=initial,**detail))
                        def decision(day,cutoff,incumbent,equity,positions,rows):
                            selected,detail = selector.choose(cutoff,fee,margin,variant,equity,overlay,
                                incumbent,rows=rows,positions=positions)
                            decisions.append(dict(**fields,stage="quarter",date=day,cutoff=cutoff,
                                equity=equity,incumbent=incumbent,selected=selected,**detail))
                            return selected
                        ledger,events,trace = replay.run(groups,fee,margin,variant,initial,
                            decision if style=="rolling" else None,regime_mode=overlay)
                        if overlay=="none":
                            root,oldmode = (PREVIOUS,"fixed_none") if style=="fixed" else (EQUITY,"actual_equity")
                            old = pd.read_parquet(root/str(year)/"ledgers"/f"{start}_{variant}_{oldmode}_{fee}_{margin}.parquet")
                            pd.testing.assert_frame_equal(ledger,old,check_dtype=False,atol=1e-6,rtol=1e-12)
                            controls += 1
                        assert trace.loc[trace.action.eq("enter"),"date"].isin(replay.calendar).all()
                        trades = trade_episodes(ledger,replay.opens,fee)
                        np.testing.assert_allclose(trades.pnl.sum(),ledger.pnl.sum(),atol=1e-6,rtol=0)
                        trades_all.append(trades.assign(**fields))
                        if spec["annual"]:
                            annual.extend(dict(**fields,**r) for r in annual_stats(ledger,trades))
                        for stats in window_stats(ledger,trace,events,start,fee,replay):
                            t = trace.iloc[:stats["days"]]
                            stats.update(regime_exits=int(t.action.eq("regime_exit").sum()),
                                regime_blocks=int(t.action.eq("regime_block").sum()),
                                regime_rejected_days=int(t.get("regime_rejected",pd.Series(0,index=t.index)).fillna(0).gt(0).sum()),
                                substitution_entries=int(t.get("regime_substitution",pd.Series(False,index=t.index)).fillna(False).sum()))
                            summaries.append(dict(**fields,**stats))
                        filename=f"{start}_{variant}_{mode}_{fee}_{margin}_ma{window}.parquet"
                        ledger.to_parquet(out/"ledgers"/filename,index=False)
                        trace.to_parquet(out/"traces"/filename,index=False)
            print("evaluated",year,window,start,selector.score.cache_info(),flush=True)
        audits.extend(selector.audit)
        # lru_cache keys include self; explicitly release large completed-window caches.
        selector.score.cache_clear()
        selector.template.cache_clear()
    for name,records in (("summary",summaries),("annual_marked",annual),("selection",decisions),("selection_audit",audits)):
        pd.DataFrame(records).to_csv(out/f"{name}.csv",index=False)
    pd.concat(trades_all,ignore_index=True).to_csv(out/"trades.csv",index=False)
    (out/"checks.json").write_text(json.dumps(dict(replayed_controls=controls)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--year",type=int)
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--skip-validation",action="store_true")
    args=parser.parse_args()
    if args.year:
        run_year(args.year)
        return
    register()
    if not args.skip_validation:
        validate()
    if args.validate_only:
        return
    def worker(year):
        subprocess.run([sys.executable,__file__,"--year",str(year)],check=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker,range(2022,2027)))
    for name in ("summary","annual_marked","trades","selection","selection_audit"):
        pd.concat([pd.read_csv(OUT/str(y)/f"{name}.csv",dtype=DATE_TYPES,low_memory=False)
                   for y in range(2022,2027)],ignore_index=True).to_csv(OUT/f"{name}.csv",index=False)


if __name__=="__main__":
    main()
