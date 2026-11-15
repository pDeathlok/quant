"""Trade attribution and separately funded recovery scenarios; no deposit as PnL."""
import json
import numpy as np
import pandas as pd
from evaluate_index_futures_principal import (
    OUT, PREVIOUS, PrincipalReplay, HistoricalSelector, CAPITAL, DATE_TYPES,
    PROFILES, trade_episodes, metrics,
)
from evaluate_index_futures_risk_paths import diagnose_trades, minimum_equity
from refine_index_futures_adaptation import entry_candidates


def selected_cases(summary):
    p=summary[summary.window.eq("to_latest")&summary.startup_mode.isin(["actual_equity","equity_score_gate"])]
    cases=[]
    for (mode,fee,margin),g in p.groupby(["startup_mode","fee","margin"]):
        if (fee,margin) not in PROFILES:continue
        for metric in ("total_return","max_drawdown"):
            cases.append(g.loc[g[metric].idxmin()])
    cases.append(p[p.start.eq("20230801")&p.variant.eq("mean5")&p.startup_mode.eq("actual_equity")&p.margin.eq(.25)&p.fee.eq(.0006)].iloc[0])
    return pd.DataFrame(cases).drop_duplicates(["start","startup_mode","variant","fee","margin"]).reset_index(drop=True)


def recent_attribution(replay, out):
    """Descriptive2x2 with existing policies, no new strategy selection."""
    groups=[g for g in replay.groups if g[0].entry_date>="20260101"]
    records=[]
    for policy in ("base","confidence60"):
        for overlay in ("none","stop5"):
            ledger,_,_=replay.run(groups,.0006,.25,"mean5",policy,risk_overlay=overlay)
            trades=trade_episodes(ledger,replay.opens,.0006)
            records.append(dict(policy=policy,overlay=overlay,**metrics(ledger,CAPITAL),trades=len(trades),
                long_pnl=trades[trades.direction.gt(0)].pnl.sum(),short_pnl=trades[trades.direction.lt(0)].pnl.sum()))
            ledger.to_parquet(out/f"2026_fixed_{policy}_{overlay}.parquet",index=False)
    pd.DataFrame(records).to_csv(out/"2026_fixed_policy_stop_attribution.csv",index=False)
    comparisons=[]
    for mode in ("actual_equity","risk_stop5"):
        root=PREVIOUS if mode=="actual_equity" else OUT
        ledger=pd.read_parquet(root/"2026/ledgers"/f"20260101_mean5_{mode}_0.0006_0.25.parquet")
        comparisons.append(trade_episodes(ledger,replay.opens,.0006).assign(mode=mode))
    pd.concat(comparisons).to_csv(OUT/"2026_mean5_trade_comparison.csv",index=False)


def main():
    out=OUT/"diagnosis";out.mkdir(exist_ok=True)
    replay=PrincipalReplay();selector=HistoricalSelector(replay,None)
    summary=pd.read_csv(PREVIOUS/"summary.csv",dtype=DATE_TYPES)
    decisions=pd.read_csv(PREVIOUS/"selection.csv",dtype=DATE_TYPES)
    cases=selected_cases(summary)
    trades_all=[];demand_all=[];topups=[];case_stats=[];attributions=[]
    for number,r in enumerate(cases.itertuples(index=False)):
        cid=f"case{number+1:02d}"
        groups=[g for g in replay.groups if g[0].entry_date>=r.start]
        byday={g[0].entry_date:g for g in groups}
        d=decisions[decisions.start.eq(r.start)&decisions.startup_mode.eq(r.startup_mode)&decisions.fee.eq(r.fee)&decisions.margin.eq(r.margin)&decisions.variant.eq(r.variant)]
        initial=d.loc[d.stage.eq("startup"),"selected"].item()
        schedule=d[d.stage.eq("quarter")].set_index("date").selected.to_dict()
        ledger,_,trace=replay.run(groups,r.fee,r.margin,r.variant,initial,lambda day,cutoff,current,*args:schedule.get(day,current))
        old=pd.read_parquet(PREVIOUS/r.start[:4]/"ledgers"/f"{r.start}_{r.variant}_{r.startup_mode}_{r.fee}_{r.margin}.parquet")
        pd.testing.assert_frame_equal(ledger,old,check_dtype=False,atol=1e-6,rtol=1e-12)
        trace.to_csv(out/f"{cid}_trace.csv",index=False)
        trades=trade_episodes(ledger,replay.opens,r.fee)
        diagnosed=diagnose_trades(ledger,trades,trace,replay.closes,replay.opens,r.fee)
        extra=[]
        for t in diagnosed.itertuples(index=False):
            row=next(v for v in byday[t.entry] if v.symbol==t.product)
            held=ledger[ledger.entry_date.ge(t.entry)&ledger.entry_date.lt(t.exit)]
            fees=held.fees.sum()
            intraday=overnight=0.
            for h in held.itertuples(index=False):
                code,side=h.positions.split(":");side=int(side)
                dailyclose=replay.closes[h.entry_date,code]
                intraday+=side*(dailyclose-replay.opens[h.entry_date,code])*t.multiplier
                overnight+=side*(replay.opens[h.exit_date,code]-dailyclose)*t.multiplier
            # Episode PnL also includes exit commissions on the following flat row.
            fees_exact=intraday+overnight-t.pnl
            crossings=trace[trace.date.gt(t.entry)&trace.date.le(t.exit)&trace.close_pnl.le(-.05*t.entry_equity)]
            first_cross=crossings.iloc[0] if len(crossings) else None
            stop_pnl=(first_cross.equity-t.entry_equity
                -replay.opens[first_cross.date,next(h.positions.split(":")[0] for h in held.itertuples(index=False) if h.exit_date==first_cross.date)]*t.multiplier*r.fee) if first_cross is not None else np.nan
            extra.append(dict(entry=t.entry,exit=t.exit,confidence=max(row.long_60,row.short_60),
                own20=row.long_20 if t.direction>0 else row.short_20,
                opposite20=row.short_20 if t.direction>0 else row.long_20,
                aux_win=row.win_long if t.direction>0 else row.win_short,
                vol20=row.f_volatility_20,etf_ret1=row.etf_ret_1,notional=row.entry_open*row.multiplier,
                minimum_equity=minimum_equity(row,t.direction,r.fee,r.margin),
                net_loss_fraction=t.pnl/t.entry_equity,intraday_pnl=intraday,overnight_pnl=overnight,all_fees=fees_exact,
                first_stop_date=first_cross.date if first_cross is not None else "",first_stop_pnl=stop_pnl,
                actual_minus_first_stop=t.pnl-stop_pnl))
        diagnosed=diagnosed.merge(pd.DataFrame(extra),on=["entry","exit"],validate="one_to_one")
        trades_all.append(diagnosed.assign(case=cid,start=r.start,mode=r.startup_mode,variant=r.variant,fee=r.fee,margin=r.margin))
        demands=[]
        for t in trace[trace.raw_candidates.gt(0)].itertuples(index=False):
            rows=byday[t.date];lookup={v.symbol:v for v in rows}
            candidates=entry_candidates(rows,t.policy,t.equity,r.fee,r.margin,funded=False)
            for confidence,symbol,side in candidates:
                row=lookup[symbol];direction=1 if side=="long" else -1;n=row.entry_open*row.multiplier
                need=minimum_equity(row,direction,r.fee,r.margin)
                demands.append(dict(case=cid,date=t.date,policy=t.policy,action=t.action,equity=t.equity,
                    symbol=symbol,side=side,confidence=confidence,notional=n,
                    bare_margin_and_open_fee=n*(r.margin+r.fee),reserve_required=need,
                    bare_topup=max(0,n*(r.margin+r.fee)-t.equity),reserve_topup=max(0,need-t.equity),
                    entry_risk_required=2*row.f_volatility_20*n/.05))
        demand=pd.DataFrame(demands);demand_all.append(demand)
        halted_date=ledger.loc[ledger.gross_lots.gt(0),"exit_date"].max()
        freeze=demand[demand.date.ge(halted_date)]
        freeze=freeze.sort_values(["date","reserve_required"]).drop_duplicates("date")
        stat=dict(case=cid,start=r.start,mode=r.startup_mode,variant=r.variant,fee=r.fee,margin=r.margin,
            total_return=r.total_return,max_drawdown=r.max_drawdown,final_equity=ledger.equity.iloc[-1],trades=len(trades),
            last_position_exit=halted_date,first_frozen_signal=freeze.date.min(),frozen_signal_days=len(freeze),
            min_frozen_topup=freeze.reserve_topup.min(),median_frozen_topup=freeze.reserve_topup.median(),max_frozen_topup=freeze.reserve_topup.max(),
            first_frozen_topup=freeze.reserve_topup.iloc[0] if len(freeze) else np.nan,
            first_frozen_bare_topup=freeze.bare_topup.iloc[0] if len(freeze) else np.nan,
            first_frozen_symbol=freeze.symbol.iloc[0] if len(freeze) else "")
        case_stats.append(stat)
        # Descriptive fixed-policy-date ablation: isolates execution risk overlay,
        # not an investable future policy schedule. Main study reselects causally.
        for overlay in ("none","entry5","stop5","fuse20","combined"):
            l,_,tr=replay.run(groups,r.fee,r.margin,r.variant,initial,
                lambda day,cutoff,current,*args:schedule.get(day,current),risk_overlay=overlay)
            attributions.append(dict(case=cid,overlay=overlay,pnl=l.pnl.sum(),final_equity=l.equity.iloc[-1],
                halted_days=int(tr.action.eq("principal_halt").sum()),active_days=int(l.gross_lots.gt(0).sum())))
        # Causal supplement trigger: first blocked entry with equity below400k.
        trigger=trace[trace.action.eq("capital_block")&trace.equity.lt(.8*CAPITAL)]
        if not len(trigger):continue
        events_to_test=[("early_block",trigger.iloc[0])]
        if r.variant=="mean5" and r.startup_mode=="actual_equity" and len(freeze):
            # Diagnostic continuation at the observed frozen state, NOT a new
            # proposed timing rule or an OOT-selected intervention.
            events_to_test.append(("frozen_state",trace[trace.date.eq(freeze.date.iloc[0])].iloc[0]))
        for event_label,event in events_to_test:
            start_index=next(i for i,g in enumerate(groups) if g[0].entry_date==event.date)
            quotes=demand[demand.date.eq(event.date)]
            amounts=dict(minimum_resume=max(0.,quotes.reserve_required.min()-event.equity)+1.,restore_500k=max(0.,CAPITAL-event.equity))
            for amount_label,addition in amounts.items():
                label=event_label+"_"+amount_label
                choices=[]
                def choose(day,cutoff,current,equity,positions,rows):
                    selected,detail=selector.choose(cutoff,r.fee,r.margin,r.variant,equity,"none",current,
                        gated=r.startup_mode=="equity_score_gate",rows=rows,positions=positions)
                    choices.append(dict(day=day,cutoff=cutoff,equity=equity,selected=selected,**detail));return selected
                # Current policy already selected before this entry was blocked.
                # If trigger is a quarterly boundary, rerun that boundary at the new
                # equity against the pre-boundary incumbent, rather than freeze choice.
                current=event.policy
                same=d[d.stage.eq("quarter")&d.date.eq(event.date)]
                if len(same):
                    current=choose(event.date,groups[start_index][0].trade_date,same.incumbent.item(),event.equity+addition,{},groups[start_index])
                tail,_,ttrace=replay.run(groups[start_index:],r.fee,r.margin,r.variant,current,choose,capital=event.equity+addition)
                external=addition
                pnl=ledger.iloc[:start_index].pnl.sum()+tail.pnl.sum()
                np.testing.assert_allclose(CAPITAL+external+pnl,tail.equity.iloc[-1],rtol=0,atol=1e-6)
                # Funding time definition1: first new position's margin release when flat.
                episodes=trade_episodes(tail,replay.opens,r.fee)
                first_trade=episodes.iloc[0] if len(episodes) else None
                # Definition2: observe enough liquidatable own equity to return all
                # added cash AND leave500k. Observational only: no hypothetical early
                # withdrawal feeds back into this no-withdrawal path.
                close_value=[]
                for row_index,v in enumerate(tail.itertuples(index=False)):
                    cost=0
                    if v.positions:
                        code,sign=v.positions.split(":");mult=300 if code[:2] in ("IF","IH") else 200
                        cost=replay.opens[v.exit_date,code]*mult*r.fee
                    close_value.append(v.equity-(cost if row_index<len(tail)-1 else 0))
                recovery=np.flatnonzero(np.asarray(close_value)-external>=CAPITAL)
                recover_date=tail.exit_date.iloc[recovery[0]] if len(recovery) else ""
                topups.append(dict(case=cid,scenario=label,date=event.date,equity_before=event.equity,added_cash=external,
                    first_entry=first_trade.entry if first_trade is not None else "",first_exit=first_trade.exit if first_trade is not None else "",
                    first_trade_sessions=first_trade.days if first_trade is not None else 0,first_trade_pnl=first_trade.pnl if first_trade is not None else 0,
                    ending_equity=tail.equity.iloc[-1],total_net_profit_excluding_deposit=pnl,
                    return_on_total_contributed=pnl/(CAPITAL+external),own_equity_after_return_of_deposit=tail.equity.iloc[-1]-external,
                    recovery_to_500k_date=recover_date,recovery_sessions=int(recovery[0]+1) if len(recovery) else np.nan,
                    recovery_calendar_days=(pd.Timestamp(recover_date)-pd.Timestamp(event.date)).days if recover_date else np.nan,
                    observed_calendar_days=(pd.Timestamp(tail.exit_date.iloc[-1])-pd.Timestamp(event.date)).days,
                    trades=len(episodes),forced=int(tail.forced_exit.sum()),blocked=int(ttrace.action.eq("capital_block").sum())))
                tail.assign(external_balance=external).to_parquet(out/f"{cid}_{label}_ledger.parquet",index=False)
                pd.DataFrame(choices).to_csv(out/f"{cid}_{label}_selection.csv",index=False)
        print("diagnosed",cid,r.start,r.variant,r.startup_mode,flush=True)
    for name,frame in (("cases",pd.DataFrame(case_stats)),("trades",pd.concat(trades_all)),("entry_funding",pd.concat(demand_all)),("topups",pd.DataFrame(topups)),("fixed_schedule_attribution",pd.DataFrame(attributions))):
        frame.to_csv(out/f"{name}.csv",index=False)
    print(pd.DataFrame(case_stats).to_string(index=False));print(pd.DataFrame(topups).to_string(index=False))
    if (OUT/"summary.csv").exists():
        recent_attribution(replay,out)


if __name__=="__main__":main()
