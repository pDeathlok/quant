"""Execution memory, cash recovery and causal activation invariants."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research"))
from index_futures_reentry import ReentryGate
from evaluate_index_futures_reentry_cash import cash_select, ReentryReplay
from evaluate_index_futures_risk_paths import risk_replay
from test_index_futures_funded import row
from test_index_futures_risk_paths import scenario


def observe(g,step,positions,confidence=.8,missing=False):
    r=row(entry=f"202201{step+2:02d}");r.trade_date=f"202201{step+1:02d}"
    r.long_60=confidence
    if missing:r.etf_ret_1=np.nan
    g.observe(r.entry_date,step,positions,[r]);return r


def test_cooldown_counts_five_executable_sessions_after_actual_exit():
    g=ReentryGate("cooldown5")
    observe(g,0,{})
    observe(g,1,{"IF2206":(1,300000)}) # position closes at this open
    observe(g,2,{})
    assert not g.admits("IF",1,2) and not g.admits("IF",1,6)
    assert g.admits("IF",1,7)
    assert g.admits("IF",-1,2) and g.admits("IH",1,2)


def test_roll_does_not_create_an_exit():
    g=ReentryGate("reset")
    observe(g,0,{"IF2206":(1,300000)})
    observe(g,1,{"IF2207":(1,300000)})
    assert not g.blocked


def test_signal_reset_needs_finite_post_exit_evidence():
    g=ReentryGate("reset")
    observe(g,0,{"IF2206":(1,300000)},confidence=.2) # pre-exit invalid signal
    observe(g,1,{},confidence=.8)
    assert not g.admits("IF",1,1)
    observe(g,2,{},confidence=.2,missing=True)
    assert not g.admits("IF",1,2)
    observe(g,3,{},confidence=.2)
    assert g.admits("IF",1,3)
    observe(g,4,{},confidence=.8)
    assert g.admits("IF",1,4)


def test_cash_quarter_and_grid_do_not_erase_reentry_memory():
    groups,opens,closes=scenario([1000]*10,[1000]*9)
    for i,g in enumerate(groups):g[0].quarter=f"Q{i}"
    # Actual exit when cash begins at step1; resume at step2, still blocked.
    schedule={groups[1][0].entry_date:"cash",groups[2][0].entry_date:"base"}
    calendar={g[0].entry_date for g in groups}
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,reentry_rule="cooldown5",
        entry_calendar=calendar,decision=lambda day,cutoff,current,*args:schedule.get(day,current))
    assert trace.action.iloc[1]=="strategy_switch"
    assert trace.action.iloc[2:7].eq("reentry_block").all()
    assert trace.action.iloc[7]=="enter"
    assert ledger.gross_lots.iloc[1:7].eq(0).all()


def scores(score=.02,daily=1000.):
    b=dict(ready=True,forced=0,median=.05,trades=8,active=30,dd90=.03,block=.1,score=score,daily_pnl=daily)
    return {p:dict(b) for p in ("base","aux55","confidence60","opposing_exit")}


def choose(values,current="cash",**kwargs):
    return cash_select(values,current,"fixed",rows=[row()],positions={},**kwargs)


def test_cash_can_pause_and_recover_without_impossible_zero_drawdown_gate():
    bad=scores(-.01)
    assert choose(bad,"base")[0]=="cash"
    assert choose(scores(),"cash")[0]=="base"


def test_cash_hysteresis_uses_existing_score_hurdle():
    v=scores(.003)
    assert choose(v,"cash")[1]=="cash_activation_score_hurdle"
    assert choose(v,"base")[0]=="base"
    assert choose(scores(0),"base")[0]=="cash"


def test_cash_requires_cash_edge_affordability_and_mature_coverage():
    assert choose(scores(daily=1))[1]=="cash_activation_cost_hurdle"
    assert choose(scores(),equity=1000)[1]=="cash_activation_not_affordable"
    v=scores();v['base']['trades']=1
    assert choose(v)[0]=="cash"
    v=scores();v['base']['ready']=False
    assert choose(v)[0]=="cash"


def test_cash_really_stays_flat_and_charges_no_fictitious_return():
    groups,opens,closes=scenario([1000,2000,3000,4000],[1000,2000,3000])
    ledger,_,trace=risk_replay(groups,opens,closes,.0006,.25,trace=True,policy="cash")
    assert ledger.pnl.eq(0).all() and ledger.equity.eq(500000).all()
    assert trace.action.eq("cash_wait").all()
    assert trace.cash_signal_candidates.iloc[0]==1
    assert trace.raw_candidates.eq(0).all()


def test_opportunity_attribution_counts_both_avoided_losses_and_missed_gains():
    import pandas as pd
    from report_index_futures_reentry_cash import attribution
    a=pd.DataFrame(dict(entry_date=['d1','d2','d3'],gross_lots=[1,1,1],pnl=[5.,-3.,2.]))
    b=pd.DataFrame(dict(entry_date=['d1','d2','d3'],gross_lots=[0,0,0],pnl=[-1.,0.,0.]))
    result=attribution(a,b)
    assert result['missed_positive_pnl']==7 and result['avoided_negative_pnl']==3
    assert result['new_flat_fees']==-1 and result['net_pnl_change']==-5
