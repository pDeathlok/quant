"""Account floor, symmetric loss budget and legacy replay invariants."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/research"))
from evaluate_index_futures_risk_paths import risk_replay
from test_index_futures_risk_paths import scenario
from test_index_futures_funded import row


def test_principal_halt_uses_previous_close_and_remains_halted_after_recovery():
    groups,opens,closes=scenario([1000,600,550,1000,1100],[1000,990,550,1000])
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,trace=True,
        risk_overlay="fuse20",principal_floor=400000,entry_calendar={r[0].entry_date for r in groups})
    assert trace.action.tolist()==["enter","hold","principal_halt","principal_halt"]
    assert ledger.equity.iloc[-1]==365000 # Gap overshoots the400k threshold.
    assert ledger.gross_lots.tolist()==[1,1,0,0]


def test_floor_is_absolute_and_not_reset_to_current_equity():
    r=row()
    ledger,_,trace=risk_replay([[r]],{},{},0,.15,capital=350000,trace=True,
        risk_overlay="fuse20",principal_floor=400000)
    assert trace.action.item()=="principal_halt" and ledger.gross_lots.item()==0


def test_extra_loss_stop_composes_with_other_exit_models():
    groups,opens,closes=scenario([1000,800,900,400],[1000,990,800])
    callback=lambda *args:(False,{})
    ledger,_,trace=risk_replay(groups,opens,closes,0,.15,variant="mean5",trace=True,
        exit_decider=callback,risk_overlay="stop5")
    assert trace.action.tolist()==["enter","hold","budget_loss_exit"]
    assert ledger.gross_lots.tolist()==[1,1,0]


def test_entry_budget_preserves_funding_and_is_symmetric():
    for side in (1,-1):
        a,b=row("IF",5000,.9),row("IH",2000,.8)
        if side<0:
            for r in (a,b):r.long_60,r.short_60=r.short_60,r.long_60
        ledger,_,trace=risk_replay([[a,b]],{},{},0,.15,trace=True,risk_overlay="entry5")
        assert ledger.positions.item()==f"IH2206:{side:+d}"
        assert trace.affordable_candidates.item()==2 and trace.risk_candidates.item()==1


def test_risk_rejection_is_not_counted_as_capital_rejection():
    r=row("IF",5000,.9)
    ledger,_,trace=risk_replay([[r]],{},{},0,.15,trace=True,risk_overlay="entry5")
    assert trace.action.item()=="risk_block"
    assert trace.raw_candidates.item()==trace.affordable_candidates.item()==1
    assert ledger.pnl.item()==0


def test_disabled_overlay_preserves_old_cash_and_grid():
    groups,opens,closes=scenario([1000,1010,1020,1030],[1000,1005,1015],roll=True)
    a,_,_=risk_replay(groups,opens,closes,.0006,.25,variant="loss_exit")
    b,_,_=risk_replay(groups,opens,closes,.0006,.25,variant="loss_exit",risk_overlay="none",principal_floor=600000)
    np.testing.assert_array_equal(a.equity,b.equity)
    assert a.positions.tolist()==b.positions.tolist()
