"""Frozen per-horizon advantage and actual-equity scoring contracts."""
import numpy as np

from test_index_futures_multistart import reference_windows, scores
import evaluate_index_futures_equity_gate as research


def scored():
    values=scores()
    values["base"].update(score_122=-.03,score_244=-.03)
    values["aux55"].update(score_122=.02,score_244=.03)
    return values


def test_startup_requires_advantage_in_both_lengths_not_only_average():
    values=scored()
    values["aux55"]["score_244"]=-.031
    assert research.gated_select(values,"base",False,initial=True)[0]=="aux55"
    selected,reason,detail=research.gated_select(values,"base",True,initial=True)
    assert selected=="base" and reason=="insufficient_horizon_score_advantage"
    assert detail["advantage_122"]>0 and detail["advantage_244"]<0


def test_same_fixed_floor_applies_at_startup_and_switch():
    values=scored()
    values["aux55"].update(score_122=-.026,score_244=-.026)
    for initial in (True,False):
        assert research.gated_select(values,"base",True,initial=initial)[1]=="insufficient_horizon_score_advantage"


def test_passing_startup_gate_needs_no_current_or_future_quotes():
    selected,_,detail=research.gated_select(scored(),"base",True,initial=True)
    assert selected=="aux55" and detail["horizon_gate_passed"]


def test_current_equity_cash_edge_uses_actual_reference_denominator():
    windows=reference_windows()
    at500=research.equity_scores(windows,"20220630",500000)
    at250=research.equity_scores(windows,"20220630",250000)
    assert np.isclose(at500["daily_pnl"],2*at250["daily_pnl"])
    assert at500["score"]==at250["score"]
    assert np.isclose(at500["score"],.5*(at500["score_122"]+at500["score_244"]))


def test_gate_off_preserves_old_selector_choices():
    from test_index_futures_funded import row
    for initial in (True,False):
        kwargs=dict(initial=initial,rows=[row()],positions={},equity=500000)
        old=research.select_policy(scored(),"base","multi_funded",**kwargs)
        new=research.gated_select(scored(),"base",False,**kwargs)
        assert old[:2]==new[:2]


def test_insufficient_evidence_or_tie_does_not_force_startup_switch():
    values=scored()
    values["aux55"]["score"]=values["base"]["score"]
    assert research.gated_select(values,"base",True,initial=True)[0]=="base"
    values["base"]["ready"]=False
    assert research.gated_select(values,"base",True,initial=True)[1]=="insufficient_mature_windows"
