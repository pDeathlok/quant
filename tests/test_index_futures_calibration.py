"""Mature-only calibration, empirical tails, equity timing and start windows."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from test_index_futures_funded import row
from test_index_futures_risk_paths import scenario
import evaluate_index_futures_calibration as research


def test_affine_calibration_shrinks_without_reversing_direction():
    a, b = research.affine_calibration([0, 1, 2], [1, 1.5, 2], [1, 1, 1])
    assert np.isclose(a, 1) and np.isclose(b, .5)
    a, b = research.affine_calibration([0, 1, 2], [2, 1, 0], [1, 1, 1])
    assert b == 0 and a == 1
    _, b = research.affine_calibration([0, 1, 2], [0, 3, 6], [1, 1, 1])
    assert b == 1


def test_quantile_obeys_weights_and_zero_tails():
    assert research.weighted_quantile([1, 100, 2], [.1, .1, .8], .9) == 2
    assert research.weighted_quantile([0, 0], [1, 2], .9) == 0


def test_calibration_uses_only_mature_past_oos_predictions():
    frame = pd.DataFrame(dict(direction=[1]*5, signal_date=["20210101", "20230101", "20230101", "20230101", "20230101"],
                              maturity=["20210105", "20230330", "20230331", "20230405", "20230330"],
                              net_5_3=[0]*5, risk_5_3=[.01]*5, pred_net_5_3=[0, 0, 0, 0, np.nan], pred_risk_5_3=[.01]*5))
    selected = research.calibration_history(frame, "20230331", 1, 3)
    assert selected.index.tolist() == [1]


def test_tail_rule_uses_current_close_equity_and_two_confirmations():
    state = SimpleNamespace(direction=1, cal_net_3=.01, risk90_3=.03, notional=1000000, cal_ready_3=True)
    lookup = {("e", "d1", "IF"): state, ("e", "d2", "IF"): state}
    target = SimpleNamespace(symbol="IF")
    low = research.calibrated_callback(lookup, "tail5", .0003)
    assert not low("d1", "e", target, 1, 500000)[0]
    exit_, detail = low("d2", "e", target, 1, 500000)
    assert exit_ and detail["exit_action"] == "tail_budget_exit"
    high = research.calibrated_callback(lookup, "tail5", .0003)
    assert not high("d1", "e", target, 1, 1000000)[0]
    assert not high("d2", "e", target, 1, 1000000)[0]
    mean_only = research.calibrated_callback(lookup, "calibrated5", .0003)
    assert not mean_only("d1", "e", target, 1, 500000)[0]
    assert not mean_only("d2", "e", target, 1, 500000)[0]


def test_tail_warmup_does_not_invent_a_bound():
    state = SimpleNamespace(direction=-1, cal_net_3=.01, risk90_3=np.nan, notional=1000000, cal_ready_3=False)
    lookup = {("e", "d1", "IF"): state, ("e", "d2", "IF"): state}
    choose = research.calibrated_callback(lookup, "tail5", .0003)
    assert not choose("d1", "e", SimpleNamespace(symbol="IF"), -1, 10000)[0]
    assert not choose("d2", "e", SimpleNamespace(symbol="IF"), -1, 10000)[0]


def test_state_callback_removes_current_open_gap():
    def run(opening):
        groups, opens, closes = scenario([1000, opening, 1800], [1000, 990])
        seen = []
        def callback(day, entry, r, side, equity):
            seen.append(equity)
            return False, {}
        research.risk_replay(groups, opens, closes, 0, .15, state_exit_decider=callback)
        return seen
    assert run(950) == run(1200) == [497000]


def test_common_length_prefix_liquidates_once():
    groups, opens, closes = scenario([1000]*5, [1000]*4)
    ledger, _, _ = research.risk_replay(groups, opens, closes, .0003, .15)
    prefix = research.liquidated_prefix(ledger, 2, opens, .0003)
    direct, _, _ = research.risk_replay(groups[:2], opens, closes, .0003, .15)
    pd.testing.assert_frame_equal(prefix, direct)
    whole = research.liquidated_prefix(ledger, len(ledger), opens, .0003)
    pd.testing.assert_frame_equal(whole, ledger)


def test_monthly_starts_and_phase_grid_are_not_selected_by_outcome():
    panel = pd.DataFrame(dict(entry_date=[f"202601{n:02}" for n in range(5, 10)]))
    specs = research.start_specs(panel, 2026)
    assert len(specs) == 14
    assert sum(s["annual"] for s in specs) == 1
    assert [s["phase"] for s in specs if s["start_kind"] == "phase"] == [1, 2, 3, 4]
    assert [s["start"] for s in specs if s["start_kind"] == "monthly"][-1] == "20261001"
