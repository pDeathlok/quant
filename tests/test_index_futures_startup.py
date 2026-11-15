"""Global calendar invariance and history-only initial policy selection."""
import pandas as pd

from test_index_futures_funded import row, scores
from test_index_futures_risk_paths import scenario
import evaluate_index_futures_startup as research


def test_calendar_is_prefix_invariant_and_not_reset_at_new_year():
    dates = ["20201228", "20201229", "20201230", "20201231", "20210104", "20210105", "20210106"]
    before = research.global_calendar(pd.DataFrame(dict(entry_date=dates[:5])))
    after = research.global_calendar(pd.DataFrame(dict(entry_date=dates)))
    assert before == frozenset(["20201228"])
    assert after == frozenset(["20201228", "20210105"])
    assert before == after.intersection(dates[:5])
    backfilled = research.global_calendar(pd.DataFrame(dict(entry_date=["20191231"]+dates)))
    assert backfilled == after


def test_same_next_global_entry_gives_identical_flat_start_paths():
    groups, opens, closes = scenario([1000]*9, [1000]*8)
    calendar = {"20220106", "20220111"}
    first, _, trace = research.risk_replay(groups, opens, closes, .0003, .15, entry_calendar=calendar, trace=True)
    later, _, trace2 = research.risk_replay(groups[1:], opens, closes, .0003, .15, entry_calendar=calendar, trace=True)
    assert trace.loc[trace.action.eq("enter"), "date"].tolist() == ["20220106"]
    assert trace2.loc[trace2.action.eq("enter"), "date"].tolist() == ["20220106"]
    pd.testing.assert_frame_equal(first.iloc[1:].reset_index(drop=True), later)


def test_exit_can_execute_between_global_entry_dates():
    groups, opens, closes = scenario([1000]*7, [1000]*6)
    for g in groups:
        g[0].short_20 = .4
    ledger, _, trace = research.risk_replay(groups, opens, closes, 0, .15,
                                           policy="opposing_exit", entry_calendar={"20220106"}, trace=True)
    assert trace.loc[trace.action.eq("enter"), "date"].tolist() == ["20220106"]
    assert trace.loc[trace.action.eq("native_opposing_exit"), "date"].tolist() == ["20220107"]
    assert ledger.gross_lots.sum() == 1


def test_initial_policy_is_applied_before_first_entry():
    r = row()
    r.win_long = .2
    base, _, _ = research.risk_replay([[r]], {}, {}, 0, .15)
    selected, _, trace = research.risk_replay([[r]], {}, {}, 0, .15, policy="aux55", trace=True)
    assert base.gross_lots.iloc[0] == 1
    assert selected.gross_lots.iloc[0] == 0
    assert trace.policy.iloc[0] == "aux55"


def test_startup_selection_uses_existing_absolute_eligibility():
    selected, reason = research.select_initial(scores())
    assert selected == "aux55"
    assert reason == "historical_risk_adjusted_selection"
    selected, _ = research.select_initial(scores(trades=2))
    assert selected == "base"


def test_startup_ties_prefer_base_and_empty_evidence_falls_back():
    values = scores(weighted_sortino=1.)
    assert research.select_initial(values)[0] == "base"
    for v in values.values():
        v["total_return"] = -.1
    selected, reason = research.select_initial(values)
    assert selected == "base" and reason == "insufficient_evidence_keep_current"


def test_startup_history_cannot_include_current_or_future_cashflows():
    frame = pd.DataFrame(dict(entry_date=["20211228", "20211229", "20211230", "20211231", "20220104"],
                              exit_date=["20211229", "20211230", "20211231", "20220104", "20220105"],
                              future_profit=[1, 2, 3, 1e9, 1e10]))
    past = research.history_before(frame, "20211231")
    assert past.future_profit.tolist() == [1, 2, 3]
    assert past.exit_date.max() == "20211231"
