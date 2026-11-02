from __future__ import annotations

import numpy as np
import pandas as pd

from quant.research.b1_wave_information import confirmed_pivots, extract_wave_information


def test_pivot_requires_right_bars_and_unique_extreme():
    values = np.array([3., 2., 1., 2., 3., 3., 3., 2., 1., 2., 3.])
    assert confirmed_pivots(values, 'low').tolist() == [2, 8]
    assert confirmed_pivots(values[:4], 'low').tolist() == []
    assert confirmed_pivots(np.ones(8), 'low').tolist() == []


def test_future_extension_cannot_rewrite_old_wave_features():
    rng = np.random.default_rng(421)
    close = 10*np.exp(np.cumsum(rng.normal(0, .035, 900)))
    dates = pd.bdate_range('2023-01-01', periods=len(close))
    f = pd.DataFrame(dict(symbol='TEST', date=dates, close=close, open=close*.995,
                          high=close*1.01, low=close*.99, pre_close=np.r_[close[0],close[:-1]],
                          volume=rng.lognormal(10, .6, len(close))))
    start = dates[200]
    full = extract_wave_information(f, start, dates[-1])
    assert len(full) > 5  # Meaningful comparison, not an empty-frame pass.
    for cut in [dates[500], dates[650], dates[750]]:
        partial = extract_wave_information(f.loc[f.date <= cut], start, cut)
        expected = full.loc[full.date <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(partial.reset_index(drop=True), expected)
        if len(partial):
            assert partial.peak_known_at.le(partial.date).all()
            assert partial.trough_date.lt(partial.impulse_date).all()


def test_unresolved_holding_keeps_price_loss_in_portfolio_nav():
    import runpy
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    simulate = runpy.run_path(str(root/'scripts/research/validate_b1_wave_information.py'))['portfolio']
    dates = pd.bdate_range('2026-01-05', periods=3)
    prices = pd.DataFrame({'TEST':[10.,9.,8.]},index=dates)
    events = pd.DataFrame(dict(symbol=['TEST'],date=[dates[0]-pd.Timedelta(days=1)],
                               entry_date=[dates[0]],exit_date=[pd.NaT],exit_price=[np.nan],
                               exit_type=['unresolved']))
    nav,trades = simulate(events,prices)
    assert trades.empty
    assert nav.positions.iloc[-1] == 1
    expected = .99+.01/(1.001*10)*8
    assert np.isclose(nav.nav.iloc[-1],expected)
    assert nav.nav.iloc[-1] < .999  # Losing open positions must not become empty cash slots.
