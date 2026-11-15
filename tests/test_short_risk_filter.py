import pandas as pd

from quant.application.short_risk_filter import unreclaimed_selloff_symbols
from quant.webapp import services


def test_unreclaimed_selloff_requires_recent_shock_and_unreclaimed_low():
    days = pd.bdate_range("2026-09-14", periods=13).strftime("%Y%m%d").tolist()
    rows = []
    for symbol, shock_index, signal_close in (
        ("recent", 6, 8.8),
        ("reclaimed", 6, 9.0),
        ("too_old", 5, 8.8),
        ("signal_day_only", 11, 8.8),
        ("split_adjusted", 6, 4.6),
    ):
        for i, day in enumerate(days):
            shock = i == shock_index
            rows.append({
                "ts_code": symbol,
                "trade_date": day,
                "close": signal_close if i == 11 else 10.0,
                "pre_close": 5.0 if i == 11 and symbol == "split_adjusted" else 10.0,
                "low": 9.0 if shock else 8.0,
                "pct_chg": -7.0 if shock else 0.0,
                "volume": 200.0 if shock else 100.0,
            })
    daily = pd.DataFrame(rows)
    assert unreclaimed_selloff_symbols(daily, set(daily.ts_code), days[11]) == {"recent"}


def test_short_selector_excludes_both_sides_and_portfolio_before_limiting(monkeypatch):
    left_blocked, right_blocked, left_kept, right_kept = "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"
    monkeypatch.setattr(services, "_row_display_quality_gate", lambda row: True)
    monkeypatch.setattr(services, "apply_selector_ranking_source", lambda rows, *_: rows)
    monkeypatch.setattr(services, "_annotate_selector_rank_release", lambda rows: None)
    monkeypatch.setattr(services, "_apply_left_side_ranked_metrics", lambda rows: None)
    pairs = [(left_blocked, "left"), (right_blocked, "right"), (left_kept, "left"), (right_kept, "right")]
    payload = {
        "signal_date": "2026-09-29",
        "total_stock_count": 4,
        "short_risk_filter": {
            "version": services.FILTER_VERSION,
            "excluded_symbols": [left_blocked, right_blocked],
        },
        "stocks": [
            {
                "symbol": symbol,
                "matched_group_sides": {"test": side},
                "selector_score": 100 - i,
                "matched_count": 1,
                "best_profit_factor": 1.0,
                "buy_score_source": "historical_return_model",
                "hold_score_source": "historical_return_model",
            }
            for i, (symbol, side) in enumerate(pairs)
        ],
        "application_portfolio": {
            "sleeves": [{"candidates": [{"symbol": s, "side": side} for s, side in pairs]}]
        },
    }
    for side, expected in (("all", [left_kept, right_kept]), ("left", [left_kept]), ("right", [right_kept])):
        view = services._display_selector_payload(payload, limit=1, side=side)
        assert [row["symbol"] for row in view["stocks"]] == expected[:1]
        assert [row["symbol"] for row in view["application_portfolio"]["sleeves"][0]["candidates"]] == expected
        assert view["total_stock_count"] == len(expected)
        assert view["complete_stock_count"] == 4
    assert len(payload["stocks"]) == 4
