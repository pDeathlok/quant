import pandas as pd
import pytest

from quant.risk import PortfolioRiskAnalyzer, RiskLimits, RiskManager


def test_portfolio_risk_concentration_and_industry_exposure() -> None:
    analyzer = PortfolioRiskAnalyzer()
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})

    report = analyzer.concentration(weights, industries={"A": "银行", "B": "科技", "C": "银行"})

    assert report["max_weight"] == pytest.approx(0.5)
    assert report["hhi"] == pytest.approx(0.38)
    assert report["effective_positions"] == pytest.approx(1 / 0.38)
    assert report["industry_exposure"]["银行"] == pytest.approx(0.7)


def test_historical_var_cvar_and_stress_losses() -> None:
    analyzer = PortfolioRiskAnalyzer(confidence_level=0.8)
    returns = pd.Series([-0.10, -0.04, 0.0, 0.02, 0.03])
    risk = analyzer.historical_var_cvar(returns)
    scenarios = pd.DataFrame(
        {"A": [-0.10, 0.05], "B": [-0.20, -0.02]},
        index=["crash", "rotation"],
    )

    assert risk["var"] == pytest.approx(0.052)
    assert risk["cvar"] == pytest.approx(0.10)
    stress = analyzer.stress_test(pd.Series({"A": 0.6, "B": 0.4}), scenarios)
    assert stress.loc["crash", "portfolio_return"] == pytest.approx(-0.14)
    assert stress.loc["crash", "loss"] == pytest.approx(0.14)


def test_capacity_reports_liquidation_days_and_participation_breaches() -> None:
    analyzer = PortfolioRiskAnalyzer(max_participation_rate=0.10)
    report = analyzer.capacity(
        positions=pd.Series({"A": 20_000, "B": 1_000}),
        prices=pd.Series({"A": 10.0, "B": 20.0}),
        average_daily_volume=pd.Series({"A": 10_000, "B": 50_000}),
    )

    assert report.loc["A", "liquidation_days"] == pytest.approx(20.0)
    assert bool(report.loc["A", "capacity_breach"])
    assert not bool(report.loc["B", "capacity_breach"])


def test_risk_manager_enforces_leverage_daily_loss_and_volume() -> None:
    manager = RiskManager(
        RiskLimits(
            max_position_size=0.5,
            max_position_pct_per_symbol=0.8,
            max_total_leverage=1.0,
            max_loss_per_day=0.02,
            max_volume_participation=0.1,
        )
    )

    passed, reason = manager.pre_order_check(
        {"symbol": "A", "side": "buy", "quantity": 200, "price": 10},
        account_value=10_000,
        positions={},
        total_exposure=9_000,
    )
    assert not passed
    assert "leverage" in reason.lower()

    passed, reason = manager.pre_order_check(
        {"symbol": "A", "side": "buy", "quantity": 200, "price": 10},
        account_value=10_000,
        positions={},
        average_daily_volume=1_000,
    )
    assert not passed
    assert "volume" in reason.lower()

    assert manager.check_daily_loss(10_000) is False
    assert manager.check_daily_loss(9_700) is True
