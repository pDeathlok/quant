import json

import pandas as pd

from quant.analysis import write_backtest_report
from quant.backtest import BacktestArtifacts


def test_write_backtest_report_publishes_complete_bundle(tmp_path) -> None:
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    artifacts = BacktestArtifacts(
        equity_curve=pd.Series([100.0, 101.0, 99.0, 103.0], index=dates),
        returns=pd.Series([0.0, 0.01, -0.01980198, 0.04040404], index=dates),
        positions=pd.DataFrame({"symbol": ["A"], "quantity": [100]}),
        orders=pd.DataFrame({"order_id": ["1"], "symbol": ["A"]}),
        executions=pd.DataFrame({"order_id": ["1"], "symbol": ["A"]}),
        trades=pd.DataFrame({"symbol": ["A"], "net_pnl": [3.0]}),
        costs=pd.DataFrame({"total_cost": [1.0]}),
        metadata={"engine": "test"},
    )
    input_path = tmp_path / "input.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(input_path)

    result = write_backtest_report(
        artifacts,
        tmp_path / "run",
        strategy_name="demo",
        parameters={"window": 5},
        data_paths=[input_path],
        random_seed=42,
        project_root=tmp_path,
    )

    assert result["status"] == "success"
    assert (tmp_path / "run/report.html").is_file()
    assert (tmp_path / "run/equity_curve.parquet").is_file()
    summary = json.loads((tmp_path / "run/metrics.json").read_text())
    manifest = json.loads((tmp_path / "run/research_manifest.json").read_text())
    assert summary["trade_count"] == 1
    assert manifest["extra"]["artifact_sha256"]["metrics.json"]
    assert "demo" in (tmp_path / "run/report.html").read_text()
