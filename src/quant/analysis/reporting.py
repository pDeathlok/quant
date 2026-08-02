"""Standard, self-contained backtest report bundles."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go

from quant.analysis.performance import PerformanceAnalyzer
from quant.data.atomic_io import atomic_write_json, atomic_write_parquet, atomic_write_text
from quant.research.manifest import build_research_manifest, file_sha256


def _series_frame(series: pd.Series, value_name: str) -> pd.DataFrame:
    frame = pd.Series(series, copy=True).rename(value_name).to_frame().reset_index()
    frame.columns = ["date", value_name]
    return frame


def write_backtest_report(
    artifacts: Any,
    output_dir: Path | str,
    *,
    strategy_name: str,
    parameters: Mapping[str, Any],
    data_paths: Iterable[Path | str],
    random_seed: int,
    project_root: Path | str,
    code_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    analyzer = PerformanceAnalyzer.from_artifacts(artifacts)
    metrics = analyzer.summary()

    artifact_paths: dict[str, Path] = {}
    artifact_paths["metrics.json"] = atomic_write_json(metrics, output / "metrics.json")
    artifact_paths["equity_curve.parquet"] = atomic_write_parquet(
        _series_frame(artifacts.equity_curve, "equity"),
        output / "equity_curve.parquet",
        index=False,
    )
    artifact_paths["returns.parquet"] = atomic_write_parquet(
        _series_frame(artifacts.returns, "return"),
        output / "returns.parquet",
        index=False,
    )
    for name in ("positions", "orders", "executions", "trades", "costs"):
        artifact_paths[f"{name}.parquet"] = atomic_write_parquet(
            getattr(artifacts, name), output / f"{name}.parquet", index=False
        )

    curve = artifacts.equity_curve
    if curve.empty:
        curve = (1.0 + artifacts.returns).cumprod()
    figure = go.Figure(
        data=[go.Scatter(x=curve.index, y=curve.values, mode="lines", name="净值")]
    )
    figure.update_layout(title="净值曲线", xaxis_title="日期", yaxis_title="净值")
    metric_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(metrics.items())
    )
    report_html = (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        f"<title>{html.escape(strategy_name)} 回测报告</title></head><body>"
        f"<h1>{html.escape(strategy_name)} 回测报告</h1>"
        f"{figure.to_html(full_html=False, include_plotlyjs=True)}"
        f"<h2>绩效指标</h2><table>{metric_rows}</table></body></html>"
    )
    artifact_paths["report.html"] = atomic_write_text(report_html, output / "report.html")

    artifact_hashes = {
        name: file_sha256(path) for name, path in sorted(artifact_paths.items())
    }
    if artifacts.returns.empty:
        raise ValueError("artifacts.returns are empty; cannot determine research sample")
    start_date = pd.Timestamp(artifacts.returns.index.min()).strftime("%Y%m%d")
    end_date = pd.Timestamp(artifacts.returns.index.max()).strftime("%Y%m%d")
    manifest = build_research_manifest(
        strategy_name=strategy_name,
        parameters=parameters,
        data_paths=data_paths,
        start_date=start_date,
        end_date=end_date,
        random_seed=random_seed,
        project_root=project_root,
        code_paths=code_paths,
        extra={
            "backtest_metadata": dict(artifacts.metadata),
            "artifact_sha256": artifact_hashes,
        },
    )
    manifest_path = atomic_write_json(manifest, output / "research_manifest.json")
    return {
        "status": "success",
        "output_dir": str(output),
        "metrics": metrics,
        "manifest_path": str(manifest_path),
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
    }
