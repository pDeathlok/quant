#!/usr/bin/env python3
"""Command-line entry point for local strategy backtests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quant import (  # noqa: E402
    B1Strategy,
    BacktestEngine,
    BreakoutStrategy,
    DualMAStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    RightSideBottomFishingStrategy,
    TemplateStrategy,
)
from quant.data.source_merge import normalize_ts_code  # noqa: E402


STRATEGIES = {
    "dual_ma": DualMAStrategy,
    "momentum": MomentumStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "b1": B1Strategy,
    "template": TemplateStrategy,
    "right_side_bottom": RightSideBottomFishingStrategy,
}


def _resolve_data_path(symbol: str, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"行情文件不存在: {path}")
        return path

    ts_code = normalize_ts_code(symbol)
    bare = ts_code.split(".", 1)[0]
    prefix = "sh" if ts_code.endswith(".SH") else "sz" if ts_code.endswith(".SZ") else "bj"
    candidates = [
        PROJECT_ROOT / "data/raw/daily" / f"{ts_code}.parquet",
        PROJECT_ROOT / "data/stocks_daily" / f"{ts_code}.parquet",
        PROJECT_ROOT / "data/stocks_daily" / f"{bare}.parquet",
        PROJECT_ROOT / "data/stocks" / f"{prefix}{bare}.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    checked = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"未找到 {symbol} 的本地行情文件，已检查:\n{checked}")


def _load_backtest_data(path: Path, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce")
    elif "trade_date" in frame.columns:
        dates = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        frame["date"] = dates
    else:
        raise ValueError(f"行情文件缺少 date/trade_date 列: {path}")
    frame["date"] = dates
    if "symbol" not in frame.columns:
        frame["symbol"] = normalize_ts_code(symbol)
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    selected = frame.loc[dates.between(start, end)].sort_values("date").reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"{path} 在 {start_date}-{end_date} 没有行情数据")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quant 本地策略回测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=["backtest"], help="执行本地事件驱动回测")
    parser.add_argument("--strategy", "-s", required=True, choices=sorted(STRATEGIES))
    parser.add_argument("--symbol", "-sym", default="600000", help="股票代码")
    parser.add_argument("--start-date", "-sd", default="20200101", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", "-ed", default="20231231", help="结束日期 YYYYMMDD")
    parser.add_argument("--data", help="可选 Parquet 行情文件；省略时从项目数据目录解析")
    parser.add_argument("--output", "-o", default="backtest_report.html", help="HTML 报告路径")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_path = _resolve_data_path(args.symbol, args.data)
    data = _load_backtest_data(data_path, args.symbol, args.start_date, args.end_date)
    strategy = STRATEGIES[args.strategy]()
    engine = BacktestEngine(data=data, strategy=strategy)

    print(f"策略: {strategy.name}")
    print(f"行情: {data_path}")
    print(f"股票: {args.symbol}")
    print(f"时间范围: {args.start_date} ~ {args.end_date}")
    engine.run(report_filename=args.output)
    print(f"回测报告已生成: {args.output}")


if __name__ == "__main__":
    main()
