from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from quant.data import TushareDataFetcher
from quant.routine.paths import PROJECT_ROOT, ROUTINE_DIR
from quant.strategies.convertible_bond import (
    ConvertibleBondFilterConfig,
    ConvertibleBondRotationConfig,
    ConvertibleBondSelector,
)


CB_CONFIG_PATH = PROJECT_ROOT / "configs/strategies/convertible_bond_rotation.yaml"


def load_convertible_bond_config(path: Path = CB_CONFIG_PATH) -> ConvertibleBondRotationConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = payload.get("strategy", payload)
    filter_raw = raw.get("filter", {})
    return ConvertibleBondRotationConfig(
        top_n=int(raw.get("top_n", 10)),
        max_position_weight=float(raw.get("max_position_weight", 0.12)),
        min_score=float(raw.get("min_score", 0.0)),
        rebalance_threshold=float(raw.get("rebalance_threshold", 0.01)),
        filter=ConvertibleBondFilterConfig(
            min_price=float(filter_raw.get("min_price", 100.0)),
            max_price=float(filter_raw.get("max_price", 135.0)),
            max_premium_rate=float(filter_raw.get("max_premium_rate", 35.0)),
            min_amount=float(filter_raw.get("min_amount", 1_000.0)),
            min_remaining_size=float(filter_raw.get("min_remaining_size", 1.0)),
            min_credit_rating=str(filter_raw.get("min_credit_rating", "AA-")),
            exclude_call_risk=bool(filter_raw.get("exclude_call_risk", True)),
            exclude_not_convertible=bool(filter_raw.get("exclude_not_convertible", True)),
        ),
    )


def build_convertible_bond_plan(
    trade_date: str,
    fetcher: TushareDataFetcher | None = None,
    config: ConvertibleBondRotationConfig | None = None,
    current_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    fetcher = fetcher or TushareDataFetcher()
    config = config or load_convertible_bond_config()
    selector = ConvertibleBondSelector(config)
    daily = fetcher.get_cb_daily(trade_date=trade_date)
    basic = fetcher.get_cb_basic(list_status="L")
    call = fetcher.get_cb_call(start_date=trade_date, end_date=trade_date)
    target = selector.target_portfolio(daily=daily, basic=basic, call=call)
    orders = selector.rebalance_orders(
        current_weights=current_weights or {},
        daily=daily,
        basic=basic,
        call=call,
    )
    return {
        "trade_date": trade_date,
        "strategy": "cb_double_low_rotation",
        "config": asdict(config),
        "targets": target.to_dict(orient="records") if not target.empty else [],
        "orders": [asdict(order) for order in orders],
    }


def write_convertible_bond_plan(
    trade_date: str,
    output_dir: Path = ROUTINE_DIR,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_convertible_bond_plan(trade_date=trade_date)
    json_path = output_dir / f"convertible_bond_plan_{trade_date}.json"
    csv_path = output_dir / f"convertible_bond_targets_{trade_date}.csv"
    json_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(plan["targets"]).to_csv(csv_path, index=False)
    return json_path
