from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from quant.data.atomic_io import atomic_write_json
from quant.data import MarketDataStore, MarketDataStoreConfig
from quant.features.project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    PROJECT_FACTOR_SCHEMA_VERSION,
)
from quant.routine.paths import CONFIG_PATH, PROJECT_ROOT, ROUTINE_DIR, WEB_DATA_DIR
from quant.routine.strategies import ExitConfig, StrategyConfig, StrategyRelease, load_strategy_release


FEATURE_PATH = PROJECT_ROOT / "data/features/b1/training_xgb_project_vars.parquet"
DAILY_PLAN_PATH = WEB_DATA_DIR / "b1_daily_plan.json"


@dataclass(frozen=True)
class EntryThresholds:
    min_up5: float | None = None
    min_up8: float | None = None
    min_up10: float | None = None
    max_down2: float | None = None
    max_down3: float | None = None


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _release_assets(
    config_path: Path = CONFIG_PATH,
) -> tuple[StrategyRelease, Path, Path, Path, Path]:
    release = load_strategy_release(config_path)
    model_dir = _resolve_project_path(release.model_dir)
    model_manifest_path = _resolve_project_path(release.model_manifest)
    summary_path = _resolve_project_path(release.backtest_summary)
    audit_path = _resolve_project_path(release.compatibility_audit)
    missing_models = [
        name for name in release.model_names if not (model_dir / f"{name}.joblib").is_file()
    ]
    if missing_models:
        raise FileNotFoundError(
            f"B1 release {release.id} is missing production models: {missing_models}"
        )
    if not model_manifest_path.is_file():
        raise FileNotFoundError(
            f"B1 release model manifest not found: {model_manifest_path}"
        )
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest.get("release_id") != release.id:
        raise RuntimeError(
            f"B1 model manifest release mismatch: expected={release.id} "
            f"actual={model_manifest.get('release_id')}"
        )
    manifest_models = model_manifest.get("models") or {}
    for name in release.model_names:
        model_path = model_dir / f"{name}.joblib"
        expected_hash = str((manifest_models.get(name) or {}).get("sha256") or "")
        if not expected_hash:
            raise RuntimeError(f"B1 model manifest has no sha256 for {name}")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise RuntimeError(
                f"B1 production model checksum mismatch for {name}: "
                f"expected={expected_hash} actual={digest}"
            )
    if not summary_path.is_file():
        raise FileNotFoundError(f"B1 release summary not found: {summary_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"B1 release compatibility audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "valid":
        raise RuntimeError(
            f"B1 release {release.id} compatibility audit is not valid: "
            f"{audit.get('reason') or audit.get('status')}"
        )
    if audit.get("release_id") != release.id:
        raise RuntimeError(
            f"B1 compatibility audit release mismatch: expected={release.id} "
            f"actual={audit.get('release_id')}"
        )
    validation = audit.get("strategy_validation") or {}
    invalid_strategies = [
        strategy.id
        for strategy in release.strategies
        if not (validation.get(strategy.backtest_combo) or {}).get("passed")
    ]
    if invalid_strategies:
        raise RuntimeError(
            f"B1 compatibility audit has no passing gate for {invalid_strategies}"
        )
    return release, model_dir, model_manifest_path, summary_path, audit_path


def predict_models(
    candidates: pd.DataFrame,
    *,
    model_dir: Path,
    model_names: tuple[str, ...],
) -> pd.DataFrame:
    out = candidates.copy()
    if "factor_schema_version" in out.columns:
        candidate_schemas = set(
            out["factor_schema_version"].dropna().astype(str).unique()
        )
        if len(candidate_schemas) > 1:
            raise RuntimeError(
                f"B1 candidate cache mixes factor schemas: {sorted(candidate_schemas)}"
            )
        candidate_schema = (
            next(iter(candidate_schemas))
            if candidate_schemas
            else LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
        )
    else:
        # Released pre-v4 caches did not carry schema metadata. Their only
        # valid interpretation is the legacy latest-scale/global-rank schema.
        candidate_schema = LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
    for model_name in model_names:
        model_path = model_dir / f"{model_name}.joblib"
        model = joblib.load(model_path)
        model_schema = (
            getattr(model, "factor_schema_version_", None)
            or LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION
        )
        if model_schema != candidate_schema:
            raise RuntimeError(
                "B1 model factor schema is incompatible: "
                f"{model_path} model={model_schema} candidates={candidate_schema}; "
                f"current_research={PROJECT_FACTOR_SCHEMA_VERSION}"
            )
        feature_cols = list(model.feature_names_in_)
        missing = [col for col in feature_cols if col not in out.columns]
        if missing:
            raise ValueError(f"{model_path} 缺少特征列: {missing[:20]}")
        features = out[feature_cols].replace([np.inf, -np.inf], np.nan)
        out[f"pred_{model_name}"] = model.predict_proba(features)[:, 1]
    prediction_columns = [f"pred_{name}" for name in model_names]
    return out.dropna(subset=prediction_columns).copy()


def _thresholds(strategy: StrategyConfig) -> EntryThresholds:
    return EntryThresholds(**asdict(strategy.entry))


def apply_entry_thresholds(
    frame: pd.DataFrame,
    thresholds: EntryThresholds,
) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    mask = pd.Series(True, index=frame.index)
    comparisons = (
        ("min_up5", "pred_up5_es", "min"),
        ("min_up8", "pred_up8_es", "min"),
        ("min_up10", "pred_up10_es", "min"),
        ("max_down2", "pred_down2_es", "max"),
        ("max_down3", "pred_down3_es", "max"),
    )
    for field, column, direction in comparisons:
        value = getattr(thresholds, field)
        if value is None:
            continue
        if column not in frame.columns:
            raise ValueError(f"strategy threshold requires missing prediction column: {column}")
        mask &= frame[column] >= value if direction == "min" else frame[column] <= value
    return mask


def _threshold_rule(strategy: StrategyConfig) -> str:
    parts: list[str] = []
    for field, label, operator in (
        ("min_up5", "up5", "ge"),
        ("min_up8", "up8", "ge"),
        ("min_up10", "up10", "ge"),
        ("max_down2", "down2", "le"),
        ("max_down3", "down3", "le"),
    ):
        value = getattr(strategy.entry, field)
        if value is not None:
            parts.append(f"{label}_{operator}_{value:.2f}")
    return "_".join(parts)


def _pct_token(value: float) -> str:
    percent = value * 100
    return str(int(percent)) if percent.is_integer() else str(percent).replace(".", "")


def _exit_mode(exit_config: ExitConfig) -> str:
    hold = exit_config.hold_days + 1
    if exit_config.kind == "expiry":
        return f"expiry_T{hold}_close"
    if exit_config.kind == "fixed":
        return (
            f"fixed_tp{_pct_token(exit_config.take_profit or 0)}"
            f"_sl{_pct_token(exit_config.stop_loss or 0)}_T{hold}"
        )
    if exit_config.kind == "trailing":
        return (
            f"trail_target{_pct_token(exit_config.take_profit or 0)}"
            f"_dd{_pct_token(exit_config.trail_drawdown or 0)}"
            f"_sl{_pct_token(exit_config.stop_loss or 0)}_T{hold}"
        )
    return f"{exit_config.kind}_T{hold}"


def _sell_plan(exit_config: ExitConfig) -> dict[str, Any]:
    hold = exit_config.hold_days + 1
    if exit_config.kind == "expiry":
        return {
            "type": "expiry",
            "summary": f"最长持有到 T+{hold}，到期按收盘价退出",
            "max_hold_days": hold,
        }
    if exit_config.kind == "fixed":
        return {
            "type": "fixed",
            "summary": (
                f"止盈 {(exit_config.take_profit or 0):.1%}；"
                f"盘中硬止损 {(exit_config.stop_loss or 0):.1%}；"
                f"最长持有到 T+{hold}"
            ),
            "take_profit_pct": (exit_config.take_profit or 0) * 100,
            "stop_loss_pct": (exit_config.stop_loss or 0) * 100,
            "max_hold_days": hold,
        }
    return {
        "type": exit_config.kind,
        "summary": (
            f"上涨达到 {(exit_config.take_profit or 0):.1%} 后，"
            f"从最高点回撤 {(exit_config.trail_drawdown or 0):.1%} 卖出；"
            f"止损 {(exit_config.stop_loss or 0):.1%}；最长持有到 T+{hold}"
        ),
        "max_hold_days": hold,
    }


def _oot_metrics(summary_path: Path, strategies: tuple[StrategyConfig, ...]) -> dict[str, dict[str, Any]]:
    summary = pd.read_csv(summary_path)
    required = {
        "period",
        "combo",
        "trades",
        "avg_return_pct",
        "win_rate",
        "max_drawdown_pct",
        "profit_factor",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"B1 release summary missing columns: {missing}")
    oot = summary[summary["period"].astype(str) == "oot_2025plus"].copy()
    metrics: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        rows = oot[oot["combo"].astype(str) == strategy.backtest_combo]
        if len(rows) != 1:
            raise ValueError(
                f"B1 strategy {strategy.id} expected one OOT row for "
                f"{strategy.backtest_combo}, found {len(rows)}"
            )
        row = rows.iloc[0]
        metrics[strategy.id] = {
            "trades": int(row["trades"]),
            "avg_return_pct": float(row["avg_return_pct"]),
            "win_rate": float(row["win_rate"]),
            "max_drawdown_pct": float(row["max_drawdown_pct"]),
            "profit_factor": float(row["profit_factor"]),
        }
    return metrics


def build_strategy_pool(
    strategies: tuple[StrategyConfig, ...],
    metrics: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "priority": strategy.priority,
            "strategy_id": strategy.id,
            "name": strategy.name,
            "entry_rule": _threshold_rule(strategy),
            "buy_filter": "model_only",
            "buy_filter_desc": "不限制 T+1 开盘涨跌幅；仅使用正式模型固定阈值",
            "exit_mode": _exit_mode(strategy.exit),
            "open_gap_rule": {
                "min_gap_pct": None,
                "max_gap_pct": None,
                "text": "不限制 T+1 开盘涨跌幅",
            },
            "entry_thresholds": asdict(strategy.entry),
            "sell_plan": _sell_plan(strategy.exit),
            "metrics": metrics[strategy.id],
        }
        for strategy in strategies
    ]


def build_daily_plan(
    signal_date: str | None = None,
    max_rows: int = 500,
    *,
    config_path: Path = CONFIG_PATH,
    feature_path: Path = FEATURE_PATH,
) -> dict[str, Any]:
    (
        release,
        model_dir,
        model_manifest_path,
        summary_path,
        audit_path,
    ) = _release_assets(config_path)
    metrics = _oot_metrics(summary_path, release.strategies)
    candidates = pd.read_parquet(feature_path)
    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates = candidates.dropna(subset=["date"])
    if candidates.empty:
        raise RuntimeError(f"B1 feature cache is empty: {feature_path}")
    if signal_date:
        target_date = pd.to_datetime(signal_date, errors="raise")
    else:
        daily_dir = PROJECT_ROOT / "data/raw/daily"
        store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
        target_date = store.latest_dataset_trade_date(daily_dir.name)
        if target_date is None:
            target_date = candidates["date"].max()
    latest = candidates[candidates["date"] == target_date].copy()
    if "name" in latest.columns:
        names = latest["name"].fillna("").astype(str)
        latest = latest[
            ~names.str.upper().str.contains("ST") & ~names.str.contains("退")
        ].copy()
    if not latest.empty:
        latest = predict_models(
            latest,
            model_dir=model_dir,
            model_names=release.model_names,
        )

    plan_rows: list[pd.DataFrame] = []
    for strategy in release.strategies:
        matched = latest[apply_entry_thresholds(latest, _thresholds(strategy))].copy()
        if matched.empty:
            continue
        strategy_metrics = metrics[strategy.id]
        matched["priority"] = strategy.priority
        matched["strategy_id"] = strategy.id
        matched["strategy_name"] = strategy.name
        matched["entry_rule"] = _threshold_rule(strategy)
        matched["buy_filter"] = "model_only"
        matched["exit_mode"] = _exit_mode(strategy.exit)
        matched["open_gap_text"] = "不限制 T+1 开盘涨跌幅"
        matched["buy_min_price"] = np.nan
        matched["buy_max_price"] = np.nan
        matched["oot_trades"] = strategy_metrics["trades"]
        matched["oot_avg_return_pct"] = strategy_metrics["avg_return_pct"]
        matched["oot_win_rate"] = strategy_metrics["win_rate"]
        matched["oot_max_drawdown_pct"] = strategy_metrics["max_drawdown_pct"]
        matched["oot_profit_factor"] = strategy_metrics["profit_factor"]
        matched["sell_summary"] = _sell_plan(strategy.exit)["summary"]
        plan_rows.append(matched)

    if plan_rows:
        plan_df = pd.concat(plan_rows, ignore_index=True)
        plan_df = plan_df.sort_values(
            ["priority", "pred_up10_es", "pred_down3_es", "symbol"],
            ascending=[True, False, True, True],
        ).head(max_rows)
        unique = (
            plan_df.sort_values(
                ["symbol", "priority", "pred_up10_es"],
                ascending=[True, True, False],
            )
            .drop_duplicates("symbol")
            .sort_values(
                ["priority", "pred_up10_es", "pred_down3_es"],
                ascending=[True, False, True],
            )
        )
    else:
        plan_df = pd.DataFrame()
        unique = pd.DataFrame()

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signal_date": target_date.strftime("%Y-%m-%d"),
        "execution_date": "下一个交易日",
        "release_id": release.id,
        "source": {
            "feature_path": str(feature_path),
            "strategy_config_path": str(config_path),
            "model_dir": str(model_dir),
            "model_manifest_path": str(model_manifest_path),
            "backtest_summary_path": str(summary_path),
            "compatibility_audit_path": str(audit_path),
        },
        "strategy_pool": build_strategy_pool(release.strategies, metrics),
        "plan_rows": _records(plan_df),
        "unique_symbols": _records(unique),
        "notes": [
            "每日计划只消费已发布的生产模型、正式 YAML 与通过门禁的同版本回测。",
            "固定模型阈值未命中时允许空仓，不再从研究 TopN 结果动态切换策略。",
            "同一股票若命中多个策略，unique_symbols 默认展示优先级最高的一条。",
        ],
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    keep_columns = [
        "date",
        "symbol",
        "name",
        "industry",
        "close",
        "priority",
        "strategy_id",
        "strategy_name",
        "entry_rule",
        "buy_filter",
        "exit_mode",
        "open_gap_text",
        "buy_min_price",
        "buy_max_price",
        "pred_up5_es",
        "pred_up8_es",
        "pred_up10_es",
        "pred_down2_es",
        "pred_down3_es",
        "kdj_d_j",
        "oot_trades",
        "oot_avg_return_pct",
        "oot_win_rate",
        "oot_max_drawdown_pct",
        "oot_profit_factor",
        "sell_summary",
    ]
    normalized = frame[[col for col in keep_columns if col in frame.columns]].copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    normalized = normalized.replace([np.inf, -np.inf], np.nan)
    return json.loads(normalized.to_json(orient="records", force_ascii=False))


def write_daily_plan(
    output_path: Path = DAILY_PLAN_PATH,
    signal_date: str | None = None,
) -> Path:
    payload = build_daily_plan(signal_date=signal_date)
    atomic_write_json(payload, output_path)
    run_dir = ROUTINE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    atomic_write_json(payload, run_dir / "b1_daily_plan.json")
    return output_path
