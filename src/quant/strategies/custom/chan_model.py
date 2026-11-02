"""Model-filtered Chan daily strategy candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CHAN_MODEL_LOGIC_VERSION = 2


def resolve_chan_signal_date(
    scored: pd.DataFrame,
    trade_date: str | pd.Timestamp | None = None,
    processed_through: str | None = None,
) -> pd.Timestamp:
    """Resolve the requested/latest processed day, including days with no signals."""
    if trade_date is not None:
        return pd.Timestamp(trade_date).normalize()
    latest = pd.to_datetime(scored["date"], errors="coerce").max()
    if processed_through:
        processed = pd.Timestamp(processed_through).normalize()
        if pd.isna(latest) or processed >= latest:
            return processed
    return latest


@dataclass(frozen=True)
class ChanModelRule:
    id: str
    name: str
    min_good_quantile: float
    require_buy3: bool
    min_chan_score: float | None
    max_entry_gap_pct: float | None
    base_position_pct: float
    description: str


CHAN_MODEL_RULES = [
    ChanModelRule(
        id="chan_model_primary",
        name="缠论三买模型主策略",
        min_good_quantile=0.80,
        require_buy3=True,
        min_chan_score=None,
        max_entry_gap_pct=3.0,
        base_position_pct=0.25,
        description="三买确认 + target_good 模型历史分位 Top20；优先胜率和盈亏比。",
    ),
    ChanModelRule(
        id="chan_model_expanded",
        name="缠论三买模型扩容",
        min_good_quantile=0.50,
        require_buy3=True,
        min_chan_score=95.0,
        max_entry_gap_pct=3.0,
        base_position_pct=0.15,
        description="三买确认 + chan_score>=95 + target_good 模型历史分位 Top50；候选不足时补充。",
    ),
    ChanModelRule(
        id="chan_model_high_conviction",
        name="缠论模型高置信池",
        min_good_quantile=0.90,
        require_buy3=False,
        min_chan_score=None,
        max_entry_gap_pct=3.0,
        base_position_pct=0.20,
        description="target_good 模型历史分位 Top10；允许非三买，但需人工复核结构。",
    ),
]


def compute_chan_model_thresholds(
    scored: pd.DataFrame,
    reference_split: str | None = None,
) -> dict[str, float]:
    """Compute model probability thresholds from scored historical candidates."""
    ref = scored.copy()
    if reference_split and "split" in ref.columns:
        ref = ref[ref["split"].eq(reference_split)]
    pred = pd.to_numeric(ref["pred_target_good"], errors="coerce").dropna()
    if pred.empty:
        raise ValueError("scored candidates must contain non-empty pred_target_good")
    return {
        "good_top50": float(pred.quantile(0.50)),
        "good_top80": float(pred.quantile(0.80)),
        "good_top90": float(pred.quantile(0.90)),
    }


def add_chan_model_strategy_columns(
    scored: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add strategy rule hits and ranking score to model-scored candidates."""
    out = scored.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if thresholds is None:
        thresholds = compute_chan_model_thresholds(out, reference_split="train")
    good_threshold_by_quantile = {
        0.50: thresholds["good_top50"],
        0.80: thresholds["good_top80"],
        0.90: thresholds["good_top90"],
    }

    pred_good = pd.to_numeric(out["pred_target_good"], errors="coerce")
    pred_big = pd.to_numeric(out.get("pred_target_big10", np.nan), errors="coerce")
    pred_win = pd.to_numeric(out.get("pred_target_win10", np.nan), errors="coerce")
    entry_gap = pd.to_numeric(out.get("entry_gap_pct", np.nan), errors="coerce")
    chan_score = pd.to_numeric(out.get("chan_score", np.nan), errors="coerce")

    out["chan_model_rule_id"] = ""
    out["chan_model_rule_name"] = ""
    out["chan_model_signal"] = 0
    out["chan_model_setup_signal"] = 0
    out["chan_model_execution_status"] = "not_selected"
    out["chan_model_execution_label"] = "未入选"
    out["chan_model_position_pct"] = np.nan
    out["chan_model_buy_plan"] = ""
    out["chan_model_sell_plan"] = ""
    out["chan_model_description"] = ""

    for rule in CHAN_MODEL_RULES:
        threshold = good_threshold_by_quantile[rule.min_good_quantile]
        mask = pred_good >= threshold
        if rule.require_buy3:
            mask &= out["chan_signal_name"].eq("三买确认")
        if rule.min_chan_score is not None:
            mask &= chan_score >= rule.min_chan_score
        update = mask & (
            out["chan_model_rule_id"].eq("")
            | ((rule.id == "chan_model_primary") & out["chan_model_rule_id"].ne("chan_model_primary"))
        )
        out.loc[update, "chan_model_rule_id"] = rule.id
        out.loc[update, "chan_model_rule_name"] = rule.name
        out.loc[update, "chan_model_setup_signal"] = 1
        executable = update & entry_gap.notna()
        if rule.max_entry_gap_pct is not None:
            executable &= entry_gap <= rule.max_entry_gap_pct
        out.loc[executable, "chan_model_signal"] = 1
        out.loc[update, "chan_model_position_pct"] = rule.base_position_pct
        out.loc[update, "chan_model_description"] = rule.description
        out.loc[update, "chan_model_buy_plan"] = (
            "T日收盘确认；T+1高开不超过3%可执行，若高开3%-6%降仓观察，高开超过6%放弃。"
        )
        out.loc[update, "chan_model_sell_plan"] = (
            "优先持有5-10个交易日；跌破信号日低点或最近中枢下沿退出，放量长阴提前减仓。"
        )

    setup = out["chan_model_setup_signal"].eq(1)
    for mask, status, label in (
        (entry_gap.isna(), "pending_open", "待开盘核验"),
        (out["chan_model_signal"].eq(1), "executable", "可执行"),
        (entry_gap.gt(3) & entry_gap.le(6), "reduce_watch", "降仓观察"),
        (entry_gap.gt(6), "skip", "高开超限，放弃"),
    ):
        out.loc[setup & mask, "chan_model_execution_status"] = status
        out.loc[setup & mask, "chan_model_execution_label"] = label

    out["chan_model_rank_score"] = (
        pred_good.rank(pct=True) * 45
        + pred_big.rank(pct=True) * 25
        + pred_win.rank(pct=True) * 15
        + chan_score.fillna(0).clip(0, 100) / 100 * 15
        - entry_gap.clip(lower=0).fillna(0) * 1.5
    )
    return out


def select_chan_model_candidates(
    scored: pd.DataFrame,
    trade_date: str | pd.Timestamp | None = None,
    top_n: int = 20,
    include_expanded: bool = True,
) -> pd.DataFrame:
    """Select model-filtered Chan candidates for a trade date.

    If ``trade_date`` is omitted, use the latest scored date without falling back
    to an earlier day. Candidates are close-confirmed setups; next-open execution
    status is reported separately, including pending and rejected entries.
    """
    out = add_chan_model_strategy_columns(scored)
    selected = out[out["chan_model_setup_signal"].eq(1)].copy()
    if not include_expanded:
        selected = selected[selected["chan_model_rule_id"].isin(["chan_model_primary", "chan_model_high_conviction"])]
    if selected.empty:
        return selected
    trade_ts = resolve_chan_signal_date(out, trade_date)
    selected = selected[selected["date"].eq(trade_ts)].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(
        ["chan_model_rule_id", "chan_model_rank_score", "pred_target_good"],
        ascending=[True, False, False],
    )
    return selected.head(top_n).reset_index(drop=True)


def summarize_chan_model_strategy(scored: pd.DataFrame) -> pd.DataFrame:
    """Summarize historical returns by model strategy rule."""
    out = add_chan_model_strategy_columns(scored)
    rows = []
    group_cols = ["chan_model_rule_id"]
    if "split" in out.columns:
        group_cols.append("split")
    for key, group in out[out["chan_model_signal"].eq(1)].groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(group_cols, key))
        returns = pd.to_numeric(group.get("hold_10d_close"), errors="coerce").dropna()
        if returns.empty:
            continue
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        rows.append(
            {
                "rule_id": key_values["chan_model_rule_id"],
                "split": key_values.get("split", "all"),
                "rows": int(len(returns)),
                "avg_return_10d": float(returns.mean()),
                "median_return_10d": float(returns.median()),
                "win_rate_10d": float((returns > 0).mean()),
                "big_win_rate_10d": float((returns >= 3).mean()),
                "profit_factor_10d": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "avg_return_10d"], ascending=[True, False])
