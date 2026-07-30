#!/usr/bin/env python3
"""Build a point-in-time, non-prescriptive A-share price-volume snapshot."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


class InputError(ValueError):
    """Raised when the OHLCV input contract is invalid."""


TICKER_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PRICE_BASES = {"unadjusted", "forward_adjusted", "backward_adjusted"}
MINIMUM_BARS = 60
DEFAULT_THRESHOLDS = {
    "volume_expansion_ratio": 1.5,
    "volume_contraction_ratio": 0.7,
    "breakout_buffer_pct": 0.0,
    "large_body_atr_ratio": 0.8,
    "ma20_tolerance_pct": 0.02,
}


@dataclass(frozen=True)
class Bar:
    """One complete daily OHLCV bar."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class AnalysisContext:
    """Validated point-in-time metadata for a price-volume calculation."""

    ticker: str
    as_of_date: date
    analysis_cutoff: datetime
    last_bar_available_at: datetime
    source: str
    price_basis: str
    adjustment_reference_date: date | None
    volume_unit: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate daily A-share OHLCV data and calculate a point-in-time "
            "price-volume snapshot. The output is an analytical aid, not a trade signal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Required root fields:\n"
            "  ticker, as_of_date, analysis_cutoff, last_bar_available_at, source,\n"
            "  price_basis, volume_unit, bars (at least 60 complete daily bars).\n"
            "Adjusted series also require adjustment_reference_date.\n\n"
            "Each bar requires date, open, high, low, close and volume.\n"
            "Optional thresholds: volume_expansion_ratio, volume_contraction_ratio,\n"
            "  breakout_buffer_pct, large_body_atr_ratio, ma20_tolerance_pct."
        ),
    )
    parser.add_argument("input", help="Input JSON path, or '-' for standard input.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Displayed decimal places, from 0 to 6 (default: 4).",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError("The JSON root must be an object.")
    return data


def require_text(mapping: dict[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{path}.{key} must be a non-empty string.")
    return value.strip()


def as_finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise InputError(f"{field} must be a finite number, not {value!r}.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{field} must be a finite number, not {value!r}.") from exc
    if not math.isfinite(number):
        raise InputError(f"{field} must be finite.")
    return number


def parse_iso_date(value: str, field: str) -> date:
    if not ISO_DATE_PATTERN.fullmatch(value):
        raise InputError(f"{field} must use YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise InputError(f"{field} must use YYYY-MM-DD.") from exc


def parse_iso_timestamp(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InputError(f"{field} must be an ISO 8601 timestamp with timezone.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise InputError(f"{field} must include an explicit UTC offset, such as +08:00.")
    return timestamp


def parse_bar(raw: Any, index: int) -> Bar:
    path = f"root.bars[{index}]"
    if not isinstance(raw, dict):
        raise InputError(f"{path} must be an object.")
    bar_date = parse_iso_date(require_text(raw, "date", path), f"{path}.date")
    open_price = as_finite_float(raw.get("open"), f"{path}.open")
    high = as_finite_float(raw.get("high"), f"{path}.high")
    low = as_finite_float(raw.get("low"), f"{path}.low")
    close = as_finite_float(raw.get("close"), f"{path}.close")
    volume = as_finite_float(raw.get("volume"), f"{path}.volume")

    if min(open_price, high, low, close) <= 0:
        raise InputError(f"{path} OHLC prices must all be greater than zero.")
    if volume < 0:
        raise InputError(f"{path}.volume must be non-negative.")
    if high < max(open_price, close, low):
        raise InputError(f"{path}.high must be at least open, close and low.")
    if low > min(open_price, close, high):
        raise InputError(f"{path}.low must be at most open, close and high.")
    return Bar(bar_date, open_price, high, low, close, volume)


def parse_thresholds(data: dict[str, Any]) -> dict[str, float]:
    raw = data.get("thresholds", {})
    if not isinstance(raw, dict):
        raise InputError("root.thresholds must be an object when provided.")
    unknown = set(raw) - set(DEFAULT_THRESHOLDS)
    if unknown:
        raise InputError(f"root.thresholds has unsupported fields: {sorted(unknown)}.")

    thresholds = dict(DEFAULT_THRESHOLDS)
    for key, value in raw.items():
        thresholds[key] = as_finite_float(value, f"root.thresholds.{key}")

    if thresholds["volume_expansion_ratio"] <= 1:
        raise InputError("root.thresholds.volume_expansion_ratio must be greater than 1.")
    if not 0 < thresholds["volume_contraction_ratio"] < 1:
        raise InputError(
            "root.thresholds.volume_contraction_ratio must be greater than 0 and less than 1."
        )
    for key in ("breakout_buffer_pct", "ma20_tolerance_pct"):
        if not 0 <= thresholds[key] <= 0.2:
            raise InputError(f"root.thresholds.{key} must be between 0 and 0.2.")
    if thresholds["large_body_atr_ratio"] <= 0:
        raise InputError("root.thresholds.large_body_atr_ratio must be greater than 0.")
    return thresholds


def validate_input(
    data: dict[str, Any],
) -> tuple[AnalysisContext, list[Bar], dict[str, float], list[str]]:
    ticker = require_text(data, "ticker", "root")
    if not TICKER_PATTERN.fullmatch(ticker):
        raise InputError("root.ticker must look like 600000.SH, 000001.SZ or 920000.BJ.")

    as_of_date = parse_iso_date(require_text(data, "as_of_date", "root"), "root.as_of_date")
    analysis_cutoff = parse_iso_timestamp(
        require_text(data, "analysis_cutoff", "root"), "root.analysis_cutoff"
    )
    if analysis_cutoff.date() != as_of_date:
        raise InputError(
            "root.as_of_date must equal the local calendar date of root.analysis_cutoff."
        )
    last_bar_available_at = parse_iso_timestamp(
        require_text(data, "last_bar_available_at", "root"),
        "root.last_bar_available_at",
    )
    if last_bar_available_at.astimezone(timezone.utc) > analysis_cutoff.astimezone(timezone.utc):
        raise InputError("root.last_bar_available_at cannot be later than root.analysis_cutoff.")

    source = require_text(data, "source", "root")
    price_basis = require_text(data, "price_basis", "root")
    if price_basis not in PRICE_BASES:
        raise InputError(f"root.price_basis must be one of {sorted(PRICE_BASES)}.")
    volume_unit = require_text(data, "volume_unit", "root")

    adjustment_reference_date: date | None = None
    if price_basis != "unadjusted":
        adjustment_reference_date = parse_iso_date(
            require_text(data, "adjustment_reference_date", "root"),
            "root.adjustment_reference_date",
        )
        if adjustment_reference_date > as_of_date:
            raise InputError(
                "root.adjustment_reference_date cannot be later than root.as_of_date."
            )
    elif "adjustment_reference_date" in data:
        raise InputError(
            "root.adjustment_reference_date must be omitted when price_basis is unadjusted."
        )

    raw_bars = data.get("bars")
    if not isinstance(raw_bars, list):
        raise InputError("root.bars must be an array.")
    if len(raw_bars) < MINIMUM_BARS:
        raise InputError(f"root.bars must contain at least {MINIMUM_BARS} complete daily bars.")
    bars = [parse_bar(raw, index) for index, raw in enumerate(raw_bars)]
    for previous, current in zip(bars, bars[1:]):
        if current.date <= previous.date:
            raise InputError("root.bars dates must be unique and strictly increasing.")
    if bars[-1].date > as_of_date:
        raise InputError("The last bar date cannot be later than root.as_of_date.")
    if (
        adjustment_reference_date is not None
        and adjustment_reference_date < bars[-1].date
    ):
        raise InputError(
            "root.adjustment_reference_date cannot be earlier than the last bar date."
        )
    if last_bar_available_at.date() < bars[-1].date:
        raise InputError(
            "root.last_bar_available_at cannot have a local date before the last bar date."
        )

    warnings: list[str] = []
    if len(bars) < 120:
        warnings.append("少于120根日线：不输出完整 MA120 长期结构。")
    if any(bar.volume == 0 for bar in bars[-20:]):
        warnings.append("最近20根日线包含零成交量；核对停牌、缺失值和交易状态。")
    if price_basis != "unadjusted":
        warnings.append(
            "当前为复权价格序列；与未复权现价或目标价比较前必须建立复权桥。"
        )

    context = AnalysisContext(
        ticker=ticker,
        as_of_date=as_of_date,
        analysis_cutoff=analysis_cutoff,
        last_bar_available_at=last_bar_available_at,
        source=source,
        price_basis=price_basis,
        adjustment_reference_date=adjustment_reference_date,
        volume_unit=volume_unit,
    )
    return context, bars, parse_thresholds(data), warnings


def average(values: list[float]) -> float:
    return sum(values) / len(values)


def moving_average(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return average(values[-window:])


def ema_series(values: list[float], span: int) -> list[float]:
    alpha = 2 / (span + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def true_ranges(bars: list[Bar]) -> list[float]:
    ranges: list[float] = []
    for index, bar in enumerate(bars):
        if index == 0:
            ranges.append(bar.high - bar.low)
            continue
        previous_close = bars[index - 1].close
        ranges.append(
            max(
                bar.high - bar.low,
                abs(bar.high - previous_close),
                abs(bar.low - previous_close),
            )
        )
    return ranges


def macd_snapshot(closes: list[float]) -> dict[str, float | str]:
    ema12 = ema_series(closes, 12)
    ema26 = ema_series(closes, 26)
    dif = [fast - slow for fast, slow in zip(ema12, ema26)]
    dea = ema_series(dif, 9)
    histogram = [2 * (d - signal) for d, signal in zip(dif, dea)]
    cross = "none"
    if dif[-2] <= dea[-2] and dif[-1] > dea[-1]:
        cross = "golden_cross"
    elif dif[-2] >= dea[-2] and dif[-1] < dea[-1]:
        cross = "dead_cross"
    return {
        "dif": round(dif[-1], 6),
        "dea": round(dea[-1], 6),
        "histogram_2x": round(histogram[-1], 6),
        "zero_axis_state": "above" if dif[-1] >= 0 else "below",
        "latest_cross": cross,
    }


def had_prior_positive_impulse(
    bars: list[Bar], volume_expansion_ratio: float
) -> bool:
    start = max(20, len(bars) - 25)
    end = len(bars) - 5
    for index in range(start, end):
        prior_average = average([bar.volume for bar in bars[index - 20 : index]])
        if prior_average <= 0:
            continue
        if (
            bars[index].close > bars[index - 1].close
            and bars[index].volume / prior_average >= volume_expansion_ratio
        ):
            return True
    return False


def optional_round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def analyze_price_volume(data: dict[str, Any]) -> dict[str, Any]:
    context, bars, thresholds, warnings = validate_input(data)
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    current = bars[-1]
    previous = bars[-2]

    ma20 = moving_average(closes, 20)
    ma60 = moving_average(closes, 60)
    ma120 = moving_average(closes, 120)
    assert ma20 is not None and ma60 is not None
    ma20_five_days_ago = average(closes[-25:-5])
    ma20_change_5d = ma20 / ma20_five_days_ago - 1

    trend_state = "mixed"
    if current.close > ma20 > ma60 and ma20_change_5d > 0:
        trend_state = "constructive"
    elif current.close < ma20 < ma60 and ma20_change_5d < 0:
        trend_state = "cautious"

    long_term_alignment = "unavailable"
    if ma120 is not None:
        if ma20 > ma60 > ma120:
            long_term_alignment = "bullish_alignment"
        elif ma20 < ma60 < ma120:
            long_term_alignment = "bearish_alignment"
        else:
            long_term_alignment = "mixed_alignment"

    return_5d = current.close / closes[-6] - 1
    return_20d = current.close / closes[-21] - 1
    daily_return = current.close / previous.close - 1
    prior_20_high = max(bar.high for bar in bars[-21:-1])
    prior_20_low = min(bar.low for bar in bars[-21:-1])
    closing_high_60 = max(closes[-60:])
    closing_low_60 = min(closes[-60:])
    range_60 = closing_high_60 - closing_low_60
    location_60 = (
        (current.close - closing_low_60) / range_60 if range_60 > 0 else None
    )

    prior_volume_20 = average(volumes[-21:-1])
    volume_ratio_20 = current.volume / prior_volume_20 if prior_volume_20 > 0 else None
    recent_volume_5 = average(volumes[-5:])
    preceding_volume_20 = average(volumes[-25:-5])
    volume_trend_ratio = (
        recent_volume_5 / preceding_volume_20 if preceding_volume_20 > 0 else None
    )

    up_volume = 0.0
    down_volume = 0.0
    for index in range(len(bars) - 20, len(bars)):
        if bars[index].close > bars[index - 1].close:
            up_volume += bars[index].volume
        elif bars[index].close < bars[index - 1].close:
            down_volume += bars[index].volume
    up_down_volume_ratio = up_volume / down_volume if down_volume > 0 else None

    quadrant = "flat_or_unavailable"
    if volume_trend_ratio is not None:
        if return_5d > 0 and volume_trend_ratio >= 1:
            quadrant = "price_up_volume_up"
        elif return_5d > 0:
            quadrant = "price_up_volume_down"
        elif return_5d < 0 and volume_trend_ratio >= 1:
            quadrant = "price_down_volume_up"
        elif return_5d < 0:
            quadrant = "price_down_volume_down"

    atr14 = average(true_ranges(bars)[-14:])
    atr14_pct = atr14 / current.close if current.close > 0 else None
    log_returns = [
        math.log(closes[index] / closes[index - 1])
        for index in range(len(closes) - 19, len(closes))
    ]
    realized_volatility_20 = statistics.stdev(log_returns) * math.sqrt(252)
    body_atr_ratio = abs(current.close - current.open) / atr14 if atr14 > 0 else None

    expansion = thresholds["volume_expansion_ratio"]
    contraction = thresholds["volume_contraction_ratio"]
    buffer = thresholds["breakout_buffer_pct"]
    large_body = thresholds["large_body_atr_ratio"]
    ma_tolerance = thresholds["ma20_tolerance_pct"]
    volume_is_expanded = volume_ratio_20 is not None and volume_ratio_20 >= expansion
    large_directional_body = body_atr_ratio is not None and body_atr_ratio >= large_body

    signals = {
        "breakout_on_expanded_volume": bool(
            current.close > prior_20_high * (1 + buffer) and volume_is_expanded
        ),
        "breakdown_on_expanded_volume": bool(
            current.close < prior_20_low * (1 - buffer) and volume_is_expanded
        ),
        "high_volume_bullish_key_bar": bool(
            current.close > current.open
            and daily_return > 0
            and volume_is_expanded
            and large_directional_body
        ),
        "high_volume_bearish_key_bar": bool(
            current.close < current.open
            and daily_return < 0
            and volume_is_expanded
            and large_directional_body
        ),
        "pullback_on_contracting_volume_after_impulse": bool(
            return_5d < 0
            and current.close >= ma20 * (1 - ma_tolerance)
            and volume_trend_ratio is not None
            and volume_trend_ratio <= contraction
            and had_prior_positive_impulse(bars, expansion)
        ),
    }

    supportive_observations: list[str] = []
    risk_observations: list[str] = []
    if trend_state == "constructive":
        supportive_observations.append("收盘价、MA20 与 MA60 构成建设性中期趋势结构。")
    elif trend_state == "cautious":
        risk_observations.append("收盘价、MA20 与 MA60 构成谨慎中期趋势结构。")
    if signals["breakout_on_expanded_volume"]:
        supportive_observations.append("收盘突破前20日高点且成交量显著放大。")
    if signals["pullback_on_contracting_volume_after_impulse"]:
        supportive_observations.append("前置放量上涨后出现缩量回撤，且尚未明显跌破MA20。")
    if quadrant == "price_down_volume_up":
        risk_observations.append("近5日价格下跌而量能放大，分歧或抛压上升。")
    if signals["breakdown_on_expanded_volume"]:
        risk_observations.append("收盘跌破前20日低点且成交量显著放大。")
    if signals["high_volume_bearish_key_bar"]:
        risk_observations.append("最新K线为放量大实体阴线，需核查事件与后续承接。")

    if volume_ratio_20 is None:
        warnings.append("前20日平均成交量为零，无法计算当日量比。")
    if volume_trend_ratio is None:
        warnings.append("此前20日平均成交量为零，无法判断近5日量能趋势。")

    return {
        "ticker": context.ticker,
        "as_of_date": context.as_of_date.isoformat(),
        "analysis_cutoff": context.analysis_cutoff.isoformat(),
        "last_bar_date": current.date.isoformat(),
        "last_bar_available_at": context.last_bar_available_at.isoformat(),
        "source": context.source,
        "price_basis": context.price_basis,
        "adjustment_reference_date": (
            context.adjustment_reference_date.isoformat()
            if context.adjustment_reference_date
            else None
        ),
        "volume_unit": context.volume_unit,
        "bar_count": len(bars),
        "thresholds": {key: round(value, 6) for key, value in thresholds.items()},
        "trend": {
            "state": trend_state,
            "close": round(current.close, 6),
            "ma20": round(ma20, 6),
            "ma60": round(ma60, 6),
            "ma120": optional_round(ma120),
            "ma20_change_5d": round(ma20_change_5d, 6),
            "long_term_alignment": long_term_alignment,
        },
        "returns_and_location": {
            "return_5d": round(return_5d, 6),
            "return_20d": round(return_20d, 6),
            "drawdown_from_60d_closing_high": round(
                current.close / closing_high_60 - 1, 6
            ),
            "location_in_60d_closing_range": optional_round(location_60),
            "prior_20d_high": round(prior_20_high, 6),
            "prior_20d_low": round(prior_20_low, 6),
        },
        "volume_price": {
            "quadrant_5d": quadrant,
            "current_volume_ratio_to_prior_20d": optional_round(volume_ratio_20),
            "avg_volume_5d_to_preceding_20d": optional_round(volume_trend_ratio),
            "up_volume_to_down_volume_20d": optional_round(up_down_volume_ratio),
        },
        "momentum": macd_snapshot(closes),
        "risk_scale": {
            "atr14": round(atr14, 6),
            "atr14_pct_of_close": optional_round(atr14_pct),
            "realized_volatility_20d_annualized": round(realized_volatility_20, 6),
            "latest_body_to_atr14": optional_round(body_atr_ratio),
        },
        "signals": signals,
        "supportive_observations": supportive_observations,
        "risk_observations": risk_observations,
        "warnings": warnings,
        "limitations": [
            "量价信号是启发式观察，不是统计胜率、内在价值或交易指令。",
            "成交量不能单独证明资金身份、吸筹或出货。",
            "支撑/压力为历史观察锚，后续价格可失效。",
            "本脚本只分析单一序列；相对强弱需对同步宽基/行业序列另行复算。",
        ],
    }


def format_number(value: Any, precision: int) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value)


def render_markdown(result: dict[str, Any], precision: int) -> str:
    trend = result["trend"]
    returns = result["returns_and_location"]
    volume_price = result["volume_price"]
    risk = result["risk_scale"]
    momentum = result["momentum"]
    lines = [
        "# A股量价技术辅助快照",
        "",
        f"- 标的：{result['ticker']}",
        f"- 分析截止：{result['analysis_cutoff']}",
        f"- 最后一根完整K线：{result['last_bar_date']}（可得于 {result['last_bar_available_at']}）",
        f"- 数据：{result['source']}｜价格口径 {result['price_basis']}｜成交量单位 {result['volume_unit']}",
        "- 定位：技术辅助层，不改变独立基本面估值，不构成交易指令。",
        "",
        "## 趋势与位置",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 技术状态 | {trend['state']} |",
        f"| 收盘 | {format_number(trend['close'], precision)} |",
        f"| MA20 / MA60 / MA120 | {format_number(trend['ma20'], precision)} / {format_number(trend['ma60'], precision)} / {format_number(trend['ma120'], precision)} |",
        f"| 5日 / 20日收益 | {format_number(returns['return_5d'], precision)} / {format_number(returns['return_20d'], precision)} |",
        f"| 距60日收盘高点 | {format_number(returns['drawdown_from_60d_closing_high'], precision)} |",
        "",
        "## 量价、动量与风险尺度",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 5日量价象限 | {volume_price['quadrant_5d']} |",
        f"| 当日量 / 前20日均量 | {format_number(volume_price['current_volume_ratio_to_prior_20d'], precision)} |",
        f"| 近5日均量 / 此前20日均量 | {format_number(volume_price['avg_volume_5d_to_preceding_20d'], precision)} |",
        f"| MACD DIF / DEA / 柱(2x) | {format_number(momentum['dif'], precision)} / {format_number(momentum['dea'], precision)} / {format_number(momentum['histogram_2x'], precision)} |",
        f"| ATR14 / 收盘 | {format_number(risk['atr14_pct_of_close'], precision)} |",
        f"| 20日年化实现波动率 | {format_number(risk['realized_volatility_20d_annualized'], precision)} |",
        "",
        "## 启发式事件",
        "",
        "| 事件 | 是否触发 |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name} | {'是' if triggered else '否'} |"
        for name, triggered in result["signals"].items()
    )
    if result["supportive_observations"]:
        lines.extend(["", "## 建设性观察", ""])
        lines.extend(f"- {item}" for item in result["supportive_observations"])
    if result["risk_observations"]:
        lines.extend(["", "## 风险观察", ""])
        lines.extend(f"- {item}" for item in result["risk_observations"])
    if result["warnings"]:
        lines.extend(["", "## 数据提示", ""])
        lines.extend(f"- {item}" for item in result["warnings"])
    lines.extend(["", "## 局限", ""])
    lines.extend(f"- {item}" for item in result["limitations"])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not 0 <= args.precision <= 6:
        print("error: --precision must be between 0 and 6.", file=sys.stderr)
        return 2
    try:
        result = analyze_price_volume(load_json(args.input))
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(result, args.precision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
