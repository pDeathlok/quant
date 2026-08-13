"""Daily Chan-theory structure strategy.

The implementation is intentionally dependency-light. It borrows the research
shape of CZSC: normalize bars, identify fractals and strokes, infer recent
centers, then express buy/sell decisions as columns that can be ranked and
backtested with the local pandas-based workflow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant.features.variable_library import build_continuous_ohlc


@dataclass(frozen=True)
class ChanDailyParams:
    min_stroke_bars: int = 4
    center_lookback_strokes: int = 7
    third_buy_tolerance: float = 0.01
    breakout_buffer: float = 0.002
    min_center_width: float = 0.015
    max_center_width: float = 0.22
    atr_window: int = 14


DEFAULT_CHAN_DAILY_PARAMS = ChanDailyParams()


def _normalize_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        raise ValueError("daily frame must contain date or trade_date")

    if "volume" not in out.columns and "vol" in out.columns:
        out = out.rename(columns={"vol": "volume"})
    if "symbol" not in out.columns and "ts_code" in out.columns:
        out["symbol"] = out["ts_code"].astype(str)

    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"daily frame missing required columns: {sorted(missing)}")

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    return out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def _is_risk_name(name: pd.Series) -> pd.Series:
    text = name.fillna("").astype(str).str.upper()
    return text.str.contains("ST", regex=False) | text.str.contains("*", regex=False) | text.str.contains("退", regex=False)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _detect_fractals(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    marks = pd.Series("", index=high.index, dtype=object)
    top = (high > high.shift(1)) & (high >= high.shift(-1)) & (low > low.shift(1))
    bottom = (low < low.shift(1)) & (low <= low.shift(-1)) & (high < high.shift(1))
    marks.loc[top.fillna(False)] = "top"
    marks.loc[bottom.fillna(False)] = "bottom"
    return pd.DataFrame({"fx_mark": marks, "fx_top": top.fillna(False), "fx_bottom": bottom.fillna(False)})


def _build_strokes(high: pd.Series, low: pd.Series, fractals: pd.DataFrame, min_bars: int) -> list[dict]:
    candidates = [
        {
            "idx": int(i),
            "mark": str(row.fx_mark),
            "price": float(high.iloc[i] if row.fx_mark == "top" else low.iloc[i]),
        }
        for i, row in fractals.loc[fractals["fx_mark"].ne("")].iterrows()
    ]
    pivots: list[dict] = []
    for point in candidates:
        if not pivots:
            pivots.append(point)
            continue
        last = pivots[-1]
        if point["mark"] == last["mark"]:
            more_extreme = (
                point["mark"] == "top" and point["price"] >= last["price"]
            ) or (
                point["mark"] == "bottom" and point["price"] <= last["price"]
            )
            if more_extreme:
                pivots[-1] = point
            continue
        if point["idx"] - last["idx"] < min_bars:
            continue
        pivots.append(point)

    strokes: list[dict] = []
    for start, end in zip(pivots, pivots[1:]):
        if start["mark"] == "bottom" and end["mark"] == "top":
            direction = "up"
            stroke_low = start["price"]
            stroke_high = end["price"]
        elif start["mark"] == "top" and end["mark"] == "bottom":
            direction = "down"
            stroke_low = end["price"]
            stroke_high = start["price"]
        else:
            continue
        if stroke_low <= 0:
            continue
        strokes.append(
            {
                "start_idx": start["idx"],
                "end_idx": end["idx"],
                "direction": direction,
                "low": float(stroke_low),
                "high": float(stroke_high),
                "amplitude": float(stroke_high / stroke_low - 1),
                "bars": int(end["idx"] - start["idx"]),
            }
        )
    return strokes


def _latest_center(strokes: list[dict], params: ChanDailyParams) -> dict | None:
    if len(strokes) < 3:
        return None
    recent = strokes[-params.center_lookback_strokes :]
    best: dict | None = None
    for offset in range(0, len(recent) - 2):
        group = recent[offset : offset + 3]
        center_low = max(stroke["low"] for stroke in group)
        center_high = min(stroke["high"] for stroke in group)
        if center_low >= center_high:
            continue
        width = center_high / center_low - 1 if center_low else np.nan
        if not np.isfinite(width) or width < params.min_center_width or width > params.max_center_width:
            continue
        best = {
            "start_idx": group[0]["start_idx"],
            "end_idx": group[-1]["end_idx"],
            "low": float(center_low),
            "high": float(center_high),
            "width": float(width),
            "stroke_count": 3,
        }
    return best


def _last_same_direction(strokes: list[dict], direction: str, before: int | None = None) -> dict | None:
    seq = strokes if before is None else strokes[:before]
    for stroke in reversed(seq):
        if stroke["direction"] == direction:
            return stroke
    return None


def add_chan_daily_signals(
    df: pd.DataFrame,
    params: ChanDailyParams = DEFAULT_CHAN_DAILY_PARAMS,
) -> pd.DataFrame:
    """Add daily Chan-theory structure signals.

    Main long signal:
    - third buy: price leaves an overlapping center upward, pulls back above
      the center top, then breaks the pullback stroke high;
    - second buy: a higher-low bottom after a prior lower-low/down-stroke
      divergence, confirmed by MA5/MA10 recovery.

    First-buy divergence is exposed as a watch signal and contributes to score,
    but it is not part of the final long signal by default.
    """
    out = _normalize_daily_frame(df)
    price = build_continuous_ohlc(out)
    open_ = price["open"].astype(float)
    high = price["high"].astype(float)
    low = price["low"].astype(float)
    close = price["close"].astype(float)
    volume = out["volume"].astype(float)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    atr = _true_range(high, low, close).rolling(params.atr_window).mean()
    volume_ma20 = volume.rolling(20).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_dif = ema12 - ema26
    macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
    macd_hist = (macd_dif - macd_dea) * 2

    fractals = _detect_fractals(high, low)
    strokes = _build_strokes(high, low, fractals, params.min_stroke_bars)

    out["chan_fx_mark"] = fractals["fx_mark"]
    out["chan_fx_top"] = fractals["fx_top"].astype(int)
    out["chan_fx_bottom"] = fractals["fx_bottom"].astype(int)
    out["chan_stroke_direction"] = ""
    out["chan_stroke_end"] = 0
    out["chan_stroke_amplitude"] = np.nan
    out["chan_center_low"] = np.nan
    out["chan_center_high"] = np.nan
    out["chan_center_width"] = np.nan
    out["chan_buy1_watch"] = 0
    out["chan_buy2_confirm"] = 0
    out["chan_buy3_confirm"] = 0
    out["signal_chan_daily_long"] = 0
    out["signal_chan_daily_exit"] = 0
    out["chan_signal_name"] = ""
    out["chan_score"] = np.nan
    out["chan_buy_plan"] = ""
    out["chan_sell_plan"] = ""
    out["chan_structure_note"] = ""

    n = len(out)
    stroke_direction = np.full(n, "", dtype=object)
    stroke_end = np.zeros(n, dtype=int)
    stroke_amplitude = np.full(n, np.nan)
    center_low_arr = np.full(n, np.nan)
    center_high_arr = np.full(n, np.nan)
    center_width_arr = np.full(n, np.nan)
    buy1_arr = np.zeros(n, dtype=int)
    buy2_arr = np.zeros(n, dtype=int)
    buy3_arr = np.zeros(n, dtype=int)
    long_arr = np.zeros(n, dtype=int)
    exit_arr = np.zeros(n, dtype=int)
    signal_name_arr = np.full(n, "", dtype=object)
    score_arr = np.full(n, np.nan)
    buy_plan_arr = np.full(n, "", dtype=object)
    sell_plan_arr = np.full(n, "", dtype=object)
    note_arr = np.full(n, "", dtype=object)

    for stroke in strokes:
        end_idx = stroke["end_idx"]
        stroke_direction[end_idx] = stroke["direction"]
        stroke_end[end_idx] = 1
        stroke_amplitude[end_idx] = stroke["amplitude"]

    if not strokes:
        return out

    trend_ok = (close > ma20) & (ma20 > ma20.shift(3)) & ((ma20 > ma60) | ma60.isna())
    liquidity_ok = (volume > 0) & ((volume_ma20.isna()) | (volume > volume_ma20 * 0.45))
    if "name" in out.columns:
        risk_name = _is_risk_name(out["name"])
    else:
        risk_name = pd.Series(False, index=out.index)
    tradable = open_.notna() & (open_ > 0) & ~risk_name & liquidity_ok

    close_values = close.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    ma5_values = ma5.to_numpy(dtype=float)
    ma10_values = ma10.to_numpy(dtype=float)
    ma20_values = ma20.to_numpy(dtype=float)
    ma60_values = ma60.to_numpy(dtype=float)
    atr_values = atr.to_numpy(dtype=float)
    trend_ok_values = trend_ok.fillna(False).to_numpy(dtype=bool)
    tradable_values = tradable.fillna(False).to_numpy(dtype=bool)
    macd_values = macd_hist.fillna(0).to_numpy(dtype=float)

    last_down_at = np.full(len(strokes) + 1, -1, dtype=int)
    prev_down_at = np.full(len(strokes) + 1, -1, dtype=int)
    last_up_at = np.full(len(strokes) + 1, -1, dtype=int)
    prev_up_at = np.full(len(strokes) + 1, -1, dtype=int)
    last_down = prev_down = last_up = prev_up = -1
    for cursor, stroke in enumerate(strokes, start=1):
        if stroke["direction"] == "down":
            prev_down = last_down
            last_down = cursor - 1
        else:
            prev_up = last_up
            last_up = cursor - 1
        last_down_at[cursor] = last_down
        prev_down_at[cursor] = prev_down
        last_up_at[cursor] = last_up
        prev_up_at[cursor] = prev_up
    for cursor in range(1, len(strokes) + 1):
        if last_down_at[cursor] == -1:
            last_down_at[cursor] = last_down_at[cursor - 1]
            prev_down_at[cursor] = prev_down_at[cursor - 1]
        if last_up_at[cursor] == -1:
            last_up_at[cursor] = last_up_at[cursor - 1]
            prev_up_at[cursor] = prev_up_at[cursor - 1]

    stroke_cursor = 0
    center = None
    for i in range(n):
        while stroke_cursor < len(strokes) and strokes[stroke_cursor]["end_idx"] <= i:
            stroke_cursor += 1
            center = _latest_center(strokes[:stroke_cursor], params)
        if center:
            center_low_arr[i] = center["low"]
            center_high_arr[i] = center["high"]
            center_width_arr[i] = center["width"]

        last_stroke = strokes[stroke_cursor - 1] if stroke_cursor else None
        # A signal can occur after the latest stroke endpoint.  The model
        # feature represents the latest completed stroke, so carry that
        # point-in-time value forward instead of exposing it only on the
        # endpoint bar and silently imputing every live signal row.
        if last_stroke is not None:
            stroke_amplitude[i] = last_stroke["amplitude"]
        last_down = strokes[last_down_at[stroke_cursor]] if last_down_at[stroke_cursor] >= 0 else None
        prev_down = strokes[prev_down_at[stroke_cursor]] if prev_down_at[stroke_cursor] >= 0 else None
        last_up = strokes[last_up_at[stroke_cursor]] if last_up_at[stroke_cursor] >= 0 else None
        prev_up = strokes[prev_up_at[stroke_cursor]] if prev_up_at[stroke_cursor] >= 0 else None

        buy1 = False
        if last_down and prev_down and last_stroke == last_down:
            lower_low = last_down["low"] < prev_down["low"] * 0.995
            price_span = max(last_down["end_idx"] - last_down["start_idx"], 1)
            prev_span = max(prev_down["end_idx"] - prev_down["start_idx"], 1)
            recent_power = abs(macd_values[last_down["start_idx"] : last_down["end_idx"] + 1].sum()) / price_span
            prev_power = abs(macd_values[prev_down["start_idx"] : prev_down["end_idx"] + 1].sum()) / prev_span
            atr_i = atr_values[i] if np.isfinite(atr_values[i]) else 0
            rebound = close_values[i] > low_values[i] + atr_i * 0.35
            buy1 = bool(lower_low and recent_power < prev_power * 0.85 and rebound)

        buy2 = False
        if last_down and prev_down:
            higher_low = last_down["low"] > prev_down["low"] * 1.005
            reclaim_ma = close_values[i] > ma5_values[i] and close_values[i] > ma10_values[i]
            trigger_price = last_down["high"] * (1 + params.breakout_buffer)
            break_last_down = close_values[i] > trigger_price and (i == 0 or close_values[i - 1] <= trigger_price)
            buy2 = bool(higher_low and reclaim_ma and break_last_down and trend_ok_values[i])

        buy3 = False
        if center and last_stroke and last_stroke["direction"] == "down":
            pullback_above_center = last_stroke["low"] > center["high"] * (1 - params.third_buy_tolerance)
            leave_center = last_up is not None and last_up["high"] > center["high"] * (1 + params.breakout_buffer)
            trigger_price = last_stroke["high"] * (1 + params.breakout_buffer)
            break_pullback = close_values[i] > trigger_price and (i == 0 or close_values[i - 1] <= trigger_price)
            buy3 = bool(pullback_above_center and leave_center and break_pullback and trend_ok_values[i])

        exit_signal = False
        if center:
            exit_signal = bool(close_values[i] < center["low"] or (np.isfinite(ma20_values[i]) and close_values[i] < ma20_values[i] * 0.985))
        if last_up and prev_up and last_stroke == last_up:
            higher_high = last_up["high"] > prev_up["high"] * 1.005
            up_power = abs(macd_values[last_up["start_idx"] : last_up["end_idx"] + 1].sum())
            prev_up_power = abs(macd_values[prev_up["start_idx"] : prev_up["end_idx"] + 1].sum())
            exit_signal = exit_signal or bool(higher_high and up_power < prev_up_power * 0.82 and close_values[i] < ma5_values[i])

        if buy1:
            buy1_arr[i] = 1
        if buy2:
            buy2_arr[i] = 1
        if buy3:
            buy3_arr[i] = 1
        if exit_signal:
            exit_arr[i] = 1

        final_long = (buy2 or buy3) and bool(tradable_values[i])
        if final_long:
            long_arr[i] = 1
            signal_name = "三买确认" if buy3 else "二买确认"
            signal_name_arr[i] = signal_name
            trend_bonus = 8.0 if np.isfinite(ma60_values[i]) and bool(ma20_values[i] > ma60_values[i]) else 4.0
            center_width = center_width_arr[i] if np.isfinite(center_width_arr[i]) else 0.12
            center_bonus = 10.0 * (1 - min(center_width, 0.2) / 0.2)
            stroke_bonus = 6.0 if last_stroke and last_stroke["amplitude"] < 0.18 else 2.0
            score_arr[i] = 78.0 + (8.0 if buy3 else 0.0) + trend_bonus + center_bonus + stroke_bonus
            buy_plan_arr[i] = (
                "日线收盘信号后，次日不高开超过3%可试仓；优先等待盘中回踩不破信号日中位价。"
            )
            sell_plan_arr[i] = (
                "跌破最近中枢下沿或20日线减仓/退出；若出现顶背驰并跌破5日线，优先锁定利润。"
            )
            if center:
                note_arr[i] = (
                    f"{signal_name}: center=[{center['low']:.2f},{center['high']:.2f}], "
                    f"width={center['width']:.2%}"
                )

    out["chan_stroke_direction"] = stroke_direction
    out["chan_stroke_end"] = stroke_end
    out["chan_stroke_amplitude"] = stroke_amplitude
    out["chan_center_low"] = center_low_arr
    out["chan_center_high"] = center_high_arr
    out["chan_center_width"] = center_width_arr
    out["chan_buy1_watch"] = buy1_arr
    out["chan_buy2_confirm"] = buy2_arr
    out["chan_buy3_confirm"] = buy3_arr
    out["signal_chan_daily_long"] = long_arr
    out["signal_chan_daily_exit"] = exit_arr
    out["chan_signal_name"] = signal_name_arr
    out["chan_score"] = pd.Series(score_arr, index=out.index).clip(upper=100)
    out["chan_buy_plan"] = buy_plan_arr
    out["chan_sell_plan"] = sell_plan_arr
    out["chan_structure_note"] = note_arr
    return out


def summarize_chan_daily(df: pd.DataFrame) -> dict[str, float | int | pd.Timestamp]:
    out = add_chan_daily_signals(df)
    signal = out["signal_chan_daily_long"] == 1
    return {
        "rows": int(len(out)),
        "long_signals": int(signal.sum()),
        "buy1_watch": int(out["chan_buy1_watch"].sum()),
        "buy2_confirm": int(out["chan_buy2_confirm"].sum()),
        "buy3_confirm": int(out["chan_buy3_confirm"].sum()),
        "exit_signals": int(out["signal_chan_daily_exit"].sum()),
        "first_signal": out.loc[signal, "date"].min() if signal.any() else pd.NaT,
        "last_signal": out.loc[signal, "date"].max() if signal.any() else pd.NaT,
    }
