"""Plot the three research candidates and common-window CSI 300 benchmark."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/strategy_lab_20260930"
SERIES = {
    "buy_nasdaq": ("Highest CAGR: Nasdaq ETF", "#1376a8"),
    "pullback_3pct_tp2_gold_300": ("Highest trade hit rate: pullback", "#d08b29"),
    "invvol60_nasdaq_trend": ("Stable: inverse vol + trend", "#167c60"),
    "buy_300": ("CSI 300 benchmark", "#787e89"),
}


def main() -> None:
    width, height = 1200, 740
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    frames = {}
    for key in SERIES:
        frame = pd.read_csv(OUT / f"{key}_daily.csv", parse_dates=["date"])
        equity = (1 + frame.daily_return).cumprod()
        frames[key] = (frame.date, equity, equity / equity.cummax().clip(lower=1) - 1)
    first = min(dates.iloc[0] for dates, _, _ in frames.values())
    last = max(dates.iloc[-1] for dates, _, _ in frames.values())
    x0, x1 = 88, 1170
    panels = [(74, 430, 0.5, 5.0, "Growth of 1 CNY"),
              (520, 670, -0.45, 0.02, "Drawdown (%)")]
    draw.text((x0, 20), "Costed ETF strategies, 2019-02 to 2026-09-24", fill="#222222", font=font)

    def xvalue(date):
        return x0 + round((date - first).days / (last - first).days * (x1 - x0))

    for top, bottom, ymin, ymax, label in panels:
        draw.text((x0, top - 18), label, fill="#333333", font=font)
        draw.line([(x0, top), (x0, bottom), (x1, bottom)], fill="#777777", width=2)
        for step in range(6):
            val = ymin + (ymax - ymin) * step / 5
            y = bottom - round((val - ymin) / (ymax - ymin) * (bottom - top))
            draw.line([(x0, y), (x1, y)], fill="#e5e5e5")
            mark = f"{val:.1f}" if top < 100 else f"{100 * val:.0f}"
            draw.text((x0 - 42, y - 6), mark, fill="#444444", font=font)
        for year in range(2019, 2027):
            date = pd.Timestamp(f"{year}-07-01")
            if first <= date <= last:
                x = xvalue(date)
                draw.line([(x, top), (x, bottom)], fill="#eeeeee")
                if top > 100:
                    draw.text((x - 14, bottom + 9), str(year), fill="#444444", font=font)

    for key, (label, color) in SERIES.items():
        dates, equity, drawdown = frames[key]
        for panel, values in zip(panels, (equity, drawdown)):
            top, bottom, ymin, ymax, _ = panel
            coords = [(xvalue(day), bottom - round((max(ymin, min(ymax, float(value))) - ymin)
                       / (ymax - ymin) * (bottom - top))) for day, value in zip(dates, values)]
            draw.line(coords, fill=color, width=3, joint="curve")
    for i, (label, color) in enumerate(SERIES.values()):
        x = x0 + (i % 2) * 360
        y = 444 + (i // 2) * 19
        draw.line([(x, y + 5), (x + 24, y + 5)], fill=color, width=4)
        draw.text((x + 32, y), label, fill="#222222", font=font)
    im.save(OUT / "equity_drawdown.png")


if __name__ == "__main__":
    main()
