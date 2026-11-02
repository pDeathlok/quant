"""Draw additive P&L attribution, with annual contribution in a second panel."""
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[2] / "reports/strategy_lab_20260930"


def main() -> None:
    summary = pd.read_csv(OUT / "stable_asset_attribution_summary.csv")
    yearly = pd.read_csv(OUT / "stable_asset_attribution_yearly.csv")
    canvas = Image.new("RGB", (1200, 650), "white")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=24)
    font = ImageFont.load_default(size=18)
    draw.text((52, 25), "Where did the stable portfolio's profit come from?", font=title, fill="#17283a")
    draw.text((52, 65), "2019-02-01 to 2026-09-24 | Additive P&L per 100 initial capital", font=font, fill="#647282")
    colors = {"513100.SH": "#226cab", "518880.SH": "#cb9229", "511010.SH": "#31917c", "cost": "#9d6671"}
    names = {"513100.SH": "Nasdaq ETF", "518880.SH": "Gold ETF", "511010.SH": "Treasury ETF", "cost": "Costs"}
    for row_no, row in enumerate(summary.itertuples()):
        y = 125 + row_no * 53
        value = row.contribution_to_initial_capital * 100
        draw.text((52, y), names[row.asset], font=font, fill="#233445")
        start, end = 257, 257 + round(value * 10.5)
        draw.rectangle((min(start, end), y, max(start, end), y + 27), fill=colors[row.asset])
        draw.text((max(start, end) + 13, y + 3), f"{value:+.2f}", font=font, fill="#233445")
    draw.text((52, 355), "Annual P&L contribution (% of that year's starting capital)", font=font, fill="#233445")
    baseline, scale = 515, 5.4
    draw.line((72, baseline, 1160, baseline), fill="#c3cbd3", width=2)
    for j, row in enumerate(yearly.itertuples(index=False, name=None)):
        year, _, *data = row
        x = 99 + j * 135
        up, down = baseline, baseline
        for asset, value in zip(colors, data[:4]):
            height = round(abs(value * 100) * scale)
            if value >= 0:
                draw.rectangle((x, up - height, x + 54, up), fill=colors[asset]); up -= height
            else:
                draw.rectangle((x, down, x + 54, down + height), fill=colors[asset]); down += height
        label = str(year) + ("*" if year in (2019, 2026) else "")
        draw.text((x, 540), label, font=font, fill="#233445")
    draw.text((52, 594), "* Partial year. Contributions sum to net return; they are not standalone asset returns.", font=font, fill="#647282")
    canvas.save(OUT / "stable_asset_attribution.png")


if __name__ == "__main__":
    main()
