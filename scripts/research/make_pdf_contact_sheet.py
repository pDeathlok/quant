#!/usr/bin/env python3
"""Build a compact contact sheet from rendered PDF page images for visual QA."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    columns = 2
    width = 900
    gap = 16
    label_height = 34
    thumbs: list[tuple[Image.Image, str]] = []
    for path in args.images:
        image = Image.open(path).convert("RGB")
        image.thumbnail((width, 1260))
        thumbs.append((image.copy(), path.stem))
    cell_height = max(image.height for image, _ in thumbs) + label_height
    rows = math.ceil(len(thumbs) / columns)
    canvas = Image.new(
        "RGB",
        (columns * width + (columns + 1) * gap, rows * cell_height + (rows + 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(thumbs):
        row, column = divmod(index, columns)
        x = gap + column * (width + gap)
        y = gap + row * cell_height
        canvas.paste(image, (x, y + label_height))
        draw.text((x, y + 6), label, fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
