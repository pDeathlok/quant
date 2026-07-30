"""Compatibility CLI for the packaged B1 formal-combo validation job."""

from quant.research.b1_formal_combos import COMBOS, combo_mask, main

__all__ = ["COMBOS", "combo_mask", "main"]


if __name__ == "__main__":
    main()
