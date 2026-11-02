"""Fetch missing watchlist daily cells from Tushare and publish audited null-only repairs."""

from quant.research.kronos_data_repair import run

if __name__ == "__main__":
    print(run())
