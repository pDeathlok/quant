import pandas as pd

from quant.routine import tradability_refresh


class FakePro:
    def __init__(self) -> None:
        self.limit_calls = 0
        self.basic_statuses = []

    def stock_basic(self, *, list_status, **kwargs):
        self.basic_statuses.append(list_status)
        frames = {
            "L": pd.DataFrame(
                {
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "list_date": ["19910403", "20260106"],
                    "delist_date": [None, None],
                    "market": ["主板", "主板"],
                }
            ),
            "D": pd.DataFrame(
                {
                    "ts_code": ["600001.SH"],
                    "list_date": ["20000101"],
                    "delist_date": ["20260105"],
                    "market": ["主板"],
                }
            ),
            "P": pd.DataFrame(),
        }
        return frames[list_status]

    def stk_limit(self, *, trade_date):
        self.limit_calls += 1
        codes = ["000001.SZ", "600001.SH"] if trade_date == "20260105" else ["000001.SZ", "000002.SZ"]
        return pd.DataFrame(
            {
                "trade_date": trade_date,
                "ts_code": codes,
                "pre_close": 10.0,
                "up_limit": 11.0,
                "down_limit": 9.0,
            }
        )

    def suspend_d(self, **kwargs):
        return pd.DataFrame(columns=["trade_date", "ts_code", "suspend_type"])

    def stock_st(self, **kwargs):
        return pd.DataFrame(columns=["trade_date", "ts_code", "type_name"])


class FakeFetcher:
    def __init__(self) -> None:
        self.pro = FakePro()


def test_tradability_backfill_uses_all_listing_statuses_and_skips_checkpoints(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        tradability_refresh,
        "load_trade_dates_from_daily",
        lambda *args: ["20260105", "20260106"],
    )
    fetcher = FakeFetcher()
    output_dir = tmp_path / "raw/tradability"
    audit_root = tmp_path / "audit"

    first = tradability_refresh.refresh_tradability_range(
        start_date="20260105",
        end_date="20260106",
        output_dir=output_dir,
        audit_root=audit_root,
        fetcher=fetcher,
        sleep_between=0,
        minimum_coverage_rate=1.0,
    )
    second = tradability_refresh.refresh_tradability_range(
        start_date="20260105",
        end_date="20260106",
        output_dir=output_dir,
        audit_root=audit_root,
        fetcher=fetcher,
        sleep_between=0,
        minimum_coverage_rate=1.0,
    )

    assert first["status"] == "success"
    assert first["success"] == 2
    assert second["skipped"] == 2
    assert fetcher.pro.limit_calls == 2
    assert fetcher.pro.basic_statuses == ["L", "D", "P", "L", "D", "P"]
    assert set(pd.read_parquet(output_dir / "20260105.parquet")["ts_code"]) == {
        "000001.SZ",
        "600001.SH",
    }
