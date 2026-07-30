import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolate_convertible_bond_watchlist(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    monkeypatch.setattr(module, "CB_WATCHLIST_PATH", tmp_path / "missing_watchlist.csv")
    monkeypatch.setattr(module, "CB_PIPELINE_ISSUE_SIZE_PATH", tmp_path / "missing_issue_size.parquet")
    monkeypatch.setattr(module, "DAILY_BASIC_DIR", tmp_path / "missing_daily_basic")


def test_convertible_bond_allotment_merges_basic_and_issue(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "cb_basic_all.parquet"
    issue_path = tmp_path / "cb_issue_all.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "cninfo_issue.parquet"
    basic = pd.DataFrame(
        [
            {
                "ts_code": "123456.SZ",
                "bond_short_name": "测试转债",
                "stk_code": "300001.SZ",
                "stk_short_name": "测试正股",
                "list_date": "20260701",
                "delist_date": "",
                "remain_size": 500000000.0,
                "conv_start_date": "20261230",
                "conv_end_date": "20300630",
                "issue_rating": "AA-",
                "newest_rating": "AA",
            }
        ]
    )
    issue = pd.DataFrame(
        [
            {
                "ts_code": "123456.SZ",
                "ann_date": "20260615",
                "onl_date": "20260620",
                "shd_ration_code": "380001",
                "shd_ration_name": "测试配债",
                "shd_ration_record_date": "20260619",
                "shd_ration_pay_date": "20260620",
                "shd_ration_ratio": 2.5,
            }
        ]
    )
    basic.to_parquet(basic_path, index=False)
    issue.to_parquet(issue_path, index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18), stage_scope="all")

    assert payload["records"]
    record = payload["records"][0]
    assert record["stock_code"] == "300001.SZ"
    assert record["bond_code"] == "123456.SZ"
    assert record["allot_code"] == "380001"
    assert record["record_date"] == "2026-06-19"
    assert record["pay_date"] == "2026-06-20"
    assert record["stage"] == "pending_listing"


def test_convertible_bond_allotment_pipeline_scope_does_not_fallback_to_listed(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "cb_basic_all.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "missing_pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": "110001.SH",
                "bond_short_name": "基础转债",
                "stk_code": "600001.SH",
                "stk_short_name": "基础正股",
                "list_date": "20260610",
                "delist_date": "",
                "remain_size": 100000000.0,
                "newest_rating": "AA+",
            }
        ]
    ).to_parquet(basic_path, index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    assert payload["data_sources"]["issue"]["available"] is False
    assert payload["stage_scope"] == "pipeline"
    assert payload["records"] == []


def test_convertible_bond_allotment_uses_pipeline_announcements(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    pd.DataFrame(
        [
            {
                "stock_code": "300001",
                "stock_name": "测试股份",
                "announcement_title": "测试股份向不特定对象发行可转换公司债券募集说明书审核问询函回复",
                "announce_date": "2026-06-10",
                "announcement_url": "http://example.test/a",
                "stage": "inquiry",
                "status": "问询回复",
            }
        ]
    ).to_parquet(pipeline_path, index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    assert payload["records"]
    record = payload["records"][0]
    assert record["stock_code"] == "300001"
    assert record["stage"] == "accepted"
    assert record["status"] == "交易所受理"
    assert "审核" in record["announcement_title"]


def test_convertible_bond_allotment_filters_expired_record_date(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "missing_pipeline.parquet"
    cninfo_issue_path = tmp_path / "cninfo_issue.parquet"
    pd.DataFrame(
        [
            {
                "债券代码": "123999",
                "债券简称": "过期转债",
                "转股代码": "300999",
                "网上申购代码": "370999",
                "网上申购简称": "过期发债",
                "公告日期": "2026-06-01",
                "网上申购日期": "2026-06-09",
                "优先申购缴款日": "2026-06-09",
                "发行对象": "向发行人原股东优先配售：发行公告公布的股权登记日（2026年6月8日，T-1日）收市后登记在册的发行人所有股东。",
                "债券名称": "过期股份向不特定对象发行可转换公司债券",
            },
            {
                "债券代码": "123998",
                "债券简称": "可配转债",
                "转股代码": "300998",
                "网上申购代码": "370998",
                "网上申购简称": "可配发债",
                "公告日期": "2026-06-01",
                "网上申购日期": "2026-06-21",
                "优先申购缴款日": "2026-06-21",
                "发行对象": "向发行人原股东优先配售：发行公告公布的股权登记日（2026年6月20日，T-1日）收市后登记在册的发行人所有股东。",
                "债券名称": "可配股份向不特定对象发行可转换公司债券",
            },
        ]
    ).to_parquet(cninfo_issue_path, index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    codes = {record["bond_code"] for record in payload["records"]}
    assert "123999" not in codes
    assert "123998" in codes


def test_convertible_bond_allotment_merges_issue_dates_before_expired_filter(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "cninfo_issue.parquet"
    pd.DataFrame(
        [
            {
                "stock_code": "300001",
                "stock_name": "测试股份",
                "announcement_title": "测试股份向不特定对象发行可转换公司债券发行提示性公告",
                "announce_date": "2026-06-17",
                "announcement_url": "http://example.test/a",
            },
            {
                "stock_code": "300002",
                "stock_name": "过期股份",
                "announcement_title": "过期股份向不特定对象发行可转换公司债券发行提示性公告",
                "announce_date": "2026-06-17",
                "announcement_url": "http://example.test/b",
            },
        ]
    ).to_parquet(pipeline_path, index=False)
    pd.DataFrame(
        [
            {
                "债券代码": "123001",
                "债券简称": "测试转债",
                "转股代码": "300001",
                "网上申购代码": "370001",
                "网上申购简称": "测试发债",
                "公告日期": "2026-06-17",
                "网上申购日期": "2026-06-19",
                "优先申购缴款日": "2026-06-19",
                "发行对象": "发行公告公布的股权登记日（2026年6月18日，T-1日）收市后登记在册的发行人所有股东。",
                "债券名称": "测试股份向不特定对象发行可转换公司债券",
            },
            {
                "债券代码": "123002",
                "债券简称": "过期转债",
                "转股代码": "300002",
                "网上申购代码": "370002",
                "网上申购简称": "过期发债",
                "公告日期": "2026-06-17",
                "网上申购日期": "2026-06-17",
                "优先申购缴款日": "2026-06-17",
                "发行对象": "发行公告公布的股权登记日（2026年6月16日，T-1日）收市后登记在册的发行人所有股东。",
                "债券名称": "过期股份向不特定对象发行可转换公司债券",
            },
        ]
    ).to_parquet(cninfo_issue_path, index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 17))

    by_code = {record["stock_code"]: record for record in payload["records"]}
    assert "300002" not in by_code
    assert by_code["300001"]["record_date"] == "2026-06-18"
    assert by_code["300001"]["pay_date"] == "2026-06-19"
    assert by_code["300001"]["issue_date"] == "2026-06-19"


def test_convertible_bond_allotment_adds_stock_price_and_kdj(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame(
        [
            {
                "stock_code": "300001",
                "stock_name": "测试股份",
                "announcement_title": "测试股份向不特定对象发行可转换公司债券审核问询函回复",
                "announce_date": "2026-06-10",
                "announcement_url": "http://example.test/a",
                "stage": "inquiry",
                "status": "问询回复",
            }
        ]
    ).to_parquet(pipeline_path, index=False)
    dates = pd.bdate_range("2025-01-02", periods=320)
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": [10 + idx * 0.03 for idx in range(len(dates))],
            "high": [10.5 + idx * 0.03 for idx in range(len(dates))],
            "low": [9.5 + idx * 0.03 for idx in range(len(dates))],
            "close": [10.2 + idx * 0.03 for idx in range(len(dates))],
            "volume": [100000 + idx for idx in range(len(dates))],
        }
    )
    daily.to_parquet(daily_dir / "300001.SZ.parquet", index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)
    monkeypatch.setattr(module, "STOCK_DAILY_DIR", daily_dir)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    record = payload["records"][0]
    assert record["stock_price"] == round(daily["close"].iloc[-1], 2)
    assert record["stock_price_date"] == dates[-1].strftime("%Y-%m-%d")
    assert record["kdj_daily_j"] is not None
    assert record["kdj_weekly_j"] is not None
    assert record["kdj_monthly_j"] is not None


@pytest.mark.parametrize(
    ("stock_code", "ts_code"),
    [
        ("300001", "300001.SZ"),
        ("920826", "920826.BJ"),
    ],
    ids=["shenzhen", "beijing_920"],
)
def test_convertible_bond_allotment_reads_partitioned_market_daily(
    monkeypatch,
    tmp_path,
    stock_code,
    ts_code,
):
    from quant.data import MarketDataStore, MarketDataStoreConfig
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    daily_dir = tmp_path / "daily"
    pd.DataFrame(
        [
            {
                "stock_code": stock_code,
                "stock_name": "测试股份",
                "announcement_title": "测试股份向不特定对象发行可转换公司债券审核问询函回复",
                "announce_date": "2026-06-10",
                "announcement_url": "http://example.test/a",
                "stage": "inquiry",
                "status": "问询回复",
            }
        ]
    ).to_parquet(pipeline_path, index=False)
    dates = pd.bdate_range("2025-01-02", periods=320)
    daily = pd.DataFrame(
        {
            "ts_code": [ts_code] * len(dates),
            "trade_date": dates.strftime("%Y%m%d"),
            "open": [10 + idx * 0.03 for idx in range(len(dates))],
            "high": [10.5 + idx * 0.03 for idx in range(len(dates))],
            "low": [9.5 + idx * 0.03 for idx in range(len(dates))],
            "close": [10.2 + idx * 0.03 for idx in range(len(dates))],
            "vol": [100000 + idx for idx in range(len(dates))],
        }
    )
    store = MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=tmp_path, mirror_parquet=True)
    )
    store.write_market_batch(daily)
    monkeypatch.setenv("MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MARKET_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("MARKET_DATA_SQL_URL", raising=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)
    monkeypatch.setattr(module, "STOCK_DAILY_DIR", daily_dir)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    record = payload["records"][0]
    assert record["stock_price"] == round(daily["close"].iloc[-1], 2)
    assert record["stock_price_date"] == dates[-1].strftime("%Y-%m-%d")
    assert record["kdj_daily_j"] is not None
    assert record["kdj_weekly_j"] is not None
    assert record["kdj_monthly_j"] is not None
    assert payload["data_sources"]["stock_daily"]["matched"] == 1


def test_convertible_bond_allotment_keeps_watchlist_metrics_as_manual_reference_only(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "missing_pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    watchlist_path = tmp_path / "watchlist.csv"
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    pd.DataFrame(
        [
            {
                "stock_code": "688200",
                "stock_name": "华峰测控",
                "shares_for_10_bonds": 181,
                "one_lot_party": True,
                "stage": "registered",
                "status": "注册批复",
                "sort_order": 1,
                "allotment_note": "科创板可一手党",
            }
        ]
    ).to_csv(watchlist_path, index=False)
    dates = pd.bdate_range("2025-01-02", periods=40)
    pd.DataFrame(
        {
            "date": dates,
            "open": [100 + idx for idx in range(len(dates))],
            "high": [101 + idx for idx in range(len(dates))],
            "low": [99 + idx for idx in range(len(dates))],
            "close": [100 + idx for idx in range(len(dates))],
            "volume": [100000 + idx for idx in range(len(dates))],
        }
    ).to_parquet(daily_dir / "688200.SH.parquet", index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)
    monkeypatch.setattr(module, "CB_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(module, "STOCK_DAILY_DIR", daily_dir)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    assert payload["records"] == []


def test_convertible_bond_allotment_estimates_one_lot_shares_from_issue_size_and_share_capital(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    daily_basic_dir = tmp_path / "daily_basic"
    daily_basic_dir.mkdir()
    pd.DataFrame(
        [
            {
                "stock_code": "688200",
                "stock_name": "华峰测控",
                "announcement_title": "华峰测控向不特定对象发行可转换公司债券同意注册批复",
                "announce_date": "2026-06-10",
                "stage": "registered",
                "status": "注册批复",
                "issue_size": 1_109_000_000.0,
            }
        ]
    ).to_parquet(pipeline_path, index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "688200.SH",
                "trade_date": "20260616",
                "total_share": 20057.5083,
                "float_share": 20057.5083,
            }
        ]
    ).to_parquet(daily_basic_dir / "20260616.parquet", index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)
    monkeypatch.setattr(module, "DAILY_BASIC_DIR", daily_basic_dir)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    record = payload["records"][0]
    assert record["total_share"] == round(20057.5083 * 10000, 0)
    assert record["issue_size_yuan"] == 1_109_000_000
    assert record["shares_for_one_lot"] == 181
    assert record["shares_for_10_bonds"] == 181
    assert record["shares_for_one_lot_source"] == "issue_size_total_share"
    assert payload["data_sources"]["daily_basic"]["matched"] == 1


def test_convertible_bond_allotment_prefers_pipeline_record_without_using_watchlist_metric_fallback(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    basic_path = tmp_path / "missing_basic.parquet"
    issue_path = tmp_path / "missing_issue.parquet"
    pipeline_path = tmp_path / "pipeline.parquet"
    cninfo_issue_path = tmp_path / "missing_cninfo_issue.parquet"
    watchlist_path = tmp_path / "watchlist.csv"
    pd.DataFrame(
        [
            {
                "stock_code": "688200",
                "stock_name": "华峰测控",
                "announcement_title": "华峰测控关于向不特定对象发行可转换公司债券申请获得中国证券监督管理委员会同意注册批复的公告",
                "announce_date": "2025-12-31",
                "announcement_url": "http://example.test/auto",
                "stage": "registered",
                "status": "注册批复",
            }
        ]
    ).to_parquet(pipeline_path, index=False)
    pd.DataFrame(
        [
            {
                "stock_code": "688200",
                "stock_name": "华峰测控",
                "shares_for_10_bonds": 181,
                "stage": "registered",
                "status": "注册批复",
                "sort_order": 1,
            }
        ]
    ).to_csv(watchlist_path, index=False)
    monkeypatch.setattr(module, "CB_BASIC_PATH", basic_path)
    monkeypatch.setattr(module, "CB_ISSUE_PATH", issue_path)
    monkeypatch.setattr(module, "CB_PIPELINE_PATH", pipeline_path)
    monkeypatch.setattr(module, "CB_CNINFO_ISSUE_PATH", cninfo_issue_path)
    monkeypatch.setattr(module, "CB_WATCHLIST_PATH", watchlist_path)

    payload = module.build_convertible_bond_allotment_payload(today=date(2026, 6, 18))

    record = payload["records"][0]
    assert record["announcement_url"] == "http://example.test/auto"
    assert record.get("data_source") is None
    assert record["manual_shares_for_10_bonds"] == 181
    assert record["shares_for_one_lot"] is None
    assert record["shares_for_one_lot_source"] is None


def test_convertible_bond_allotment_refreshes_issue_size_for_legacy_inquiry_stage(monkeypatch, tmp_path):
    import quant.routine.convertible_bond_allotment as module

    issue_size_path = tmp_path / "issue_size.parquet"
    calls = []

    def fake_refresh(record, today=None):
        calls.append(record["stock_code"])
        return {
            "stock_code": record["stock_code"],
            "stock_name": record["stock_name"],
            "issue_size_yuan": 600_000_000,
            "issue_size_source": "cninfo_pdf",
            "issue_size_title": "测试募集说明书",
            "issue_size_url": "http://example.test/doc.pdf",
        }

    monkeypatch.setattr(module, "CB_PIPELINE_ISSUE_SIZE_PATH", issue_size_path)
    monkeypatch.setattr(module, "_refresh_issue_size_for_record", fake_refresh)
    records = [
        {
            "stock_code": "300001",
            "stock_name": "测试股份",
            "stage": "inquiry",
            "issue_size": None,
        }
    ]

    refreshed, meta = module._attach_pipeline_issue_sizes(records, refresh=True, today=date(2026, 6, 18))

    assert calls == ["300001"]
    assert refreshed[0]["issue_size_yuan"] == 600_000_000
    assert refreshed[0]["issue_size_source"] == "cninfo_pdf"
    assert meta["refreshed"] == 1


def test_convertible_bond_allotment_extracts_issue_dates_from_issuing_text():
    import quant.routine.convertible_bond_allotment as module

    text = """
    可转债代码 118070 可转债简称 南芯转债
    原股东配售代码 726484 原股东配售简称 南芯配债
    转债申购代码 718484 转债申购简称 南芯发债
    发行日期及时间 （2026 年 6 月 18 日）（9:30-11:30,13:00-15:00）
    股权登记日 2026 年 6 月 17 日 原股东缴款日 2026 年 6 月 18 日
    """

    parsed = module._extract_issue_dates_from_text(text)

    assert parsed["record_date"] == "2026-06-17"
    assert parsed["pay_date"] == "2026-06-18"
    assert parsed["issue_date"] == "2026-06-18"
    assert parsed["allot_code"] == "726484"
    assert parsed["allot_name"] == "南芯配债"
    assert parsed["bond_code"] == "118070"
    assert parsed["bond_name"] == "南芯转债"
