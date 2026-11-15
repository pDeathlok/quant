import pandas as pd

from quant.infrastructure.publication import PublicationStore
from quant.routine import short_etf_flow


class FundSource:
    def __init__(self):
        self.calls = []
        self.stale = False

    def trade_cal(self, **kwargs):
        calendar = pd.bdate_range(kwargs["start_date"], kwargs["end_date"])
        return pd.DataFrame({"cal_date": calendar.strftime("%Y%m%d"), "is_open": "1"})

    def etf_basic(self, **kwargs):
        return pd.DataFrame([
            {"ts_code": code, "index_code": index_code, "list_date": "20200101"}
            for index_code, codes in short_etf_flow.ETF_BASKETS.values() for code in codes
        ])

    def __getattr__(self, name):
        date_col, values = short_etf_flow.SOURCES[name]

        def call(ts_code, start_date, end_date, **kwargs):
            self.calls.append((name, ts_code, start_date, end_date))
            calendar = pd.bdate_range(start_date, end_date)
            if self.stale and name == "fund_daily":
                calendar = calendar[calendar < pd.Timestamp(end_date)]
            frame = pd.DataFrame({"ts_code": ts_code, date_col: calendar.strftime("%Y%m%d")})
            for value in values:
                frame[value] = calendar.strftime("%Y%m%d") if value == "ann_date" else 10.0
            return frame

        return call


def test_short_etf_refresh_is_incremental_date_bound_and_preserves_snapshot(tmp_path):
    source = FundSource()
    read = lambda date: short_etf_flow.read_short_etf_flow(date, project_root=tmp_path)
    refresh = lambda date: short_etf_flow.refresh_short_etf_flow(date, project_root=tmp_path, pro=source)
    assert refresh("20261105")["status"] == "success"
    assert len(source.calls) == 60
    assert read("2026-11-04") == {"status": "unavailable"}

    assert refresh("20261106")["status"] == "success"
    assert len(source.calls) == 120
    assert {call[2] for call in source.calls[60:]} == {"20261023"}
    current = read("2026-11-06")
    assert current["signal_date"] == "2026-11-06"
    assert len(current["baskets"]) == 9
    assert current["baskets"][0]["flow_yuan_5"] == 0
    assert current["baskets"][0]["through_5"] == "2026-11-04"

    source.stale = True
    result = refresh("20261109")
    assert result["status"] == "degraded"
    assert read("2026-11-06") == current
    assert read("2026-11-09") == {"status": "unavailable"}


def test_short_etf_context_is_published_with_the_generation(tmp_path):
    from quant.infrastructure.publication import publication_context

    store = PublicationStore(tmp_path, (short_etf_flow.SNAPSHOT_DIRECTORY,))
    date = "2026-11-06"
    with store.begin("short-etf-test") as staged:
        result = short_etf_flow.refresh_short_etf_flow(date, project_root=tmp_path, pro=FundSource())
        assert result["status"] == "success"
        with publication_context(store.view()):
            assert short_etf_flow.read_short_etf_flow(date, project_root=tmp_path)["status"] == "unavailable"
        store.commit(staged, {"status": "success", "freshness_audit": {"status": "success"}})
    with publication_context(store.view()):
        assert short_etf_flow.read_short_etf_flow(date, project_root=tmp_path)["status"] == "success"
