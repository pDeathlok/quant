from __future__ import annotations

import re
import math
import io
import signal
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quant.data import MarketDataStore, MarketDataStoreConfig, read_partitioned_symbol_file
from quant.data.source_merge import normalize_ts_code
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.features.variable_library import calculate_project_extra_features
from quant.routine.convertible_bond_grid_plan import CB_BASIC_PATH, CB_DATA_DIR
from quant.routine.paths import DAILY_DIR, PROJECT_ROOT


CB_ISSUE_PATH = CB_DATA_DIR / "cb_issue_all.parquet"
CB_PIPELINE_PATH = CB_DATA_DIR / "cb_pipeline_candidates.parquet"
CB_CNINFO_ISSUE_PATH = CB_DATA_DIR / "cb_cninfo_issue.parquet"
CB_PIPELINE_ISSUE_SIZE_PATH = CB_DATA_DIR / "cb_pipeline_issue_size.parquet"
CB_PIPELINE_ISSUE_DATE_PATH = CB_DATA_DIR / "cb_pipeline_issue_dates.parquet"
STOCK_DAILY_DIR = DAILY_DIR
DAILY_BASIC_DIR = PROJECT_ROOT / "data/raw/daily_basic"
CB_WATCHLIST_PATH = PROJECT_ROOT / "configs/convertible_bond_allotment_watchlist.csv"
PIPELINE_STAGES = {
    "board_plan",
    "shareholder_approved",
    "accepted",
    "exchange_approved",
    "registered",
    "issuing",
}

PIPELINE_STAGE_STATUS = {
    "board_plan": "董事会预案",
    "shareholder_approved": "股东大会通过",
    "accepted": "交易所受理",
    "exchange_approved": "上市委通过",
    "registered": "同意注册",
    "issuing": "发行公告",
}


class _PdfExtractTimeout(Exception):
    pass


def _raise_pdf_extract_timeout(signum: int, frame: Any) -> None:
    raise _PdfExtractTimeout()


def _today_text(today: date | None = None) -> str:
    return (today or date.today()).strftime("%Y%m%d")


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"<[^>]+>", "", str(value)).strip()
    if not text or text.lower() in {"nan", "none", "nat", "null"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _is_pipeline_issuance_title(title: Any) -> bool:
    text = _clean_text(title) or ""
    non_public_issue = [
        "向特定对象发行可转债",
        "向特定对象发行可转换公司债券",
        "定向发行可转债",
        "定向发行可转换公司债券",
        "定向可转债",
        "发行股份及可转换公司债券购买资产",
        "发行股份、可转换公司债券购买资产",
    ]
    if any(key in text for key in non_public_issue):
        return False
    include = [
        "向不特定对象发行可转债",
        "向不特定对象发行可转换公司债券",
        "公开发行可转换公司债券",
        "发行可转换公司债券",
        "发行可转债",
        "可转债预案",
        "可转换公司债券预案",
    ]
    if not any(key in text for key in include):
        return False
    exclude = [
        "转股价格调整",
        "调整可转债转股价",
        "下修可转债转股价格",
        "转股价格条件",
        "转股结果",
        "付息",
        "赎回",
        "回售",
        "兑付",
        "摘牌",
        "持有人会议",
        "跟踪评级",
        "评级报告",
        "受托管理",
        "募投项目结项",
        "募集资金投资项目",
        "节余募集资金",
        "募集资金专项账户",
        "重新论证并延期",
        "永久性补充流动资金",
    ]
    return not any(key in text for key in exclude)


def _date_text(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    chinese_date = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text)
    if chinese_date:
        year, month, day = chinese_date.groups()
        return f"{year}{int(month):02d}{int(day):02d}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return text


def _display_date(value: Any) -> str | None:
    text = _date_text(value)
    if not text or len(text) != 8 or not text.isdigit():
        return text
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rounded(value: Any, digits: int = 2) -> float | None:
    number = _number(value)
    if number is None:
        return None
    return round(number, digits)


def _bool_value(value: Any) -> bool | None:
    text = _clean_text(value)
    if text is None:
        return None
    return text.lower() in {"1", "true", "yes", "y", "一手党", "是"}


def _first(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column in row.index:
            value = row.get(column)
            if _clean_text(value) is not None:
                return value
    return None


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _read_any_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    return pd.read_parquet(path)


def _ensure_optional_pdf_import_path() -> None:
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib"
    if not bundled.exists():
        return
    for site_packages in bundled.glob("python*/site-packages"):
        if str(site_packages) not in sys.path:
            sys.path.append(str(site_packages))


def _stock_code_candidates(stock_code: Any) -> list[str]:
    raw = _clean_text(stock_code)
    if not raw:
        return []
    code = raw.split(".")[0]
    candidates = [raw]
    if "." not in raw and len(code) == 6:
        candidates.append(normalize_ts_code(code))
    candidates.append(code)
    return list(dict.fromkeys(candidates))


def _legacy_stock_daily_path(stock_code: Any, daily_dir: Path) -> Path | None:
    for candidate in _stock_code_candidates(stock_code):
        path = daily_dir / f"{candidate}.parquet"
        if path.exists():
            return path
    return None


def _load_stock_daily(
    stock_code: Any,
    daily_dir: Path | None = None,
    canonical_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    daily_dir = daily_dir or STOCK_DAILY_DIR
    candidates = _stock_code_candidates(stock_code)
    legacy_path = _legacy_stock_daily_path(stock_code, daily_dir)
    if legacy_path is not None:
        return read_partitioned_symbol_file(legacy_path)
    if canonical_daily is not None:
        if canonical_daily.empty or "ts_code" not in canonical_daily.columns:
            return pd.DataFrame()
        symbol_rows = canonical_daily[
            canonical_daily["ts_code"].astype(str).isin(candidates)
        ]
        return symbol_rows.reset_index(drop=True)
    store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
    return store.read_market_range(daily_dir.name, symbols=candidates)


def _prepare_stock_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "volume" not in out.columns and "vol" in out.columns:
        out["volume"] = out["vol"]
    if "pct_chg" not in out.columns:
        out["pct_chg"] = pd.to_numeric(out.get("close"), errors="coerce").pct_change() * 100
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required <= set(out.columns):
        return pd.DataFrame()
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return out


def _market_snapshot_from_daily(daily: pd.DataFrame) -> dict[str, Any]:
    try:
        features = calculate_project_extra_features(daily)
    except Exception:
        return {
            "stock_price": _rounded(daily["close"].iloc[-1]),
            "stock_price_date": _display_date(daily["date"].iloc[-1]),
        }
    last = daily.iloc[-1]
    last_features = features.iloc[-1] if not features.empty else pd.Series(dtype=float)
    return {
        "stock_price": _rounded(last.get("close")),
        "stock_price_date": _display_date(last.get("date")),
        "kdj_daily_j": _rounded(last_features.get("kdj_d_j")),
        "kdj_weekly_j": _rounded(last_features.get("kdj_w_j")),
        "kdj_monthly_j": _rounded(last_features.get("kdj_m_j")),
    }


def _attach_stock_market_snapshots(records: list[dict[str, Any]], daily_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    daily_dir = daily_dir or STOCK_DAILY_DIR
    stock_codes = {
        stock_code
        for item in records
        if (stock_code := _clean_text(item.get("stock_code"))) is not None
    }
    canonical_codes = {
        stock_code
        for stock_code in stock_codes
        if _legacy_stock_daily_path(stock_code, daily_dir) is None
    }
    canonical_daily = pd.DataFrame()
    storage_backend = "legacy"
    storage_error = None
    if canonical_codes:
        candidates = sorted(
            {
                candidate
                for stock_code in canonical_codes
                for candidate in _stock_code_candidates(stock_code)
            }
        )
        store = MarketDataStore(MarketDataStoreConfig.from_env(root=daily_dir.parent))
        storage_backend = store.config.backend
        try:
            canonical_daily = store.read_market_range(daily_dir.name, symbols=candidates)
        except Exception as exc:
            storage_error = str(exc)
    cache: dict[str, dict[str, Any]] = {}
    hits = 0
    for item in records:
        stock_code = _clean_text(item.get("stock_code"))
        if not stock_code:
            continue
        if stock_code not in cache:
            daily = _load_stock_daily(
                stock_code,
                daily_dir=daily_dir,
                canonical_daily=canonical_daily,
            )
            prepared = _prepare_stock_daily(daily)
            if prepared.empty:
                cache[stock_code] = {}
            else:
                cache[stock_code] = _market_snapshot_from_daily(prepared)
        snapshot = cache[stock_code]
        if snapshot:
            hits += 1
            item.update(snapshot)
    return records, {
        "source": str(daily_dir),
        "storage_backend": storage_backend,
        "error": storage_error,
        "requested": len(stock_codes),
        "matched": hits,
    }


def _latest_daily_basic_frame(daily_basic_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily_basic_dir = daily_basic_dir or DAILY_BASIC_DIR
    meta: dict[str, Any] = {"source": str(daily_basic_dir), "available": daily_basic_dir.exists(), "file": None, "error": None}
    if not daily_basic_dir.exists():
        return pd.DataFrame(), meta
    files = sorted(daily_basic_dir.glob("*.parquet"))
    if not files:
        meta["available"] = False
        return pd.DataFrame(), meta
    latest = files[-1]
    meta["file"] = str(latest)
    try:
        frame = pd.read_parquet(latest)
        meta.update({"available": not frame.empty, "rows": int(len(frame))})
        return frame, meta
    except Exception as exc:
        meta.update({"available": False, "error": str(exc)})
        return pd.DataFrame(), meta


def _attach_share_capital(records: list[dict[str, Any]], daily_basic_dir: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frame, meta = _latest_daily_basic_frame(daily_basic_dir=daily_basic_dir)
    if frame.empty or "ts_code" not in frame.columns or "total_share" not in frame.columns:
        meta["matched"] = 0
        return records, meta
    lookup = {
        str(row.get("ts_code")): row
        for _, row in frame.iterrows()
        if _clean_text(row.get("ts_code")) and _number(row.get("total_share")) is not None
    }
    hits = 0
    for item in records:
        row = None
        for candidate in _stock_code_candidates(item.get("stock_code")):
            if candidate in lookup:
                row = lookup[candidate]
                break
        if row is None:
            continue
        total_share = _number(row.get("total_share"))
        float_share = _number(row.get("float_share"))
        if total_share is None or total_share <= 0:
            continue
        hits += 1
        item["total_share"] = _rounded(total_share * 10000, 0)
        item["float_share"] = _rounded(float_share * 10000, 0) if float_share is not None else None
        item["share_capital_date"] = _display_date(row.get("trade_date"))
    meta["matched"] = hits
    return records, meta


def _amount_to_yuan(value: Any, unit: str | None) -> float | None:
    number = _number(str(value).replace(",", "").replace("，", ""))
    if number is None:
        return None
    unit_text = unit or "元"
    if "亿" in unit_text:
        return number * 100_000_000
    if "万" in unit_text:
        return number * 10000
    return number


def _extract_issue_size_yuan_from_text(text: str) -> float | None:
    compact = re.sub(r"\s+", "", text or "")
    patterns = [
        r"(?:募集资金总额|募集资金总量|发行总额|发行规模|可转债总额|可转换公司债券总额)[^。；]{0,80}?(?:不超过|不超|不多于|为)?(?:人民币)?([\d,，.]+)(亿元|万元|元)",
        r"(?:不超过|不超|不多于)(?:人民币)?([\d,，.]+)(亿元|万元|元)[^。；]{0,80}?(?:可转换公司债券|可转债|募集资金)",
    ]
    candidates: list[float] = []
    for pattern in patterns:
        for match in re.finditer(pattern, compact):
            amount = _amount_to_yuan(match.group(1), match.group(2))
            if amount and amount >= 10_000_000:
                candidates.append(amount)
    return max(candidates) if candidates else None


def _extract_issue_size_yuan_from_pdf_bytes(content: bytes) -> float | None:
    use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
    previous_handler = None
    if use_alarm:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _raise_pdf_extract_timeout)
        signal.alarm(6)
    try:
        try:
            _ensure_optional_pdf_import_path()
            import pypdf
        except Exception:
            return None
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
        except Exception:
            return None
        texts = []
        for page in reader.pages[:40]:
            try:
                texts.append(page.extract_text() or "")
            except _PdfExtractTimeout:
                return None
            except Exception:
                continue
            amount = _extract_issue_size_yuan_from_text("\n".join(texts))
            if amount:
                return amount
        return _extract_issue_size_yuan_from_text("\n".join(texts))
    except _PdfExtractTimeout:
        return None
    finally:
        if use_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)


def _cninfo_pdf_url(adjunct_url: Any) -> str | None:
    text = _clean_text(adjunct_url)
    if not text:
        return None
    if text.startswith("http://"):
        return "https://" + text.removeprefix("http://")
    if text.startswith("https://"):
        return text
    return f"https://static.cninfo.com.cn/{text.lstrip('/')}"


def _query_cninfo_issue_documents(record: dict[str, Any], today: date | None = None) -> list[dict[str, Any]]:
    stock_name = _clean_text(record.get("stock_name"))
    stock_code = _clean_text(record.get("stock_code"))
    if not stock_name and not stock_code:
        return []
    try:
        import requests
    except Exception:
        return []
    end = today or date.today()
    start = end - timedelta(days=1200)
    search_terms = [
        f"{stock_name or stock_code} 向不特定对象发行可转换公司债券 募集说明书",
        f"{stock_name or stock_code} 向不特定对象发行可转债 预案",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in search_terms:
        payload = {
            "pageNum": "1",
            "pageSize": "20",
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": term,
            "secid": "",
            "category": "category_kzzq_szsh",
            "trade": "",
            "seDate": f"{start:%Y-%m-%d}~{end:%Y-%m-%d}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        try:
            response = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=payload, timeout=8)
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue
        for item in data.get("announcements") or []:
            announcement_id = _clean_text(item.get("announcementId"))
            if not announcement_id or announcement_id in seen:
                continue
            title = _clean_text(item.get("announcementTitle")) or ""
            if not _is_pipeline_issuance_title(title):
                continue
            if not any(key in title for key in ["预案", "募集说明书", "可行性分析", "申报稿"]):
                continue
            seen.add(announcement_id)
            rows.append(
                {
                    "announcement_id": announcement_id,
                    "announcement_title": title,
                    "announcement_time": item.get("announcementTime"),
                    "pdf_url": _cninfo_pdf_url(item.get("adjunctUrl")),
                }
            )
    return rows


def _query_cninfo_issuing_documents(record: dict[str, Any], today: date | None = None) -> list[dict[str, Any]]:
    stock_name = _clean_text(record.get("stock_name"))
    stock_code = _clean_text(record.get("stock_code"))
    if not stock_name and not stock_code:
        return []
    try:
        import requests
    except Exception:
        return []
    end = today or date.today()
    start = end - timedelta(days=90)
    search_terms = [
        f"{stock_name or stock_code} 向不特定对象发行可转换公司债券 发行公告",
        f"{stock_name or stock_code} 向不特定对象发行可转换公司债券 发行提示性公告",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in search_terms:
        payload = {
            "pageNum": "1",
            "pageSize": "20",
            "column": "szse",
            "tabName": "fulltext",
            "plate": "",
            "stock": "",
            "searchkey": term,
            "secid": "",
            "category": "category_kzzq_szsh",
            "trade": "",
            "seDate": f"{start:%Y-%m-%d}~{end:%Y-%m-%d}",
            "sortName": "",
            "sortType": "",
            "isHLtitle": "true",
        }
        try:
            response = requests.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=payload, timeout=8)
            response.raise_for_status()
            data = response.json()
        except Exception:
            continue
        for item in data.get("announcements") or []:
            announcement_id = _clean_text(item.get("announcementId"))
            if not announcement_id or announcement_id in seen:
                continue
            title = _clean_text(item.get("announcementTitle")) or ""
            if "上市公告" in title or "网上路演" in title:
                continue
            if "发行公告" not in title and "发行提示性公告" not in title:
                continue
            seen.add(announcement_id)
            rows.append(
                {
                    "announcement_id": announcement_id,
                    "announcement_title": title,
                    "announcement_time": item.get("announcementTime"),
                    "pdf_url": _cninfo_pdf_url(item.get("adjunctUrl")),
                }
            )
    return rows


def _extract_near_date(text: str, label_pattern: str, window: int = 80) -> str | None:
    match = re.search(label_pattern, text)
    if not match:
        return None
    snippet = text[match.end() : match.end() + window]
    date_match = re.search(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{8}", snippet)
    return _display_date(date_match.group(0)) if date_match else None


def _extract_issue_dates_from_text(text: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", " ", text or "")
    record_date = _extract_near_date(compact, r"股权登记日")
    pay_date = _extract_near_date(compact, r"(?:原股东缴款日|认购及缴款日为|缴款日)")
    issue_date = _extract_near_date(compact, r"(?:发行日期及时间|网上申购日|申购日)")
    allot_code_match = re.search(r"(?:原股东配售代码|配售代码)[^\d]{0,20}(\d{6})", compact)
    allot_name_match = re.search(r"(?:原股东配售简称|配售简称)\s*([\u4e00-\u9fa5A-Za-z0-9]+)", compact)
    bond_code_match = re.search(r"可转债代码[^\d]{0,20}(\d{6})", compact)
    bond_name_match = re.search(r"可转债简称\s*([\u4e00-\u9fa5A-Za-z0-9]+)", compact)
    return {
        "record_date": record_date,
        "pay_date": pay_date,
        "issue_date": issue_date,
        "allot_code": allot_code_match.group(1) if allot_code_match else None,
        "allot_name": allot_name_match.group(1) if allot_name_match else None,
        "bond_code": bond_code_match.group(1) if bond_code_match else None,
        "bond_name": bond_name_match.group(1) if bond_name_match else None,
    }


def _extract_issue_dates_from_pdf_bytes(content: bytes) -> dict[str, Any] | None:
    try:
        _ensure_optional_pdf_import_path()
        import pypdf
    except Exception:
        return None
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
    except Exception:
        return None
    texts = []
    for page in reader.pages[:8]:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            continue
    parsed = _extract_issue_dates_from_text("\n".join(texts))
    if any(parsed.get(key) for key in ["record_date", "pay_date", "issue_date"]):
        return parsed
    return None


def _load_issue_size_cache() -> pd.DataFrame:
    return _read_any_frame(CB_PIPELINE_ISSUE_SIZE_PATH)


def _save_issue_size_cache(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    CB_PIPELINE_ISSUE_SIZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.drop_duplicates("stock_code", keep="last").to_parquet(CB_PIPELINE_ISSUE_SIZE_PATH, index=False)


def _refresh_issue_size_for_record(record: dict[str, Any], today: date | None = None) -> dict[str, Any] | None:
    try:
        import requests
    except Exception:
        return None
    for doc in _query_cninfo_issue_documents(record, today=today):
        pdf_url = doc.get("pdf_url")
        if not pdf_url:
            continue
        try:
            response = requests.get(pdf_url, timeout=8)
            response.raise_for_status()
        except Exception:
            continue
        if len(response.content) > 8 * 1024 * 1024:
            continue
        amount = _extract_issue_size_yuan_from_pdf_bytes(response.content)
        if not amount:
            continue
        return {
            "stock_code": _clean_text(record.get("stock_code")),
            "stock_name": _clean_text(record.get("stock_name")),
            "issue_size": amount,
            "issue_size_yuan": amount,
            "issue_size_source": "cninfo_pdf",
            "issue_size_title": doc.get("announcement_title"),
            "issue_size_url": pdf_url,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    return None


def _attach_pipeline_issue_sizes(
    records: list[dict[str, Any]],
    refresh: bool = False,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = _load_issue_size_cache()
    lookup: dict[str, dict[str, Any]] = {}
    if not cache.empty and "stock_code" in cache.columns:
        lookup = {
            str(row.get("stock_code")): row.to_dict()
            for _, row in cache.dropna(subset=["stock_code"]).iterrows()
        }
    refreshed: list[dict[str, Any]] = []
    if refresh:
        for item in records:
            code = _clean_text(item.get("stock_code"))
            if _normalize_pipeline_stage(item.get("stage")) not in PIPELINE_STAGES:
                continue
            if not code or _number(item.get("issue_size")) is not None or code in lookup:
                continue
            result = _refresh_issue_size_for_record(item, today=today)
            if result:
                lookup[code] = result
                refreshed.append(result)
                _save_issue_size_cache(pd.DataFrame([*lookup.values()]))
    hits = 0
    for item in records:
        code = _clean_text(item.get("stock_code"))
        cached = lookup.get(str(code)) if code else None
        if not cached:
            continue
        issue_size = _number(cached.get("issue_size_yuan")) or _number(cached.get("issue_size"))
        if issue_size is None:
            continue
        hits += 1
        item["issue_size"] = issue_size
        item["issue_size_yuan"] = issue_size
        item["issue_size_source"] = cached.get("issue_size_source") or "issue_size_cache"
        item["issue_size_title"] = cached.get("issue_size_title")
        item["issue_size_url"] = cached.get("issue_size_url")
    return records, {
        "source": str(CB_PIPELINE_ISSUE_SIZE_PATH),
        "available": not cache.empty or bool(refreshed),
        "rows": int(len(lookup)),
        "matched": hits,
        "refreshed": len(refreshed),
    }


def _load_issue_date_cache() -> pd.DataFrame:
    return _read_any_frame(CB_PIPELINE_ISSUE_DATE_PATH)


def _save_issue_date_cache(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    CB_PIPELINE_ISSUE_DATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.drop_duplicates("stock_code", keep="last").to_parquet(CB_PIPELINE_ISSUE_DATE_PATH, index=False)


def _refresh_issue_dates_for_record(record: dict[str, Any], today: date | None = None) -> dict[str, Any] | None:
    try:
        import requests
    except Exception:
        return None
    for doc in _query_cninfo_issuing_documents(record, today=today):
        pdf_url = doc.get("pdf_url")
        if not pdf_url:
            continue
        try:
            response = requests.get(pdf_url, timeout=8)
            response.raise_for_status()
        except Exception:
            continue
        if len(response.content) > 8 * 1024 * 1024:
            continue
        parsed = _extract_issue_dates_from_pdf_bytes(response.content)
        if not parsed:
            continue
        return {
            "stock_code": _clean_text(record.get("stock_code")),
            "stock_name": _clean_text(record.get("stock_name")),
            **parsed,
            "issue_date_source": "cninfo_issuing_pdf",
            "issue_date_title": doc.get("announcement_title"),
            "issue_date_url": pdf_url,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    return None


def _attach_pipeline_issue_dates(
    records: list[dict[str, Any]],
    refresh: bool = False,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = _load_issue_date_cache()
    lookup: dict[str, dict[str, Any]] = {}
    if not cache.empty and "stock_code" in cache.columns:
        lookup = {
            str(row.get("stock_code")): row.to_dict()
            for _, row in cache.dropna(subset=["stock_code"]).iterrows()
        }
    refreshed: list[dict[str, Any]] = []
    if refresh:
        for item in records:
            code = _clean_text(item.get("stock_code"))
            if item.get("stage") != "issuing" or not code or code in lookup:
                continue
            if all(_clean_text(item.get(field)) is not None for field in ["record_date", "pay_date", "issue_date"]):
                continue
            result = _refresh_issue_dates_for_record(item, today=today)
            if result:
                lookup[code] = result
                refreshed.append(result)
                _save_issue_date_cache(pd.DataFrame([*lookup.values()]))
    hits = 0
    for item in records:
        code = _clean_text(item.get("stock_code"))
        cached = lookup.get(str(code)) if code else None
        if not cached:
            continue
        applied = False
        for field in ["record_date", "pay_date", "issue_date", "allot_code", "allot_name", "bond_code", "bond_name"]:
            value = cached.get(field)
            if _clean_text(item.get(field)) is None and _clean_text(value) is not None:
                item[field] = _display_date(value) if field.endswith("_date") else value
                applied = True
        if applied:
            hits += 1
            item["issue_date_source"] = cached.get("issue_date_source") or "issue_date_cache"
            item["issue_date_title"] = cached.get("issue_date_title")
            item["issue_date_url"] = cached.get("issue_date_url")
    return records, {
        "source": str(CB_PIPELINE_ISSUE_DATE_PATH),
        "available": not cache.empty or bool(refreshed),
        "rows": int(len(lookup)),
        "matched": hits,
        "refreshed": len(refreshed),
    }


def _issue_size_yuan(item: dict[str, Any]) -> float | None:
    for key in ["plan_issue_size", "issue_size", "remain_size"]:
        value = _number(item.get(key))
        if value is None or value <= 0:
            continue
        if value >= 100_000_000:
            return value
        if value >= 10_000:
            return value * 10000
        return value * 100_000_000
    return None


def _attach_allotment_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for item in records:
        rights_per_share = _number(item.get("rights_per_share")) or _number(item.get("allot_ratio"))
        metric_source = "allot_ratio" if rights_per_share is not None else None
        issue_size = _issue_size_yuan(item)
        total_share = _number(item.get("total_share"))
        if rights_per_share is None and issue_size and total_share and total_share > 0:
            rights_per_share = issue_size / total_share
            metric_source = "issue_size_total_share"
        shares_for_10 = _number(item.get("shares_for_10_bonds"))
        if rights_per_share is None and shares_for_10 and shares_for_10 > 0:
            rights_per_share = 1000 / shares_for_10
            metric_source = "watchlist"
        if shares_for_10 is None and rights_per_share and rights_per_share > 0:
            shares_for_10 = math.ceil(1000 / rights_per_share)
        stock_price = _number(item.get("stock_price"))
        rights_value_pct = None
        if rights_per_share is not None and stock_price and stock_price > 0:
            rights_value_pct = rights_per_share / stock_price * 100
        item["issue_size_yuan"] = _rounded(issue_size, 0)
        item["rights_per_share"] = _rounded(rights_per_share, 4)
        item["shares_for_one_lot"] = int(shares_for_10) if shares_for_10 is not None and shares_for_10 > 0 else None
        item["shares_for_10_bonds"] = item["shares_for_one_lot"]
        item["shares_for_one_lot_source"] = metric_source
        item["rights_value_pct"] = _rounded(rights_value_pct)
    return records


def _load_basic(refresh: bool = False, fetcher: TushareDataFetcher | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "source": str(CB_BASIC_PATH),
        "refreshed": False,
        "poll_status": "not_requested",
        "error": None,
    }
    if refresh or not CB_BASIC_PATH.exists():
        fetcher = fetcher or TushareDataFetcher(cache_dir=CB_DATA_DIR / "tushare_cache")
        try:
            basic = fetcher.get_cb_basic(list_status="all")
            basic = basic if isinstance(basic, pd.DataFrame) else pd.DataFrame()
            meta["poll_status"] = "success"
            if not basic.empty:
                CB_BASIC_PATH.parent.mkdir(parents=True, exist_ok=True)
                basic.to_parquet(CB_BASIC_PATH, index=False)
                meta["refreshed"] = True
                return basic, meta
        except Exception as exc:
            meta["poll_status"] = "failed"
            meta["error"] = str(exc)
    return _read_frame(CB_BASIC_PATH), meta


def _load_issue(refresh: bool = False, fetcher: TushareDataFetcher | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "source": str(CB_ISSUE_PATH),
        "refreshed": False,
        "poll_status": "not_requested",
        "available": CB_ISSUE_PATH.exists(),
        "error": None,
    }
    if refresh:
        fetcher = fetcher or TushareDataFetcher(cache_dir=CB_DATA_DIR / "tushare_cache")
        try:
            issue = fetcher.get_cb_issue()
            issue = issue if isinstance(issue, pd.DataFrame) else pd.DataFrame()
            meta["poll_status"] = "success"
            if not issue.empty:
                CB_ISSUE_PATH.parent.mkdir(parents=True, exist_ok=True)
                issue.to_parquet(CB_ISSUE_PATH, index=False)
                meta.update({"refreshed": True, "available": True})
                return issue, meta
        except Exception as exc:
            meta["poll_status"] = "failed"
            meta["error"] = str(exc)
    frame = _read_frame(CB_ISSUE_PATH)
    meta["available"] = not frame.empty
    return frame, meta


def _load_pipeline_candidates(refresh: bool = False, today: date | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "source": str(CB_PIPELINE_PATH),
        "refreshed": False,
        "poll_status": "not_requested",
        "available": CB_PIPELINE_PATH.exists(),
        "error": None,
    }
    if refresh:
        try:
            import akshare as ak

            end = today or date.today()
            start = end - timedelta(days=900)
            frames = []
            for keyword in ["可转债", "可转换公司债券"]:
                frame = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol="",
                    market="沪深京",
                    keyword=keyword,
                    category="可转债",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                if frame is not None and not frame.empty:
                    frames.append(frame)
            announcements = pd.concat(frames, ignore_index=True).drop_duplicates() if frames else pd.DataFrame()
            pipeline = _pipeline_from_announcements(announcements)
            meta["poll_status"] = "success"
            if not pipeline.empty:
                CB_PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
                pipeline.to_parquet(CB_PIPELINE_PATH, index=False)
                meta.update({"refreshed": True, "available": True, "raw_rows": int(len(announcements)), "keywords": ["可转债", "可转换公司债券"]})
                return pipeline, meta
            meta.update({"refreshed": True, "available": False, "raw_rows": int(len(announcements)), "keywords": ["可转债", "可转换公司债券"]})
        except Exception as exc:
            meta["poll_status"] = "failed"
            meta["error"] = str(exc)
    frame = _read_any_frame(CB_PIPELINE_PATH)
    meta["available"] = not frame.empty
    return frame, meta


def _load_cninfo_issue(refresh: bool = False, today: date | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "source": str(CB_CNINFO_ISSUE_PATH),
        "refreshed": False,
        "poll_status": "not_requested",
        "available": CB_CNINFO_ISSUE_PATH.exists(),
        "error": None,
    }
    if refresh:
        try:
            import akshare as ak

            end = today or date.today()
            start = end - timedelta(days=900)
            issue = ak.bond_cov_issue_cninfo(
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            issue = issue if isinstance(issue, pd.DataFrame) else pd.DataFrame()
            meta["poll_status"] = "success"
            if not issue.empty:
                CB_CNINFO_ISSUE_PATH.parent.mkdir(parents=True, exist_ok=True)
                issue.to_parquet(CB_CNINFO_ISSUE_PATH, index=False)
                meta.update({"refreshed": True, "available": True})
                return issue, meta
            meta.update({"refreshed": True, "available": False})
        except Exception as exc:
            meta["poll_status"] = "failed"
            meta["error"] = str(exc)
    frame = _read_any_frame(CB_CNINFO_ISSUE_PATH)
    meta["available"] = not frame.empty
    return frame, meta


def _load_watchlist() -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {
        "source": str(CB_WATCHLIST_PATH),
        "available": CB_WATCHLIST_PATH.exists(),
        "error": None,
    }
    try:
        frame = _read_any_frame(CB_WATCHLIST_PATH)
        meta["available"] = not frame.empty
        return frame, meta
    except Exception as exc:
        meta["error"] = str(exc)
        return pd.DataFrame(), meta


def _classify_stage(row: pd.Series, today_text: str) -> tuple[str, str]:
    list_date = _date_text(row.get("list_date"))
    delist_date = _date_text(row.get("delist_date"))
    pay_date = _date_text(_first(row, ["shd_ration_pay_date", "pay_date", "payment_date", "allot_pay_date"]))
    record_date = _date_text(_first(row, ["shd_ration_record_date", "record_date", "reg_date", "allot_record_date"]))
    issue_date = _date_text(_first(row, ["onl_date", "issue_date", "sub_date", "purchase_date"]))
    ann_date = _date_text(_first(row, ["ann_date", "res_ann_date"]))

    if delist_date and delist_date <= today_text:
        return "delisted", "已退市"
    if list_date and list_date > today_text:
        return "pending_listing", "待上市"
    if pay_date and pay_date >= today_text:
        return "payment", "缴款核对"
    if record_date and record_date >= today_text:
        return "record", "等待股权登记"
    if issue_date and issue_date >= today_text:
        return "issue", "发行申购"
    if list_date and list_date <= today_text:
        return "listed", "已上市"
    if ann_date:
        return "announced", "已公告"
    return "watching", "待公告"


def _stage_from_title(title: Any) -> tuple[str, str]:
    text = _clean_text(title) or ""
    if any(key in text for key in ["终止", "停止", "撤回", "失效"]):
        return "terminated", "已终止"
    if any(key in text for key in ["上市公告", "上市的公告"]):
        return "listed", "已上市"
    if any(key in text for key in ["发行公告", "发行提示", "网上路演", "申购", "募集说明书提示性公告"]):
        return "issuing", PIPELINE_STAGE_STATUS["issuing"]
    if any(key in text for key in ["申报稿", "文件更新", "募集说明书等申请文件"]):
        return "accepted", PIPELINE_STAGE_STATUS["accepted"]
    if any(key in text for key in ["同意注册", "注册批复", "予以注册"]):
        return "registered", PIPELINE_STAGE_STATUS["registered"]
    if any(key in text for key in ["审核通过", "上市委", "审议通过"]):
        return "exchange_approved", PIPELINE_STAGE_STATUS["exchange_approved"]
    if any(key in text for key in ["问询", "审核问询", "落实函", "回复"]):
        return "accepted", PIPELINE_STAGE_STATUS["accepted"]
    if any(key in text for key in ["受理", "获受理"]):
        return "accepted", PIPELINE_STAGE_STATUS["accepted"]
    if "股东大会" in text:
        return "shareholder_approved", PIPELINE_STAGE_STATUS["shareholder_approved"]
    if any(key in text for key in ["预案", "董事会", "方案", "可行性分析", "论证分析"]):
        return "board_plan", PIPELINE_STAGE_STATUS["board_plan"]
    return "announced", "已公告"


def _normalize_pipeline_stage(stage: Any) -> str | None:
    text = _clean_text(stage)
    if text == "inquiry":
        return "accepted"
    if text in {"issue", "record", "payment", "pending_listing"}:
        return "issuing"
    return text


def _pipeline_from_announcements(announcements: pd.DataFrame) -> pd.DataFrame:
    if announcements is None or announcements.empty:
        return pd.DataFrame()
    frame = announcements.copy()
    rename = {
        "代码": "stock_code",
        "简称": "stock_name",
        "公告标题": "announcement_title",
        "公告时间": "announce_date",
        "公告链接": "announcement_url",
    }
    frame = frame.rename(columns=rename)
    required = ["stock_code", "stock_name", "announcement_title", "announce_date", "announcement_url"]
    for column in required:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[frame["announcement_title"].map(_is_pipeline_issuance_title)].copy()
    if frame.empty:
        return pd.DataFrame(columns=required + ["stage", "status"])
    frame["stage"], frame["status"] = zip(*frame["announcement_title"].map(_stage_from_title))
    frame = frame[~frame["stage"].isin({"listed", "terminated"})].copy()
    frame["announce_date"] = pd.to_datetime(frame["announce_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.sort_values(["stock_code", "announce_date"], ascending=[True, False])
    frame = frame.drop_duplicates("stock_code", keep="first")
    return frame[required + ["stage", "status"]].copy()


def _cninfo_issue_records(issue: pd.DataFrame, today_text: str) -> list[dict[str, Any]]:
    if issue is None or issue.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in issue.iterrows():
        record_date = _cninfo_shareholder_record_date(row)
        list_stage, status = _stage_from_issue_dates(row, today_text)
        if list_stage not in PIPELINE_STAGES:
            continue
        records.append(
            {
                "status": status,
                "stage": list_stage,
                "bond_code": _clean_text(row.get("债券代码")),
                "bond_name": _clean_text(_first(row, ["债券简称", "债券名称"])),
                "stock_code": _clean_text(row.get("转股代码")),
                "stock_name": None,
                "allot_code": _clean_text(row.get("网上申购代码")),
                "allot_name": _clean_text(row.get("网上申购简称")),
                "record_date": _display_date(record_date),
                "pay_date": _display_date(row.get("优先申购缴款日")),
                "issue_date": _display_date(_first(row, ["网上申购日期", "发行起始日"])),
                "announce_date": _display_date(row.get("公告日期")),
                "list_date": None,
                "delist_date": None,
                "convert_start_date": _display_date(row.get("转股开始日期")),
                "convert_end_date": _display_date(row.get("转股终止日期")),
                "issue_size": _number(row.get("计划发行总量")),
                "remain_size": _number(row.get("实际发行总量")),
                "allot_ratio": None,
                "rating": None,
                "announcement_title": _clean_text(row.get("债券名称")),
                "announcement_url": None,
                "allotment_note": _issue_allotment_note(row),
                "risk_note": _clean_text(row.get("募资用途说明")) or "发行阶段；关注申购日、股权登记日和正股波动",
            }
        )
    return records


def _stage_from_issue_dates(row: pd.Series, today_text: str) -> tuple[str, str]:
    return "issuing", PIPELINE_STAGE_STATUS["issuing"]


def _cninfo_shareholder_record_date(row: pd.Series) -> str | None:
    explicit = _clean_text(
        _first(
            row,
            [
                "原股东股权登记日",
                "股权登记日",
                "股东登记日",
                "优先配售股权登记日",
                "配售股权登记日",
            ],
        )
    )
    if explicit:
        return explicit
    target_text = " ".join(
        text
        for text in [
            _clean_text(row.get("发行对象")),
            _clean_text(row.get("发行范围")),
        ]
        if text
    )
    if not target_text:
        return None
    match = re.search(r"股权登记日[^\d]*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{8})", target_text)
    return match.group(1) if match else None


def _pipeline_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        announcement_title = row.get("announcement_title")
        if not _is_pipeline_issuance_title(announcement_title):
            continue
        inferred_stage, inferred_status = _stage_from_title(announcement_title)
        stage = _normalize_pipeline_stage(inferred_stage or row.get("stage"))
        status = PIPELINE_STAGE_STATUS.get(stage or "", inferred_status or _clean_text(row.get("status")))
        if stage not in PIPELINE_STAGES:
            continue
        records.append(
            {
                "status": status,
                "stage": stage,
                "bond_code": _clean_text(row.get("bond_code")),
                "bond_name": _clean_text(row.get("bond_name")),
                "stock_code": _clean_text(row.get("stock_code")),
                "stock_name": _clean_text(row.get("stock_name")),
                "allot_code": _clean_text(row.get("allot_code")),
                "allot_name": _clean_text(row.get("allot_name")),
                "record_date": _display_date(row.get("record_date")),
                "pay_date": _display_date(row.get("pay_date")),
                "issue_date": _display_date(row.get("issue_date")),
                "announce_date": _display_date(row.get("announce_date")),
                "list_date": _display_date(row.get("list_date")),
                "delist_date": None,
                "convert_start_date": _display_date(row.get("convert_start_date")),
                "convert_end_date": _display_date(row.get("convert_end_date")),
                "issue_size": _number(row.get("issue_size")),
                "remain_size": None,
                "allot_ratio": _number(row.get("allot_ratio")),
                "rating": _clean_text(row.get("rating")),
                "announcement_title": _clean_text(announcement_title),
                "announcement_url": _clean_text(row.get("announcement_url")),
                "allotment_note": _clean_text(row.get("allotment_note")) or _pipeline_note(stage),
                "risk_note": _clean_text(row.get("risk_note")) or "前置阶段；重点跟踪下一份受理/上市委/注册/发行公告",
            }
        )
    return records


def _watchlist_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        stock_code = _clean_text(row.get("stock_code"))
        stock_name = _clean_text(row.get("stock_name"))
        stage = _normalize_pipeline_stage(row.get("stage")) or "registered"
        if not stock_code or stage not in PIPELINE_STAGES:
            continue
        note = _clean_text(row.get("allotment_note"))
        records.append(
            {
                "status": PIPELINE_STAGE_STATUS.get(stage) or _clean_text(row.get("status")) or "观察池",
                "stage": stage,
                "bond_code": _clean_text(row.get("bond_code")),
                "bond_name": _clean_text(row.get("bond_name")),
                "stock_code": stock_code,
                "stock_name": stock_name,
                "allot_code": _clean_text(row.get("allot_code")),
                "allot_name": _clean_text(row.get("allot_name")),
                "record_date": _display_date(row.get("record_date")),
                "pay_date": _display_date(row.get("pay_date")),
                "issue_date": _display_date(row.get("issue_date")),
                "announce_date": _display_date(row.get("announce_date")),
                "list_date": _display_date(row.get("list_date")),
                "delist_date": None,
                "convert_start_date": _display_date(row.get("convert_start_date")),
                "convert_end_date": _display_date(row.get("convert_end_date")),
                "issue_size": _number(row.get("issue_size")),
                "remain_size": None,
                "allot_ratio": _number(row.get("rights_per_share")),
                "manual_shares_for_10_bonds": _number(row.get("shares_for_10_bonds")),
                "one_lot_party": _bool_value(row.get("one_lot_party")),
                "rating": _clean_text(row.get("rating")),
                "announcement_title": _clean_text(row.get("announcement_title")) or f"{stock_name or stock_code}可转债排队观察",
                "announcement_url": _clean_text(row.get("announcement_url")),
                "allotment_note": "；".join(part for part in [note, "手工观察池补充"] if part),
                "risk_note": _clean_text(row.get("risk_note")) or "排队配债股；等待下一份上市委/注册/发行公告校准配售比例",
                "sort_order": _number(row.get("sort_order")),
                "data_source": "watchlist",
            }
        )
    return records


def _merge_basic_issue(basic: pd.DataFrame, issue: pd.DataFrame) -> pd.DataFrame:
    if basic.empty:
        return pd.DataFrame()
    basic = basic.copy()
    basic["ts_code"] = basic["ts_code"].astype(str)
    if issue.empty or "ts_code" not in issue.columns:
        return basic
    issue = issue.copy()
    issue["ts_code"] = issue["ts_code"].astype(str)
    issue = issue.drop_duplicates("ts_code", keep="last")
    return basic.merge(issue, on="ts_code", how="left", suffixes=("", "_issue"))


def _build_record(row: pd.Series, today_text: str) -> dict[str, Any]:
    stage, status = _classify_stage(row, today_text)
    allot_code = _clean_text(_first(row, ["shd_ration_code", "ration_code", "allot_code", "placing_code"]))
    allot_name = _clean_text(_first(row, ["shd_ration_name", "ration_name", "allot_name", "placing_name"]))
    ratio = _number(_first(row, ["shd_ration_ratio", "ration_ratio", "allot_ratio", "priority_ratio"]))
    issue_size = _number(_first(row, ["plan_issue_size", "issue_size", "remain_size"]))
    return {
        "status": status,
        "stage": stage,
        "bond_code": _clean_text(row.get("ts_code")),
        "bond_name": _clean_text(_first(row, ["bond_short_name", "bond_full_name"])),
        "stock_code": _clean_text(_first(row, ["stk_code", "stock_code"])),
        "stock_name": _clean_text(_first(row, ["stk_short_name", "stock_name"])),
        "allot_code": allot_code,
        "allot_name": allot_name,
        "record_date": _display_date(_first(row, ["shd_ration_record_date", "record_date", "reg_date", "allot_record_date"])),
        "pay_date": _display_date(_first(row, ["shd_ration_pay_date", "pay_date", "payment_date", "allot_pay_date"])),
        "issue_date": _display_date(_first(row, ["onl_date", "issue_date", "sub_date", "purchase_date"])),
        "announce_date": _display_date(_first(row, ["ann_date", "res_ann_date"])),
        "list_date": _display_date(row.get("list_date")),
        "delist_date": _display_date(row.get("delist_date")),
        "convert_start_date": _display_date(row.get("conv_start_date")),
        "convert_end_date": _display_date(row.get("conv_end_date")),
        "issue_size": issue_size,
        "remain_size": _number(row.get("remain_size")),
        "allot_ratio": ratio,
        "rating": _clean_text(_first(row, ["newest_rating", "issue_rating", "rate"])),
        "allotment_note": _allotment_note(allot_code, ratio),
        "risk_note": _risk_note(row),
    }


def _allotment_note(allot_code: str | None, ratio: float | None) -> str:
    parts = []
    if allot_code:
        parts.append(f"配债代码 {allot_code}")
    else:
        parts.append("等待发行公告确认配债代码")
    if ratio is not None:
        parts.append(f"原股东配售比例 {ratio:g}")
    else:
        parts.append("配售比例待补齐")
    return "；".join(parts)


def _pipeline_note(stage: str) -> str:
    notes = {
        "board_plan": "董事会预案：跟踪股东大会审议",
        "shareholder_approved": "股东大会通过：等待交易所受理",
        "accepted": "交易所受理：跟踪问询、回复和上市委会议",
        "exchange_approved": "上市委通过：等待证监会同意注册",
        "registered": "同意注册：等待发行公告和配债代码",
        "issuing": "发行公告：关注股权登记日、优先配售和缴款",
    }
    return notes.get(stage, "前置阶段：等待下一份可转债进度公告")


def _issue_allotment_note(row: pd.Series) -> str:
    pieces = []
    if _clean_text(row.get("网上申购代码")):
        pieces.append(f"申购代码 {_clean_text(row.get('网上申购代码'))}")
    record_date = _cninfo_shareholder_record_date(row)
    if record_date:
        pieces.append(f"股权登记日 {_display_date(record_date)}")
    if _clean_text(row.get("优先申购缴款日")):
        pieces.append(f"优先缴款日 {_display_date(row.get('优先申购缴款日'))}")
    if _number(row.get("配售价格")) is not None:
        pieces.append(f"配售价格 {_number(row.get('配售价格')):g}")
    return "；".join(pieces) or "发行阶段：等待配售和申购细节"


def _risk_note(row: pd.Series) -> str:
    notes = []
    if _clean_text(row.get("delist_date")):
        notes.append("已退市或已结束交易")
    if _clean_text(row.get("call_clause")):
        notes.append("关注强赎条款")
    if _clean_text(row.get("reset_clause")):
        notes.append("关注下修条款")
    rating = _clean_text(_first(row, ["newest_rating", "issue_rating"]))
    if rating:
        notes.append(f"评级 {rating}")
    return "；".join(notes) or "关注正股波动、忘记缴款和上市首日溢价回落"


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_rank = {
        "registered": 0,
        "exchange_approved": 1,
        "accepted": 2,
        "shareholder_approved": 3,
        "board_plan": 4,
        "issuing": 5,
        "announced": 11,
        "listed": 90,
        "terminated": 91,
        "delisted": 92,
    }
    return sorted(
        records,
        key=lambda item: (
            1 if item.get("data_source") == "watchlist" else 0,
            stage_rank.get(item.get("stage") or "", 99),
            _number(item.get("sort_order")) or 9999,
            -_record_sort_date(item),
            item.get("stock_code") or item.get("bond_code") or "",
        ),
    )


def _record_sort_date(item: dict[str, Any]) -> int:
    text = item.get("pay_date") or item.get("record_date") or item.get("issue_date") or item.get("announce_date") or ""
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    return int(digits[:8]) if len(digits) >= 8 else 0


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for item in _sort_records(records):
        key = item.get("stock_code") or item.get("bond_code") or item.get("announcement_title")
        if not key:
            continue
        key_text = str(key)
        if key_text not in deduped:
            deduped[key_text] = item
            continue
        existing = deduped[key_text]
        for field, value in item.items():
            if field in {"data_source", "sort_order"}:
                continue
            if _clean_text(existing.get(field)) is None and _clean_text(value) is not None:
                existing[field] = value
    return list(deduped.values())


def _record_date_expired(item: dict[str, Any], today_text: str) -> bool:
    record_date = _date_text(item.get("record_date"))
    return bool(record_date and record_date < today_text)


def _event_poll_contract(
    *,
    refresh: bool,
    today_text: str,
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    statuses = {
        name: meta.get("poll_status")
        for name, meta in sources.items()
    }
    status = (
        "success"
        if refresh and all(value == "success" for value in statuses.values())
        else ("failed" if refresh else "not_requested")
    )
    return {
        "event_poll_status": status,
        "event_poll_sources": statuses,
        "event_polled_through": (
            _display_date(today_text) if status == "success" else None
        ),
    }


def build_convertible_bond_allotment_payload(
    limit: int = 80,
    include_listed_days: int = 90,
    refresh: bool = False,
    stage_scope: str = "pipeline",
    fetcher: TushareDataFetcher | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    today_text = _today_text(today)
    pipeline, pipeline_meta = _load_pipeline_candidates(refresh=refresh, today=today)
    cninfo_issue, cninfo_issue_meta = _load_cninfo_issue(refresh=refresh, today=today)
    watchlist, watchlist_meta = _load_watchlist()
    basic, basic_meta = _load_basic(refresh=refresh, fetcher=fetcher)
    issue, issue_meta = _load_issue(refresh=refresh, fetcher=fetcher)
    merged = _merge_basic_issue(basic, issue)
    records = [
        *_watchlist_records(watchlist),
        *_pipeline_records(pipeline),
        *_cninfo_issue_records(cninfo_issue, today_text),
    ]
    if stage_scope == "all":
        records.extend(_build_record(row, today_text) for _, row in merged.iterrows())

    records = _dedupe_records(records)
    records, pipeline_issue_date_meta = _attach_pipeline_issue_dates(records, refresh=refresh, today=today)
    cutoff = pd.to_datetime(today_text, format="%Y%m%d") - pd.Timedelta(days=include_listed_days)
    filtered = []
    for item in records:
        list_text = item.get("list_date")
        list_ts = pd.to_datetime(list_text, errors="coerce") if list_text else pd.NaT
        if stage_scope == "pipeline" and item.get("stage") not in PIPELINE_STAGES:
            continue
        if _record_date_expired(item, today_text):
            continue
        if item["stage"] == "delisted":
            continue
        if item["stage"] == "watching" and not item.get("announce_date") and not item.get("list_date"):
            continue
        if stage_scope == "pipeline" or item["stage"] != "listed" or pd.isna(list_ts) or list_ts >= cutoff:
            filtered.append(item)

    filtered = [item for item in filtered if item.get("data_source") != "watchlist"][:limit]
    filtered, market_meta = _attach_stock_market_snapshots(filtered)
    filtered, share_capital_meta = _attach_share_capital(filtered)
    filtered, pipeline_issue_size_meta = _attach_pipeline_issue_sizes(filtered, refresh=refresh, today=today)
    filtered = _attach_allotment_metrics(filtered)
    stage_counts = pd.Series([item["stage"] for item in filtered]).value_counts().to_dict() if filtered else {}
    event_poll = _event_poll_contract(
        refresh=refresh,
        today_text=today_text,
        sources={
            "pipeline": pipeline_meta,
            "cninfo_issue": cninfo_issue_meta,
            "basic": basic_meta,
            "issue": issue_meta,
        },
    )
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "asof": _display_date(today_text),
        # Event rows may legitimately be empty.  Only completed provider polls
        # advance this marker; cached fallbacks and caught exceptions do not.
        **event_poll,
        "stage_scope": stage_scope,
        "records": filtered,
        "stage_counts": {str(key): int(value) for key, value in stage_counts.items()},
        "field_schema": [
            {"key": "stock_code", "label": "正股代码", "source": "cb_basic.stk_code"},
            {"key": "bond_code", "label": "转债代码", "source": "cb_basic.ts_code"},
            {"key": "allot_code", "label": "配债代码", "source": "cb_issue.shd_ration_code / alias"},
            {"key": "record_date", "label": "股权登记日", "source": "巨潮发行明细 / 发行公告PDF"},
            {"key": "pay_date", "label": "配售缴款日", "source": "巨潮发行明细 / 发行公告PDF"},
            {"key": "list_date", "label": "上市日", "source": "cb_basic.list_date"},
            {"key": "stock_price", "label": "股价", "source": "data/raw/daily.close"},
            {"key": "kdj_daily_j", "label": "日线J值", "source": "calculate_project_extra_features.kdj_d_j"},
            {"key": "kdj_weekly_j", "label": "周线J值", "source": "calculate_project_extra_features.kdj_w_j"},
            {"key": "kdj_monthly_j", "label": "月线J值", "source": "calculate_project_extra_features.kdj_m_j"},
            {"key": "rights_value_pct", "label": "含权量", "source": "rights_per_share / stock_price"},
            {
                "key": "shares_for_one_lot",
                "label": "配一手至少股数",
                "source": "ceil(1000 / 每股可配售额)，由发行规模 / 总股本自动估算，不使用手工观察池",
            },
        ],
        "data_sources": {
            "watchlist": {**watchlist_meta, "rows": int(len(watchlist))},
            "pipeline": {**pipeline_meta, "rows": int(len(pipeline))},
            "cninfo_issue": {**cninfo_issue_meta, "rows": int(len(cninfo_issue))},
            "basic": {**basic_meta, "rows": int(len(basic))},
            "issue": {**issue_meta, "rows": int(len(issue))},
            "stock_daily": market_meta,
            "daily_basic": share_capital_meta,
            "pipeline_issue_size": pipeline_issue_size_meta,
            "pipeline_issue_date": pipeline_issue_date_meta,
        },
        "notes": [
            "默认按董事会预案、股东大会通过、交易所受理、上市委通过、同意注册、发行公告六个主阶段展示。",
            "前置阶段来自巨潮信披可转债公告标题推断；发行日、股权登记日和缴款日优先来自巨潮发行明细，缺失时从发行公告PDF解析。",
            "配一手至少股数和含权量只使用发行规模、配售比例、总股本和股价等自动数据；手工观察池不参与计算。",
            "已上市转债不会出现在默认列表；需要诊断历史样本时可使用 stage_scope=all。",
        ],
    }
