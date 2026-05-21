from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
import akshare as ak


class DataFetcher:
    def __init__(self, cache_dir: Union[str, Path] = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, pd.DataFrame] = {}

    def get_stock_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_{start_date}_{end_date}_{adjust}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        df = ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        df = self._normalize(df, symbol)
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df

        return df

    def get_stock_minute(
        self,
        symbol: str,
        period: str = "5",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_min_{period}_{start_date}_{end_date}_{adjust}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        if period in ["1", "5", "15", "30", "60"]:
            df = ak.stock_zh_a_minute(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
        else:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )

        df = self._normalize(df, symbol)
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df

        return df

    def get_index_daily(self, symbol: str = "000001", start_date: str = "20200101", end_date: str = "20260101") -> pd.DataFrame:
        cache_key = f"index_{symbol}_{start_date}_{end_date}"
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        file_path = self.cache_dir / f"{cache_key}.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            self._memory_cache[cache_key] = df
            return df

        if symbol.startswith("sh") or symbol.startswith("sz"):
            symbol_code = symbol[2:]
        else:
            symbol_code = symbol

        df = ak.index_zh_a_hist(symbol=symbol_code, period="daily", start_date=start_date, end_date=end_date)
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "turnover",
            "涨跌幅": "pct_change"
        })
        df["symbol"] = symbol
        df.to_parquet(file_path, index=False)
        self._memory_cache[cache_key] = df

        return df

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        column_mapping = {
            "日期": "date",
            "时间": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "turnover",
            "涨跌幅": "pct_change",
            "code": "symbol"
        }

        df = df.rename(columns=column_mapping)

        required_cols = ["date", "open", "close", "high", "low", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        if "symbol" not in df.columns:
            df["symbol"] = symbol

        if "pct_change" not in df.columns and "close" in df.columns and "open" in df.columns:
            df["pct_change"] = (df["close"] - df["open"]) / df["open"] * 100

        df = df.sort_values("date").reset_index(drop=True)
        return df

    def clear_cache(self, symbol: Optional[str] = None):
        if symbol:
            keys_to_remove = [k for k in self._memory_cache.keys() if k.startswith(symbol)]
            for k in keys_to_remove:
                del self._memory_cache[k]
        else:
            self._memory_cache.clear()

        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.parquet"):
                if symbol is None or symbol in f.name:
                    f.unlink()
