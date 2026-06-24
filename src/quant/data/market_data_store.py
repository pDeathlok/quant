"""Market data storage helpers.

The production storage target is MySQL. A parquet mirror can be enabled for
research scripts that still scan local files directly while the project is being
incrementally migrated to SQL-backed reads.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MarketDataStoreConfig:
    backend: str = "mysql"
    root: Path = Path("data/raw")
    sql_url: str | None = None
    mirror_parquet: bool = True

    @classmethod
    def from_env(cls, root: Path | str = "data/raw") -> "MarketDataStoreConfig":
        return cls(
            backend=os.getenv("MARKET_DATA_BACKEND", "mysql").lower(),
            root=Path(os.getenv("MARKET_DATA_ROOT", str(root))),
            sql_url=os.getenv("MARKET_DATA_SQL_URL"),
            mirror_parquet=os.getenv("MARKET_DATA_MIRROR_PARQUET", "1").lower() not in {"0", "false", "no"},
        )


class MarketDataStore:
    """Read/write market data using parquet or SQL tables."""

    _sql_write_lock = threading.Lock()

    def __init__(self, config: MarketDataStoreConfig | None = None):
        self.config = config or MarketDataStoreConfig.from_env()

    def write_frame(self, frame: pd.DataFrame, dataset: str, key: str) -> None:
        if self.config.backend in {"mysql", "sql"}:
            if self.config.sql_url:
                self._write_sql(frame, dataset, key)
            if self.config.mirror_parquet:
                self._write_parquet(frame, dataset, key)
                return
            if not self.config.sql_url:
                raise ValueError("MARKET_DATA_SQL_URL is required when MARKET_DATA_BACKEND=mysql")
            return
        self._write_parquet(frame, dataset, key)

    def read_frame(self, dataset: str, key: str) -> pd.DataFrame:
        if self.config.backend in {"mysql", "sql"}:
            if self.config.sql_url:
                sql_frame = self._read_sql(dataset, key)
                if not sql_frame.empty:
                    return sql_frame
            if self.config.mirror_parquet:
                return self._read_parquet(dataset, key)
            if not self.config.sql_url:
                raise ValueError("MARKET_DATA_SQL_URL is required when MARKET_DATA_BACKEND=mysql")
            return pd.DataFrame()
        return self._read_parquet(dataset, key)

    def latest_trade_date(self, dataset: str, key: str) -> pd.Timestamp | None:
        if self.config.backend in {"mysql", "sql"} and self.config.sql_url:
            latest = self._latest_trade_date_sql(dataset, key)
            if latest is not None:
                return latest
            if not self.config.mirror_parquet:
                return None
        return self._latest_trade_date_parquet(dataset, key)

    def _path(self, dataset: str, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "").replace(" ", "_")
        return self.config.root / dataset / f"{safe_key}.parquet"

    def _write_parquet(self, frame: pd.DataFrame, dataset: str, key: str) -> None:
        path = self._path(dataset, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _read_parquet(self, dataset: str, key: str) -> pd.DataFrame:
        path = self._path(dataset, key)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _latest_trade_date_parquet(self, dataset: str, key: str) -> pd.Timestamp | None:
        path = self._path(dataset, key)
        if not path.exists():
            return None
        for column in ("trade_date", "date"):
            try:
                frame = pd.read_parquet(path, columns=[column])
            except Exception:
                continue
            if frame.empty:
                return None
            if column == "trade_date":
                dates = pd.to_datetime(frame[column].astype(str), format="%Y%m%d", errors="coerce")
            else:
                dates = pd.to_datetime(frame[column], errors="coerce")
            latest = dates.max()
            return latest if pd.notna(latest) else None
        return None

    def _engine(self):
        if not self.config.sql_url:
            raise ValueError("MARKET_DATA_SQL_URL is required when MARKET_DATA_BACKEND=mysql")
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ImportError("SQL backend requires sqlalchemy and a database driver such as pymysql") from exc
        connect_args = {
            "connect_timeout": int(os.getenv("MARKET_DATA_SQL_CONNECT_TIMEOUT", "10")),
            "read_timeout": int(os.getenv("MARKET_DATA_SQL_READ_TIMEOUT", "60")),
            "write_timeout": int(os.getenv("MARKET_DATA_SQL_WRITE_TIMEOUT", "60")),
        }
        return create_engine(
            self.config.sql_url,
            connect_args=connect_args,
            pool_pre_ping=True,
            pool_recycle=300,
        )

    def _write_sql(self, frame: pd.DataFrame, dataset: str, key: str) -> None:
        table = self._table_name(dataset, key)
        engine = self._engine()
        try:
            with self._sql_write_lock:
                frame.to_sql(table, engine, if_exists="replace", index=False, chunksize=5000, method="multi")
        finally:
            engine.dispose()

    def _read_sql(self, dataset: str, key: str) -> pd.DataFrame:
        table = self._table_name(dataset, key)
        engine = self._engine()
        try:
            return pd.read_sql_table(table, engine)
        except Exception:
            return pd.DataFrame()
        finally:
            engine.dispose()

    def _latest_trade_date_sql(self, dataset: str, key: str) -> pd.Timestamp | None:
        table = self._table_name(dataset, key)
        engine = self._engine()
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                for column in ("trade_date", "date"):
                    try:
                        value = conn.execute(text(f"SELECT MAX({column}) FROM `{table}`")).scalar()
                    except Exception:
                        continue
                    if value is None:
                        return None
                    if column == "trade_date":
                        latest = pd.to_datetime(str(value), format="%Y%m%d", errors="coerce")
                    else:
                        latest = pd.to_datetime(value, errors="coerce")
                    return latest if pd.notna(latest) else None
        finally:
            engine.dispose()
        return None

    @staticmethod
    def _table_name(dataset: str, key: str) -> str:
        raw = f"{dataset}_{key}".lower()
        return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")[:60]
