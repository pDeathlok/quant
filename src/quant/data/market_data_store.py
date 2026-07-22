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
            if not self.config.sql_url and not self.config.mirror_parquet:
                raise ValueError("MARKET_DATA_SQL_URL is required when MARKET_DATA_BACKEND=mysql")
            return
        self._write_parquet(frame, dataset, key)

    def write_market_batch(
        self,
        frame: pd.DataFrame,
        dataset: str = "daily",
        partition_column: str = "trade_date",
    ) -> dict[str, int | str]:
        """Idempotently persist a cross-sectional market batch in one SQL transaction and date partitions."""

        if frame.empty:
            return {"rows": 0, "sql_rows": 0, "parquet_partitions": 0, "table": self._dataset_table_name(dataset)}
        required = {"ts_code", partition_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"market batch missing required columns: {sorted(missing)}")
        normalized = frame.copy()
        normalized["ts_code"] = normalized["ts_code"].astype(str)
        normalized[partition_column] = (
            normalized[partition_column].astype(str).str.replace("-", "", regex=False)
        )
        normalized = normalized.drop_duplicates(["ts_code", partition_column], keep="last")

        sql_rows = 0
        if self.config.backend in {"mysql", "sql"}:
            if self.config.sql_url:
                sql_rows = self._write_sql_batch(normalized, dataset, partition_column)
            elif not self.config.mirror_parquet:
                raise ValueError("MARKET_DATA_SQL_URL is required when MARKET_DATA_BACKEND=mysql")
        parquet_partitions = 0
        if self.config.backend not in {"mysql", "sql"} or self.config.mirror_parquet:
            parquet_partitions = self._write_partitioned_parquet(normalized, dataset, partition_column)
        return {
            "rows": int(len(normalized)),
            "sql_rows": int(sql_rows),
            "parquet_partitions": int(parquet_partitions),
            "table": self._dataset_table_name(dataset),
        }

    def read_frame(self, dataset: str, key: str) -> pd.DataFrame:
        if self.config.backend in {"mysql", "sql"}:
            if self.config.sql_url:
                sql_frame = self._read_sql(dataset, key)
                if not sql_frame.empty:
                    return sql_frame
            if self.config.mirror_parquet:
                return self._read_parquet(dataset, key)
            if not self.config.sql_url and not self.config.mirror_parquet:
                raise ValueError("MARKET_DATA_SQL_URL is required when MARKET_DATA_BACKEND=mysql")
            return pd.DataFrame()
        return self._read_parquet(dataset, key)

    def read_market_range(
        self,
        dataset: str = "daily",
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | set[str] | tuple[str, ...] | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Read a canonical date range in one query or one pass over month partitions."""

        if self.config.backend in {"mysql", "sql"} and self.config.sql_url:
            frame = self._read_sql_range(dataset, start_date, end_date, symbols, columns)
            if not frame.empty or not self.config.mirror_parquet:
                return frame
        return self._read_parquet_range(dataset, start_date, end_date, symbols, columns)

    def list_symbols(self, dataset: str = "daily") -> list[str]:
        if self.config.backend in {"mysql", "sql"} and self.config.sql_url:
            engine = self._engine()
            try:
                from sqlalchemy import text

                frame = pd.read_sql_query(
                    text(f"SELECT DISTINCT ts_code FROM `{self._dataset_table_name(dataset)}`"),
                    engine,
                )
                if not frame.empty:
                    return sorted(frame["ts_code"].dropna().astype(str).tolist())
            except Exception:
                if not self.config.mirror_parquet:
                    return []
            finally:
                engine.dispose()
        symbols: set[str] = set()
        for path in self._partition_root(dataset).glob("year_month=*/data.parquet"):
            frame = pd.read_parquet(path, columns=["ts_code"])
            symbols.update(frame["ts_code"].dropna().astype(str).unique().tolist())
        return sorted(symbols)

    def latest_trade_date(self, dataset: str, key: str) -> pd.Timestamp | None:
        if self.config.backend in {"mysql", "sql"} and self.config.sql_url:
            try:
                latest = self._latest_trade_date_sql(dataset, key)
            except Exception:
                latest = None
            if latest is not None:
                return latest
            if not self.config.mirror_parquet:
                return None
        return self._latest_trade_date_parquet(dataset, key)

    def latest_dataset_trade_date(self, dataset: str) -> pd.Timestamp | None:
        if self.config.backend in {"mysql", "sql"} and self.config.sql_url:
            try:
                latest = self._latest_dataset_trade_date_sql(dataset)
            except Exception:
                latest = None
            if latest is not None:
                return latest
            if not self.config.mirror_parquet:
                return None
        latest = self._latest_partition_trade_date(dataset)
        if latest is not None:
            return latest
        return None

    def _path(self, dataset: str, key: str) -> Path:
        safe_key = key.replace("/", "_").replace(":", "").replace(" ", "_")
        return self.config.root / dataset / f"{safe_key}.parquet"

    def _partition_root(self, dataset: str) -> Path:
        return self.config.root / f"{dataset}_partitioned"

    def _write_parquet(self, frame: pd.DataFrame, dataset: str, key: str) -> None:
        path = self._path(dataset, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def _write_partitioned_parquet(self, frame: pd.DataFrame, dataset: str, partition_column: str) -> int:
        partition_root = self._partition_root(dataset)
        partitions = 0
        year_month = frame[partition_column].astype(str).str.replace("-", "", regex=False).str[:6]
        for partition_value, incoming in frame.groupby(year_month, sort=True):
            partition_dir = partition_root / f"year_month={partition_value}"
            path = partition_dir / "data.parquet"
            partition_dir.mkdir(parents=True, exist_ok=True)
            if path.exists():
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
            else:
                combined = incoming.copy()
            combined = (
                combined.drop_duplicates(["ts_code", partition_column], keep="last")
                .sort_values("ts_code")
                .reset_index(drop=True)
            )
            temp_path = partition_dir / ".data.parquet.tmp"
            combined.to_parquet(temp_path, index=False)
            os.replace(temp_path, path)
            partitions += 1

        return partitions

    def _read_parquet(self, dataset: str, key: str) -> pd.DataFrame:
        frame = self._read_parquet_range(dataset, symbols=[str(key)])
        if not frame.empty or any(self._partition_root(dataset).glob("year_month=*/data.parquet")):
            return frame
        path = self._path(dataset, key)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def _read_parquet_range(
        self,
        dataset: str,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | set[str] | tuple[str, ...] | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        start = str(start_date).replace("-", "") if start_date else None
        end = str(end_date).replace("-", "") if end_date else None
        start_month = start[:6] if start else None
        end_month = end[:6] if end else None
        symbol_values = {str(value) for value in symbols} if symbols else None
        frames: list[pd.DataFrame] = []
        for path in sorted(self._partition_root(dataset).glob("year_month=*/data.parquet")):
            month = path.parent.name.partition("=")[2]
            if start_month and month < start_month:
                continue
            if end_month and month > end_month:
                continue
            read_columns = list(dict.fromkeys([*(columns or []), "ts_code", "trade_date"])) if columns else None
            frame = pd.read_parquet(path, columns=read_columns)
            if symbol_values is not None and "ts_code" in frame.columns:
                frame = frame[frame["ts_code"].astype(str).isin(symbol_values)]
            if start and "trade_date" in frame.columns:
                frame = frame[frame["trade_date"].astype(str).str.replace("-", "", regex=False) >= start]
            if end and "trade_date" in frame.columns:
                frame = frame[frame["trade_date"].astype(str).str.replace("-", "", regex=False) <= end]
            if not frame.empty:
                frames.append(frame)
        result = self._merge_frames(frames)
        if columns and not result.empty:
            return result[[column for column in columns if column in result.columns]]
        return result

    def _latest_trade_date_parquet(self, dataset: str, key: str) -> pd.Timestamp | None:
        for path in reversed(sorted(self._partition_root(dataset).glob("year_month=*/data.parquet"))):
            try:
                frame = pd.read_parquet(
                    path,
                    columns=["ts_code", "trade_date"],
                    filters=[("ts_code", "=", str(key))],
                )
            except Exception:
                frame = pd.read_parquet(path, columns=["ts_code", "trade_date"])
                frame = frame[frame["ts_code"].astype(str).eq(str(key))]
            if frame.empty:
                continue
            latest = pd.to_datetime(
                frame["trade_date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            ).max()
            if pd.notna(latest):
                return latest
        return self._latest_trade_date_from_path(self._path(dataset, key))

    @staticmethod
    def _latest_trade_date_from_path(path: Path) -> pd.Timestamp | None:
        if not path.exists():
            return None
        for column in ("trade_date", "date"):
            try:
                frame = pd.read_parquet(path, columns=[column])
            except Exception:
                continue
            if frame.empty:
                return None
            dates = (
                pd.to_datetime(frame[column].astype(str), format="%Y%m%d", errors="coerce")
                if column == "trade_date"
                else pd.to_datetime(frame[column], errors="coerce")
            )
            latest = dates.max()
            return latest if pd.notna(latest) else None
        return None

    def _latest_partition_trade_date(self, dataset: str) -> pd.Timestamp | None:
        paths = sorted(self._partition_root(dataset).glob("year_month=*/data.parquet"))
        if not paths:
            return None
        frame = pd.read_parquet(paths[-1], columns=["trade_date"])
        latest = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce").max()
        return latest if pd.notna(latest) else None

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

    def _write_sql_batch(
        self,
        frame: pd.DataFrame,
        dataset: str,
        partition_column: str,
        update_existing: bool = True,
    ) -> int:
        from sqlalchemy import Date, DateTime, Float, MetaData, String, Table, inspect
        from sqlalchemy.dialects.mysql import VARCHAR

        table_name = self._dataset_table_name(dataset)
        engine = self._engine()
        try:
            dtype = {
                "ts_code": VARCHAR(9, charset="ascii", collation="ascii_bin"),
                "symbol": VARCHAR(9, charset="ascii", collation="ascii_bin"),
                partition_column: Date(),
                "date": DateTime(),
                "name": String(128),
                "industry": String(128),
            }
            for column in (
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "change",
                "pct_chg",
                "vol",
                "volume",
                "amount",
                "turnover",
            ):
                dtype[column] = Float()
            frame.head(0).to_sql(table_name, engine, if_exists="append", index=False, dtype=dtype)
            inspector = inspect(engine)
            indexes = inspector.get_indexes(table_name)
            unique_columns = {tuple(item.get("column_names") or []) for item in indexes if item.get("unique")}
            indexed_columns = {tuple(item.get("column_names") or []) for item in indexes}
            if ("ts_code", partition_column) not in unique_columns:
                with engine.begin() as conn:
                    conn.exec_driver_sql(
                        f"ALTER TABLE `{table_name}` ADD UNIQUE KEY "
                        f"`uq_ts_code_{partition_column}` (`ts_code`, `{partition_column}`)"
                    )
            if (partition_column,) not in indexed_columns:
                with engine.begin() as conn:
                    conn.exec_driver_sql(
                        f"ALTER TABLE `{table_name}` ADD INDEX "
                        f"`idx_{partition_column}` (`{partition_column}`)"
                    )
            table = Table(table_name, MetaData(), autoload_with=engine)
            sql_frame = frame.copy()
            sql_frame[partition_column] = pd.to_datetime(
                sql_frame[partition_column].astype(str).str.replace("-", "", regex=False),
                format="%Y%m%d",
                errors="raise",
            ).dt.date
            records = sql_frame.astype(object).where(pd.notna(sql_frame), None).to_dict("records")
            chunk_size = max(100, int(os.getenv("MARKET_DATA_SQL_BATCH_SIZE", "5000")))
            with self._sql_write_lock, engine.begin() as conn:
                if engine.dialect.name == "mysql":
                    from sqlalchemy.dialects.mysql import insert as mysql_insert

                    statement = mysql_insert(table)
                    if update_existing:
                        updates = {
                            column.name: statement.inserted[column.name]
                            for column in table.columns
                            if column.name not in {"ts_code", partition_column}
                        }
                        statement = statement.on_duplicate_key_update(**updates)
                    else:
                        statement = statement.prefix_with("IGNORE")
                    for offset in range(0, len(records), chunk_size):
                        conn.execute(statement, records[offset : offset + chunk_size])
                else:
                    dates = sorted(frame[partition_column].astype(str).unique().tolist())
                    conn.execute(table.delete().where(table.c[partition_column].in_(dates)))
                    conn.execute(table.insert(), records)
            return len(records)
        finally:
            engine.dispose()

    def _read_sql(self, dataset: str, key: str) -> pd.DataFrame:
        return self._read_sql_range(dataset, symbols=[str(key)])

    def _read_sql_range(
        self,
        dataset: str,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | set[str] | tuple[str, ...] | None = None,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        from sqlalchemy import bindparam, text

        clauses: list[str] = []
        params: dict[str, object] = {}
        if start_date:
            clauses.append("trade_date >= :start_date")
            params["start_date"] = str(start_date).replace("-", "")
        if end_date:
            clauses.append("trade_date <= :end_date")
            params["end_date"] = str(end_date).replace("-", "")
        selected = "*"
        if columns:
            safe_columns = [column for column in columns if column.replace("_", "").isalnum()]
            selected = ", ".join(f"`{column}`" for column in safe_columns)
        statement = f"SELECT {selected} FROM `{self._dataset_table_name(dataset)}`"
        symbol_values = [str(value) for value in symbols] if symbols else []
        if symbol_values:
            clauses.append("ts_code IN :symbols")
            params["symbols"] = symbol_values
        if clauses:
            statement += " WHERE " + " AND ".join(clauses)
        query = text(statement)
        if symbol_values:
            query = query.bindparams(bindparam("symbols", expanding=True))
        engine = self._engine()
        try:
            frame = pd.read_sql_query(query, engine, params=params)
            if "trade_date" in frame.columns:
                frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
            return frame
        except Exception:
            return pd.DataFrame()
        finally:
            engine.dispose()

    def _latest_trade_date_sql(self, dataset: str, key: str) -> pd.Timestamp | None:
        engine = self._engine()
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                value = conn.execute(
                    text(
                        f"SELECT MAX(`trade_date`) FROM `{self._dataset_table_name(dataset)}` "
                        "WHERE ts_code = :ts_code"
                    ),
                    {"ts_code": str(key)},
                ).scalar()
        finally:
            engine.dispose()
        latest = pd.to_datetime(str(value).replace("-", ""), format="%Y%m%d", errors="coerce")
        return latest if pd.notna(latest) else None

    def _latest_dataset_trade_date_sql(self, dataset: str) -> pd.Timestamp | None:
        engine = self._engine()
        try:
            from sqlalchemy import text

            with engine.connect() as conn:
                value = conn.execute(
                    text(f"SELECT MAX(`trade_date`) FROM `{self._dataset_table_name(dataset)}`")
                ).scalar()
            latest = pd.to_datetime(str(value).replace("-", ""), format="%Y%m%d", errors="coerce")
            return latest if pd.notna(latest) else None
        except Exception:
            return None
        finally:
            engine.dispose()

    @staticmethod
    def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        usable = [frame for frame in frames if frame is not None and not frame.empty]
        if not usable:
            return pd.DataFrame()
        combined = pd.concat(usable, ignore_index=True, sort=False)
        keys = [column for column in ("ts_code", "trade_date") if column in combined.columns]
        if "trade_date" in combined.columns:
            combined["trade_date"] = combined["trade_date"].astype(str).str.replace("-", "", regex=False)
        if keys:
            combined = combined.drop_duplicates(keys, keep="last")
        sort_columns = [column for column in ("date", "trade_date") if column in combined.columns]
        return combined.sort_values(sort_columns).reset_index(drop=True) if sort_columns else combined.reset_index(drop=True)

    @staticmethod
    def _table_name(dataset: str, key: str) -> str:
        raw = f"{dataset}_{key}".lower()
        return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")[:60]

    @staticmethod
    def _dataset_table_name(dataset: str) -> str:
        raw = f"market_{dataset}".lower()
        return "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")[:60]


def read_partitioned_symbol_file(
    path: Path | str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Read one legacy symbol file plus all newer trade-date partitions beside it."""

    symbol_path = Path(path)
    store = MarketDataStore(
        MarketDataStoreConfig(
            backend="parquet",
            root=symbol_path.parent.parent,
            mirror_parquet=True,
        )
    )
    if start_date is None and end_date is None:
        return store.read_frame(symbol_path.parent.name, symbol_path.stem)
    start = pd.to_datetime(start_date).strftime("%Y%m%d") if start_date is not None else None
    end = pd.to_datetime(end_date).strftime("%Y%m%d") if end_date is not None else None
    return store.read_market_range(
        symbol_path.parent.name,
        start_date=start,
        end_date=end,
        symbols=[symbol_path.stem],
    )


def list_partitioned_symbol_paths(daily_dir: Path | str) -> list[Path]:
    """Return synthetic per-symbol paths for code that processes one time series per task."""

    directory = Path(daily_dir)
    store = MarketDataStore(
        MarketDataStoreConfig(backend="parquet", root=directory.parent, mirror_parquet=True)
    )
    symbols = store.list_symbols(directory.name)
    if symbols:
        return [directory / f"{symbol}.parquet" for symbol in symbols]
    return sorted(directory.glob("*.parquet"))
