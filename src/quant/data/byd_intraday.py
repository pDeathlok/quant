"""Local-only BYD training artifact contract shared by workspace and TTL evidence.

The fixed authoritative file wins over all legacy exports, even when corrupt.
Its checkpoint is the last COMPLETE session's Shanghai-local midnight, never
the download time or the requested end date. This is historical training data,
not a current intraday feature. No function in this module accesses the network.
"""

from __future__ import annotations

import copy
import hashlib
import io
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from quant.core.paths import PROJECT_ROOT
from quant.infrastructure.publication import publication_path

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data/cache"
TRAINING_FILENAME = "byd_002594_5min_training_qfq.parquet"
LEGACY_GLOB = "baostock_002594_5min_*_qfq.parquet"
MAX_STALENESS_DAYS = 60
PRICE_COLUMNS = ["open", "high", "low", "close"]
VALUE_COLUMNS = [*PRICE_COLUMNS, "volume", "amount"]
SESSION_MINUTES = tuple(range(575, 691, 5)) + tuple(range(785, 901, 5))


def local_timestamp(value: object) -> pd.Timestamp:
    """Interpret naive clocks as Asia/Shanghai; convert aware clocks first."""
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError("BYD timestamp is missing")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Shanghai").tz_localize(None)
    return parsed


def validate_training_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject bad rows and partial sessions instead of silently dropping them."""
    required = {"datetime", "date", "code", "adjustflag", *VALUE_COLUMNS}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"BYD training bars empty or missing columns: {required - set(frame.columns)}")
    out = frame.copy()
    for column in ("datetime", "date"):
        out[column] = pd.to_datetime(out[column], errors="raise")
        if out[column].isna().any() or out[column].dt.tz is not None:
            raise ValueError(f"BYD invalid local {column}")
    if not out["date"].eq(out["datetime"].dt.normalize()).all():
        raise ValueError("BYD date/datetime mismatch")
    if "time" in out:
        raw_time = out["time"].astype(str)
        if not raw_time.str.fullmatch(r"\d{17}").all():
            raise ValueError("BYD malformed BaoStock time")
        parsed = pd.to_datetime(raw_time.str[:14], format="%Y%m%d%H%M%S", errors="raise")
        if not parsed.eq(out["datetime"]).all() or not raw_time.str.endswith("000").all():
            raise ValueError("BYD time/datetime mismatch")
    if not out["code"].eq("sz.002594").all() or not out["adjustflag"].astype(str).eq("2").all():
        raise ValueError("BYD requires sz.002594 and qfq adjustflag=2")
    if out["datetime"].duplicated().any():
        raise ValueError("BYD duplicate minute timestamps")
    for column in VALUE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="raise")
    values = out[VALUE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (out[PRICE_COLUMNS] <= 0).any().any():
        raise ValueError("BYD non-finite or non-positive prices")
    if (out[["volume", "amount"]] < 0).any().any():
        raise ValueError("BYD negative volume/amount")
    if (out["high"] < out[PRICE_COLUMNS].max(axis=1)).any() or (
        out["low"] > out[PRICE_COLUMNS].min(axis=1)
    ).any():
        raise ValueError("BYD invalid OHLC ordering")
    timestamps = out["datetime"]
    minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
    if not timestamps.eq(timestamps.dt.floor("min")).all() or not minutes.isin(SESSION_MINUTES).all():
        raise ValueError("BYD invalid five-minute session grid")
    if not out.groupby("date").size().eq(48).all():
        raise ValueError("BYD incomplete session: expected 48 five-minute bars")
    return out.sort_values("datetime").reset_index(drop=True)


def _signature(path: Path) -> tuple[str, int, int, int, int]:
    stat = path.stat()
    return str(path), stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _inventory(cache_dir: Path) -> tuple[tuple[str, int, int, int, int], ...]:
    logical_path = cache_dir / TRAINING_FILENAME
    authoritative = publication_path(logical_path)
    if authoritative.is_symlink():
        raise ValueError("BYD authoritative training file must not be a symlink")
    if authoritative.exists():
        return (_signature(authoritative),)
    if authoritative != logical_path:
        # A new managed path may be absent from older generations. Root legacy
        # files cannot prove that generation's inputs or satisfy a staging skip.
        raise FileNotFoundError(f"BYD training data missing from publication: {authoritative}")
    paths = sorted(publication_path(cache_dir).glob(LEGACY_GLOB))
    if not paths:
        raise FileNotFoundError("BYD historical five-minute training data missing")
    return tuple(_signature(path) for path in paths)


@lru_cache(maxsize=4)
def _read_inventory(inventory: tuple[tuple[str, int, int, int, int], ...]) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    for signature in inventory:
        path = Path(signature[0])
        raw = path.read_bytes()
        frame = validate_training_bars(pd.read_parquet(io.BytesIO(raw)))
        frame.attrs.update(
            intraday_training_source=str(path),
            intraday_training_fingerprint="sha256:" + hashlib.sha256(raw).hexdigest(),
        )
        frame.attrs.setdefault("provider", "baostock" if path.name.startswith("baostock_") else "unknown")
        candidates.append(frame)
    # A newer short research export must not displace the full training history.
    earliest = min(frame["date"].min() for frame in candidates)
    candidates = [frame for frame in candidates if frame["date"].min() == earliest]
    latest = max(frame["date"].max() for frame in candidates)
    candidates = [frame for frame in candidates if frame["date"].max() == latest]
    selected = candidates[0]
    for other in candidates[1:]:
        if not selected[["datetime", *VALUE_COLUMNS]].equals(other[["datetime", *VALUE_COLUMNS]]):
            raise ValueError("BYD ambiguous legacy training histories; select an authoritative export")
    return selected


def read_byd_training_frame(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Read the same validated artifact as evidence, with replacement-aware caching."""
    cache_dir = Path(cache_dir).absolute()
    for _ in range(3):
        before = _inventory(cache_dir)
        frame = _read_inventory(before)
        if before == _inventory(cache_dir):
            result = frame.copy(deep=True)
            result.attrs = copy.deepcopy(frame.attrs)
            return result
    raise RuntimeError("BYD training artifact changed repeatedly during read")


def read_byd_intraday_training_evidence(
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[date, datetime, str]:
    """daily_dependency_runtime adapter result: watermark, checked_at, fingerprint.

    Register a dedicated adapter (e.g. ``byd_intraday_training``) calling this
    with ``project_root / 'data/cache'``. Do NOT combine it with glob_parquet_max.
    Missing/corrupt input must become missing evidence, not a healthy fallback.
    The naive checkpoint is Shanghai-local midnight, matching runtime's clock.
    """
    frame = read_byd_training_frame(cache_dir)
    checkpoint = frame["date"].max().to_pydatetime()
    return checkpoint.date(), checkpoint, frame.attrs["intraday_training_fingerprint"]
