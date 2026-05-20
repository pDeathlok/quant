import pandas as pd
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import hashlib


class DataCache:
    def __init__(self, max_size: int = 100, ttl_hours: int = 24):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._access_count: Dict[str, int] = {}
        self.max_size = max_size
        self.ttl_hours = ttl_hours

    def get(self, key: str) -> Optional[pd.DataFrame]:
        if key not in self._cache:
            return None

        if self._is_expired(key):
            self.invalidate(key)
            return None

        self._access_count[key] = self._access_count.get(key, 0) + 1
        return self._cache[key].copy()

    def set(self, key: str, data: pd.DataFrame):
        self._evict_if_needed()

        self._cache[key] = data.copy()
        self._timestamps[key] = datetime.now()
        self._access_count[key] = 0

    def invalidate(self, key: str):
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._access_count:
            del self._access_count[key]

    def clear(self):
        self._cache.clear()
        self._timestamps.clear()
        self._access_count.clear()

    def _is_expired(self, key: str) -> bool:
        if key not in self._timestamps:
            return True
        age = datetime.now() - self._timestamps[key]
        return age > timedelta(hours=self.ttl_hours)

    def _evict_if_needed(self):
        if len(self._cache) >= self.max_size:
            lru_key = min(self._access_count, key=self._access_count.get)
            self.invalidate(lru_key)

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
