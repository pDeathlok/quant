import pandas as pd
from pathlib import Path
from typing import Union, Optional
from datetime import datetime


class DataStorage:
    def __init__(self, data_dir: Union[str, Path] = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_parquet(self, df: pd.DataFrame, filename: str, partition_by: Optional[str] = None):
        if partition_by and partition_by in df.columns:
            partition_dir = self.data_dir / f"{filename}.parquet"
            partition_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(partition_dir, index=False, partition_cols=[partition_by])
        else:
            filepath = self.data_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)

    def load_parquet(self, filename: str, partition_filter: Optional[dict] = None) -> pd.DataFrame:
        filepath = self.data_dir / f"{filename}.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        if filepath.is_dir():
            return pd.read_parquet(filepath, filters=self._build_filter(partition_filter))
        else:
            return pd.read_parquet(filepath)

    def _build_filter(self, partition_filter: Optional[dict]) -> Optional[list[tuple[str, str, object]]]:
        if not partition_filter:
            return None
        return [(key, "==", value) for key, value in partition_filter.items()]

    def save_csv(self, df: pd.DataFrame, filename: str):
        filepath = self.data_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8")

    def load_csv(self, filename: str) -> pd.DataFrame:
        filepath = self.data_dir / f"{filename}.csv"
        return pd.read_csv(filepath)

    def list_files(self, pattern: str = "*.parquet") -> list:
        return list(self.data_dir.rglob(pattern))
