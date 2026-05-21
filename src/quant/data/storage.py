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
            partition_dir = self.data_dir / filename
            for partition_value, group in df.groupby(partition_by):
                group_path = partition_dir / f"{partition_by}={partition_value}"
                group.drop(columns=[partition_by]).to_parquet(group_path, index=False)
        else:
            filepath = self.data_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)

    def load_parquet(self, filename: str, partition_filter: Optional[dict] = None) -> pd.DataFrame:
        filepath = self.data_dir / f"{filename}.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        if filepath.is_dir():
            import pyarrow.dataset as ds
            dataset = ds.dataset(str(filepath), format="parquet")
            table = dataset.to_table(filter=self._build_filter(partition_filter))
            return table.to_pandas()
        else:
            return pd.read_parquet(filepath)

    def _build_filter(self, partition_filter: Optional[dict]) -> Optional[str]:
        if not partition_filter:
            return None
        filters = [f"{k}={v}" for k, v in partition_filter.items()]
        return " AND ".join(filters)

    def save_csv(self, df: pd.DataFrame, filename: str):
        filepath = self.data_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False, encoding="utf-8")

    def load_csv(self, filename: str) -> pd.DataFrame:
        filepath = self.data_dir / f"{filename}.csv"
        return pd.read_csv(filepath)

    def list_files(self, pattern: str = "*.parquet") -> list:
        return list(self.data_dir.rglob(pattern))
