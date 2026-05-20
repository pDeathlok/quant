from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class Factor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        pass


class TechnicalFactor(Factor, ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name
