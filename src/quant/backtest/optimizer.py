import multiprocessing as mp
from typing import Dict, List, Any, Optional
from itertools import product
import pandas as pd

from quant.backtest.engine import BacktestEngine
from quant.research.validation import PurgedWalkForwardSplitter


class GridSearchOptimizer:
    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
        workers: int = 4,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003,
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.metric = metric
        self.workers = workers
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self._results: List[Dict] = []

    def run(self) -> Dict:
        param_combinations = [
            dict(zip(self.param_grid.keys(), v))
            for v in product(*self.param_grid.values())
        ]

        if self.workers > 1:
            with mp.Pool(self.workers) as pool:
                self._results = pool.map(self._evaluate, param_combinations)
        else:
            self._results = [self._evaluate(params) for params in param_combinations]

        if not self._results:
            raise ValueError("param_grid produces no parameter combinations")
        best_idx = max(range(len(self._results)), key=lambda i: self._results[i].get(self.metric, -float("inf")))
        best_result = self._results[best_idx]

        return {
            "best_params": {key: best_result[key] for key in self.param_grid},
            "best_score": best_result.get(self.metric),
            "best_result": best_result,
            "all_results": sorted(self._results, key=lambda x: x.get(self.metric, -float("inf")), reverse=True)
        }

    def _evaluate(self, params: Dict) -> Dict:
        try:
            strategy = self.strategy_class(**params)
            engine = BacktestEngine(
                data=self.data,
                strategy=strategy,
                initial_cash=self.initial_cash,
                commission_rate=self.commission_rate,
            )
            engine.run(show_progress=False)
            metrics = engine.get_metrics()
            return {**params, self.metric: metrics.get(self.metric, 0)}
        except Exception as e:
            return {**params, self.metric: -float("inf"), "error": str(e)}


class WalkForwardOptimizer:
    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        n_folds: int = 3,
        metric: str = "sharpe_ratio",
        purge_periods: int = 0,
        embargo_periods: int = 0,
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.train_ratio = train_ratio
        self.n_folds = n_folds
        self.metric = metric
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods

    def _date_series(self) -> pd.Series:
        if "date" in self.data.columns:
            dates = pd.to_datetime(self.data["date"], errors="coerce")
        elif "trade_date" in self.data.columns:
            dates = pd.to_datetime(
                self.data["trade_date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
        else:
            dates = pd.Series(pd.to_datetime(self.data.index, errors="coerce"), index=self.data.index)
        if dates.isna().any():
            raise ValueError("backtest data contain invalid dates")
        return pd.Series(dates.to_numpy(), index=self.data.index)

    def run(self) -> Dict:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be in (0, 1)")
        if self.n_folds <= 0:
            raise ValueError("n_folds must be positive")
        dates = self._date_series()
        unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
        train_size = int(len(unique_dates) * self.train_ratio)
        step_size = (len(unique_dates) - train_size) // self.n_folds
        if train_size <= 0 or step_size <= 0:
            raise ValueError("data do not contain enough periods for walk-forward validation")
        splitter = PurgedWalkForwardSplitter(
            train_periods=train_size,
            test_periods=step_size,
            purge_periods=self.purge_periods,
            embargo_periods=self.embargo_periods,
            expanding=True,
        )

        results = []
        for split in splitter.split(unique_dates)[: self.n_folds]:
            train_data = self.data.loc[
                dates.between(split.train_start, split.train_end)
            ]
            test_data = self.data.loc[
                dates.between(split.test_start, split.test_end)
            ]

            optimizer = GridSearchOptimizer(
                self.strategy_class,
                self.param_grid,
                train_data,
                metric=self.metric,
                workers=1
            )

            train_result = optimizer.run()
            best_params = train_result["best_params"]

            test_strategy = self.strategy_class(**best_params)
            test_engine = BacktestEngine(
                data=test_data,
                strategy=test_strategy,
                initial_cash=1000000.0,
            )
            test_engine.run(show_progress=False)
            test_metrics = test_engine.get_metrics()

            results.append({
                "fold": split.fold,
                "train_start": split.train_start.isoformat(),
                "train_end": split.train_end.isoformat(),
                "test_start": split.test_start.isoformat(),
                "test_end": split.test_end.isoformat(),
                "train_score": train_result["best_score"],
                "test_score": test_metrics.get(self.metric, 0),
                "best_params": best_params,
                "test_metrics": test_metrics
            })

        if not results:
            raise ValueError("walk-forward splitter produced no folds")
        return {
            "fold_results": results,
            "avg_train_score": sum(r["train_score"] for r in results) / len(results),
            "avg_test_score": sum(r["test_score"] for r in results) / len(results)
        }
