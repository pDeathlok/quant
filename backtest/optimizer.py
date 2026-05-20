import multiprocessing as mp
from typing import Dict, List, Any, Optional
from itertools import product
import pandas as pd
import akquant as aq


class GridSearchOptimizer:
    def __init__(
        self,
        strategy_class,
        param_grid: Dict[str, List[Any]],
        data: pd.DataFrame,
        metric: str = "sharpe_ratio",
        workers: int = 4,
        initial_cash: float = 1000000.0,
        commission_rate: float = 0.0003
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

        best_idx = max(range(len(self._results)), key=lambda i: self._results[i].get(self.metric, -float("inf")))
        best_result = self._results[best_idx]

        return {
            "best_params": {k: v for k, v in best_result.items() if k != self.metric},
            "best_score": best_result.get(self.metric),
            "best_result": best_result,
            "all_results": sorted(self._results, key=lambda x: x.get(self.metric, -float("inf")), reverse=True)
        }

    def _evaluate(self, params: Dict) -> Dict:
        try:
            strategy = self.strategy_class(**params)
            result = aq.run_backtest(
                data=self.data,
                strategy=strategy,
                initial_cash=self.initial_cash,
                commission_rate=self.commission_rate,
                show_progress=False
            )

            metrics = result.summary()
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
        metric: str = "sharpe_ratio"
    ):
        self.strategy_class = strategy_class
        self.param_grid = param_grid
        self.data = data
        self.train_ratio = train_ratio
        self.n_folds = n_folds
        self.metric = metric

    def run(self) -> Dict:
        n_samples = len(self.data)
        train_size = int(n_samples * self.train_ratio)
        step_size = (n_samples - train_size) // self.n_folds

        results = []
        for fold in range(self.n_folds):
            train_end = train_size + fold * step_size
            train_data = self.data.iloc[:train_end]
            test_data = self.data.iloc[train_end:train_end + step_size]

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
            test_backtest = aq.run_backtest(
                data=test_data,
                strategy=test_strategy,
                initial_cash=1000000.0,
                show_progress=False
            )

            test_metrics = test_backtest.summary()

            results.append({
                "fold": fold + 1,
                "train_score": train_result["best_score"],
                "test_score": test_metrics.get(self.metric, 0),
                "best_params": best_params,
                "test_metrics": test_metrics
            })

        return {
            "fold_results": results,
            "avg_train_score": sum(r["train_score"] for r in results) / len(results),
            "avg_test_score": sum(r["test_score"] for r in results) / len(results)
        }
