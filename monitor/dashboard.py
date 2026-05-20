from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime


class QuantDashboard:
    def __init__(self, title: str = "Quantitative Trading Dashboard"):
        self.title = title
        self.metrics: Dict[str, float] = {}
        self.positions: Dict[str, int] = {}
        self.orders: List[Dict] = []
        self.equity_curve: List[float] = []

    def update_metrics(self, metrics: Dict[str, float]):
        self.metrics.update(metrics)

    def update_positions(self, positions: Dict[str, int]):
        self.positions = positions

    def add_order(self, order: Dict):
        self.orders.append({
            **order,
            "timestamp": datetime.now()
        })

    def update_equity(self, equity: float):
        self.equity_curve.append(equity)

    def get_summary(self) -> Dict:
        return {
            "metrics": self.metrics,
            "positions": self.positions,
            "pending_orders": len([o for o in self.orders if o.get("status") == "pending"]),
            "total_trades": len(self.orders),
            "current_equity": self.equity_curve[-1] if self.equity_curve else 0
        }

    def render_html(self) -> str:
        summary = self.get_summary()

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>{self.title}</h1>
            <h2>Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="metric-label">Current Equity</div>
                    <div class="metric-value">{summary['current_equity']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{summary['metrics'].get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value">{summary['metrics'].get('max_drawdown', 0):.2%}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{summary['total_trades']}</div>
                </div>
            </div>
            <h2>Positions</h2>
            <pre>{pd.DataFrame([summary['positions']]).to_string()}</pre>
        </body>
        </html>
        """
        return html
