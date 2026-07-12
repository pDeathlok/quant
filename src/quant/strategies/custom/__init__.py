"""
策略模块 - 用户自定义策略
"""

from .b1 import B1Strategy
from .template import TemplateStrategy
from .right_side_bottom import RightSideBottomFishingStrategy
from .triple_volume_breakout import (
    add_triple_volume_breakout_signals,
    add_triple_volume_research_signals,
    add_triple_volume_strategy_pool_signals,
)
from .vegas_tunnel import add_vegas_tunnel_signals
from .chan_daily import add_chan_daily_signals, summarize_chan_daily
from .chan_model import (
    add_chan_model_strategy_columns,
    select_chan_model_candidates,
    summarize_chan_model_strategy,
)

__all__ = [
    "B1Strategy",
    "TemplateStrategy",
    "RightSideBottomFishingStrategy",
    "add_triple_volume_breakout_signals",
    "add_triple_volume_research_signals",
    "add_triple_volume_strategy_pool_signals",
    "add_vegas_tunnel_signals",
    "add_chan_daily_signals",
    "summarize_chan_daily",
    "add_chan_model_strategy_columns",
    "select_chan_model_candidates",
    "summarize_chan_model_strategy",
]
