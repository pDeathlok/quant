# 双低网格 + 趋势增强 Overlay 验证（2026-06-17）

## 实验目的

将上一轮“可转债趋势增强”加入现有双低网格策略，验证它是否能提升收益或风险收益比。

接入原则：

- 不改变双低网格的核心低位入场逻辑。
- 趋势信号只影响新开仓和网格加仓。
- 已有持仓仍按原网格退出规则管理，避免趋势信号把低位仓位过早卖掉。

## 代码变更

- `HoldingGridConfig` 新增可选趋势入场过滤参数：
  - `min_entry_trend_strength`
  - `min_entry_six_sword`
  - `min_entry_consecutive_six_sword`
  - `min_entry_return_5d/max_entry_return_5d`
  - `min_entry_return_1d/max_entry_return_1d`
  - `max_entry_price_position_60d`
  - `max_entry_market_median_double_low`
  - `min_entry_market_trend_20d`
  - `min_entry_market_trend_breadth`
- 网格回测预处理同时生成：
  - 双低网格低位特征：`price_position_252`、`drawdown_from_252_high`、`momentum_20d`
  - 趋势增强特征：`trend_strength`、`six_sword_daily`、`return_5d`、`market_trend_breadth`
- 修复一个历史数据边界：`close <= 0` 时不再计算网格加仓除法。

## Overlay 方案

本次测试 3 个网格底座：

- `core_market_scaled`
- `success_balanced_scaled`
- `return_core`

每个底座比较 4 个版本：

- baseline：原双低网格
- `trend_confirm`：个券趋势确认较强后才允许新开仓
- `trend_rebound`：允许低位反弹初期买入，过滤明显弱势
- `market_gate`：只加市场温度闸门，不加个券趋势强过滤

回测维度：

- 起点：2018、2020、2024
- 调仓：weekly、monthly
- 共 72 组

结果文件：

- `reports/convertible_bond/grid_trend_overlay/iteration_summary.csv`
- `reports/convertible_bond/grid_trend_overlay/paired_comparison.csv`

## 核心结果

按 overlay 汇总的平均改善：

| Overlay | 平均总收益差 | 平均最大回撤差 | 平均 Sharpe 差 | 总收益改善比例 | Sharpe 改善比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| market_gate | +5.20% | +1.10% | +0.150 | 83.3% | 83.3% |
| trend_rebound | +2.62% | +1.60% | +0.132 | 66.7% | 83.3% |
| trend_confirm | -1.32% | +3.15% | +0.134 | 50.0% | 72.2% |

最佳改善样本：

| 版本 | 起点 | 调仓 | 总收益 | 年化 | 最大回撤 | Sharpe | 相对 baseline 总收益差 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `return_core_trend_rebound` | 2020 | weekly | 36.40% | 5.20% | -15.97% | 0.754 | +24.79% |
| `return_core_trend_confirm` | 2020 | weekly | 34.36% | 4.95% | -17.22% | 0.675 | +22.75% |
| `return_core_market_gate` | 2020 | weekly | 31.21% | 4.54% | -15.39% | 0.688 | +19.60% |
| `core_market_scaled_market_gate` | 2018 | weekly | 37.08% | 4.00% | -12.17% | 0.672 | +14.63% |
| `core_market_scaled_market_gate` | 2020 | weekly | 34.77% | 5.00% | -15.41% | 0.814 | +10.55% |

## 结论

趋势增强加入双低网格是有效的，但最稳的形态不是“强趋势个券确认”，而是“市场闸门 + 温和反弹过滤”。

- `market_gate` 最稳定：多数样本提升收益和 Sharpe，逻辑上是避免在可转债整体过热或市场趋势转弱时继续开新仓。
- `trend_rebound` 对进攻型 `return_core` 最有价值：它保留低位网格买点，同时过滤仍在明显走弱的标的。
- `trend_confirm` 不适合作为通用规则：它能降低回撤，但经常错过网格策略最有收益弹性的低位建仓阶段。

当前推荐：

- 若追求稳健：优先使用 `core_market_scaled_market_gate`。
- 若追求收益：使用 `return_core_trend_rebound`，但接受 -16% 左右历史最大回撤。
- 不建议把上一轮独立趋势策略的严格条件原样套到双低网格里。

## 复跑命令

```bash
python scripts/research/iterate_convertible_bond_grid_trend_overlay.py
```
