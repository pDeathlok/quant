# 2026-06-08 复权价格口径重训与策略网格复测记录

## 背景

前端股票池中曾出现除权断点导致的 KDJ/价格滚动指标失真案例。为避免同类问题继续影响模型训练、标签构建和买卖点回测，本轮将价格衍生逻辑统一为连续复权价格口径：使用 Tushare 原始日线字段读取数据，再在本地通过 `adj_factor` 构建连续 OHLC，成交量和基础面字段不做价格复权。

## 已完成代码调整

- `src/quant/ml/label_maker.py`：B1 训练标签的未来收益、未来高低点、T+1/T+2 标签全部改为连续复权 OHLC。
- `scripts/research/analyze_b1_entry_exit_grid.py`：买卖点网格中的未来开高低收和信号日收盘价改为连续复权口径。
- `scripts/research/analyze_z_skill_entry_exit_backtest.py`：扩展策略的 KDJ、BBI、均线、形态判断改为连续复权价格。
- `scripts/research/build_training_data_parallel.py`：通用技术因子计算改为连续复权价格，避免 MA/KDJ/MACD 等因子受除权断点污染。
- `scripts/research/train_b1_tushare_models.py`：B1 入模样本筛选中的振幅、涨跌幅、BBI/MA60 判断改为连续复权价格。
- `scripts/research/train_z_skill_models_and_backtest.py`：多策略模型训练集支持进程并发，并将 close_pos 改为连续复权口径。
- `scripts/research/rebuild_strategy_signal_cache.py`：新增轻量脚本，只重建全市场策略规则信号缓存，不跑完整回测。
- `scripts/research/score_latest_strategy_models.py`：新增轻量脚本，每天基于最新规则候选构建当日特征并计算模型分。
- `src/quant/routine/pipeline.py`、`src/quant/webapp/services.py`：前端“获取最新数据”流程增加两步：重建策略规则信号、计算当日策略模型分，并展示进度。

## 日常刷新链路

点击前端“获取最新数据”后，流程变为：

1. 拉取 Tushare 最新日线数据。
2. 生成最新策略每日计划和 dashboard。
3. 重建全市场策略规则信号缓存。
4. 使用已训练模型计算最新交易日模型分。
5. 生成核心策略股票池和扩展策略股票池。
6. 写入 MySQL 策略股票池快照，供历史日期复盘查询。

这样每天选股时不再沿用旧日期模型分；即使完整模型重训还没执行，当日候选也会用当前已上线模型重新评分。

## 本轮数据与训练

- 最新交易日：2026-06-05。
- 规则缓存重建结果：
  - B1/B2/B3/SB1 家族候选：全量 2,740 行，最新日 282 行。
  - 扩展策略候选：全量 783,446 行，最新日 491 行。
- 当日模型分：最新候选特征 481 行，策略-股票得分 495 行，模型过滤通过 73 行。
- 多策略模型训练集：484,930 行，112 个特征。
- B1 专用模型训练集：185,679 行，daily_basic 匹配率 99.20%。

## 多策略模型表现

按股票代码随机切分 train/test，2025-01-01 起作为 OOT。

| 策略 | up5 OOT AUC | up8 OOT AUC | down3 OOT AUC | 说明 |
|---|---:|---:|---:|---|
| B2 | 0.4853 | 0.5043 | 0.4527 | OOT 辨别力弱，暂不宜单独依赖模型分 |
| BREATHING | 0.6304 | 0.6559 | 0.6720 | 相对稳定 |
| NANA | 0.6410 | 0.6596 | 0.6296 | 样本较少但可继续观察 |
| YIDONG_DILIAN | 0.6177 | 0.6418 | 0.6329 | 高频策略，模型有过滤价值 |
| KEY_K | 0.6117 | 0.6301 | 0.6936 | 下行风险模型较强 |
| GOLDEN_BOWL | 0.6199 | 0.6537 | 0.6424 | 相对稳定 |

多策略 OOT 中，PF 最高的组合集中在 KEY_K、NANA、YIDONG_DILIAN，但部分组合最大回撤仍很深。示例：

- KEY_K：`up5_ge_0.65`，T+1 开盘 0%-2%，`expiry_T7_close`，OOT 719 笔，均值 4.11%，胜率 72.18%，PF 3.16，最大回撤 -66.76%。
- NANA：`up5_ge_0.60_down3_le_0.50`，T+1 开盘 0%-2%，`fixed_tp4.0%_sl1.5%_close_T5`，OOT 84 笔，均值 1.61%，胜率 69.05%，PF 2.57，最大回撤 -14.18%。
- YIDONG_DILIAN：`up5_ge_0.60_down3_le_0.50`，T+1 开盘 0%-2%，`expiry_T7_close`，OOT 298 笔，均值 3.22%，胜率 59.73%，PF 2.59，最大回撤 -43.52%。

## B1 专用模型表现

| 模型 | Test AUC | OOT AUC |
|---|---:|---:|
| up5_es | 0.6918 | 0.6220 |
| up8_es | 0.7059 | 0.6514 |
| up10_es | 0.7107 | 0.6654 |
| down2_es | 0.6586 | 0.6318 |
| down3_es | 0.6784 | 0.6432 |

B1 非重叠持仓策略网格输出 290,400 行。高 PF 排名中，到期卖出收益较强但回撤偏深；更适合实操的是回撤受控组合。

### B1 OOT 高收益但高回撤示例

- `up8_ge_0.55_down3_le_0.40 + expiry_T12_close`：476 笔，均值 3.04%，胜率 56.51%，PF 2.51，最大回撤 -37.22%。
- `up8_ge_0.50_down3_le_0.40 + expiry_T12_close`：831 笔，均值 2.48%，胜率 55.48%，PF 2.18，最大回撤 -44.66%。
- `up8_ge_0.70_down3_le_0.50 + expiry_T7_close`：111 笔，均值 2.35%，胜率 56.76%，PF 2.09，最大回撤 -17.50%。

### B1 回撤受控候选

- `up8_ge_0.70_down3_le_0.50 + expiry_T7_close`：111 笔，均值 2.35%，胜率 56.76%，PF 2.09，最大回撤 -17.50%。样本偏少，但收益/回撤相对均衡。
- `up8_ge_0.55_down3_le_0.40 + fixed_tp10.0%_sl1.0%_T12`：556 笔，均值 0.83%，PF 1.62，最大回撤 -19.59%。止损触发率 75.36%，说明这是强风控、低胜率、靠少数大收益覆盖止损的组合。
- `up8_ge_0.65_down3_le_0.40 + fixed_tp6.0%_sl1.0%_T9/T12`：129 笔，均值约 0.60%，PF 1.45，最大回撤 -14.19%。样本少，适合观察。

## 当前结论

1. 复权价格口径下，B1 模型仍有可用辨别力，尤其 up8/up10 和 down3。
2. B1 到期卖出组合容易取得高均值和高 PF，但最大回撤仍偏深；前端默认排序不应只看 PF。
3. 若偏稳健，应优先选择模型分严格、样本不过少、最大回撤受控的组合，例如 `up8_ge_0.70_down3_le_0.50 + expiry_T7_close`。
4. 若必须带止损，B1 可考虑 `up8_ge_0.55_down3_le_0.40 + fixed_tp10.0%_sl1.0%_T12` 作为观察组合，但胜率低、止损频繁，实盘体验会比较磨人。
5. 多策略中 KEY_K、NANA、YIDONG_DILIAN 的模型过滤效果更值得继续挖；B2 当前 OOT 模型效果弱，暂不建议作为模型分主力策略。

## 输出文件

- B1 模型训练报告：`reports/b1/research/xgb_project_vars/training_report.json`
- B1 非重叠持仓网格：`reports/b1/research/xgb_project_vars_strategy/b1_xgb_entry_exit_grid_non_overlap_summary_20260608_092501.csv`
- B1 网格 Markdown：`reports/b1/research/xgb_project_vars_strategy/b1_xgb_entry_exit_grid_non_overlap_report_20260608_092501.md`
- 多策略模型训练报告：`reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_training_report.csv`
- 多策略模型网格：`reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_entry_exit_backtest.csv`
- 多策略模型 playbook：`reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_operational_playbook.csv`
- 最新日模型分：`reports/b1/research/xgb_project_vars_strategy/latest_z_skill_model_scored_candidates.parquet`

## 验证

- `PYTHONPATH=src pytest -q`：12 passed。
- 语法检查：`python -m compileall` 已通过。
