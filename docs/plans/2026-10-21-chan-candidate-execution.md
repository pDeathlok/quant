# 缠论收盘候选与开盘执行分离

目标：当天收盘结构和模型达标即进入候选，次日开盘另行核验；无候选时保留所选日期。历史统计继续只计入已知开盘且跳空不超过 3% 的交易。

修改顺序：
1. `/Users/didi/Project/quant/tests/test_chan_model.py`：覆盖开盘未知、3%/6% 边界、最新日期空池、历史统计排除待核验和超限交易。
2. `/Users/didi/Project/quant/src/quant/strategies/custom/chan_model.py`：新增独立收盘信号与执行状态；候选按收盘信号筛选，历史执行信号保留原门槛；默认取全部评分数据的最新日期。
3. `/Users/didi/Project/quant/src/quant/webapp/services.py`、`/Users/didi/Project/quant/scripts/research/build_chan_model_strategy.py`：以成功刷新记录的 processed_through 确定最新日期；空池日期不回退；输出状态并淘汰旧版本缓存。
4. `/Users/didi/Project/quant/web/app.js`：表格和详情展示执行状态、未知开盘提示。
5. `/Users/didi/Project/quant/docs/strategies/chan_daily_strategy.md`：记录新字段与历史统计口径。

验证：`PYTHONPATH=src python3 -m pytest -q tests/test_chan_model.py tests/test_chan_live_refresh_reference.py tests/test_chan_strategy_payload.py` 应全部通过；`node --check web/app.js` 应成功。用当前评分文件重新生成候选，核对 2026-10-21 海欣食品进入待开盘核验池。

仓库已有其他未提交修改；只编辑上述相关区域，不提交、不回滚其他工作。变更可通过针对上述文件的新补丁撤销，生成报告可从原评分文件重建。
