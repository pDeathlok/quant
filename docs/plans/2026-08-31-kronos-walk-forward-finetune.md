# Kronos 日线滚动微调与样本外测试

## 目标

在不改变线上选股和交易逻辑的前提下，用本地 `002594` 日线验证 Kronos-small 的零样本预测能力，以及只微调 predictor 后相对零样本和随机游走基线的增量。

## 前置条件

- `models/research/kronos/source` 必须是官方 `shiyu-coder/Kronos` 提交 `67b630e67f6a18c9e9be918d9b4337c960db1e9a`。
- `models/research/kronos/pretrained/Kronos-small/model.safetensors` 的 SHA-256 必须是 `b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020`。
- `models/research/kronos/pretrained/Kronos-Tokenizer-base/model.safetensors` 的 SHA-256 必须是 `59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee`。
- `data/stocks_daily/002594.parquet` 必须含 `date/open/high/low/close/volume`，并覆盖至 `2026-05-26`。

## 实验合同

- 输入：90 个交易日 OHLCV 与成交额；本地 `turnover` 映射为 Kronos `amount`。
- 输出：未来 10 个交易日的 K 线；主评价字段为 `close`。
- 微调：冻结官方 tokenizer，只更新 Kronos-small predictor；每折均从相同预训练权重开始，禁止继承上一折参数。
- 训练样本：窗口必须完整落在训练期；验证窗口必须完整落在验证期。
- 测试样本：决策日和全部未来 10 个目标交易日必须落在测试期；90 日上下文可以来自测试期以前。
- 三折：`2015-2022/2023/2024`、`2016-2023/2024/2025`、`2017-2024/2025/2026YTD`，分别为训练/验证/测试。
- 对照：同一测试原点比较 `random_walk`、`pretrained` 和 `finetuned`。
- 指标：相对当前收盘价归一化的路径 MAE/RMSE、终点方向准确率、终点收益相关系数、平均路径收益相关系数、正信号扣费后收益。
- 结论边界：公开预训练权重可能见过历史市场数据，因此结果只称为“本地滚动样本外”，不能称为严格无预训练污染的历史回测。

## 实现文件

- `configs/kronos_byd_walk_forward.yaml`：模型版本、权重校验、训练参数和三折边界。
- `src/quant/research/kronos_walk_forward.py`：数据合同、窗口构造、权重审计、微调、推理、指标和报告。
- `scripts/research/run_kronos_walk_forward.py`：`audit`、`run` 命令与 smoke 限额。
- `tests/test_kronos_walk_forward.py`：数据边界、窗口无泄漏、测试原点与指标合同。

## 验证顺序

```bash
PYTHONPATH=src python -m pytest -q tests/test_kronos_walk_forward.py
models/research/kronos/venv/bin/ruff check src/quant/research/kronos_walk_forward.py scripts/research/run_kronos_walk_forward.py tests/test_kronos_walk_forward.py
PYTHONPATH=src models/research/kronos/venv/bin/python scripts/research/run_kronos_walk_forward.py audit --config configs/kronos_byd_walk_forward.yaml
PYTHONPATH=src models/research/kronos/venv/bin/python scripts/research/run_kronos_walk_forward.py run --config configs/kronos_byd_walk_forward.yaml --folds test_2024 --epochs 1 --max-train-samples 32 --max-eval-origins 2 --output-dir reports/research/kronos_byd_walk_forward_smoke
PYTHONPATH=src models/research/kronos/venv/bin/python scripts/research/run_kronos_walk_forward.py run --config configs/kronos_byd_walk_forward.yaml
```

验收时，三种模型变体必须拥有完全相同的测试键；所有训练与验证窗口结束日必须早于各折测试开始日；最终报告和 `manifest.json` 必须记录源码提交、权重 SHA-256、数据截止日、设备、随机种子和运行参数。

## 回滚

所有新增源代码和配置均为研究路径，不接入生产。删除上述四个跟踪文件即可回滚代码；`models/research/kronos/` 与 `reports/research/kronos_byd_walk_forward*` 已位于项目忽略的模型/报告目录，可单独移入归档或删除，不影响线上状态。
