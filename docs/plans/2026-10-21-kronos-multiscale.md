# Kronos 多周期量价表征与可空仓策略实验

工作根目录：`/Users/didi/Project/quant`。此任务只生成研究产物，达到样本外准入条件才讨论生产替换。

## 固定协议

- 只使用 OHLC 和成交量。日线128、周线104、月线60；周/月只使用已经结束的自然周期，不使用之后才知道的整周/整月数据。
- 使用本地 Kronos TransformerBlock（RoPE、因果注意力、RMSNorm、SwiGLU）。以5维连续量价投影替代现有六维 tokenizer，避免引入成交额及来源截止日期未知的 tokenizer 权重。严格实验从随机初始化预训练；小规模先验证4层128维结构，再按资源扩展。
- 三周期共享编码器，周期身份向量和独立可训练 EOS；不输入日期、股票身份、财务或其他因子。EOS 必须经过 Transformer，并参与下一根量价预测的训练损失。三个 EOS 拼接后使用 MLP 产生收益、盈利概率、下行风险子分。
- 无监督预训练目标：各周期下一根 OHLCV；训练窗口及目标时间均不晚于2022-12-31。下游2024训练、2025选择、2026封存测试；标签跨边界必须 purge。
- 对原模型的增益用留出的时间段比较 baseline、sequence-only、baseline+sequence、简单规则。融合层训练使用原模型历史 OOT 分数；不得用当前生产模型回填历史分数。
- 原始股票行情的复权及交易限制口径需要审计。固定 next_close / 5-session 标签及往返20bps成本用于与左侧原指标一致的事件研究；另行输出允许空仓、同股不重叠、资金占用约束的组合回测。不得将事件均值当作组合收益。
- 简单规则在训练期用深度不超过3的树拟合历史模型分数，验证期选择叶子；每叶样本至少30，盈利概率 Wilson 下限至少0.5、净收益为正、PF>1。没有合格叶子则全现金。
- case只从训练/验证区间提炼；2026测试case只诊断，不用于当前版本再调参。输出输赢case、规则、覆盖率、胜率、PF、净收益、回撤、空仓比例和逐日资金曲线。
- 首轮按股票代码稳定哈希取64只（先抽样再检查覆盖），只作为工程和小样本验证；不得宣称全市场增益。全量模式不限制股票数。

## 文件及执行顺序

1. `/Users/didi/Project/quant/src/quant/research/kronos_multiscale.py`：因果多周期数据、EOS编码及训练函数。
2. `/Users/didi/Project/quant/src/quant/research/sequence_strategy_validation.py`：时间切分、规则蒸馏、现金与持仓回测、增益统计。
3. `/Users/didi/Project/quant/scripts/research/run_kronos_multiscale.py`：数据审计、预训练、embedding缓存、左右侧融合、报告。
4. `/Users/didi/Project/quant/configs/kronos_multiscale.yaml`：冻结配置。
5. `/Users/didi/Project/quant/tests/test_kronos_multiscale.py`：追加未来数据不改变历史输入、周月未完成周期排除、EOS梯度、时间purge、现金/资金占用测试。

验证命令：`.venv-kronos/bin/python -m pytest tests/test_kronos_multiscale.py -q`，预期全部通过。执行命令：`PYTHONPATH=src .venv-kronos/bin/python scripts/research/run_kronos_multiscale.py --config configs/kronos_multiscale.yaml`。输出到全新 `reports/research/kronos_multiscale_v1`，已有目录必须显式恢复或更换，防止覆盖研究。

## 运行前提与回滚

研究环境 `.venv-kronos` 安装 torch、numpy、pandas、pyarrow、scikit-learn、pyyaml、joblib、einops、huggingface-hub、safetensors。原实验 events/labels 曾按用户要求删除，使用保留的 OOT 预测键及现存历史行情重建标签，必须记录缺失覆盖及差异。生产模型与配置不改动；回滚只需停止本任务研究进程，研究产物保留供核查。没有自动提交/发布步骤。

## 执行结果与修订

首轮64只股票快照覆盖不足，扩样改为2022年12月原始股票池稳定哈希取256只，使用2010—2023原始分区和2024—2026原始日线串接连续价格指数。256只均有行情，222只有合格多周期候选，共39700条事件；26974条右侧收益标签与旧实验最大差2.45e-15。预训练5轮完成，输出384维embedding、左右侧各3个子分、融合模型及两轮浅树规则。测试期截至2026年8月，不是截至当前日期的全市场结果。

输出：`/Users/didi/Project/quant/reports/research/kronos_multiscale_expanded_v2`。左侧融合负增益，右侧改善但配对区块置信区间跨零；所有版本未达到高成功率研究准入条件。反馈规则没有验证期合格叶子，保留全现金结果。未修改生产模型。

9项时间边界、EOS训练、归一化、空仓、订单限制及增益检验测试通过，新增代码ruff检查通过。运行时源文件已保存到输出的source_snapshot，之后只做格式和未使用import清理；formatting_verification.json核实非import AST完全一致。保留原始协议哈希；格式化后的脚本对旧目录resume将按严格来源校验拒绝，应新建输出目录或使用已存档的原版本恢复。environment.lock.txt记录实际执行环境。
