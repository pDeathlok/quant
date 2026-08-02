# A 股执行策略实施计划

## Goal

把 A 股佣金、印花税、过户费、最低佣金、100 股整手、T+1 和成交量参与率变成一个经过校验、可覆盖、会写入回测元数据的执行策略，并保持 `BacktestEngine` 现有构造方式兼容。

## Reuse and data-source boundary

- 先复用项目自有 `MarketDataStore`、`TushareDataFetcher`、字段标准化和缓存能力，不新建第二套数据获取链路。
- 行情与证券状态以 Tushare 为第一来源；只有 Tushare 不提供等价字段时才新增其他来源适配器，并记录来源与降级原因。
- `AShareExecutionConfig` 保持引擎无关；akquant 只在 `BacktestEngine` 边界接收翻译后的参数，不作为市场规则事实来源。

## Official basis checked on 2026-07-31

- 财政部、税务总局公告 2023 年第 39 号仍为全文有效，证券交易印花税自 2023-08-28 起减半征收：<https://shanghai.chinatax.gov.cn/tax/zcfw/zcfgk/yhs/202308/t468451.html>。
- 中国结算 2025 年上海、深圳市场收费表均列示 A 股交易过户费按成交金额 `0.01‰` 向买卖双方收取：<https://www.chinaclear.cn/zdjs/fbzyls/202506/9d22b74d9f2e40edb67b44d1f6596f18/files/%E4%B8%8A%E6%B5%B7%E5%B8%82%E5%9C%BA%E8%AF%81%E5%88%B8%E7%99%BB%E8%AE%B0%E7%BB%93%E7%AE%97%E4%B8%9A%E5%8A%A1%E6%94%B6%E8%B4%B9%E5%8F%8A%E4%BB%A3%E6%94%B6%E7%A8%8E%E8%B4%B9%E4%B8%80%E8%A7%88%E8%A1%A8.pdf>、<https://www.chinaclear.cn/zdjs/fbzyls/202506/ab6384ba25514554a7eceaee3e521032/files/%E6%B7%B1%E5%9C%B3%E5%B8%82%E5%9C%BA%E8%AF%81%E5%88%B8%E7%99%BB%E8%AE%B0%E7%BB%93%E7%AE%97%E4%B8%9A%E5%8A%A1%E6%94%B6%E8%B4%B9%E5%8F%8A%E4%BB%A3%E6%94%B6%E7%A8%8E%E8%B4%B9%E4%B8%80%E8%A7%88%E8%A1%A8.pdf>。
- 上交所和深交所规则要求普通股票竞价买入以 100 股或其整数倍申报；零股卖出有单独规则：<https://www.sse.com.cn/lawandrules/guide/stock/jyglywznylc/tz/c/c_20230209_5716007.shtml>、<https://www.szse.cn/www/investor/knowledge/t20230306_599093.html>。
- 上交所说明内地股票当日买入不能 T+0 卖出：<https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20150912_3988692.shtml>。

佣金和最低佣金由券商协议决定，不是统一官方费率。因此项目默认值 `commission_rate=0.0003`、`min_commission=5.0` 仅是保守研究假设，必须允许策略或账户配置覆盖。

## Steps

### Step 1 — 定义经过校验的执行策略

**File:** `/Users/didi/Project/quant/src/quant/backtest/execution.py`

```python
@dataclass(frozen=True)
class AShareExecutionConfig:
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    min_commission: float = 5.0
    slippage: float = 0.0
    volume_limit_pct: float = 0.10
    lot_size: int = 100
    t_plus_one: bool = True

    def __post_init__(self) -> None:
        for name in (
            "commission_rate",
            "stamp_tax_rate",
            "transfer_fee_rate",
            "min_commission",
            "slippage",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < self.volume_limit_pct <= 1.0:
            raise ValueError("volume_limit_pct must be in (0, 1]")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
```

**Verify:** `PYTHONPATH=src pytest -q tests/test_a_share_execution.py` → 默认值、覆盖值和非法输入测试通过。

### Step 2 — 接入引擎且保持旧调用兼容

**File:** `/Users/didi/Project/quant/src/quant/backtest/engine.py`

构造函数新增 `execution_config: Optional[AShareExecutionConfig] = None`。配置为空时继续使用原 `commission_rate` 和 `slippage` 参数；配置存在时仅在外部引擎边界将 `to_dict()` 翻译并合并到 `akquant.run_backtest()` 参数，同时把完整配置写入 `BacktestArtifacts.metadata["execution_policy"]`。

**File:** `/Users/didi/Project/quant/main.py`

CLI 明确使用 `AShareExecutionConfig()`，使 README 中的正式本地 A 股示例默认包含税费、T+1、整手和成交量参与率。

**Files:**

- `/Users/didi/Project/quant/src/quant/backtest/__init__.py`
- `/Users/didi/Project/quant/src/quant/__init__.py`

导出 `AShareExecutionConfig`。

**Verify:** `PYTHONPATH=src pytest -q tests/test_a_share_execution.py tests/test_backtest_engine_contract.py tests/test_main_cli.py` → kwargs 透传、元数据、兼容调用和 CLI 默认策略测试通过。

### Step 3 — 文档化范围和剩余约束

**Files:**

- `/Users/didi/Project/quant/README.md`
- `/Users/didi/Project/quant/docs/architecture.md`

明确当前策略已覆盖费用、T+1、整手和成交量参与率；涨跌停、停牌、风险警示板块和上市初期无涨跌幅限制仍需证券状态与每日涨跌停价数据，不能宣称已经覆盖。

**Verify:** `rg -n 'AShareExecutionConfig|涨跌停|成交量参与率' README.md docs/architecture.md` → 范围和缺口均有记录。

### Step 4 — 完整验证

```bash
PYTHONPATH=src pytest -q tests/test_a_share_execution.py tests/test_backtest_engine_contract.py tests/test_main_cli.py tests/test_quant.py
PYTHONPATH=src pytest -q
PYTHONPATH=src python -m compileall -q src tests
```

预期：定向测试和全量测试无失败，compileall 返回 0。

## Rollback

本轮没有迁移或外部写入。回滚时删除 `src/quant/backtest/execution.py` 和对应测试，恢复 `engine.py`、`main.py`、导出文件及文档的本轮局部改动，不覆盖此前稳定回测契约或用户已有 Web 修改。
