# 开发、验证与提交

本文维护源码开发流程；生产日更见 [operations.md](operations.md)，研究入口见 [research_workflows.md](research_workflows.md)。所有命令从仓库根目录执行。

## 环境与运行产物

项目声明 Python >=3.9；CI 分别检查 Python 3.9 和 3.13 的选定回归集。完整应用与新增研究脚本建议使用 Python 3.13，不能把选定 CI 检查视为所有研究脚本都兼容 3.9。运行时依赖、可选的 `ml` / `kronos` / `data-fallback` 依赖见 `pyproject.toml`，CI 的受限安装环境见 [ci-runtime.md](ci-runtime.md)。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
test -f .env || cp .env.example .env
PYTHONPATH=src python scripts/run_webapp.py
```

源码仓库不含 `data/`、`models/`、`reports/` 和 `web/data/`。空数据目录可以启动服务，但不会自动获得正式候选、模型评分或冻结研究页面；生产配置引用的模型及其 manifest 必须另行恢复并核对哈希。不要通过放宽日期、缺失数据或模型身份检查来让空环境看起来已就绪。

## 本地回归

用独立数据根目录、空凭证和文件后端运行回归，避免测试读取正式行情和 SQL。测试本身仍须为其他产物路径使用临时目录或依赖替身；这组环境变量不是整个文件系统的隔离沙箱。

```bash
test_root=$(mktemp -d /tmp/quant-tests.XXXXXX)
mkdir -p "$test_root/raw" "$test_root/factors"
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  MARKET_DATA_BACKEND=file MARKET_DATA_SQL_URL='' TUSHARE_TOKEN='' \
  MARKET_DATA_ROOT="$test_root/raw" DAILY_FACTOR_ROOT="$test_root/factors" \
  python -m pytest -q --tb=short -rs
node --test tests/frontend/*.test.js
python -m ruff check --select E9,F63,F7,F82 src/quant scripts tests
python -m compileall -q src/quant scripts tests
git diff --check
```

Node 用于前端回归；`tests/frontend/*.test.js` 不需要启动 Web 服务，也不等同于真实浏览器验收。缺少 PyTorch 等可选依赖时应保留并报告 skip 原因。完整 Ruff 规则仍可用于局部清理，但上述命令针对语法错误和未定义名称，不宣称所有历史代码均满足格式规范。

涉及因子注册表时运行 `PYTHONPATH=src python scripts/generate_factor_catalog.py` 并核对差异。涉及刷新、模型发布或页面交互时，优先覆盖输入日期、候选覆盖、版本身份、失败保留旧产物和历史快照隔离。

## 文件归属与提交

| 内容 | 位置 | Git 约定 |
| --- | --- | --- |
| 可复用实现与测试 | `src/quant/`、`tests/` | 提交 |
| 研究命令 | `scripts/research/` | 保留原入口，按研究流程建立索引 |
| 生效策略 / 发布与回滚 / 研究参数 | `configs/strategies/`、`configs/releases/`、`configs/research/` | 提交；目录名本身不能证明已发布 |
| 评分校准、报告合同、服务模板 | `config/` | 提交；暂不与 `configs/` 合并，以免破坏路径契约 |
| 当前手册 / 历史证据 | `docs/`、`docs/plans/`、`docs/reviews/`、`docs/research/` | 提交；带日期文档保留当时结论 |
| 自选池人工操作留痕 | `outputs/YYYY-MM-DD/` | 经内容检查后按既有归档方式提交，不代替实时自选池 |
| 数据、模型、运行报告、页面快照 | `data/`、`models/`、`reports/`、`web/data/` | 不提交，单独备份 |
| 临时依赖、下载资料、工具缓存 | `.tmp/`、`.run/`、`.pytest_cache/`、`.ruff_cache/` | 不提交；有保存价值的正文先归档 |

历史已跟踪的 `.tmp/verify_disclosure_*.py` 不因忽略规则而删除；后续修改仍会正常显示在 Git 中。新增工具应放入 `scripts/`，不要继续把源码放在临时目录。

提交前检查 `git status --short`、暂存差异和文件清单，避免加入 `.env`、真实凭证或大型运行产物。按独立研究、生产集成、记录归档、文档维护分批提交；共享依赖及调用方应保持同一批次可理解。使用 `feat`、`fix`、`perf`、`docs`、`chore` 等提交类型，在正文记录验证范围及未执行项。提交只保存版本，不代表模型已激活或服务已部署。
