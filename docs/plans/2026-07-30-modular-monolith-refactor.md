# 模块化单体第一批重构实施计划

## Goal

在不改变 FastAPI 路由、页面行为、策略输出和快照格式的前提下，建立可持续扩展的应用层边界，消除 `quant.routine` 对 `quant.webapp` 的依赖，把工作区并发刷新改为显式依赖，并拆出前端公共基础模块。

## Pre-conditions

- [x] `git status --short` 已确认工作区有 9 个用户未提交文件；本批次保留这些改动，不执行重置、还原或覆盖。
- [x] `python -m pytest -q` 已通过，基线为 `322 passed in 16.19s`。
- [x] 当前环境没有安装 `ruff`；静态检查改用 `python -m compileall` 和架构测试，最终仍保留 `ruff` 命令供完整开发环境执行。
- [x] 本批次不执行数据库迁移，不修改 MySQL、Parquet、JSON 快照 schema，不访问外部行情源。

## Steps

### Step 1 — 建立应用层工作区刷新契约

**Files:**

- `/Users/didi/Project/quant/src/quant/application/__init__.py`
- `/Users/didi/Project/quant/src/quant/application/workspace_refresh.py`
- `/Users/didi/Project/quant/tests/test_workspace_refresh.py`

新增不可变 `WorkspaceRefreshOperations` 数据类，显式声明：

```python
@dataclass(frozen=True)
class WorkspaceRefreshOperations:
    latest_signal_date: Callable[[], str | None]
    refresh_chan: Callable[..., dict[str, Any]]
    refresh_long: Callable[..., dict[str, Any]]
    refresh_convertible_bonds: Callable[..., dict[str, Any]]
    refresh_allotments: Callable[..., dict[str, Any]]
    refresh_byd: Callable[..., dict[str, Any]]
    refresh_similar_patterns: Callable[[], dict[str, Any]]
```

新增：

```python
def refresh_daily_workspaces(
    operations: WorkspaceRefreshOperations,
    *,
    max_workers: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Refresh six independent workspaces with bounded concurrency."""
```

测试断言六个工作区可以并行完成，并断言单个工作区失败只生成该工作区的失败结果。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_workspace_refresh.py` → `2 passed`。

### Step 2 — 打断 routine/webapp 反向依赖

**Files:**

- `/Users/didi/Project/quant/src/quant/routine/pipeline.py`
- `/Users/didi/Project/quant/tests/test_pipeline.py`
- `/Users/didi/Project/quant/tests/test_architecture_boundaries.py`

删除 `pipeline.refresh_daily_web_workspaces()` 内部对 `quant.webapp.services` 的运行时导入。`run_daily_pipeline` 新增显式可选依赖：

```python
def run_daily_pipeline(
    skip_data: bool = True,
    skip_backtest: bool = False,
    *,
    workspace_operations: WorkspaceRefreshOperations | None = None,
) -> dict[str, Any]:
    """Run the legacy diagnostic pipeline with optional workspace refresh."""
```

传入 `workspace_operations` 时调用应用层工作区刷新；未传入时写入：

```python
{
    "status": "skipped",
    "reason": "workspace operations were not provided; canonical production refresh runs through web-refresh",
}
```

架构测试使用 Python AST 扫描 `src/quant/routine/**/*.py`，拒绝任何 `quant.webapp` 导入；同时拒绝 `src/quant/**/*.py` 对 `scripts` 包的普通 Python 导入。字符串形式的兼容命令路径暂时允许，并由后续生产任务迁移批次消除。

**Verify:**

- `PYTHONPATH=src python -m pytest -q tests/test_pipeline.py tests/test_architecture_boundaries.py`
- `rg -n 'from quant\.webapp|import quant\.webapp' src/quant/routine` → 无结果。

### Step 3 — 统一刷新状态契约

**Files:**

- `/Users/didi/Project/quant/src/quant/application/refresh_contracts.py`
- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/tests/test_refresh_contracts.py`

把刷新范围、步骤定义、范围标准化和步骤列表构造迁入应用层：

```python
def normalize_refresh_scope(scope: str | None) -> str:
    """Return a supported refresh scope or all."""


def build_progress_steps(scope: str | None = None) -> list[dict[str, Any]]:
    """Build a fresh progress list for the selected scope."""
```

`webapp.services` 保留 `_normalize_refresh_scope` 和 `_progress_steps` 兼容别名，现有 API 测试和外部调用不变。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_refresh_contracts.py tests/test_webapp_api.py` → 全部通过。

### Step 4 — 拆出前端公共模块

**Files:**

- `/Users/didi/Project/quant/web/core/formatters.js`
- `/Users/didi/Project/quant/web/core/api-client.js`
- `/Users/didi/Project/quant/web/app.js`
- `/Users/didi/Project/quant/web/index.html`
- `/Users/didi/Project/quant/tests/test_web_frontend.py`

`formatters.js` 导出百分比、比率、数字、价格、权重、金额、区间和 HTML 转义函数。`api-client.js` 导出：

```javascript
export function createApiClient(apiBase) {
  return async function fetchJson(path, options = {}) {
    // 保持现有超时、AbortController、错误详情和 no-store 行为。
  };
}
```

`app.js` 通过 ES Modules 导入这些函数；`index.html` 将入口脚本改为 `type="module"`。前端测试读取入口与公共模块的组合源码，继续验证当前页面契约。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_web_frontend.py` → 全部通过。

### Step 5 — 把项目路径提升到 core 层

**Files:**

- `/Users/didi/Project/quant/src/quant/core/__init__.py`
- `/Users/didi/Project/quant/src/quant/core/paths.py`
- `/Users/didi/Project/quant/src/quant/config/settings.py`
- `/Users/didi/Project/quant/src/quant/routine/paths.py`
- `/Users/didi/Project/quant/tests/test_core_paths.py`

新增不可变 `ProjectPaths`，从仓库根目录派生 `data`、`cache`、`logs`、`reports`、`models`、`configs` 和 `web` 路径；目录创建只允许通过显式 `ensure_runtime_directories()` 调用。`routine.paths` 保留当前常量作为兼容导出，但根路径来源改为 `quant.core.paths.PROJECT_ROOT`。旧 `Settings` 使用同一个根路径，不再在构造时写文件系统。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_core_paths.py tests/test_routine_cli.py` → 全部通过。

### Step 6 — 迁移首个完整工作区切片

**Files:**

- `/Users/didi/Project/quant/src/quant/application/workspaces/__init__.py`
- `/Users/didi/Project/quant/src/quant/application/workspaces/byd.py`
- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/tests/test_byd_workspace.py`

把 BYD 日线计划生成、行情加载和数据标准化迁入 `application/workspaces/byd.py`。快照读取、快照写入、日线加载和验证数据加载通过显式函数参数传入。`webapp.services.get_byd_daily_strategy` 保留原签名，作为兼容代理调用新的应用用例。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_byd_workspace.py tests/test_webapp_api.py -k 'byd or health'` → 全部通过。

### Step 7 — 迁移首个生产研究任务

**Files:**

- `/Users/didi/Project/quant/src/quant/research/b1_backtest.py`
- `/Users/didi/Project/quant/src/quant/research/b1_formal_combos.py`
- `/Users/didi/Project/quant/scripts/research/analyze_b1_entry_exit_grid.py`
- `/Users/didi/Project/quant/scripts/research/analyze_b1_formal_combos.py`
- `/Users/didi/Project/quant/src/quant/routine/pipeline.py`
- `/Users/didi/Project/quant/tests/test_research_market_backtest.py`
- `/Users/didi/Project/quant/tests/test_pipeline.py`

把 `ExitRule`、未来价格拼接、退出模拟和收益汇总迁入 `quant.research.b1_backtest`，让研究网格和正式组合校验共同复用。把正式组合校验主体迁入 `quant.research.b1_formal_combos`，原脚本只保留兼容导出和 `main()` 调用。生产流水线改为 `python -m quant.research.b1_formal_combos`，不再执行该研究脚本路径。

**Verify:**

- `PYTHONPATH=src:scripts/research python -m pytest -q tests/test_research_market_backtest.py tests/test_pipeline.py`
- `PYTHONPATH=src python -m quant.research.b1_formal_combos --help` 不作为验证命令，因为正式任务没有 CLI 参数且会运行真实回测；只验证模块可导入和命令构造。

### Step 8 — 拆出工作区快照仓储

**Files:**

- `/Users/didi/Project/quant/src/quant/infrastructure/__init__.py`
- `/Users/didi/Project/quant/src/quant/infrastructure/workspace_snapshots.py`
- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/tests/test_workspace_snapshots.py`

新增 `WorkspaceSnapshotRepository`，集中负责参数键、快照键、日期标准化、文件系统最近历史快照读取、原子写入和 SQL 回退。行情存储通过 `store_factory` 显式注入，仓储本身不读取项目环境。`webapp.services` 保留全部 `_workspace_*` 兼容函数，并在每次调用时按当前模块常量构造仓储，保证现有 monkeypatch 测试与调用方不变。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_workspace_snapshots.py tests/test_webapp_api.py -k 'workspace_snapshot or health'` → 全部通过。

### Step 9 — 更新架构文档并执行完整回归

**Files:**

- `/Users/didi/Project/quant/docs/architecture.md`
- `/Users/didi/Project/quant/docs/project_structure_and_storage.md`
- `/Users/didi/Project/quant/README.md`

文档增加模块化单体依赖方向、应用层职责、研究脚本迁移规则和下一批任务清单。

**Verify:**

- `PYTHONPATH=src python -m pytest -q` → 不少于 `326 passed`。
- `PYTHONPATH=src python -m compileall -q src tests` → 退出码 `0`。
- `rg -n 'from quant\.webapp|import quant\.webapp' src/quant/routine` → 无结果。
- `git diff --check` → 无空白错误。

## Rollback

- 本批次没有数据库或外部系统写入，不需要数据回滚。
- 每一步均保留现有公开函数或兼容别名；若某一步失败，只使用反向补丁删除该步新增模块并恢复对应导入，不执行 `git reset --hard`、`git checkout --` 或覆盖用户未提交改动。
- 前端模块化失败时，可把 `index.html` 入口恢复为普通 `app.js`，并将公共函数原样内联回入口；页面和 API 数据不受影响。

## Execution result

- 2026-07-30：Steps 1–9 已完成。
- `PYTHONPATH=src python -m pytest -q` → `349 passed in 18.28s`。
- `PYTHONPATH=src python -m compileall -q src tests` → 退出码 `0`。
- routine/webapp 反向依赖扫描 → 无匹配。
- `git diff --check` → 无空白错误。
- 当前环境未安装 Ruff，因此未执行 `ruff check src tests`。

## Continuation goal

继续缩小 Web 服务层，把可转债与配债股迁成可独立测试的应用工作区；为静态资源增加压缩、缓存与模块预加载，并使用真实浏览器验证七个工作区的功能、响应式布局和运行时健康，最终在独立分支提交并推送远端。

## Continuation pre-conditions

- [x] 当前分支已从 `main` 切换为 `codex/modular-architecture-frontend`，保留现有工作区改动。
- [x] `origin` 指向 `git@github.com:pDeathlok/quant.git`，基线为 `8760969`。
- [x] `PYTHONPATH=src python -m pytest -q` 基线为 `349 passed in 18.28s`。
- [x] `http://127.0.0.1:8088/api/health` 返回 `{"status":"ok","service":"quant-webapp"}`。
- [x] 真实浏览器基线：标题为“策略工作台”、默认激活 `similarPage`、控制台无错误。
- [x] 本批次不触发真实行情刷新，不修改数据库 schema，不写入外部业务系统。

## Continuation steps

### Step 10 — 拆出可转债与配债股应用工作区

**Files:**

- `/Users/didi/Project/quant/src/quant/application/workspaces/convertible_bonds.py`
- `/Users/didi/Project/quant/src/quant/application/workspaces/__init__.py`
- `/Users/didi/Project/quant/src/quant/webapp/services.py`
- `/Users/didi/Project/quant/tests/test_convertible_bond_workspace.py`
- `/Users/didi/Project/quant/tests/test_webapp_api.py`

新增不可变依赖契约：

```python
@dataclass(frozen=True)
class ConvertibleBondGridDependencies:
    read_snapshot: Callable[..., dict[str, Any] | None]
    read_legacy_snapshot: Callable[[], dict[str, Any] | None]
    write_filesystem_snapshot: Callable[..., None]
    write_snapshot: Callable[..., None]
    refresh_daily: Callable[..., dict[str, Any]]
    build_plan: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class ConvertibleBondAllotmentDependencies:
    read_snapshot: Callable[..., dict[str, Any] | None]
    read_daily_cache: Callable[[], dict[str, Any] | None]
    write_daily_cache: Callable[[dict[str, Any]], None]
    write_snapshot: Callable[..., None]
    build_payload: Callable[..., dict[str, Any]]
    is_daily_current: Callable[[dict[str, Any]], bool]
```

应用层提供 `build_convertible_bond_grid_workspace()`、`build_convertible_bond_allotment_workspace()` 和 `evaluate_convertible_bond_allotment_quality()`；`webapp.services` 保留现有函数名，只在调用时组装依赖，确保 monkeypatch 和 API 兼容。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_convertible_bond_workspace.py tests/test_webapp_api.py -k 'convertible_bond or allotment or health'` → 全部通过。

### Step 11 — 优化静态资源传输

**Files:**

- `/Users/didi/Project/quant/src/quant/webapp/static_delivery.py`
- `/Users/didi/Project/quant/src/quant/webapp/app.py`
- `/Users/didi/Project/quant/web/index.html`
- `/Users/didi/Project/quant/tests/test_web_frontend.py`

新增纯 ASGI `StaticAssetCacheMiddleware`：

```python
class StaticAssetCacheMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = str(scope.get("path") or "")

        async def send_with_cache(message: Message) -> None:
            if message["type"] == "http.response.start" and int(message["status"]) == 200:
                headers = MutableHeaders(scope=message)
                if path == "/" or path.endswith(".html"):
                    headers["Cache-Control"] = "no-cache"
                elif path.endswith((".js", ".css")):
                    headers["Cache-Control"] = "public, max-age=3600"
            await send(message)

        await self.app(scope, receive, send_with_cache)
```

FastAPI 增加 `GZipMiddleware(minimum_size=1024, compresslevel=6)`，`index.html` 为两个 core 模块增加与实际 import URL 一致的 `modulepreload`。测试断言 HTML 不缓存、JS/CSS 缓存一小时、支持 gzip，并验证模块预加载路径。

**Verify:** `PYTHONPATH=src python -m pytest -q tests/test_web_frontend.py` → 全部通过；使用带响应体的 `curl --compressed -D - -o /dev/null http://127.0.0.1:8088/app.js`，响应包含 `content-encoding: gzip` 和 `cache-control: public, max-age=3600`。

### Step 12 — 完整自动化与真实浏览器验证

**Files:** 不新增仓库内测试产物；截图仅保存到 `/private/tmp/quant-frontend-qa/`。

自动化验证：

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m compileall -q src tests
git diff --check
```

真实浏览器验证路径：

```text
/ -> 自选池数据可见
自选池 -> 短线策略 -> 股票池或明确空态可见
短线策略 -> 缠论策略 -> 候选或明确空态可见
缠论策略 -> 长线策略 -> 策略卡和股票池可见
长线策略 -> 可转债策略 -> 候选或明确空态可见
可转债策略 -> 配债股 -> 质量指标和记录表可见
配债股 -> BYD 做T -> 持仓输入与日线计划可见
```

桌面视口使用浏览器默认尺寸，移动视口使用 `390×844`；每个视口检查页面非空、无错误覆盖层、无相关 console error/warn，并保存截图证据。

### Step 13 — 完善文档

**Files:**

- `/Users/didi/Project/quant/README.md`
- `/Users/didi/Project/quant/docs/architecture.md`
- `/Users/didi/Project/quant/docs/project_structure_and_storage.md`
- `/Users/didi/Project/quant/docs/operations.md`
- `/Users/didi/Project/quant/docs/plans/2026-07-30-modular-monolith-refactor.md`

记录可转债工作区边界、静态资源压缩/缓存策略、真实浏览器验收结果、验证命令和剩余演进项。

**Verify:** `rg -n 'ConvertibleBond|gzip|Cache-Control|真实浏览器|358 passed' README.md docs` → 命中文档记录。

### Step 14 — 提交并推送远端分支

**Commit:**

```text
refactor(architecture): modularize workspaces and frontend delivery

Why:
- web services mixed delivery, orchestration and persistence concerns
- large uncompressed static assets slowed the strategy workspace

What:
- add application and infrastructure boundaries with compatibility facades
- share B1 research/backtest implementations
- add gzip, cache policy and module preload for frontend assets
- add architecture, API, frontend and real-browser regression coverage
```

执行：

```bash
git add README.md docs scripts src tests web
git commit
git push -u origin codex/modular-architecture-frontend
```

推送前再次确认 `.env`、`data/`、`models/`、`reports/` 和截图未进入暂存区。

## Continuation rollback

- Step 10 或 Step 11 验证失败时，只使用反向补丁恢复对应兼容代理和新增模块，不重置工作区。
- 真实浏览器验证发现回归时，保留分支且不提交，先修复并重新执行 Step 12。
- 推送后如需撤销，保留远端分支审计历史并新增 `git revert <commit-sha>`；不对共享远端执行强制推送。

## Continuation execution result

- 2026-07-30：Steps 10–13 已完成，Step 14 在同一批验证通过后执行。
- 可转债/配债股定向测试：`26 passed, 70 deselected in 9.62s`。
- 前端静态与响应式测试：`33 passed in 1.78s`。
- 全量自动化：`PYTHONPATH=src python -m pytest -q` → `358 passed in 14.80s`。
- 编译检查：`PYTHONPATH=src python -m compileall -q src tests scripts` → 退出码 `0`。
- 架构扫描：`application` 未导入 `infrastructure`、`webapp` 或 `routine`；`git diff --check` 无错误。
- 真实服务：健康检查成功，`app.js` 与 `styles.css` 的 GET 响应均包含 gzip、`Vary: Accept-Encoding` 和一小时缓存。
- 本机传输量：`app.js` 从 173,407 B 降至 41,014 B（减少 76.3%）；`styles.css` 从 84,839 B 降至 14,514 B（减少 82.9%）。
- 真实浏览器桌面端与 `390×844` 移动端均遍历 7 个工作区；短线搜索交互正常，页面无错误覆盖层和 console 日志。
- 移动端修复了主网格最小宽度、缠论指标单列与刷新工具栏换行问题；活动页和文档均无横向溢出，刷新按钮为 44px 高且文字不换行。
- 未触发真实行情刷新、交易操作或外部数据写入；这部分仍按生产运维窗口独立验收。
