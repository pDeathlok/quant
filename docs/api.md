# API 参考

服务默认监听 `http://127.0.0.1:8088`，所有业务接口使用 `/api` 前缀。启动服务后可访问：

- Swagger UI：`http://127.0.0.1:8088/docs`
- OpenAPI JSON：`http://127.0.0.1:8088/openapi.json`
- 健康检查：`GET /api/health`

本项目当前面向本机研究工作台，没有认证层，不应直接暴露到公网。

## 工作区接口

| 方法 | 路径 | 主要参数 | 用途 |
| --- | --- | --- | --- |
| `GET` | `/api/selector/stocks` | `strategies`、`signal_date`、`include_extended`、`refresh` | 短线股票池 |
| `GET` | `/api/selector/calendar` | `start`、`end` | 可复盘交易日 |
| `GET` | `/api/b1/plan` | `refresh`、`signal_date` | B1 每日计划 |
| `POST` | `/api/b1/plan/refresh` | JSON `signal_date` | 强制刷新 B1 计划 |
| `GET` | `/api/chan/strategy-plan` | `top_n`、`signal_date`、`refresh` | 缠论候选 |
| `POST` | `/api/chan/strategy-plan/refresh` | `top_n`、`signal_date` | 强制刷新缠论候选 |
| `GET` | `/api/long/stock-pool` | `variant`、`signal_date`、`refresh` | 长线股票池 |
| `GET` | `/api/long/blood-chip` | `signal_date`、`refresh` | 带血筹长线计划 |
| `GET` | `/api/convertible-bonds/plan` | `trade_date`、`limit`、`refresh` | 可转债计划 |
| `GET` | `/api/convertible-bonds/allotments` | `limit`、`include_listed_days`、`refresh`、`stage_scope` | 配债股跟踪 |
| `GET` | `/api/byd/daily-plan` | 持仓、成本和当日做T参数 | BYD 日线计划 |
| `GET` | `/api/transition/plan` | `strategy=return\|win`、`refresh` | 多状态右侧收益型或胜率型 |
| `PUT` | `/api/transition/account/{strategy}` | `revision`、`cash`、`positions` | 保存用户提供的过渡账户状态；版本冲突返回 409 |
| `GET` | `/api/similar-patterns/analysis` | `refresh` | 相似走势分析 |
| `POST` | `/api/similar-patterns/analysis/refresh` | 无 | 强制刷新相似走势 |

日期参数中，股票和策略信号日期使用 `YYYY-MM-DD`；可转债返回的 `trade_date` 为 `YYYYMMDD`，验收时先统一格式再比较。

## 相似走势自选池

```bash
# 查看
curl http://127.0.0.1:8088/api/similar-patterns/watchlist

# 加入
curl -X POST http://127.0.0.1:8088/api/similar-patterns/watchlist \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"002594.SZ"}'

# 删除
curl -X DELETE http://127.0.0.1:8088/api/similar-patterns/watchlist/002594.SZ
```

## 后台刷新

`POST /api/selector/refresh-latest` 启动后台任务；同一时刻只允许一个刷新任务运行。

```bash
curl -X POST http://127.0.0.1:8088/api/selector/refresh-latest \
  -H 'Content-Type: application/json' \
  -d '{"scope":"all"}'

curl http://127.0.0.1:8088/api/selector/refresh-latest/status
```

指定历史补跑日期时，在请求体传 `"trade_date":"YYYY-MM-DD"`；例行运行优先使用[统一脚本](operations.md#手工刷新页面工作区)，由脚本检查交易日并处理重试。可用作用域：

| `scope` | 页面 |
| --- | --- |
| `all` | 7 个日常策略工作区及当天股指开仓预案 |
| `short` | 短线策略 |
| `chan` | 缠论策略 |
| `long` | 长线策略 |
| `cb` | 可转债策略 |
| `cbAllotment` | 配债股 |
| `byd` | 多状态右侧（沿用内部兼容名称，不是历史 BYD 分钟策略） |
| `similar` | 相似走势 |

状态响应包含 `run_id`、`trade_date`、`status`、`percent`、`current_step`、`steps`、`result` 和 `error`。网页按 `steps` 显示已完成步数，`percent` 保留为兼容字段。终态为 `success` 或 `failed`，可通过 `manifest_path` 查看运行清单。当日全量任务将股指步骤记在 `steps` 的 `index_futures` 项，预案结果写入 `result.index_futures`。

`/api/selector/stocks` 还返回同日 ETF 资金背景 `etf_flow`；历史查询按信号日读取。已启用的模型组合由正式发布产物读取，不在 GET 请求中重新训练。模型/日期不匹配的候选结果不能替代历史发布快照。

### 核验页面实际读取

先查询交易日历，记录响应头 `X-Quant-Generation`，再给相关 GET 请求携带同一代标识。版本不可用时按 409 处理，重新取交易日历；不要把不同版本的响应拼在一起。示例（将变量替换为待验证发布代和日期）：

```bash
base=http://127.0.0.1:8088
generation=$(curl -fsS -D - -o /dev/null "$base/api/selector/calendar" |
  awk 'tolower($1) == "x-quant-generation:" {sub(/\r$/, "", $2); print $2}')
test -n "$generation" || exit 1
target=2026-11-20
curl -fsS -D - -H "X-Quant-Generation: $generation" \
  "$base/api/selector/stocks?signal_date=$target&include_extended=true"
```

| 页面 | 核对字段和路径 |
| --- | --- |
| 短线、缠论、长线 | 各接口的 `signal_date`；长线分别核对股票池与带血筹 |
| 可转债、配债股 | `trade_date`、`asof`；配债股另核对实际明细的行情和股本日期 |
| 多状态右侧 | 分别请求 `strategy=return` 和 `strategy=win`，核对 `signal_date`、下一交易日 `trade_date` 及响应耗时 |
| 自选池与向量 | `/api/similar-patterns/analysis` 的 `target_date` 与参考库来源日；`/api/similar-patterns/watchlist?include_scores=true` 的逐股 `score_date` |
| 股指战法 | `/data/index-futures/signal.json` 的信号日和执行日；`research.json` 的冻结回测日期不当作信号日 |

操作计划和复盘是用户记录，不要求记录日期等于日更日期。仅核验时不要传 `refresh=true` 或提交刷新请求。

## 复盘记录与操作计划

| 方法 | 路径 | 输入与行为 |
| --- | --- | --- |
| `GET` | `/api/review-records` | 返回 `records` 数组 |
| `POST` | `/api/review-records` | 必填 `symbol`；可带 `name`、`source`、`signal_date`、`information`、`snapshot`、`note`，返回 `record` |
| `PUT` | `/api/review-records/{record_id}/note` | 仅接受 `note`，保留加入时的快照和依据；其他字段返回参数校验错误 |
| `DELETE` | `/api/review-records/{record_id}` | 删除指定记录，返回 `{"deleted":true}` |
| `GET` / `POST` | `/api/operation-plans` | 查询 / 新建用户操作计划，字段以 OpenAPI 为准 |
| `PUT` / `DELETE` | `/api/operation-plans/{plan_id}` | 更新 / 删除指定计划 |

复盘股票代码支持六位代码及 `.SH` / `.SZ` / `.BJ` 后缀；非法代码返回 400，未知记录的笔记修改和删除返回 404。数据存于 `web/data/review_records.json`，写操作使用文件锁和原子替换。损坏文件会显式报错，不会作为空记录覆盖。操作计划和复盘均为用户记录，日常刷新不会重写，部署前应备份 `web/data/`。

## 股指战法手动刷新与已发布预案

| 方法 | 路径 | 行为 |
| --- | --- | --- |
| `POST` | `/api/index-futures/refresh` | 启动独立后台行情与预案刷新；正在运行时复用当前任务 |
| `GET` | `/api/index-futures/refresh/status` | 返回手动刷新任务的 `status` 和可选 `message` |
| `GET` | `/data/index-futures/signal.json` | 读取最近发布的预案，包括 `generated_at`、`signal_date` 和 `execution_date` |

此接口的状态为 `idle`、`running`、`complete`、`failed`，与日常刷新使用的 `success` 不同。后台子进程时限为 300 秒，失败保留此前公开结果。历史页面来自冻结研究导出；详细数据时钟和季度模型要求见 [股指战法工作台](strategies/index_futures_workspace.md)。

## 研究与历史接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/b1/strategies` | B1 策略清单 |
| `GET` | `/api/b1/signals` | B1 唯一股票或全部信号 |
| `GET` | `/api/b1/history` | 历史 dashboard 数据 |
| `GET` | `/api/research/b1` | 研究结果索引 |

错误响应使用 FastAPI 标准结构：

```json
{"detail":"错误说明"}
```
