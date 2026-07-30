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
| `GET` | `/api/convertible-bonds/plan` | `trade_date`、`limit`、`refresh` | 可转债计划 |
| `GET` | `/api/convertible-bonds/allotments` | `limit`、`include_listed_days`、`refresh`、`stage_scope` | 配债股跟踪 |
| `GET` | `/api/byd/daily-plan` | 持仓、成本和当日做T参数 | BYD 日线计划 |
| `GET` | `/api/similar-patterns/analysis` | `refresh` | 相似走势分析 |
| `POST` | `/api/similar-patterns/analysis/refresh` | 无 | 强制刷新相似走势 |

日期参数中，股票和策略信号日期使用 `YYYY-MM-DD`；可转债 `trade_date` 使用 `YYYYMMDD`。

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

可用作用域：

| `scope` | 页面 |
| --- | --- |
| `all` | 全部工作区 |
| `short` | 短线策略 |
| `chan` | 缠论策略 |
| `long` | 长线策略 |
| `cb` | 可转债策略 |
| `cbAllotment` | 配债股 |
| `byd` | BYD 做T |
| `similar` | 相似走势 |

状态响应包含 `status`、`percent`、`current_step`、`steps`、`result` 和 `error`。终态为 `success` 或 `failed`。

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
