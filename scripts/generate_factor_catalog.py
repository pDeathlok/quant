"""Generate the governed factor catalog from the canonical registry."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from quant.features.factor_registry import (
    FACTOR_REGISTRY,
    FACTOR_REGISTRY_SCHEMA_VERSION,
    FactorDefinition,
)
from quant.features.factor_execution import validate_factor_execution_registry


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "factor_catalog.md"

LAYER_ORDER = (
    "project_daily",
    "project_daily_candidate",
    "right_side_rule",
    "right_side_identity",
    "selector_live",
    "chan_live",
    "long_snapshot",
    "long_research",
    "long_external_candidate",
)

ROLE_ORDER = ("feature", "compatibility_alias", "strategy_identity")
LIFECYCLE_ORDER = (
    "production_model",
    "production_materialized",
    "research_candidate",
    "compatibility_alias",
    "strategy_identity",
)


def _cell(value: object) -> str:
    if isinstance(value, (tuple, list)):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ") or "-"


def _counter_rows(counter: Counter[str], order: Iterable[str]) -> list[str]:
    rows = []
    seen = set()
    for key in order:
        if key in counter:
            rows.append(f"| `{_cell(key)}` | {counter[key]} |")
            seen.add(key)
    for key in sorted(set(counter) - seen):
        rows.append(f"| `{_cell(key)}` | {counter[key]} |")
    return rows


def _sorted_records() -> list[FactorDefinition]:
    layer_position = {name: index for index, name in enumerate(LAYER_ORDER)}
    return sorted(
        FACTOR_REGISTRY,
        key=lambda item: (
            layer_position.get(item.layer, len(layer_position)),
            item.semantic_category,
            item.factor_level,
            item.name,
        ),
    )


def build_catalog() -> str:
    role_counts = Counter(item.role for item in FACTOR_REGISTRY)
    lifecycle_counts = Counter(item.lifecycle for item in FACTOR_REGISTRY)
    layer_counts = Counter(item.layer for item in FACTOR_REGISTRY)
    family_counts = Counter(item.family for item in FACTOR_REGISTRY)
    semantic_counts = Counter(item.semantic_category for item in FACTOR_REGISTRY)
    level_counts = Counter(item.factor_level for item in FACTOR_REGISTRY)
    calculator_counts = Counter(item.calculator_id for item in FACTOR_REGISTRY)
    cadence_counts = Counter(item.refresh_cadence for item in FACTOR_REGISTRY)

    lines = [
        "# 完整因子目录",
        "",
        "> 本文由 `scripts/generate_factor_catalog.py` 从 `factor_registry.py` 自动生成，请勿手工编辑。治理规则见[因子治理与生命周期](factor_governance.md)。",
        "",
        f"- 注册表 schema：`{FACTOR_REGISTRY_SCHEMA_VERSION}`",
        f"- 注册记录总数：**{len(FACTOR_REGISTRY)}**",
        f"- 规范因子：**{role_counts['feature']}**",
        f"- 兼容别名：**{role_counts['compatibility_alias']}**",
        f"- 策略身份字段：**{role_counts['strategy_identity']}**",
        "",
        "## 生命周期统计",
        "",
        "| 生命周期 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(lifecycle_counts, LIFECYCLE_ORDER),
        "",
        "## 角色统计",
        "",
        "| 角色 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(role_counts, ROLE_ORDER),
        "",
        "## 计算层统计",
        "",
        "| 计算层 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(layer_counts, LAYER_ORDER),
        "",
        "## 刷新节奏统计",
        "",
        "| 刷新节奏 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(cadence_counts, ("trade_daily", "on_demand")),
        "",
        "## 业务语义类别统计",
        "",
        "| 类别 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(semantic_counts, sorted(semantic_counts)),
        "",
        "## 因子层级统计",
        "",
        "| 层级 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(level_counts, sorted(level_counts)),
        "",
        "## 计算器统计",
        "",
        "| 计算器 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(calculator_counts, sorted(calculator_counts)),
        "",
        "## 历史来源族统计（兼容字段）",
        "",
        "| 来源族 | 数量 |",
        "| --- | ---: |",
        *_counter_rows(family_counts, sorted(family_counts)),
        "",
        "## 全部注册记录",
        "",
        "| 名称 | 规范名称 | 业务类别 | 因子层级 | 生命周期 | 计算器 | 计算归属 | 物化方式 | 观察频率 | 刷新节奏 | PIT | 当前消费者 | 历史来源族 | 数据来源 | 计算入口 | 计算版本 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for item in _sorted_records():
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_cell(item.name)}`",
                    f"`{_cell(item.canonical_name)}`",
                    f"`{_cell(item.semantic_category)}`",
                    f"`{_cell(item.factor_level)}`",
                    f"`{_cell(item.lifecycle)}`",
                    f"`{_cell(item.calculator_id)}`",
                    f"`{_cell(item.calculation_owner)}`",
                    f"`{_cell(item.materialization)}`",
                    f"`{_cell(item.frequency)}`",
                    f"`{_cell(item.refresh_cadence)}`",
                    "yes" if item.point_in_time else "no",
                    _cell(item.active_consumers),
                    f"`{_cell(item.family)}`",
                    f"`{_cell(item.source)}`",
                    f"`{_cell(item.calculation_entrypoint)}`",
                    f"`{_cell(item.calculation_version)}`",
                )
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    validate_factor_execution_registry()
    OUTPUT_PATH.write_text(build_catalog(), encoding="utf-8")
    print(f"wrote {len(FACTOR_REGISTRY)} records to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
