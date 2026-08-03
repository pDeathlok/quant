"""Project-wide feature and variable library."""

from .variable_library import (
    PROJECT_FACTOR_COLUMNS,
    EXTRA_FEATURE_COLUMNS,
    calc_bbi,
    calculate_project_extra_features,
    load_daily_basic_features,
    merge_daily_basic_features,
)
from .market_regime import classify_market_regime
from .factor_registry import FACTOR_REGISTRY, LONG_FACTOR_COLUMNS, registry_frame
from .project_factor_layer import (
    LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION,
    PROJECT_FACTOR_SCHEMA_VERSION,
    admit_factors_by_sample,
    calculate_project_factor_frame,
    calculate_project_market_factors,
    resolve_project_factor_schema,
)

__all__ = [
    "PROJECT_FACTOR_COLUMNS",
    "EXTRA_FEATURE_COLUMNS",
    "calc_bbi",
    "calculate_project_extra_features",
    "load_daily_basic_features",
    "merge_daily_basic_features",
    "classify_market_regime",
    "PROJECT_FACTOR_SCHEMA_VERSION",
    "LEGACY_PRODUCTION_FACTOR_SCHEMA_VERSION",
    "FACTOR_REGISTRY",
    "LONG_FACTOR_COLUMNS",
    "registry_frame",
    "admit_factors_by_sample",
    "calculate_project_factor_frame",
    "calculate_project_market_factors",
    "resolve_project_factor_schema",
]
