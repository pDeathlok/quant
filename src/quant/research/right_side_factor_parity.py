"""Versioned production-predicate to unified-model factor contracts.

The signal flag and the inputs that produced it are different facts.  The
former establishes membership in the research universe; the latter lets the
pooled model learn how far an event is from each rule boundary.  This module
keeps that distinction explicit for all fourteen current right/mixed members.

Ten Z-skill members are recomputed directly from the live detector contract.
B2, B3, Vegas, and triple-volume membership remains authoritative in the Web
family cache; their rule columns reconstruct the *current* generator exactly
but must not be presented as proof that an old cache was built by the same
code/config revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from quant.research.right_side_unified_features import (
    RIGHT_SIDE_SIGNALS,
    RULE_FEATURE_COLUMNS,
    SIGNAL_FEATURE_REQUIREMENTS,
)
from quant.research.right_side_unified_signals import (
    B2_FAMILY_SOURCE_COLUMNS,
    B3_FAMILY_SOURCE_COLUMNS,
    CANONICAL_SIGNAL_SCHEMA_VERSION,
    FAMILY_DIRECT_SOURCE_COLUMNS,
)
from quant.strategies.custom.vegas_tunnel import OPTIMIZED_VEGAS_TUNNEL_PARAMS


PREDICATE_FACTOR_SCHEMA_VERSION = "right_side_predicate_factor_v2_20260813"

TRIPLE_VOLUME_CONFIG_SHA256 = (
    "a99b00574759071a1fd012d3bc9f0490cf4d388a91ac4ad3e48eb8aa11ad82d8"
)
VEGAS_OPTIMIZED_PARAMS_SHA256 = (
    "14bba199bf4242fddd0f039ab0162291922c3577989e9f630d9091faed09daa0"
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRIPLE_VOLUME_CONFIG_PATH = (
    PROJECT_ROOT / "configs/strategies/triple_volume_breakout.yaml"
)

Authority = Literal["live_detector", "web_family_cache"]
Representation = Literal[
    "continuous_margin",
    "boolean_state",
    "aggregate_flag",
]
Parity = Literal[
    "exact_live",
    "exact_current_generator",
    "cache_reconstruction",
    "proxy",
]


@dataclass(frozen=True)
class PredicateFactor:
    """One stable production predicate and its model-visible representation."""

    predicate_id: str
    description: str
    factors: tuple[str, ...]
    threshold_semantics: str
    required_history: str
    representation: Representation
    parity: Parity


@dataclass(frozen=True)
class PredicateFactorContract:
    """Versioned factor contract for one selector member."""

    signal: str
    authority: Authority
    rule_version: str
    source: str
    minimum_history: int | None
    predicates: tuple[PredicateFactor, ...]
    cache_source_columns: tuple[str, ...] = ()
    generator_fingerprint: str | None = None
    caveat: str = ""

    @property
    def factors(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                factor
                for predicate in self.predicates
                for factor in predicate.factors
            )
        )


def _p(
    predicate_id: str,
    description: str,
    factors: Sequence[str],
    threshold_semantics: str,
    required_history: str,
    representation: Representation,
    parity: Parity,
) -> PredicateFactor:
    return PredicateFactor(
        predicate_id=predicate_id,
        description=description,
        factors=tuple(factors),
        threshold_semantics=threshold_semantics,
        required_history=required_history,
        representation=representation,
        parity=parity,
    )


LIVE_SOURCE = "src/quant/strategies/custom/z_skill_patterns.py"
FAMILY_SOURCE = "scripts/research/analyze_b1_family_rule_backtest.py::compute_signal_flags"


PREDICATE_FACTOR_CONTRACTS: tuple[PredicateFactorContract, ...] = (
    PredicateFactorContract(
        signal="B2",
        authority="web_family_cache",
        rule_version="web_family_b2_current_20260813",
        source=FAMILY_SOURCE,
        minimum_history=None,
        cache_source_columns=B2_FAMILY_SOURCE_COLUMNS,
        caveat="Cache membership is authoritative; rule columns reconstruct current code and do not date-version an old cache.",
        predicates=(
            _p(
                "b2.b1_anchor",
                "B1 anchor: oversold, relative-volume contraction, fewer than four yin bars, and BBI/MA60 support.",
                (
                    "rs_vol_ratio_20_inclusive",
                    "rs_recent_yin_count_4",
                    "rs_close_to_ma60_pct",
                    "rs_b1_support_ok",
                    "rs_b1_like_prev3",
                    "rs_family_bbi_distance_pct",
                ),
                "J<=-10, vol20<1, yin4<4, close>=97% of BBI or MA60; anchor in prior 3 bars",
                "60 bars for the complete MA60 margin; aggregate flag remains causal before that",
                "boolean_state",
                "exact_current_generator",
            ),
            _p(
                "b2.current_confirmation",
                "Current-bar right-side confirmation margins.",
                (
                    "rs_pct_chg_1d",
                    "rs_vol_ratio_5_inclusive",
                    "rs_close_pos",
                    "rs_family_kdj_j",
                    "rs_upper_shadow_pct",
                ),
                "branch-specific pct>=3/4/5, vol5>=1.2/1.5, close_pos>=0.65/0.70/0.75, J<55/60/80",
                "5-9 bars depending on volume and KDJ margin",
                "continuous_margin",
                "exact_current_generator",
            ),
            _p(
                "b2.alternate_anchor",
                "Oversold-five-bar and BBI reclaim anchor states.",
                ("rs_pre_oversold_prev5", "rs_bbi_reclaim"),
                "prior-5 J<0; prior close<=101% BBI and current close>BBI",
                "5/24 bars",
                "boolean_state",
                "exact_current_generator",
            ),
            _p(
                "b2.branch_flags",
                "All current B2 generator branches, including the four Web-consumed branches.",
                (
                    "rs_b2_from_b1_pchg3",
                    "rs_b2_from_b1_pchg4",
                    "rs_b2_from_b1_pchg5",
                    "rs_b2_any",
                    "rs_b2_oversold",
                    "rs_b2_bbi_reclaim",
                ),
                "exact conjunctions from compute_signal_flags; Web consumes pchg4, any, oversold, BBI",
                "inherits branch inputs",
                "aggregate_flag",
                "cache_reconstruction",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="B3",
        authority="web_family_cache",
        rule_version="web_family_b3_current_20260813",
        source=FAMILY_SOURCE,
        minimum_history=None,
        cache_source_columns=B3_FAMILY_SOURCE_COLUMNS,
        caveat="Cache membership is authoritative; the three explicit rs_b3 flags reconstruct current code.",
        predicates=(
            _p(
                "b3.b2_anchor",
                "Standard/broad B2 occurrence in the prior three bars.",
                ("rs_b2_recent_prev3", "rs_b2_broad_recent_prev3", "rs_days_since_b2"),
                "prior-3 standard or broad B2; days-since is continuous context",
                "inherits B2 history",
                "boolean_state",
                "exact_current_generator",
            ),
            _p(
                "b3.current_consolidation",
                "Small-positive or calm-pullback current-bar margins.",
                ("rs_pct_chg_1d", "rs_amplitude_pct", "rs_close_pos", "rs_vol_ratio_5_inclusive"),
                "pct in (0,2) or [-1,2), amplitude<7, close_pos>=0.5, vol5<=1.3 as branch requires",
                "5 bars",
                "continuous_margin",
                "exact_current_generator",
            ),
            _p(
                "b3.branch_flags",
                "The exact three B3 columns consumed by the Web family cache.",
                ("rs_b3_small_pos_amp7", "rs_b3_broad_small_pos", "rs_b3_broad_calm_pullback"),
                "exact conjunctions for b3_small_pos_amp7/broad_small_pos/broad_calm_pullback",
                "inherits B2 history",
                "aggregate_flag",
                "cache_reconstruction",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="KEY_K",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_key_k",
        minimum_history=20,
        predicates=(
            _p(
                "key_k.body_close_volume",
                "Bull body, close position, return and dynamic volume threshold.",
                ("rs_body_abs_pct", "rs_is_rise", "rs_close_pos", "rs_pct_chg_1d", "rs_vol_ratio_prev5"),
                "body>=3, rise, close_pos>=0.75, pct>=2; vol5>=1.1 if body>=7 else >=1.3",
                "20 bars",
                "continuous_margin",
                "exact_live",
            ),
            _p(
                "key_k.key_location",
                "Within 2% of the prior-20 high or low boundary.",
                ("rs_high_to_prev20_high_pct", "rs_low_to_prev20_low_pct", "rs_at_key_20d"),
                "high>=98% prior20 high OR low<=102% prior20 low",
                "20 bars",
                "boolean_state",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="VIOLENCE_K",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_violence_k",
        minimum_history=20,
        predicates=(
            _p(
                "violence_k.bottom",
                "Bottom proximity to the prior-20 low.",
                ("rs_low_to_prev20_low_pct", "rs_at_bottom_20d"),
                "low<=105% prior20 low",
                "20 bars",
                "boolean_state",
                "exact_live",
            ),
            _p(
                "violence_k.impulse",
                "Bullish large-body and volume impulse.",
                ("rs_is_rise", "rs_pct_chg_1d", "rs_close_pos", "rs_body_abs_pct", "rs_body_vs_prev6", "rs_vol_ratio_prev5"),
                "rise, pct>0, close_pos>=0.70, body>=5 and >2x prior-six mean, vol5>=2",
                "20 bars",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="PINGHANG",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_pinghang",
        minimum_history=12,
        predicates=(
            _p(
                "pinghang.two_cannons",
                "Two strong-yang cannons in eight bars, with the second on the signal bar.",
                ("rs_strong_yang", "rs_strong_yang_count_8d", "rs_two_yang_gap_days", "rs_mid_bar_count"),
                "at least two; >=2 middle bars; strong yang is rise,pct>=3,prior5-volume ratio>=1.5",
                "12 bars",
                "boolean_state",
                "exact_live",
            ),
            _p(
                "pinghang.middle_and_second",
                "Middle yin/volume contraction and second-cannon quality.",
                ("rs_mid_yin_share", "rs_first_vol_to_mid_max", "rs_second_vol_to_mid_max", "rs_second_to_first_vol", "rs_second_yang_pct", "rs_kdj_j"),
                "yin_share>=0.5, each cannon/mid_max>=1.15, second/first>=0.9, second pct>=4, J<55",
                "two anchored cannons",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="DOUBLE_GUN",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_double_gun",
        minimum_history=18,
        predicates=(
            _p(
                "double_gun.anchors",
                "Two gun anchors and their volume/J state.",
                ("rs_double_gun_gap_days", "rs_double_gun_first_vol_ratio_prev", "rs_double_gun_second_vol_ratio_prev", "rs_double_gun_pre_second_kdj_j"),
                "each gun pct>=3,rise,prev-volume ratio>=1.8; gap 3..10; pre-second J<20",
                "18 bars",
                "continuous_margin",
                "exact_live",
            ),
            _p(
                "double_gun.post_second",
                "Post-second timing, middle contraction, and low support.",
                ("rs_double_gun_mid_avg_vol_ratio_prev", "rs_double_gun_days_since_second", "rs_double_gun_close_to_second_low_pct", "rs_double_gun_active"),
                "middle avg<1.2, 1..4 bars after second, close>=second low",
                "up to 15 trailing bars",
                "aggregate_flag",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="CHANGAN",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_changan",
        minimum_history=12,
        predicates=(
            _p(
                "changan.three_day_sequence",
                "Oversold day, prior-day impulse, and current small-yang half-volume confirmation.",
                ("rs_kdj_j_lag2", "rs_pct_chg_lag1", "rs_is_rise_lag1", "rs_vol_ratio_prev5_lag1", "rs_kdj_j_lag1_minus_lag2", "rs_pct_chg_1d", "rs_amplitude_pct", "rs_vol_ratio_prev"),
                "J[t-2]<-13; pct[t-1]>=4,rise,vol5>=1.4,J rising; 0<pct[t]<2.2,amp<7,vol/prev<=0.55",
                "12 bars",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="KENGQI",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_kengqi",
        minimum_history=25,
        predicates=(
            _p(
                "kengqi.pit_anchor",
                "Trailing-18 pit depth, location, bearish pit bar and pit volume expansion.",
                ("rs_pit_depth_18d", "rs_pit_recent_3_14d", "rs_days_since_pit", "rs_last_pit_vol_ratio_prev"),
                "pit has >=5 prior bars, depth>=0.12, bearish, pit vol ratio>=1.25",
                "25 bars",
                "boolean_state",
                "exact_live",
            ),
            _p(
                "kengqi.fill",
                "Fill ratio and post-pit volume contraction.",
                ("rs_pit_fill_ratio", "rs_post_to_pre_volume_ratio", "rs_pct_chg_1d"),
                "fill in [0.78,1.12], post/pre volume<0.8, current pct<=3",
                "pit plus up to five post bars",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="VEGAS",
        authority="web_family_cache",
        rule_version="optimized_vegas_tunnel_20260813",
        source="src/quant/strategies/custom/vegas_tunnel.py::add_vegas_tunnel_signals",
        minimum_history=181,
        cache_source_columns=(FAMILY_DIRECT_SOURCE_COLUMNS["VEGAS"],),
        generator_fingerprint=VEGAS_OPTIMIZED_PARAMS_SHA256,
        caveat="The exact current optimized generator is reconstructed; stored cache membership remains authoritative.",
        predicates=(
            _p(
                "vegas.moving_averages",
                "Optimized fast/momentum/tunnel exponential averages.",
                ("rs_ema10", "rs_ema20", "rs_ema144", "rs_ema169", "rs_vegas_tunnel_upper"),
                "EMA spans 10/20/144/169, adjust=False",
                "169 bars",
                "continuous_margin",
                "exact_current_generator",
            ),
            _p(
                "vegas.trend_pullback_rebound",
                "Tunnel distance/slope, trend stack, recent pullback and right-side rebound.",
                ("rs_vegas_tunnel_distance", "rs_vegas_tunnel_slope_20d", "rs_vegas_fast_spread", "rs_vegas_recent_pullback_8d", "rs_vegas_trend_stack", "rs_vegas_tunnel_up", "rs_vegas_right_side_rebound"),
                "near<=2.5% within 8 bars; close>EMA10>EMA20>tunnel; tunnel rising; close>EMA10 and rise",
                "181 bars",
                "boolean_state",
                "exact_current_generator",
            ),
            _p(
                "vegas.volume_heat_tradability",
                "Volume, overheat, history, and name/open tradability gates.",
                ("rs_vegas_volume_strength", "rs_vegas_volume_confirm", "rs_vegas_not_overheated", "rs_vegas_history_ok", "rs_vegas_tradable"),
                "1.05<vol/MA20<3; close/tunnel<=1.18; position>=180; open>0 and non-ST/*/退",
                "181 bars",
                "boolean_state",
                "exact_current_generator",
            ),
            _p(
                "vegas.final_signal",
                "Exact conjunction emitted as signal_vegas_tunnel.",
                ("rs_vegas_signal",),
                "all optimized Vegas predicates",
                "181 bars",
                "aggregate_flag",
                "cache_reconstruction",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="TRIPLE_VOLUME_BREAKOUT",
        authority="web_family_cache",
        rule_version="triple_volume_breakout_yaml_20260813",
        source="src/quant/strategies/custom/triple_volume_breakout.py::add_triple_volume_strategy_pool_signals",
        minimum_history=None,
        cache_source_columns=(FAMILY_DIRECT_SOURCE_COLUMNS["TRIPLE_VOLUME_BREAKOUT"],),
        generator_fingerprint=TRIPLE_VOLUME_CONFIG_SHA256,
        caveat="The 2.5x/3.0x current YAML variants are reconstructed; historical cache version is not inferred.",
        predicates=(
            _p(
                "tvb.anchor",
                "Prior-bar 2.5x/3.0x volume anchors and anchor-relative state.",
                ("rs_tvb_anchor_volume_multiple", "rs_tvb_days_since_anchor_25", "rs_tvb_days_since_anchor_30", "rs_tvb_anchor_price_25", "rs_tvb_anchor_price_30"),
                "volume[t-1]/volume[t-2]>=2.5 or 3.0; days_since>0",
                "two bars plus anchor state",
                "continuous_margin",
                "exact_current_generator",
            ),
            _p(
                "tvb.consolidation_breakout",
                "Average pre-shrink, tight range, bull stack and breakout margins for both variants.",
                ("rs_tvb_consolidation_range_25", "rs_tvb_consolidation_range_30", "rs_tvb_avg_pre_shrink_25", "rs_tvb_avg_pre_shrink_30", "rs_tvb_breakout_pct_25", "rs_tvb_breakout_pct_30", "rs_tvb_bull_no60", "rs_ma20_slope_5d_pct"),
                "range<1.15, avg prior volume<avg MA5, current volume<pre-anchor volume, close>anchor and open, MA5>MA10>MA20 rising",
                "20 bars plus anchor state",
                "boolean_state",
                "exact_current_generator",
            ),
            _p(
                "tvb.variant_and_merge",
                "Exact 2.5x expanded, 3.0x conservative, and merged generator flags.",
                ("rs_tvb_candidate_25", "rs_tvb_candidate_30", "rs_tvb_merged"),
                "merged = candidate_25 OR candidate_30",
                "inherits variant inputs",
                "aggregate_flag",
                "cache_reconstruction",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="GOLDEN_BOWL",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_golden_bowl",
        minimum_history=120,
        predicates=(
            _p(
                "golden_bowl.lines_and_support",
                "White/yellow trend lines and close position inside the bowl support zone.",
                ("rs_zg_white", "rs_dg_yellow", "rs_white_yellow_spread_pct", "rs_close_to_yellow_pct", "rs_close_bowl_position", "rs_kdj_j"),
                "white>yellow*1.005; yellow<=close<=(white+yellow)/2; close/yellow-1<=4%; J<80",
                "120 bars",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="ZAIHOU",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_zaihou",
        minimum_history=60,
        predicates=(
            _p(
                "zaihou.anchor",
                "Oldest qualifying prior fangliang anchor in the trailing 15 bars.",
                ("rs_fangliang_recent_3_12d", "rs_days_since_fangliang", "rs_fangliang_ref_volume_15d"),
                "anchor pct>5 and volume>1.5x prior-five mean; current excluded",
                "60 bars",
                "boolean_state",
                "exact_live",
            ),
            _p(
                "zaihou.rebuild",
                "Rising BBI, proximity, and shrink relative to the anchor.",
                ("rs_volume_to_fangliang_ref", "rs_bbi_slope_5d_pct", "rs_bbi_distance_pct", "rs_pct_chg_1d"),
                "BBI slope>0, abs(close/BBI-1)<2.5%, volume/anchor<0.6; pct is context",
                "anchor plus BBI history",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="BREATHING",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_breathing",
        minimum_history=16,
        predicates=(
            _p(
                "breathing.rhythm",
                "Seven-bar exhale/inhale rhythm and higher-low N structure.",
                ("rs_phase_exhale", "rs_phase_inhale", "rs_exhale_count_7d", "rs_inhale_count_7d", "rs_higher_low_ratio"),
                "current exhale; >=2 exhale and >=2 inhale; current low>98% min(low[t-3],low[t-6])",
                "16 bars",
                "boolean_state",
                "exact_live",
            ),
            _p(
                "breathing.current_confirmation",
                "Current return, previous-day volume ratio, and close-position confirmation.",
                ("rs_pct_chg_1d", "rs_vol_ratio_prev", "rs_close_pos"),
                "pct>=1, vol/prev>=1.2, close_pos>=0.60",
                "16 bars",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
    PredicateFactorContract(
        signal="YUEYUE",
        authority="live_detector",
        rule_version=CANONICAL_SIGNAL_SCHEMA_VERSION,
        source=f"{LIVE_SOURCE}::_detect_yueyue",
        minimum_history=35,
        predicates=(
            _p(
                "yueyue.platform",
                "Twenty-bar platform range and current distance to its high.",
                ("rs_platform_range_20d", "rs_close_to_platform_high_pct"),
                "(high20-low20)/max(low20,1)<=0.16",
                "35 bars",
                "continuous_margin",
                "exact_live",
            ),
            _p(
                "yueyue.volume_tests",
                "Huge-volume count and bullish share in the platform.",
                ("rs_huge_volume_count_20d", "rs_huge_yang_share_20d", "rs_pct_chg_1d", "rs_close_pos"),
                "volume>2x inclusive MA10 on >=2 bars and bullish share>=0.5; pct/close_pos are ranking context",
                "35 bars",
                "continuous_margin",
                "exact_live",
            ),
        ),
    ),
)


CONTRACT_BY_SIGNAL: Mapping[str, PredicateFactorContract] = {
    contract.signal: contract for contract in PREDICATE_FACTOR_CONTRACTS
}


FAMILY_CACHE_TO_RULE_FACTOR: Mapping[str, str] = {
    "b2_any_pchg4_vol15": "rs_b2_any",
    "b2_oversold_pchg3_vol12": "rs_b2_oversold",
    "b2_bbi_reclaim_vol12": "rs_b2_bbi_reclaim",
    "b2_pchg4_vol15": "rs_b2_from_b1_pchg4",
    "b3_broad_small_pos": "rs_b3_broad_small_pos",
    "b3_broad_calm_pullback": "rs_b3_broad_calm_pullback",
    "b3_small_pos_amp7": "rs_b3_small_pos_amp7",
    "signal_vegas_tunnel": "rs_vegas_signal",
    "signal_tvb_merged": "rs_tvb_merged",
}


def vegas_optimized_params_fingerprint() -> str:
    payload = json.dumps(
        OPTIMIZED_VEGAS_TUNNEL_PARAMS,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def triple_volume_config_fingerprint(
    path: Path = TRIPLE_VOLUME_CONFIG_PATH,
) -> str:
    return sha256(path.read_bytes()).hexdigest()


def validate_generator_fingerprints() -> None:
    actual_vegas = vegas_optimized_params_fingerprint()
    actual_tvb = triple_volume_config_fingerprint()
    errors: list[str] = []
    if actual_vegas != VEGAS_OPTIMIZED_PARAMS_SHA256:
        errors.append(
            f"Vegas optimized params changed: {actual_vegas} != {VEGAS_OPTIMIZED_PARAMS_SHA256}"
        )
    if actual_tvb != TRIPLE_VOLUME_CONFIG_SHA256:
        errors.append(
            f"triple-volume config changed: {actual_tvb} != {TRIPLE_VOLUME_CONFIG_SHA256}"
        )
    if errors:
        raise ValueError("; ".join(errors))


def contract_factor_audit(
    contracts: Sequence[PredicateFactorContract] = PREDICATE_FACTOR_CONTRACTS,
    *,
    model_features: Sequence[str] = RULE_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return one audit row per predicate without silently dropping errors."""

    available = set(model_features)
    rows: list[dict[str, object]] = []
    for contract in contracts:
        for predicate in contract.predicates:
            missing = sorted(set(predicate.factors) - available)
            rows.append(
                {
                    "signal": contract.signal,
                    "authority": contract.authority,
                    "predicate_id": predicate.predicate_id,
                    "representation": predicate.representation,
                    "parity": predicate.parity,
                    "factor_count": len(predicate.factors),
                    "missing_factors": missing,
                    "status": "missing_factor" if missing else "ok",
                }
            )
    return pd.DataFrame(rows)


def validate_predicate_factor_contracts(
    contracts: Sequence[PredicateFactorContract] = PREDICATE_FACTOR_CONTRACTS,
    *,
    model_features: Sequence[str] = RULE_FEATURE_COLUMNS,
) -> None:
    """Fail on missing/duplicate members, unmapped factors, or proxy claims."""

    signals = [contract.signal for contract in contracts]
    errors: list[str] = []
    if tuple(signals) != RIGHT_SIDE_SIGNALS:
        errors.append(
            f"signal order/coverage differs: actual={signals}, expected={list(RIGHT_SIDE_SIGNALS)}"
        )
    predicate_ids = [
        predicate.predicate_id
        for contract in contracts
        for predicate in contract.predicates
    ]
    duplicate_ids = sorted(
        predicate_id
        for predicate_id in set(predicate_ids)
        if predicate_ids.count(predicate_id) > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate predicate ids: {duplicate_ids}")

    available = set(model_features)
    mapped = {
        factor
        for contract in contracts
        for predicate in contract.predicates
        for factor in predicate.factors
    }
    missing = sorted(mapped - available)
    orphan = sorted(available - mapped)
    if missing:
        errors.append(f"predicate factors absent from model: {missing}")
    if orphan:
        errors.append(f"model rule factors have no predicate explanation: {orphan}")

    for contract in contracts:
        required = set(SIGNAL_FEATURE_REQUIREMENTS.get(contract.signal, ()))
        unmapped_required = sorted(required - set(contract.factors))
        if unmapped_required:
            errors.append(
                f"{contract.signal} required factors not mapped: {unmapped_required}"
            )
        if not contract.predicates:
            errors.append(f"{contract.signal} has no predicates")
        for predicate in contract.predicates:
            if not predicate.factors:
                errors.append(f"{predicate.predicate_id} has no factors")

    if any(
        predicate.parity == "proxy"
        for contract in contracts
        for predicate in contract.predicates
    ):
        errors.append("proxy predicate remains in the production factor contract")
    if errors:
        raise ValueError("right-side predicate-factor contract invalid: " + "; ".join(errors))


def reconstruct_web_family_flags(rule_features: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct current Web family generator columns from model factors.

    This checks current-code parity only.  Callers must continue to use the
    persisted family cache as the historical membership authority.
    """

    missing = set(FAMILY_CACHE_TO_RULE_FACTOR.values()) - set(rule_features.columns)
    if missing:
        raise ValueError(
            f"rule_features missing family reconstruction factors: {sorted(missing)}"
        )
    out = pd.DataFrame(index=rule_features.index)
    for cache_column, factor in FAMILY_CACHE_TO_RULE_FACTOR.items():
        out[cache_column] = rule_features[factor].fillna(False).astype(bool)
    out["B2"] = out[list(B2_FAMILY_SOURCE_COLUMNS)].any(axis=1)
    out["B3"] = out[list(B3_FAMILY_SOURCE_COLUMNS)].any(axis=1)
    out["VEGAS"] = out[FAMILY_DIRECT_SOURCE_COLUMNS["VEGAS"]]
    out["TRIPLE_VOLUME_BREAKOUT"] = out[
        FAMILY_DIRECT_SOURCE_COLUMNS["TRIPLE_VOLUME_BREAKOUT"]
    ]
    return out


def audit_signal_factor_slice(
    frame: pd.DataFrame,
    signal: str,
) -> pd.DataFrame:
    """Audit required factors on rows where one selector is active."""

    if signal not in CONTRACT_BY_SIGNAL:
        raise ValueError(f"unknown right-side signal: {signal}")
    if signal not in frame.columns:
        raise ValueError(f"event frame missing signal column: {signal}")
    active = frame[signal].fillna(False).astype(bool)
    selected = frame.loc[active]
    event_rows = len(selected)
    rows: list[dict[str, object]] = []
    for factor in SIGNAL_FEATURE_REQUIREMENTS[signal]:
        if factor not in selected.columns:
            rows.append(
                {
                    "signal": signal,
                    "factor": factor,
                    "event_rows": event_rows,
                    "non_null_rate": 0.0,
                    "finite_rate": 0.0,
                    "unique_values": 0,
                    "status": "missing",
                }
            )
            continue
        values = selected[factor]
        numeric = pd.to_numeric(values, errors="coerce")
        original_non_null = values.notna()
        finite = pd.Series(
            np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)),
            index=values.index,
        )
        non_finite_non_null = original_non_null & ~finite
        if event_rows == 0:
            status = "empty_signal"
        elif not original_non_null.any():
            status = "all_null"
        elif non_finite_non_null.any():
            status = "non_finite"
        else:
            status = "ok"
        rows.append(
            {
                "signal": signal,
                "factor": factor,
                "event_rows": event_rows,
                "non_null_rate": (
                    float(original_non_null.mean()) if event_rows else 0.0
                ),
                "finite_rate": float(finite.mean()) if event_rows else 0.0,
                "unique_values": int(values.nunique(dropna=True)),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def validate_signal_factor_slice(frame: pd.DataFrame, signal: str) -> None:
    audit = audit_signal_factor_slice(frame, signal)
    failures = audit[audit["status"].ne("ok")]
    if not failures.empty:
        details = failures[["factor", "status"]].to_dict("records")
        raise ValueError(f"{signal} active-slice factor audit failed: {details}")


def predicate_contract_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for contract in PREDICATE_FACTOR_CONTRACTS:
        parities = [predicate.parity for predicate in contract.predicates]
        representations = [
            predicate.representation for predicate in contract.predicates
        ]
        rows.append(
            {
                "signal": contract.signal,
                "authority": contract.authority,
                "rule_version": contract.rule_version,
                "factor_count": len(contract.factors),
                "continuous_predicates": representations.count("continuous_margin"),
                "state_predicates": representations.count("boolean_state"),
                "aggregate_predicates": representations.count("aggregate_flag"),
                "cache_reconstruction_predicates": parities.count("cache_reconstruction"),
                "proxy_predicates": parities.count("proxy"),
                "cache_source_columns": ",".join(contract.cache_source_columns),
                "caveat": contract.caveat,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CONTRACT_BY_SIGNAL",
    "FAMILY_CACHE_TO_RULE_FACTOR",
    "PREDICATE_FACTOR_CONTRACTS",
    "PREDICATE_FACTOR_SCHEMA_VERSION",
    "PredicateFactor",
    "PredicateFactorContract",
    "TRIPLE_VOLUME_CONFIG_SHA256",
    "VEGAS_OPTIMIZED_PARAMS_SHA256",
    "audit_signal_factor_slice",
    "contract_factor_audit",
    "predicate_contract_summary",
    "reconstruct_web_family_flags",
    "triple_volume_config_fingerprint",
    "validate_generator_fingerprints",
    "validate_predicate_factor_contracts",
    "validate_signal_factor_slice",
    "vegas_optimized_params_fingerprint",
]
