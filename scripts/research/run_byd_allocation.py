#!/usr/bin/env python3
"""Frozen finite experiment: multi-horizon experts and nested online allocation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from quant.research.byd_allocation import (
    AllocationConfig, allocation_path, bayesian_search, expanded_experts, expert_groups,
    inner_score, market_context, past_pruned_mask, signal_from_snapshot, simulate_path,
)
from quant.research.byd_mwu import ExecutionConfig, ROOT, evaluate_experts, metrics


OUT = ROOT / "reports/research/byd_allocation_20261013"
PREVIOUS = ROOT / "reports/research/byd_mwu_20261012"
YEARS = tuple(range(2020, 2027))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def identity(config: AllocationConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True)


def candidate_bank() -> tuple[dict[str, AllocationConfig], dict[str, list[str]], list[str]]:
    bank: dict[str, AllocationConfig] = {}
    families: dict[str, list[str]] = {}

    def add(family: str, config: AllocationConfig) -> str:
        key = identity(config)
        bank[key] = config
        families.setdefault(family, [])
        if key not in families[family]:
            families[family].append(key)
        return key

    for group in ("legacy", "short", "long", "diverse", "all"):
        for method in ("equal", "mwu", "share", "discount", "adah2011", "squint_grid", "multiwindow", "context", "omd", "ftrl"):
            rates = (.03, .1, .3) if method == "mwu" else (.1, .3) if method in ("multiwindow", "context", "omd", "ftrl") else (.3,)
            for eta in rates:
                settings = [{"eta": eta}]
                if method == "share":
                    settings = [{"eta": eta, "share": share} for share in (.01, .05)]
                elif method == "discount":
                    settings = [{"eta": eta, "forget": memory} for memory in (.97, .99)]
                elif method == "context":
                    settings = [{"eta": eta, "forget": .99}]
                for setting in settings:
                    add(method, AllocationConfig(method=method, group=group, **setting))
        for group_cost in ("legacy", "long", "diverse"):
            if group != group_cost:
                continue
            for gamma in (1.0, 4.0):
                add("cost_prox", AllocationConfig(group=group, cost_gamma=gamma, deadband=0))
                add("robust_prox", AllocationConfig(group=group, cost_gamma=gamma, risk=50, deadband=0))
    for eta in (.03, .1, .3):
        add("pruned_mwu", AllocationConfig(group="pruned", eta=eta))
    search = []
    for eta, share, forget, controls in itertools.product(
        (.03, .1, .3, .6), (0., .02), (.97, .99, 1.), ((.03, .1), (.06, .2)),
    ):
        cfg = AllocationConfig(group="diverse", eta=eta, share=share, forget=forget,
                               deadband=controls[0], max_step=controls[1])
        search.append(add("grid_search", cfg))
    return bank, families, search


def markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| 方法 | 累计净收益 | 年化 | Sharpe | 最大回撤 | 平均仓位 | 订单数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in frame.itertuples():
        lines.append(f"| {row.name} | {row.total_return:.2%} | {row.cagr:.2%} | {row.sharpe:.2f} | {row.max_drawdown:.2%} | {row.average_position:.2%} | {row.orders} |")
    return "\n".join(lines)


def svg_chart(ledgers: dict[str, pd.DataFrame], path: Path) -> None:
    colors = ("#0f766e", "#2563eb", "#d97706", "#9333ea", "#64748b")
    curves = {}
    for name, ledger in ledgers.items():
        frame = ledger[ledger.date.ge("2020-01-01")]
        curves[name] = np.r_[1., np.cumprod(1 + frame["return"].to_numpy())]
    dates = frame.date
    minimum = min(.8, min(v.min() for v in curves.values()) * .95)
    maximum = max(v.max() for v in curves.values()) * 1.05
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="540" viewBox="0 0 1120 540">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#334155}</style>',
             '<text x="75" y="32" style="font-size:21px;font-weight:bold">BYD | chronological annual selection, 2020+ | net equity</text>']
    for value in np.linspace(minimum, maximum, 6):
        y = 60 + (maximum - value) / (maximum - minimum) * 375
        parts += [f'<path d="M75 {y:.1f} H1090" stroke="#e2e8f0"/>', f'<text x="15" y="{y+5:.1f}">{value:.2f}x</text>']
    for j, (name, values) in enumerate(curves.items()):
        x = np.linspace(75, 1090, len(values))
        y = 60 + (maximum - values) / (maximum - minimum) * 375
        points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
        parts += [f'<polyline points="{points}" fill="none" stroke="{colors[j]}" stroke-width="{3 if j == 0 else 1.8}"/>',
                  f'<text x="{75+j*205}" y="510" style="fill:{colors[j]}">{name}</text>']
    for ratio in (0, .25, .5, .75, 1):
        date = dates.iloc[int(ratio * (len(dates) - 1))]
        parts.append(f'<text x="{75+ratio*1015-20:.1f}" y="470">{date:%Y-%m}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, name: str) -> dict:
    aa = a[a.date.ge("2020-01-01")].set_index("date")["return"]
    bb = b[b.date.ge("2020-01-01")].set_index("date")["return"]
    if not aa.index.equals(bb.index):
        raise ValueError("bootstrap dates differ")
    delta = (aa - bb).to_numpy()
    rng = np.random.default_rng(20261013)
    starts = rng.integers(0, len(delta), size=(2000, math.ceil(len(delta) / 20)))
    indices = ((starts[:, :, None] + np.arange(20)) % len(delta)).reshape(2000, -1)[:, :len(delta)]
    low, high = np.quantile(delta[indices].mean(axis=1) * 252, [.025, .975])
    return {"comparator": name, "annual_mean_difference": float(delta.mean() * 252),
            "low95": float(low), "high95": float(high)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    bank, families, search_keys = candidate_bank()
    initial_protocol = out / "initial_replay/protocol.json"
    if initial_protocol.exists():
        original_protocol = json.loads(initial_protocol.read_text())
        if original_protocol["candidates"] != {key: cfg.to_dict() for key, cfg in bank.items()}:
            raise RuntimeError("Correctness replay must retain the initial candidate bank")
    write_json(out / "protocol.json", {
        "candidate_count": len(bank), "candidates": {key: cfg.to_dict() for key, cfg in bank.items()},
        "families": families, "outer_years": YEARS,
        "inner": "last three complete years, minimum 2016, before each outer year",
        "score": "mean(Sharpe+.5*CAGR-.5*abs(MDD))-.35*std",
        "stopping": "finite preregistered bank completed; no outer-result-driven additions",
        "history_previously_observed": True, "blind_test": False,
        "search_budget": 12, "successive_halving_keep_fraction": 1/3,
    })
    print(f"Frozen {len(bank)} candidates in {len(families)} families", flush=True)
    market = pd.read_parquet(PREVIOUS / "market_snapshot.parquet")
    legacy = pd.read_parquet(PREVIOUS / "expert_targets.parquet").drop(columns="date")
    index = pd.read_parquet(ROOT / "data/raw/index_000300.SH.parquet")
    signal = signal_from_snapshot(market)
    print("Building 1/21/63-day rolling models and multi-horizon rules", flush=True)
    targets, features, training = expanded_experts(signal, index, legacy)
    training.to_csv(out / "training_audit.csv", index=False)
    targets.assign(date=market.date).to_parquet(out / "expert_targets.parquet", index=False)
    features.assign(date=market.date).to_parquet(out / "features.parquet", index=False)
    # Verify a historical prefix before any evaluation tables are examined.
    prefix = int(market.date.le("2023-12-31").sum())
    short, short_features, _ = expanded_experts(signal.iloc[:prefix], index, legacy.iloc[:prefix])
    pd.testing.assert_frame_equal(targets.iloc[:prefix], short)
    pd.testing.assert_frame_equal(features.iloc[:prefix], short_features)
    if not (pd.to_datetime(training.last_label_end) <= pd.to_datetime(training.fit_date)).all():
        raise RuntimeError("future labels")
    execution = ExecutionConfig()
    print(f"Experts={len(targets.columns)}; evaluating integer virtual accounts", flush=True)
    rewards, expert_ledgers = evaluate_experts(market, targets, execution)
    rewards.assign(date=market.date).to_parquet(out / "expert_net_returns.parquet", index=False)
    groups = expert_groups(list(targets.columns))
    groups["rules"] = tuple(name for name in groups["diverse"] if not name.startswith(("ridge", "tree")))
    groups["pruned"] = tuple(targets.columns)
    mask, pruning = past_pruned_mask(market, targets, rewards)
    write_json(out / "pruning_audit.json", pruning)
    weights_cache, candidate_ledgers, proposals = {}, {}, {}

    def path_for(cfg: AllocationConfig) -> tuple[np.ndarray, pd.DataFrame]:
        allocation_fields = {key: value for key, value in cfg.to_dict().items()
                             if key not in ("deadband", "max_step", "cost_gamma", "risk", "vol_target")}
        cache_key = json.dumps(allocation_fields, sort_keys=True)
        if cache_key not in weights_cache:
            names = groups[cfg.group]
            request, weights = allocation_path(market, targets.loc[:, names], rewards.loc[:, names], features, cfg,
                                               mask if cfg.group == "pruned" else None)
            weights_cache[cache_key] = request, pd.DataFrame(weights, columns=names)
        return weights_cache[cache_key]

    inner_rows = []
    for number, (key, cfg) in enumerate(bank.items(), 1):
        request, _ = path_for(cfg)
        proposals[key] = request
        ledger = simulate_path(market, request, [cfg] * len(market), execution)
        candidate_ledgers[key] = ledger
        for year in YEARS:
            inner_rows.append({"year": year, "key": key, "score": inner_score(ledger, year)})
        if number % 20 == 0:
            print(f"Evaluated {number}/{len(bank)} fixed candidates", flush=True)
            pd.DataFrame(inner_rows).to_csv(out / "inner_scores.csv", index=False)
    inner = pd.DataFrame(inner_rows)
    inner.to_csv(out / "inner_scores.csv", index=False)
    score_lookup = {(row.year, row.key): row.score for row in inner.itertuples()}
    selections, routes = [], {}
    for family, keys in families.items():
        routes[family] = {}
        for year in YEARS:
            winner = max(keys, key=lambda key: score_lookup[year, key])
            routes[family][year] = winner
            selections.append({"family": family, "year": year, "cutoff": f"{year-1}-12-31",
                               "inner_score": score_lookup[year, winner], "key": winner, **bank[winner].to_dict()})
    # Meta-method selects across all standard methods based only on inner scores.
    standard_keys = sorted({key for family, keys in families.items() if family != "grid_search" for key in keys})
    routes["selected_pipeline"] = {year: max(standard_keys, key=lambda key: score_lookup[year, key]) for year in YEARS}
    for year, key in routes["selected_pipeline"].items():
        selections.append({"family": "selected_pipeline", "year": year, "cutoff": f"{year-1}-12-31",
                           "inner_score": score_lookup[year, key], "key": key, **bank[key].to_dict()})

    # Search algorithms see a past-only inner-score oracle; no outer returns.
    search_features = np.array([[np.log10(bank[key].eta), bank[key].share * 50,
                                (1-bank[key].forget) * 100, bank[key].deadband * 20,
                                bank[key].max_step * 5] for key in search_keys])
    search_features = (search_features - search_features.mean(axis=0)) / np.maximum(search_features.std(axis=0), 1e-9)
    search_rows = []
    for name in ("bayes_search", "random_search", "halving_search"):
        routes[name] = {}
    for year in YEARS:
        oracle = np.array([score_lookup[year, key] for key in search_keys])
        gp, seen = bayesian_search(search_features, oracle, 12)
        rng = np.random.default_rng(20261013)
        random_seen = rng.choice(len(search_keys), size=12, replace=False).tolist()
        random_best = max(random_seen, key=lambda i: oracle[i])
        # Low-fidelity uses the earliest of the three annual inner blocks.
        first_year = max(2016, year - 3)
        cheap = []
        for key in search_keys:
            frame = candidate_ledgers[key]
            mm = metrics(frame[frame.date.dt.year.eq(first_year)])
            cheap.append(mm["sharpe"] + .5 * mm["cagr"] - .5 * abs(mm["max_drawdown"]))
        survivors = np.argsort(cheap)[-math.ceil(len(search_keys)/3):].tolist()
        half_best = max(survivors, key=lambda i: oracle[i])
        for name, chosen, accessed, budget in (
            ("grid_search", int(np.argmax(oracle)), list(range(len(search_keys))), len(search_keys)),
            ("bayes_search", gp, seen, len(seen)),
            ("random_search", random_best, random_seen, len(random_seen)),
            ("halving_search", half_best, survivors, len(search_keys)/3 + len(survivors)),
        ):
            routes.setdefault(name, {})[year] = search_keys[chosen]
            search_rows.append({"year": year, "search": name, "full_fidelity_equivalent_queries": budget,
                                "inner_score": float(oracle[chosen]), "gap_to_grid": float(oracle.max()-oracle[chosen]),
                                "accessed_indices": accessed, "chosen": search_keys[chosen]})
    write_json(out / "routes_frozen.json", routes)
    pd.DataFrame(selections).to_csv(out / "annual_selections.csv", index=False)
    pd.DataFrame(search_rows).to_csv(out / "search_comparison.csv", index=False)
    print("Past-only annual routes frozen; replaying continuous outer accounts", flush=True)
    scheduled_year = market.date.shift(-1).fillna(market.date.iloc[-1]).dt.year.to_numpy()
    ledgers, route_configs, route_weights, route_requests = {}, {}, {}, {}

    def run_route(name: str, route: dict[int, str]) -> None:
        initial = AllocationConfig()  # pre-2020 initialization, never retroactively optimized
        request, weights = path_for(initial)
        request = request.copy()
        combined_weights = pd.DataFrame(0., index=market.index, columns=targets.columns)
        combined_weights.loc[:, weights.columns] = weights.to_numpy()
        configs = [initial] * len(market)
        for year, key in route.items():
            mask_year = scheduled_year == year
            cfg = bank[key]
            request[mask_year] = proposals[key][mask_year]
            configs_year = path_for(cfg)[1]
            combined_weights.loc[mask_year] = 0.
            combined_weights.loc[mask_year, configs_year.columns] = configs_year.to_numpy()[mask_year]
            for i in np.flatnonzero(mask_year):
                configs[i] = cfg
        ledgers[name] = simulate_path(market, request, configs, execution, combined_weights)
        route_configs[name], route_weights[name], route_requests[name] = configs, combined_weights, request

    for name, route in routes.items():
        run_route(name, route)
    baseline_cfg = AllocationConfig(method="equal", deadband=.03, max_step=.1)
    base_policies = {"original_range_matched": targets.original_range.to_numpy(),
                     "constant_one_third": np.full(len(market), 1/3),
                     "constant_half": np.full(len(market), .5), "buy_hold": np.ones(len(market))}
    for name, request in base_policies.items():
        cfg = replace(baseline_cfg, deadband=0, max_step=1) if name == "buy_hold" else baseline_cfg
        ledgers[name] = simulate_path(market, request, [cfg] * len(market), execution)
    frozen = json.loads((PREVIOUS / "frozen_config.json").read_text())["config"]
    old_cfg = AllocationConfig(method="mwu", group="legacy", eta=frozen["eta"], deadband=frozen["deadband"], max_step=frozen["max_step"])
    old_request, old_weights = path_for(old_cfg)
    ledgers["previous_frozen_MWU"] = simulate_path(market, old_request, [old_cfg] * len(market), execution, old_weights)
    # Verify old frozen baseline numerically, not just by comparing names.
    old_reference = pd.read_csv(PREVIOUS / "ledger_MWU.csv")
    np.testing.assert_allclose(ledgers["previous_frozen_MWU"]["return"], old_reference["return"], atol=1e-12)
    for name, group in (("fixed_mwu_legacy", "legacy"), ("fixed_mwu_diverse", "diverse"),
                        ("fixed_mwu_all", "all"), ("fixed_mwu_rules", "rules"), ("fixed_mwu_pruned", "pruned")):
        cfg = AllocationConfig(group=group)
        request, weights = path_for(cfg)
        ledgers[name] = simulate_path(market, request, [cfg] * len(market), execution, weights)
    for name, origin, transform in (
        ("prox_without_cost", "cost_prox", lambda cfg: replace(cfg, cost_gamma=0)),
        ("prox_fixed_band", "cost_prox", lambda cfg: replace(cfg, cost_gamma=0, deadband=.03)),
        ("robust_without_risk", "robust_prox", lambda cfg: replace(cfg, risk=0)),
    ):
        ledgers[name] = simulate_path(market, route_requests[origin], [transform(cfg) for cfg in route_configs[origin]], execution)
    # Common annual volatility ceiling, fixed without looking at evaluation results.
    for name in ("selected_pipeline", "mwu", "cost_prox", "robust_prox", "original_range_matched", "constant_half", "buy_hold"):
        configs = route_configs.get(name, [replace(baseline_cfg, deadband=0, max_step=1) if name == "buy_hold" else baseline_cfg] * len(market))
        request = route_requests.get(name, base_policies.get(name))
        ledgers[f"vol20_{name}"] = simulate_path(market, request, [replace(cfg, vol_target=.2) for cfg in configs], execution)

    # Prefix replay for selected path includes the account-specific cost controller.
    prefix_ledger = simulate_path(market.iloc[:prefix], route_requests["selected_pipeline"][:prefix],
                                 route_configs["selected_pipeline"][:prefix], execution)
    np.testing.assert_allclose(prefix_ledger["return"], ledgers["selected_pipeline"]["return"].iloc[:len(prefix_ledger)], atol=1e-12)
    summaries, yearly, concentration = [], [], []
    for name, ledger in ledgers.items():
        ledger.to_csv(out / f"ledger_{name}.csv", index=False)
        for period, start in (("outer_2020plus", "2020-01-01"), ("observed_2024plus", "2024-01-01"), ("full_2015plus", "2015-01-01")):
            summaries.append({"name": name, "period": period, **metrics(ledger[ledger.date.ge(start)])})
        for year, frame in ledger[ledger.date.ge("2020-01-01")].groupby(ledger.date.dt.year):
            yearly.append({"name": name, "year": int(year), **metrics(frame)})
        w = ledger[ledger.date.ge("2020-01-01")].filter(like="weight_")
        if not w.empty:
            concentration.append({"name": name, "mean_daily_max_weight": float(w.max(axis=1).mean()),
                                  "mean_effective_experts": float((1/(w*w).sum(axis=1)).mean()),
                                  "dominant": w.mean().idxmax().removeprefix("weight_"),
                                  "dominant_mean_weight": float(w.mean().max())})
    summary = pd.DataFrame(summaries)
    correction_audit = {}
    initial_metrics = out / "initial_replay/metrics.csv"
    if initial_metrics.exists():
        initial = pd.read_csv(initial_metrics)
        unchanged_names = [name for name in families if name not in ("cost_prox", "robust_prox")]
        unchanged_names += ["original_range_matched", "constant_one_third", "constant_half", "buy_hold", "previous_frozen_MWU"]
        for name in unchanged_names:
            old = initial[initial.name.eq(name)].set_index("period")
            new = summary[summary.name.eq(name)].set_index("period")
            numeric = list(new.select_dtypes(include=np.number).columns)
            np.testing.assert_allclose(new.loc[old.index, numeric], old[numeric], atol=1e-12)
        correction_audit = {
            "candidate_bank_unchanged": True,
            "unaffected_methods_and_baselines_unchanged": True,
            "reason": "no-trade proximal choice must preserve share count across overnight gaps",
            "initial_artifacts": "initial_replay/",
        }
    summary.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(yearly).to_csv(out / "yearly_metrics.csv", index=False)
    pd.DataFrame(concentration).to_csv(out / "concentration.csv", index=False)
    expert_rows, conditional = [], []
    context = market_context(features)
    for name, ledger in expert_ledgers.items():
        expert_rows.append({"name": name, **metrics(ledger[ledger.date.ge("2020-01-01")])})
        for state in range(6):
            mask_state = (context[:-1] == state) & market.date.iloc[1:].ge("2020-01-01").to_numpy()
            returns = rewards[name].to_numpy()[1:][mask_state]
            conditional.append({"expert": name, "state": state, "sessions": len(returns),
                                "annual_mean_return": float(returns.mean()*252) if len(returns) else 0})
    pd.DataFrame(expert_rows).to_csv(out / "expert_metrics.csv", index=False)
    pd.DataFrame(conditional).to_csv(out / "conditional_expert_returns.csv", index=False)
    rewards[market.date.ge("2020-01-01")].corr().to_csv(out / "expert_return_correlation.csv")
    targets[market.date.ge("2020-01-01")].corr().to_csv(out / "expert_position_correlation.csv")
    bootstrap = [paired_bootstrap(ledgers["selected_pipeline"], ledgers[name], name)
                 for name in ("original_range_matched", "previous_frozen_MWU", "constant_one_third", "constant_half")]
    pd.DataFrame(bootstrap).to_csv(out / "paired_bootstrap.csv", index=False)

    # Slippage stress re-computes all virtual rewards, keeps routes/parameters frozen.
    print("Computing double-slippage stress without re-selection", flush=True)
    stress_execution = replace(execution, slippage=.001)
    stress_rewards, _ = evaluate_experts(market, targets, stress_execution)
    stress_request = route_requests["selected_pipeline"].copy()
    stress_prune, _ = past_pruned_mask(market, targets, stress_rewards)
    for year, key in routes["selected_pipeline"].items():
        cfg = bank[key]
        names = groups[cfg.group]
        request, _ = allocation_path(market, targets.loc[:, names], stress_rewards.loc[:, names], features, cfg,
                                     stress_prune if cfg.group == "pruned" else None)
        mask_year = scheduled_year == year
        stress_request[mask_year] = request[mask_year]
    stress_initial = AllocationConfig()
    initial_request, _ = allocation_path(market, targets.loc[:, groups["diverse"]], stress_rewards.loc[:, groups["diverse"]], features, stress_initial)
    stress_request[scheduled_year < 2020] = initial_request[scheduled_year < 2020]
    stress = simulate_path(market, stress_request, route_configs["selected_pipeline"], stress_execution)
    stress.to_csv(out / "ledger_double_slippage.csv", index=False)
    stress_metrics = metrics(stress[stress.date.ge("2020-01-01")])
    write_json(out / "stress_metrics.json", stress_metrics)
    chart_names = ("selected_pipeline", "original_range_matched", "previous_frozen_MWU", "constant_half", "buy_hold")
    svg_chart({name: ledgers[name] for name in chart_names}, out / "outer_equity.svg")
    outer = summary[summary.period.eq("outer_2020plus")].set_index("name")
    selected = outer.loc["selected_pipeline"]
    increments = {name: float(selected.total_return-outer.loc[name,"total_return"])
                  for name in ("original_range_matched", "previous_frozen_MWU", "constant_half")}
    accepted = bool(all(value > 0 for value in increments.values())
                    and selected.sharpe > outer.loc["original_range_matched", "sharpe"]
                    and abs(selected.max_drawdown) <= abs(outer.loc["original_range_matched", "max_drawdown"]))
    manifest = {
        "data_end": str(market.date.max().date()), "rows": len(market), "experts": list(targets.columns),
        "candidate_count": len(bank), "families": list(routes), "outer_years": YEARS,
        "snapshot_sha256": hashlib.sha256((PREVIOUS / "market_snapshot.parquet").read_bytes()).hexdigest(),
        "execution": execution.__dict__, "bootstrap": bootstrap, "increments": increments,
        "passed_comparison_gate": accepted, "recommended_production_replacement": False,
        "blind_test": False, "production_changed": False,
        "verification": {"model_and_features_prefix_equal": True, "ledger_prefix_equal": True,
                         "previous_MWU_reproduced": True, "all_model_labels_mature": True},
        "method_variants": {"adah2011": "paper algorithm 1, phi=2",
                            "squint_grid": "25 log-spaced learning rates, discrete uniform prior",
                            "multiwindow": "21/63/126/252 rolling Hedge plus meta Hedge; no strongly-adaptive theorem claim",
                            "robust_prox": "variance-penalized conservative prox; not a distributionally robust minimax solver"},
        "runtime": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__},
        "correctness_replay": correction_audit,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
            ROOT / "src/quant/research/byd_allocation.py", Path(__file__).resolve(),
            ROOT / "tests/test_byd_allocation.py", ROOT / "docs/plans/2026-10-13-byd-allocation-methods.md",
        )},
    }
    write_json(out / "manifest.json", manifest)
    family_names = [name for name in routes if name != "grid_search"]
    headline_names = ["selected_pipeline", "original_range_matched", "previous_frozen_MWU", "constant_one_third", "constant_half", "buy_hold"]
    main_table = markdown_table(summary[summary.period.eq("outer_2020plus") & summary.name.isin(headline_names)])
    family_table = markdown_table(summary[summary.period.eq("outer_2020plus") & summary.name.isin(family_names)])
    diagnostic_names = [name for name in ledgers if name.startswith(("fixed_", "prox_", "robust_without", "vol20_"))]
    diagnostics_table = markdown_table(summary[summary.period.eq("outer_2020plus") & summary.name.isin(diagnostic_names)])
    observed_table = markdown_table(summary[summary.period.eq("observed_2024plus") & summary.name.isin(headline_names)])
    concentration_table = pd.DataFrame(concentration).to_csv(index=False)
    selection_lines = "\n".join(f"- {year}：`{bank[key].method}` / `{bank[key].group}`，eta={bank[key].eta}，成本gamma={bank[key].cost_gamma}，风险={bank[key].risk}。" for year,key in routes["selected_pipeline"].items())
    bootstrap_lines = "\n".join(f"- 相对 {row['comparator']}：年化日均差 {row['annual_mean_difference']:.2%}，描述性95%区间 [{row['low95']:.2%}, {row['high95']:.2%}]。" for row in bootstrap)
    group_lines = "\n".join(f"- {name}：{', '.join(group)}。" for name, group in groups.items())
    report = f"""# BYD 长短期专家与优化方法验证

数据截止 {manifest['data_end']}；{len(market)}条原始日线；{len(targets.columns)}个专家；{len(bank)}个固定候选配置。

## 结论与评估边界

逐年流程对原区间策略的2020+累计收益差 {increments['original_range_matched']:.2%}（百分点），对上一轮MWU差 {increments['previous_frozen_MWU']:.2%}，对固定半仓差 {increments['constant_half']:.2%}。
预定义的收益、Sharpe与回撤联合比较门槛：{'通过' if accepted else '未通过'}。这不构成生产替换建议。
2020+是本轮按年份逐年仅使用过去数据选型的外层历史回放；研究设计已经受到项目历史和上一轮结果影响，尤其2024+已经观察，不能称为新盲测。算法家庭之间的外层排名也不能作为再次选型的独立证据。

{main_table}

![逐年选型净值](outer_equity.svg)

## 每个方法的独立逐年选型

{family_table}

每个方法只用该年前三个完整年度选择参数/专家池，上层selected_pipeline也只在过去内层分数中选择算法。所有候选从2015在线学习；真实选型账户跨年连续，2015-2019初始化统一为默认diverse MWU，不拼接不同候选的收益。年度配置在下一年第一笔开盘订单之前、上一交易日收盘时生效。
内层每年评分为Sharpe+.5*CAGR-.5*abs(MDD)，三个年度取均值-.35标准差。可用训练/初始数据为2011-2015；内层验证为2016起到外层前一年。标签成熟约束贯穿整个过程，年度边界不需要未来标签。候选拟合与内层验证是在固定历史路径上重放，未把未来年度分数传入选型器。

{selection_lines}

## 专家设计

{group_lines}

短期反转2/5日、中期均值回归20日、趋势20/60/120/250日、突破20/60日、大盘相对强弱63日、252日回撤防守和量价确认；新增Ridge1/21/63日、浅树21日，原5日模型保留。长短期指信号和预测窗口，都是每日收盘更新目标，并非互不相关的资产。
新增模型每21报价日重训，最多1260个历史成熟标签，最少300；特征j的h日标签为open(j+h+1)/open(j+1)，只在j+h+1<=训练日时使用。Ridge alpha50，树40棵深度2最小叶60，预先固定。模型输入含多窗口收益/均线距离、波动、区间位置、回撤、大盘相对强弱、量比、跳空和日内收益；无事后财报或文本发布时间假设。
专家去重每年只看之前756条专家净收益和仓位，按过去Sharpe排序；收益相关>=.95且仓位相关>=.97才视为重复。`pruning_audit.json`保存截止日期和保留名单。全外层相关矩阵与状态收益只是诊断，未反馈到选型。

## 公式与工程变体

MWU、分享和折扣复用同一收益尺度.02。AdaHedge2011使用mixability gap预算和phi2重启。Squint用25个[.001,.5]对数学习率的离散先验；多窗口候选是滚动Hedge的工程融合，不是完整强自适应论文算法。
条件Hedge有6个固定状态：120日均线距离<-3%、中间、>3%，各自按年化波动40%分两档。当天反馈只更新生成当日订单的前一收盘状态；然后读取当前状态生成下一订单。
OMD是带均匀先验L2回拉的欧氏投影梯度，步长eta/sqrt(t)；FTRL是L2正则累计效用投影，正则强度随sqrt(t)增大。
成本近端控制在实际账户收盘仓位上选择目标：min (x-proposal)^2/(2gamma)+risk*variance*x^2/2+预计费用/权益；包含不交易和原始目标，以及0到1的2%网格。预计费用按当日已知价/权益计算，含最低佣金、滑点、卖方税；随后仍按次日真实开盘价和整手约束成交，不读取次日开盘作决策。不交易分支保持股数，只处理公司行为与收盘估值，不因隔夜跳空再平衡。
robust_prox是风险惩罚版本，没有实现分布不确定集的minimax求解，因此不声称严格分布鲁棒保证。所有方法的理论专家线性损失保证均不直接等同于本账本净收益保证。

## 消融与共同波动上限

{diagnostics_table}

fixed_mwu保持eta.3及相同交易控制，仅改变专家池；rules去掉新增模型，pruned仅移除高度重复专家。prox_without_cost移除近端成本控制且保持原路径与其他参数；prox_fixed_band改用固定3%阈值；robust_without_risk仅移除风险罚。
vol20对预定方法及简单基线使用相同20%年化波动上限，只按当时20日市场波动缩减目标，未强制各方法实现完全相同的波动或仓位。风险匹配后的真实差异仍见平均仓位和回撤。

## 搜索方法

固定48配置空间，比较全网格、12次GP期望改进、12次固定随机，以及先看最早内层年度再全保留1/3候选的多保真淘汰。GP固定Matern核不再训练核超参数。搜索只能读取该年前的内层评分。
`search_comparison.csv`报告每年查询轨迹、预算和与全网格的分数差；预算是等价评分查询数，所有研究共享缓存，没有宣称实测运行加速。淘汰的低保真分数可能把后续更好的配置提前丢掉。

## 稳健性与已观察时段

{bootstrap_lines}

分块bootstrap长度20，2000次，只是描述配对差异，未校正多重搜索与项目历史观察。双倍滑点重新计算专家净收益、在线权重和成本控制，但保持每年方法/参数选型冻结；2020+收益{stress_metrics['total_return']:.2%}，Sharpe{stress_metrics['sharpe']:.2f}，回撤{stress_metrics['max_drawdown']:.2%}。

2024+历史回放：

{observed_table}

权重集中度（CSV，逐日最大权重均值与有效专家数）：

```csv
{concentration_table}```

## 检查与限制

模型/规则/市场特征2023前缀完整重放一致；年度过去选型、标签成熟、在线权重与真实账本前缀一致；上一轮冻结MWU逐日收益误差<=1e-12。
初次运行曾将成本控制的不交易选择误作次日比例目标，已修正为保持股数并增加隔夜跳空/送转测试。同一150候选库重跑，初次指标和协议保存在initial_replay；非成本方法与基线指标保持一致，不根据外层结果增加参数。
100万元、无杠杆、现金利息0；买入100股整手、全卖可处理零股；单日一次开盘净交易；参与率使用前一天成交量1%；成本与分红税储备沿用上一轮。大量历史涨跌停由主板10%参考价推算，分红到账及税务为简化假设，不能认证真实订单成交。只有日线，没有盘口、分钟排队、TWAP或RL成交验证；EXP3/Thompson等bandit没有必要丢弃这里可观察的全部专家反馈，本轮未将其当作融合算法。没有训练执行RL。
完成固定候选库即停止，没有根据外层结果扩大参数库，也没有宣称全局最优或未来收益收敛。唯一可作为未来验证方案的是预定义的逐年选型流程，而不是回头挑外层收益最高的方法。待新的前向数据积累再评估是否替换。

运行：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 scripts/research/run_byd_allocation.py`。
检查：`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -p 'test_byd_allocation.py' -v`。

## 研究来源

- [AdaHedge2011](https://papers.neurips.cc/paper/4191-adaptive-hedge.pdf)
- [Squint](https://proceedings.mlr.press/v40/Koolen15a.html)
- [Strongly Adaptive Online Learning](https://proceedings.mlr.press/v37/daniely15.html)
- [成本感知在线组合](https://ojs.aaai.org/index.php/AAAI/article/view/8693)
- [成本与约束组合优化](https://stanford.edu/~boyd/papers/portfolio.html)
"""
    (out / "report.md").write_text(report, encoding="utf-8")
    print(main_table, flush=True)
    print(f"Saved {out / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
