"""Active application policy and immutable selector queue presentation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from quant.ml.rank_fusion import file_sha256

ROOT = Path(__file__).resolve().parents[3]
CONFIG = 'configs/strategies/application_portfolio.json'


def load_portfolio_config(root: Path = ROOT) -> dict:
    path = root / CONFIG
    if not path.exists():
        return {'enabled': False}
    value = json.loads(path.read_text())
    if type(value.get('enabled')) is not bool:
        raise ValueError('Portfolio enabled flag must be boolean')
    if value['enabled']:
        for key in ['policy', 'rankings', 'inputs', 'manifest']:
            target = (root / value[key]).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Portfolio path escapes project')
        if file_sha256(root / value['policy']) != value['policy_sha256']:
            raise ValueError('Active portfolio policy identity changed')
    return value


def project_portfolio(portfolio: dict | None, *, side: str = 'all', symbols: set[str] | None = None, limit: int = 20) -> dict | None:
    if not portfolio:
        return None
    sleeves = []
    for sleeve in portfolio['sleeves']:
        rows = [row for row in sleeve['candidates']
                if (side == 'all' or row['side'] == side) and (symbols is None or row['symbol'] in symbols)]
        sleeves.append({**sleeve, 'visible_eligible_count': len(rows), 'candidates': rows[:limit] if limit > 0 else rows})
    return {**portfolio, 'sleeves': sleeves, 'view_side': side, 'view_filtered': symbols is not None or side != 'all'}


def attach_portfolio(payload: dict, *, resolve_path: Callable[[Path], Path],
                     require_full_coverage: bool = False, root: Path = ROOT, config: dict | None = None) -> dict:
    config = config or load_portfolio_config(root)
    date = str(payload.get('signal_date') or '')
    if not config['enabled'] or not date or pd.Timestamp(date) < pd.Timestamp(config['active_from']):
        return payload
    path = resolve_path(root / config['manifest'])
    metadata = json.loads(path.read_text())
    rankings_path = resolve_path(root / config['rankings'])
    if metadata.get('target_date') != date or metadata.get('policy_sha256') != config['policy_sha256']:
        raise RuntimeError('Application portfolio date or release differs from selector')
    if metadata.get('status') != 'success' or file_sha256(rankings_path) != metadata['rankings_sha256']:
        raise RuntimeError('Application portfolio rankings are incomplete or changed')
    rankings = pd.read_parquet(rankings_path)
    if not rankings.empty and (not pd.to_datetime(rankings.date).eq(date).all() or rankings.duplicated(['sleeve', 'symbol']).any()):
        raise RuntimeError('Invalid application portfolio ranking keys')
    stocks = {row['symbol']: row for row in payload.get('stocks', [])}
    missing = set(rankings.symbol) - set(stocks)
    if missing and require_full_coverage:
        raise RuntimeError(f'Selector omitted application candidates: {sorted(missing)[:20]}')
    sleeves = []
    for name, weight in metadata['weights'].items():
        rows = []
        for ranking in rankings.loc[rankings.sleeve.eq(name)].sort_values('priority').to_dict('records'):
            stock = stocks.get(ranking['symbol'])
            if stock is None:
                continue
            expected = ranking.get('expected_net')
            rows.append({'symbol': ranking['symbol'], 'name': stock.get('name', ranking['symbol']),
                         'side': ranking['side'], 'score': float(ranking['score']),
                         'priority': int(ranking['priority']), 'close': stock.get('close'),
                         'expected_net': None if expected is None or pd.isna(expected) else float(expected)})
        sleeves.append({'key': name, 'label': metadata['labels'].get(name, name),
                        'initial_capital_weight': weight, 'eligible_count': int(rankings.sleeve.eq(name).sum()),
                        'candidates': rows, 'conditions': metadata.get('conditions', {}).get(name, {})})
    return {**payload, 'application_portfolio': {'schema_version': 'selector-portfolio-v1',
        'release_id': metadata['release_id'], 'policy_sha256': metadata['policy_sha256'],
        'signal_date': date, 'primary_application': True, 'sleeves': sleeves, 'account': metadata['account'],
        'execution_note': '各路独立资金记账。候选队列需先排除已有持仓，每路每天最多尝试新开20只，持有5个交易日。'}}
