"""Read-only HTTP acceptance of new and historical published selector scores."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]


def fetch(**parameters: str) -> tuple[dict, dict]:
    url = 'http://127.0.0.1:8088/api/selector/stocks?' + urlencode(parameters)
    with urlopen(url, timeout=120) as response:
        assert response.status == 200
        return json.load(response), dict(response.headers)


def main() -> None:
    status = json.loads((ROOT / 'data/routine/latest_refresh_status.json').read_text())
    assert status['status'] == 'success' and status['trade_date'].replace('-', '') == '20261111'
    pointer = json.loads((ROOT / 'data/publications/current.json').read_text())
    artifact = ROOT / 'models/production/left_side_ranknet_v6/manifest.json'
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    historical = json.loads((ROOT / 'reports/research/left_ranknet_rollout/api_after_display_fix_historical.json').read_text())
    current, headers = fetch(side='left', include_extended='true', signal_date='2026-11-11')
    assert current['signal_date'] == '2026-11-11'
    header_generation = {k.lower(): v for k, v in headers.items()}['x-quant-generation']
    assert header_generation == pointer['generation']
    left = [r for r in current['stocks'] if r.get('ranking_source') == 'left_side_unified']
    assert left, 'No new left rows are visible in the published API'
    assert all(r['ranking_model_artifact_sha256'] == artifact_sha for r in left)
    assert all(r['ranking_model_engine'] == 'rank_fusion' for r in left)
    assert all(r['ranking_model_release_id'] == 'left-ranknet-v6-20261111' for r in left)
    calibrations = [c['key'] for c in current['score_probability_bands']['calibrations']]
    assert 'model_left' not in calibrations and 'model_left_v5_shadow' not in calibrations
    old, old_headers = fetch(strategies='B1', signal_date='2026-11-10')
    assert old['signal_date'] == '2026-11-10'
    old_left = {r['symbol']: r for r in old['stocks'] if r.get('ranking_source') == 'left_side_unified'}
    assert set(old_left) == {r['symbol'] for r in historical['left']}
    for before in historical['left']:
        after = old_left[before['symbol']]
        for key in ['ranking_score', 'ranking_model_artifact_sha256']:
            assert after[key] == before[key], (before['symbol'], key)
    right_payload, _ = fetch(side='right', include_extended='true', signal_date='2026-11-11')
    right_hashes = sorted({r['ranking_model_artifact_sha256'] for r in right_payload['stocks']
                           if r.get('ranking_source') == 'right_side_unified'})
    assert right_hashes == ['6168648ea8a3c3a44f5f4624604ac44f49bf13510e923737faf18750cd46c50e']
    report = dict(passed=True, generation=header_generation, date=current['signal_date'],
                  current_left_count=len(left), current_left_artifact_sha256=artifact_sha,
                  current_left_symbols=[r['symbol'] for r in left],
                  current_calibration_keys=calibrations,
                  historical_date=old['signal_date'], historical_scores_unchanged=True,
                  historical_generation={k.lower(): v for k, v in old_headers.items()}.get('x-quant-generation'),
                  right_artifact_sha256=right_hashes)
    output = ROOT / 'reports/research/left_ranknet_rollout/api_release_acceptance.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
