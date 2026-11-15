"""Materialize the selected release against the currently published candidates."""
import json
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from quant.application.daily_dependencies import DEFAULT_DAILY_DEPENDENCY_REGISTRY
from quant.infrastructure.publication import PublicationStore, publication_context
from quant.ml.rank_fusion import file_sha256
from quant.routine.application_portfolio_production import run_portfolio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', help='Defaults to the committed generation target date')
    args = parser.parse_args()
    out = ROOT / 'reports/research/model_iteration_v2'
    decision = json.loads((out / 'deployment_decision.json').read_text())
    version = decision['release'].rsplit('_', 1)[-1]
    policy = f'models/candidates/application_policy_{version}/policy.json'
    config = dict(enabled=True, policy=policy, policy_sha256=file_sha256(ROOT / policy),
                  **{k: f'reports/research/model_iteration_v2/live_preview/{v}' for k, v in
                     [('rankings', 'rankings.parquet'), ('inputs', 'inputs.parquet'), ('manifest', 'manifest.json')]})
    managed = tuple(sorted({p for n in DEFAULT_DAILY_DEPENDENCY_REGISTRY.nodes.values() for p in n.publication_outputs}))
    view = PublicationStore(ROOT, managed).view()
    pointer = json.loads((ROOT / 'data/publications/current.json').read_text())
    target_date = args.date or pointer['target_trade_date']
    with publication_context(view):
        result = run_portfolio(target_date, config=config)
    result['source_generation'] = view.generation
    (out / 'live_preview/provenance.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: result[k] for k in ['status', 'release_id', 'candidate_rows', 'ranking_rows', 'source_generation']}))


if __name__ == '__main__':
    main()
