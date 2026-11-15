"""Retrieve dated industry memberships through the configured data provider."""
import json
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(Path(__file__).resolve().parent)]
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.ml.rank_fusion import file_sha256
from model_iteration_v2_protocol import OUT


def main():
    load_dotenv(ROOT / '.env')
    folder = OUT / 'industry_source'
    folder.mkdir(parents=True, exist_ok=True)
    fetcher = TushareDataFetcher(cache_dir=ROOT / 'data/cache/source_merge/tushare')
    classifications = fetcher.pro.index_classify(level='L1', src='SW2021')
    if classifications.empty:
        raise RuntimeError('No SW2021 classifications')
    classifications.to_parquet(folder / 'classifications.parquet', index=False)
    pieces = []
    for code in sorted(classifications.index_code.unique()):
        for latest in ['Y', 'N']:
            path = folder / f'{code}_{latest}.parquet'
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                frame = fetcher.pro.index_member_all(l1_code=code, is_new=latest)
                if len(frame) >= 2000:
                    raise RuntimeError(f'Membership response may be truncated: {code}')
                frame.to_parquet(path, index=False)
                time.sleep(.55)
            pieces.append(frame)
        print('INDUSTRY', code, flush=True)
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates()
    required = {'ts_code', 'l1_code', 'in_date', 'out_date'}
    if not required <= set(combined):
        raise RuntimeError('Membership interval fields absent')
    combined.to_parquet(folder / 'memberships.parquet', index=False)
    (folder / 'manifest.json').write_text(json.dumps({'provider': 'Tushare index_member_all',
        'classification': 'SW2021', 'rows': len(combined), 'industry_count': len(classifications),
        'sha256': file_sha256(folder / 'memberships.parquet'),
        'limitation': 'Retrospective vendor membership intervals; not archived announcement/revision vintages.'}, indent=2))


if __name__ == '__main__':
    main()
