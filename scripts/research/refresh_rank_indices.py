"""Bootstrap missing index history once, then use the shared incremental owner."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.features.active_market_value import KEY_INDEX_SPECS
from quant.routine.reference_data_refresh import refresh_index_daily
from quant.data.atomic_io import atomic_write_parquet


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);args=parser.parse_args()
    raw=ROOT/'data/raw';fetcher=TushareDataFetcher(cache_dir=ROOT/'data/cache/source_merge/tushare')
    for spec in KEY_INDEX_SPECS:
        path=raw/f'index_{spec.ts_code}.parquet'
        if not path.exists():
            seed=ROOT/f'reports/research/right_t012_index_sources_v1/{spec.ts_code}_index_daily.parquet'
            atomic_write_parquet(pd.read_parquet(seed),path,index=False)
        try:
            result=refresh_index_daily(fetcher,raw,args.date,index_code=spec.ts_code)
        except Exception as exc:
            raise SystemExit(f'Index refresh failed: {spec.ts_code}, {type(exc).__name__}') from None
        print(spec.ts_code,result['status'],result.get('new_rows'),flush=True)
        if result['status']!='success':raise SystemExit('Provider has not published the required index date')


if __name__=='__main__':main()
