"""Repair the required prior-session flow context using cached trade dates."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from quant.data.tushare_fetcher import TushareDataFetcher
from quant.data.long_factor_backfill import backfill_trade_date_partitions,RequestPolicy
from dotenv import load_dotenv


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True);args=parser.parse_args()
    raw=ROOT/'data/raw';dates=sorted(p.stem for p in (raw/'daily_basic').glob('*.parquet') if p.stem.isdigit() and p.stem<args.date)[-60:]
    load_dotenv(ROOT/'.env',override=False)
    fetcher=TushareDataFetcher(cache_dir=ROOT/'data/cache/source_merge/tushare')
    audit=raw/'source_audit'/f'rank_flows_{args.date}';audit.mkdir(parents=True,exist_ok=True)
    for kind in ['margin_detail','moneyflow']:
        result=backfill_trade_date_partitions(fetcher.pro,kind,dates[0],dates[-1],raw,audit,
            trade_dates=dates,policy=RequestPolicy(retries=2,sleep_seconds=.15,max_retry_wait_seconds=30))
        (audit/f'{kind}_result.json').write_text(json.dumps(result,default=str,indent=2))
        print(kind,result['status'],flush=True)
        if result['status']!='success':raise SystemExit('Flow source context incomplete; see audit')


if __name__=='__main__':main()
