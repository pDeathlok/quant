"""Run one configured left release without fetching data from the scorer."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from quant.application.left_side_ranking import load_left_side_ranking_config
from quant.routine.left_side_unified_production import run_left_side_production


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--date',required=True)
    parser.add_argument('--config',default='configs/strategies/left_side_unified.yaml')
    args=parser.parse_args();config=load_left_side_ranking_config(config_path=args.config)
    result=run_left_side_production(args.date,config=config)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str),flush=True)


if __name__=='__main__':main()
