"""Resumable Tushare option history snapshot; explicit pagination to empty page."""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

OUT = Path("data/research/index_futures/option_iv/sources")
END = "20261106"


class Client:
    def __init__(self):
        load_dotenv()
        self.token = os.environ["TUSHARE_TOKEN"]
        self.session = requests.Session()
        self.audit = []

    def query(self,api,params):
        time.sleep(.25)
        for attempt in range(5):
            try:
                response = self.session.post("https://api.tushare.pro",json=dict(api_name=api,token=self.token,params=params),timeout=40)
                response.raise_for_status()
                data = response.json()
                if data.get("code") != 0:
                    message = str(data.get("msg", "")).replace(self.token,"[redacted]")[:200]
                    raise RuntimeError(f"{api}: {data.get('code')} {message}")
                return pd.DataFrame(data["data"]["items"],columns=data["data"]["fields"])
            except (requests.RequestException,ValueError,RuntimeError) as exc:
                if attempt == 4:
                    if isinstance(exc,RuntimeError): raise
                    raise RuntimeError(f"{api}: {type(exc).__name__}") from None
                time.sleep(min(2**attempt,8))

    def pages(self,api,params,key,identity):
        directory = OUT/"pages"/key
        directory.mkdir(parents=True,exist_ok=True)
        frames,offset,seen = [],0,set()
        while True:
            path = directory/f"{offset}.parquet"
            if path.exists(): frame = pd.read_parquet(path)
            else:
                frame = self.query(api,dict(**params,limit=10000,offset=offset))
                tmp = path.with_suffix(".tmp");frame.to_parquet(tmp,index=False);tmp.replace(path)
            self.audit.append(dict(api=api,params=params,offset=offset,rows=len(frame),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if frame.empty: break
            ids = set(map(tuple,frame[identity].to_numpy()))
            if len(ids)!=len(frame) or ids&seen:
                raise RuntimeError(f"{key}: duplicate identity or pagination ignored at {offset}")
            seen |= ids;frames.append(frame);offset += len(frame)
            if offset>2_000_000: raise RuntimeError(f"Unbounded pagination: {key}")
        return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def run(only=None):
    OUT.mkdir(parents=True,exist_ok=True)
    client = Client()
    for exchange in (only,) if only else ("CFFEX","SSE"):
        basic = client.pages("opt_basic",dict(exchange=exchange),f"basic_{exchange}",["ts_code"])
        basic.to_parquet(OUT/f"contracts_{exchange}.parquet",index=False)
        print("contracts",exchange,len(basic),basic.list_date.min(),basic.list_date.max(),flush=True)
        begin = "2019-12" if exchange=="CFFEX" else "2015-02"
        months = list(pd.period_range(begin,"2026-11",freq="M"))[::-1]
        for month in months:
            first = month.start_time.strftime("%Y%m%d")
            last = min(month.end_time.strftime("%Y%m%d"),END)
            target = OUT/f"daily_{exchange}_{month}.parquet"
            frame = client.pages("opt_daily",dict(exchange=exchange,start_date=first,end_date=last),f"daily_{exchange}_{month}",["ts_code","trade_date"])
            if not frame.empty:
                if not frame.trade_date.between(first,last).all():
                    raise RuntimeError(f"Out-of-range dates: {exchange} {month}")
                frame.sort_values(["trade_date","ts_code"]).to_parquet(target,index=False)
            print("daily",exchange,str(month),len(frame),flush=True)
        (OUT/f"download_audit_{exchange}.json").write_text(json.dumps(dict(retrieved_at=datetime.now(timezone.utc).isoformat(),end=END,pages=client.audit),ensure_ascii=False,indent=2))
    rates = client.pages("shibor",dict(start_date="20150101",end_date=END),"shibor",["date"])
    rates.sort_values("date").to_parquet(OUT/"shibor.parquet",index=False)
    print("shibor",len(rates),flush=True)
    (OUT/"download_audit.json").write_text(json.dumps(dict(retrieved_at=datetime.now(timezone.utc).isoformat(),end=END,pages=client.audit),ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--exchange",choices=("CFFEX","SSE"))
    run(parser.parse_args().exchange)
