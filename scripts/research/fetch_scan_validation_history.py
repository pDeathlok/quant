"""Read-only, resumable research supplement using existing Tushare configuration.

Downloads only missing delisted daily series and yearly name-change records;
never updates canonical stores or emits credential-bearing errors.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import httpx
import pandas as pd

from quant.routine.paths import load_project_env


async def fetch(missing: Path, output: Path) -> None:
    load_project_env()
    token = os.environ.get("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Existing market-data credential unavailable")
    output.mkdir(parents=True,exist_ok=True)
    (output/"delisted").mkdir(exist_ok=True)
    (output/"names").mkdir(exist_ok=True)
    stocks=pd.read_csv(missing,dtype={"ts_code":str})
    tasks=[]
    for row in stocks.itertuples(index=False):
        tasks.append(("daily",{"ts_code":row.ts_code,"start_date":"20170101","end_date":"20260930"},
                      "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
                      output/"delisted"/(row.ts_code+".parquet")))
    for year in range(1990,2027):
        tasks.append(("namechange",{"start_date":f"{year}0101","end_date":min(f"{year}1231","20261007")},
                      "ts_code,name,start_date,end_date,ann_date,change_reason",output/"names"/(str(year)+".parquet")))
    lock=asyncio.Lock();parallel=asyncio.Semaphore(4);next_start=0.;audit=[]
    async with httpx.AsyncClient(timeout=25) as client:
        async def one(api: str,params: dict,fields: str,path: Path) -> None:
            nonlocal next_start
            async with parallel:
                if path.exists():
                    audit.append({"api":api,"params":params,"status":"cache","rows":len(pd.read_parquet(path))})
                    return
                for attempt in range(2):
                    async with lock:
                        await asyncio.sleep(max(0,next_start-time.monotonic()))
                        next_start=time.monotonic()+.2
                    try:
                        r=await client.post("https://api.tushare.pro",json={"api_name":api,"token":token,"params":params,"fields":fields})
                        r.raise_for_status();body=r.json()
                        if body.get("code")!=0 or body.get("data") is None:
                            audit.append({"api":api,"params":params,"status":"provider_error","code":body.get("code")})
                            break
                        data=body["data"];frame=pd.DataFrame(data["items"],columns=data["fields"])
                        if api=="daily" and len(frame):
                            if not frame.ts_code.eq(params["ts_code"]).all() or frame.duplicated(["ts_code","trade_date"]).any():
                                raise ValueError("Unexpected stock identity or duplicate key")
                        if len(frame)>=5000:
                            audit.append({"api":api,"params":params,"status":"possible_truncation","rows":len(frame)})
                            break
                        frame.to_parquet(path,index=False)
                        audit.append({"api":api,"params":params,"status":"success" if len(frame) else "empty","rows":len(frame)})
                        break
                    except (httpx.HTTPError,ValueError) as exc:
                        if attempt:
                            audit.append({"api":api,"params":params,"status":"failed","error_type":type(exc).__name__})
                        else:
                            await asyncio.sleep(1)
                if len(audit)%20==0:
                    print(f"History supplement {len(audit)}/{len(tasks)}",flush=True)
                (output/"fetch_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
        await asyncio.gather(*(one(*task) for task in tasks))
    result={"requests":len(tasks),"success":sum(a["status"] in ["success","cache"] for a in audit),
            "empty":sum(a["status"]=="empty" for a in audit),"failed":sum(a["status"] not in ["success","cache","empty"] for a in audit),
            "scope":"Missing known delisted stocks plus ann-date-filtered namechanges; not independent proof of exhaustive lifecycle or name history",
            "canonical_store_modified":False}
    (output/"complete.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--missing",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(fetch(args.missing,args.output))
