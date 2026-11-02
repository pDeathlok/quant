import concurrent.futures, datetime, json, threading, time
from pathlib import Path
import akshare as ak
import pandas as pd
import requests

OUT=Path('reports/disclosure_date_validation_20261029')
local=threading.local()
original=requests.sessions.Session.request

def bounded(self, method, url, **kwargs):
    kwargs.setdefault('timeout', (8, 20))
    local.requests=getattr(local,'requests',0)+1
    return original(self,method,url,**kwargs)
requests.sessions.Session.request=bounded

def run(job):
    source,period,market=job
    local.requests=0
    started=time.monotonic()
    result=dict(source=source,period=period,market=market)
    try:
        if source=='eastmoney':
            df=ak.stock_yysj_em(symbol=market,date=period)
            df=df.rename(columns={'首次预约时间':'first','一次变更日期':'change1','二次变更日期':'change2','三次变更日期':'change3','实际披露时间':'actual'})
        else:
            df=ak.stock_report_disclosure(market=market,period=period)
            df=df.rename(columns={'首次预约':'first','初次变更':'change1','二次变更':'change2','三次变更':'change3','实际披露':'actual'})
        df.to_csv(OUT/f'{source}_{period}_{market}.csv',index=False)
        for c in ['first','change1','change2','change3','actual']:
            df[c]=pd.to_datetime(df[c],errors='coerce')
        latest=df[['first','change1','change2','change3']].ffill(axis=1).iloc[:,-1]
        changed=df[['change1','change2','change3']].notna().any(axis=1)
        paired=df.actual.notna() & latest.notna()
        gap=(df.actual-latest).dt.days
        firstgap=(df.actual-df['first']).dt.days
        result.update(rows=len(df),duplicate_codes=int(df['股票代码'].duplicated().sum()),first_available=int(df['first'].notna().sum()),actual_available=int(df.actual.notna().sum()),changed_count=int(changed.sum()),changed_pct=round(float(changed.mean()*100),3),paired=int(paired.sum()),actual_latest_mismatch=int((gap[paired]!=0).sum()),actual_latest_gap_days=gap[paired].value_counts().sort_index().to_dict(),actual_first_mismatch=int((firstgap[df.actual.notna()&df['first'].notna()]!=0).sum()),latest_future=int((latest>pd.Timestamp('2026-10-29')).sum()))
    except Exception as e:
        result['error']=f'{type(e).__name__}: {e}'
    result.update(requests=local.requests,seconds=round(time.monotonic()-started,2))
    print(json.dumps(result,ensure_ascii=False,default=str),flush=True)
    return result
jobs=[('eastmoney',p,'沪深A股') for p in ['20260930','20260630','20251231','20240930']]+[('eastmoney','20260930','京市A股'),('cninfo','2026三季','沪深京')]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,jobs))
(OUT/'summary.json').write_text(json.dumps({'akshare_version':ak.__version__,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':results},ensure_ascii=False,indent=2,default=str))
