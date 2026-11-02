import concurrent.futures,json,time
from pathlib import Path
import requests,pandas as pd
OUT=Path('reports/disclosure_date_validation_20261029')
MAP={'SECURITY_CODE':'股票代码','SECURITY_NAME_ABBR':'股票简称','FIRST_APPOINT_DATE':'first','FIRST_CHANGE_DATE':'change1','SECOND_CHANGE_DATE':'change2','THIRD_CHANGE_DATE':'change3','ACTUAL_PUBLISH_DATE':'actual'}
def run(period):
 start=time.monotonic(); calls=0; frames=[]
 params={'reportName':'RPT_PUBLIC_BS_APPOIN','columns':'ALL','pageSize':6000,'pageNumber':1,'sortColumns':'SECURITY_CODE','sortTypes':'1','filter':f"(REPORT_DATE='{period}')"}
 try:
  while True:
   data=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params=params,timeout=(8,20)).json(); calls+=1
   result=data.get('result')
   if not result: raise ValueError(str(data)[:250])
   frames+=result['data']
   if params['pageNumber']>=result['pages']:break
   params['pageNumber']+=1
  (OUT/f'raw_eastmoney_{period}.json').write_text(json.dumps(frames,ensure_ascii=False))
  df=pd.DataFrame(frames).rename(columns=MAP)
  df.to_csv(OUT/f'raw_eastmoney_{period}.csv',index=False)
  for c in ['first','change1','change2','change3','actual']:df[c]=pd.to_datetime(df[c],errors='coerce')
  latest=df[['first','change1','change2','change3']].ffill(axis=1).iloc[:,-1]
  changed=df[['change1','change2','change3']].notna().any(axis=1)
  paired=df.actual.notna()&latest.notna();gap=(df.actual-latest).dt.days
  firstpaired=df.actual.notna()&df['first'].notna();fgap=(df.actual-df['first']).dt.days
  row=dict(period=period,rows=len(df),declared_count=result['count'],duplicate_codes=int(df['股票代码'].duplicated().sum()),first_available=int(df['first'].notna().sum()),actual_available=int(df.actual.notna().sum()),changed_count=int(changed.sum()),changed_pct=round(changed.mean()*100,3),paired=int(paired.sum()),actual_latest_mismatch=int((gap[paired]!=0).sum()),actual_latest_gap_days=gap[paired].value_counts().sort_index().to_dict(),actual_first_mismatch=int((fgap[firstpaired]!=0).sum()),actual_first_gap_abs_median=float(fgap[firstpaired].abs().median()),actual_first_gap_abs_p95=float(fgap[firstpaired].abs().quantile(.95)),markets=df['TRADE_MARKET'].value_counts().to_dict())
 except Exception as e:row=dict(period=period,error=str(e))
 row.update(requests=calls,seconds=round(time.monotonic()-start,2));print(json.dumps(row,ensure_ascii=False,default=str),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 rows=list(pool.map(run,['2026-09-30','2026-06-30','2025-12-31','2024-09-30','2008-12-31']))
(OUT/'raw_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,default=str))
