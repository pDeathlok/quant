import json,time
from pathlib import Path
import requests,pandas as pd,akshare as ak
OUT=Path('reports/disclosure_date_validation_20261029')
rows=[]
for size in [1000,5000]:
 t=time.monotonic()
 try:
  r=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',params={'reportName':'RPT_PUBLIC_BS_APPOIN','columns':'ALL','pageSize':size,'pageNumber':1,'filter':"(REPORT_DATE='2026-09-30')"},timeout=20).json()
  result=r['result'];row={'requested_page_size':size,'returned_rows':len(result['data']),'pages':result['pages'],'count':result['count']}
 except Exception as e:row={'requested_page_size':size,'error':str(e)}
 row['seconds']=round(time.monotonic()-t,2); rows.append(row); print(row,flush=True)
comparisons=[]
for period,rawdate in [('2026半年报','2026-06-30'),('2025年报','2025-12-31'),('2026三季','2026-09-30')]:
 t=time.monotonic()
 try:
  path=OUT/f'cninfo_{period}_沪深京.csv'
  if path.exists():df=pd.read_csv(path,dtype={'股票代码':str})
  else:
   df=ak.stock_report_disclosure(market='沪深京',period=period).rename(columns={'首次预约':'first','初次变更':'change1','二次变更':'change2','三次变更':'change3','实际披露':'actual'})
   df.to_csv(path,index=False)
  em=pd.read_csv(OUT/f'raw_eastmoney_{rawdate}.csv',dtype={'股票代码':str})
  merged=em.merge(df,on='股票代码',suffixes=('_em','_cn'))
  mismatch={}
  for col in ['first','change1','change2','change3','actual']:
   left=pd.to_datetime(merged[col+'_em'],errors='coerce');right=pd.to_datetime(merged[col+'_cn'],errors='coerce')
   mismatch[col]=int((~(left.eq(right)|(left.isna()&right.isna()))).sum())
  row={'period':period,'cninfo_rows':len(df),'common':len(merged),'only_em':sorted(set(em['股票代码'])-set(df['股票代码'])),'only_cn':sorted(set(df['股票代码'])-set(em['股票代码'])),'field_mismatches':mismatch,'seconds':round(time.monotonic()-t,2)}
 except Exception as e:row={'period':period,'error':str(e)}
 comparisons.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
(OUT/'comparison.json').write_text(json.dumps({'page_sizes':rows,'cross_source':comparisons},ensure_ascii=False,indent=2))
