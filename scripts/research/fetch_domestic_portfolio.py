"""Fetch a frozen domestic ETF snapshot without exposing credentials."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/research/domestic_portfolio_20261013/source'
CODES=('510300.SH','510500.SH','518880.SH','511010.SH','512880.SH','510880.SH')
END='20261012'


def token_value() -> str:
    for line in (ROOT/'.env').read_text().splitlines():
        key,sep,value=line.strip().partition('=')
        if sep and key=='TUSHARE_TOKEN':return value.strip().strip('\"\'')
    raise RuntimeError('Configured Tushare token unavailable')


def query(client: requests.Session, token: str, api: str, params: dict, fields: str='') -> pd.DataFrame:
    try:
        response=client.post('https://api.tushare.pro',json={'api_name':api,'token':token,'params':params,'fields':fields},timeout=35.)
        response.raise_for_status();data=response.json()
    except Exception as error:
        raise RuntimeError(f'{api} network failure: {type(error).__name__}') from None
    if data.get('code')!=0:
        raise RuntimeError(f"{api}: {str(data.get('msg','')).replace(token,'[redacted]')[:180]}")
    return pd.DataFrame(data['data']['items'],columns=data['data']['fields'])


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True);token=token_value();coverage=[]
    with requests.Session() as client:
        for code in CODES:
            name=code.replace('.','_');destination=OUT/f'{name}.parquet'
            if destination.exists():
                frame=pd.read_parquet(destination)
            else:
                frames=[]
                for year in range(2011,2027):
                    parts={}
                    for api,fields in [('fund_daily','ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount'),('fund_adj','ts_code,trade_date,adj_factor')]:
                        cache=OUT/f'{name}_{year}_{api}.parquet'
                        if cache.exists():parts[api]=pd.read_parquet(cache)
                        else:
                            parts[api]=query(client,token,api,{'ts_code':code,'start_date':f'{year}0101','end_date':min(f'{year}1231',END)},fields)
                            parts[api].to_parquet(cache,index=False)
                    if len(parts['fund_daily']):
                        f=parts['fund_daily'].merge(parts['fund_adj'],on=['ts_code','trade_date'],how='left',validate='one_to_one')
                        if f.adj_factor.isna().any():raise RuntimeError(f'Missing factors for {code} {year}')
                        frames.append(f)
                    if year%4==0:print(f'{code}: fetched through {year}',flush=True)
                frame=pd.concat(frames,ignore_index=True).sort_values('trade_date').reset_index(drop=True)
                if frame.duplicated('trade_date').any():raise RuntimeError('Duplicate quotes')
                frame.to_parquet(destination,index=False)
            div=OUT/f'{name}_dividends.parquet'
            if not div.exists():query(client,token,'fund_div',{'ts_code':code}).to_parquet(div,index=False)
            coverage.append({'code':code,'first_quote':str(frame.trade_date.min()),'last_quote':str(frame.trade_date.max()),'rows':len(frame)})
            print(coverage[-1],flush=True)
        cal=OUT/'calendar.parquet'
        if not cal.exists():query(client,token,'trade_cal',{'exchange':'SSE','start_date':'20110101','end_date':END},'exchange,cal_date,is_open').to_parquet(cal,index=False)
    (OUT/'coverage.json').write_text(json.dumps(coverage,indent=2))

if __name__=='__main__':main()
