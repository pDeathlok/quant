"""Frozen transition-right workspace: actual holdings and next-close instructions."""
from __future__ import annotations
import json,os,re,fcntl,hashlib,threading,inspect
from datetime import datetime,date,timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from quant.data.atomic_io import atomic_write_json,atomic_write_parquet
from quant.data.source_merge import normalize_tushare_daily,normalize_ts_code
from quant.features.variable_library import build_continuous_ohlc
from quant.data.amount_units import resolve_amount_kcny

POLICIES={'return':dict(label='收益型',fraction=1.,trail=.06,tp=0.,sl=0.),'win':dict(label='胜率型',fraction=.5,trail=0.,tp=.03,sl=.05)}
INDICES={'000001.SH':'上证指数','399001.SZ':'深证成指','399006.SZ':'创业板指','000016.SH':'上证50','000300.SH':'沪深300','000905.SH':'中证500','000852.SH':'中证1000'}
RULES=['仅市场均线处于过渡阶段时产生新买入信号，不要求连续三日确认。','创业板/科创板；至少250根行情、股价≥3元、20日成交额中位数≥5000万元、信号日涨幅<9.5%。','EMA144>EMA169、收盘>EMA144、3日涨幅>0、距MA20在-2%至8%；按20日波动从低到高排序。','收盘确认后，下一交易日收盘尝试成交；先卖后买，最多5仓，空槽留现金，不加仓、不按排名强制换股。','收益型：每仓预算为权益20%，最高收盘回撤6%触发退出；胜率型：每仓10%，收盘涨3%/跌5%触发退出。','最长持有入场后20个交易日；触发价格退出后下一交易日收盘卖，市场状态变化不强制卖。','成交需满足T+1、实际交易价格、整手和成交容量；止盈止损为收盘触发条件，不是保证成交价。']
_LOCK=threading.RLock()
_DATA_LOCK=threading.RLock()

def today():return datetime.now(ZoneInfo('Asia/Shanghai'))
def iso(x):return pd.Timestamp(x).date().isoformat()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,(pd.Timestamp,date)):return iso(x)
 if isinstance(x,(np.integer,np.bool_)):return x.item()
 if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
 return x

def account_path(root):return Path(root)/'data/transition/accounts.json'
def get_account(root,policy):
 if policy not in POLICIES:raise ValueError('未知策略')
 p=account_path(root)
 if not p.exists():return dict(revision=0,cash=0.,positions=[],confirmed=False)
 data=json.loads(p.read_text());return data.get(policy,dict(revision=0,cash=0.,positions=[],confirmed=False))
def save_account(root,policy,body,calendar=None):
 if policy not in POLICIES:raise ValueError('未知策略')
 cash=float(body['cash'])
 if not np.isfinite(cash) or cash<0:raise ValueError('可用现金必须为非负有限数值')
 positions=body.get('positions',[]);seen=set();normalized=[]
 if len(positions)>50:raise ValueError('持仓记录过多')
 for row in positions:
  symbol=normalize_ts_code(row['symbol'])
  if re.fullmatch(r'(?:3\d{5}\.SZ|688\d{3}\.SH)',symbol) is None:raise ValueError('此策略持仓限创业板、科创板普通股')
  if symbol in seen:raise ValueError('同一股票只能有一条持仓，请合并数量和成本')
  seen.add(symbol);entry=pd.Timestamp(row['entry_date']).normalize();qty=int(row['quantity']);cost=float(row['cost'])
  if entry.date()>today().date():raise ValueError('不能把未来计划记录为实际持仓')
  if calendar is not None and entry not in pd.DatetimeIndex(calendar):raise ValueError('入场日期不是已确认的交易日')
  if qty<=0 or qty!=row['quantity'] or not np.isfinite(cost) or cost<=0:raise ValueError('持仓数量和成本必须大于0')
  normalized.append(dict(symbol=symbol,entry_date=iso(entry),quantity=qty,cost=cost))
 p=account_path(root);p.parent.mkdir(parents=True,exist_ok=True)
 with _LOCK,p.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  data=json.loads(p.read_text()) if p.exists() else {};old=data.get(policy,dict(revision=0))
  if int(body.get('revision',-1))!=old['revision']:raise ValueError('持仓记录已被其他页面更新，请重新加载后保存')
  result=dict(revision=old['revision']+1,cash=cash,positions=normalized,confirmed=True,updated_at=today().isoformat());data[policy]=result;atomic_write_json(data,p)
  return result

def symbol_features(frame,symbol):
 n=normalize_tushare_daily(frame,symbol).sort_values('date').drop_duplicates('date',keep='last').reset_index(drop=True)
 adj=build_continuous_ohlc(n);c=adj.close;amount,_=resolve_amount_kcny(n)
 ma=c.rolling(20).mean();e144=c.ewm(span=144,adjust=False,min_periods=144).mean();e169=c.ewm(span=169,adjust=False,min_periods=169).mean()
 r=c.pct_change(fill_method=None);dist=c/ma-1;mom=c.pct_change(3,fill_method=None)
 latest=dict(symbol=symbol,date=iso(n.date.iloc[-1]),close=float(n.close.iloc[-1]),ema144=float(e144.iloc[-1]),ema169=float(e169.iloc[-1]),ma20=float(ma.iloc[-1]),distance=float(np.float32(dist.iloc[-1])),mom3=float(np.float32(mom.iloc[-1])),vol20=float(np.float32(r.rolling(20).std().iloc[-1])),amount=float(amount.iloc[-1]*1000),median_amount20=float(amount.rolling(20).median().iloc[-1]*1000),observed_bars=len(n))
 latest['eligible']=bool(len(n)>=250 and n.close.iloc[-1]>=3 and latest['median_amount20']>=50e6 and r.iloc[-1]<.095 and e144.iloc[-1]>e169.iloc[-1] and c.iloc[-1]>e144.iloc[-1] and mom.iloc[-1]>0 and -.02<=dist.iloc[-1]<=.08)
 history=pd.DataFrame(dict(symbol=symbol,date=n.date,close=n.close,continuous_close=c,factor=c/n.close)).tail(260)
 return latest,history

def market_state(indices,asof):
 rows=[];counts={20:0,60:0,250:0};issues=[]
 for code,label in INDICES.items():
  f=indices.get(code,pd.DataFrame()).copy()
  if f.empty:issues.append(f'{label}缺少行情');continue
  f['date']=pd.to_datetime(f['date'].astype(str));f=f[f.date.le(pd.Timestamp(asof))].sort_values('date').drop_duplicates('date',keep='last')
  if len(f)<250 or iso(f.date.iloc[-1])!=asof:issues.append(f'{label}均线预热不足或未更新到{asof}');continue
  row=dict(code=code,name=label,close=float(f.close.iloc[-1]))
  for window in counts:
   value=float(f.close.tail(window).mean());above=bool(f.close.iloc[-1]>value);counts[window]+=above;row[f'ma{window}']=value;row[f'above{window}']=above
  rows.append(row)
 if issues:return dict(state='unavailable',label='数据待更新',counts=counts,indices=rows,issues=issues)
 a,b,c=(counts[n]>=4 for n in (250,60,20))
 state='bull' if a and b and c else ('pullback' if a and not c else ('rebound' if not a and c else ('bear' if not a and not b and not c else 'transition')))
 return dict(state=state,label={'bull':'牛市','bear':'熊市','pullback':'牛市回调','rebound':'熊市反弹','transition':'过渡阶段'}[state],counts=counts,indices=rows,issues=[])

def build_plan(policy,account,summary,history,market,calendar,signal_date,trade_date,issues=None):
 cfg=POLICIES[policy];issues=list(issues or [])+market.get('issues',[]);calendar=pd.DatetimeIndex(calendar).sort_values().unique();asof=pd.Timestamp(signal_date);target=pd.Timestamp(trade_date) if trade_date else None
 rows={r['symbol']:r for r in summary};history=history.copy();history['date']=pd.to_datetime(history.date) if len(history) else pd.Series(dtype='datetime64[ns]')
 sells=[];holds=[];held=set();value=0.;cash=account['cash']
 for pos in account['positions']:
  symbol=pos['symbol'];held.add(symbol);entry=pd.Timestamp(pos['entry_date']);row=rows.get(symbol);f=history[(history.symbol==symbol)&history.date.between(entry,asof)] if len(history) else history
  item={**pos,'name':(row or {}).get('name',symbol),'action':'HOLD','reason':'继续持有，未触发退出','sell_quantity':0,'execution':'下一交易日收盘；先确认可卖数量及成交条件'}
  mark=(row or {}).get('close',pos['cost']);value+=mark*pos['quantity'];item['close']=mark
  if row is None or row['date']!=signal_date:
   item.update(action='CHECK',reason='缺少该股最新收盘，不能可靠判断退出');holds.append(item);issues.append(f'{symbol}持仓行情缺失或停牌，买入计划暂停');continue
  if entry>asof:
   item.update(reason='新入场，等待入场日收盘；遵守T+1');holds.append(item);continue
  if entry not in calendar:
   item.update(action='CHECK',reason='无法核对入场交易日');holds.append(item);issues.append(f'{symbol}入场日不在交易日历');continue
  entry_index=calendar.get_indexer([entry])[0];deadline=calendar[entry_index+20] if entry_index+20<len(calendar) else None
  due=deadline is not None and target is not None and target>=deadline
  item['deadline']=iso(deadline) if deadline is not None else None;item['holding_sessions']=int(((calendar>entry)&(calendar<=asof)).sum())
  reference=f[f.date.eq(entry)]
  if len(reference):
   base=float(reference.continuous_close.iloc[0]);factor=float(f.factor.iloc[-1]);last=float(f.continuous_close.iloc[-1]);peak=float(f.continuous_close.max());gain=last/base-1
   item.update(entry_reference=float(reference.close.iloc[0]),gain=gain,peak_reference=peak/factor,trailing_price=peak*(1-cfg['trail'])/factor if cfg['trail'] else None,take_profit_price=base*(1+cfg['tp'])/factor if cfg['tp'] else None,stop_loss_price=base*(1-cfg['sl'])/factor if cfg['sl'] else None)
   # Keep a price trigger sticky while a previous requested sell remains
   # unfilled; a later recovery must not silently cancel the sell order.
   peaks=f.continuous_close.cummax();gains=f.continuous_close/base-1
   triggered=((cfg['trail']>0)&(f.continuous_close/peaks-1<=-cfg['trail']))|((cfg['tp']>0)&gains.ge(cfg['tp']))|((cfg['sl']>0)&gains.le(-cfg['sl']))
   events=f.loc[triggered,'date'];first_trigger=events.iloc[0] if len(events) else None
   reason='最高收盘回撤达到6%' if cfg['trail'] else ('收盘上涨达到3%止盈' if first_trigger is not None and float(gains.loc[events.index[0]])>=cfg['tp'] else '收盘下跌达到5%止损')
   if first_trigger is not None:due=True;item.update(trigger_date=iso(first_trigger),reason=reason+'；未成交前继续卖出')
  elif not due:
   item.update(action='CHECK',reason='入场日参考行情缺失，不能计算价格退出');issues.append(f'{symbol}缺少入场日参考收盘')
  if due:
   item.update(action='SELL',sell_quantity=pos['quantity'])
   if not item.get('trigger_date'):item['reason']='已到入场后第20个交易日，执行时间退出'
   sells.append(item)
  else:holds.append(item)
 equity=cash+value;budget=equity*cfg['fraction']/5
 can_buy=market['state']=='transition' and not issues and bool(trade_date) and account.get('confirmed',False)
 candidates=sorted([r for r in summary if r.get('eligible') and r['date']==signal_date],key=lambda r:(r['vol20'],r['symbol']))[:10]
 selling={p['symbol'] for p in sells}
 buy=[];reserved=0.;projected_slots=max(0,5-len(account['positions']));cap_cash=cash;remaining=max(0,equity*cfg['fraction']-value)
 for rank,row in enumerate(candidates,1):
  symbol=row['symbol'];price=row['close']*1.0005;limit=min(budget,max(0,cap_cash-reserved),max(0,remaining-reserved),row['amount']*.01)
  qty=int(max(0,limit-5)/(price*1.00031)//100)*100;minimum=200 if symbol.startswith('688') else 100
  is_held=symbol in held;available=can_buy and not is_held and projected_slots>0 and qty>=minimum
  after_sell=can_buy and bool(sells) and (not is_held or symbol in selling) and min(budget,row['amount']*.01)>price*minimum+5
  reason='现持仓若先全部卖出，可按队列重新评估独立新买单；入场日期需重置' if is_held and symbol in selling and after_sell else ('已持仓，不重复买入' if is_held else ('市场非过渡阶段，不开新仓' if market['state']!='transition' else ('资料待确认，暂停买入' if not can_buy else ('空槽或可用现金不足，仅备选' if not available else '符合冻结规则，列入次日条件买入队列'))))
  if after_sell and not available and not is_held:reason='先完成卖出、核实到账现金和空槽，再按队列评估，不能预支卖出款'
  buy.append({**row,'rank':rank,'action':'BUY' if available else ('AFTER_SELL' if after_sell else 'WATCH'),'reason':reason,'budget':budget,'estimated_quantity':qty if available else 0,'execution':'次日收盘按当时价格、可用现金和成交额1%上限重新计算；涨停/停牌不买'})
  if available:projected_slots-=1;reserved+=qty*price+max(5,qty*price*.0003)+qty*price*.00001
 # Do not budget unfilled sell proceeds or free slots before actual sales.
 if sells:issues.append('买入数量按当前现金和空槽估算；卖出实际成交后才可释放资金/槽位，并继续按备选顺序评估')
 status='blocked' if market['state']=='unavailable' or not trade_date or any(i for i in issues if not i.startswith('买入数量')) else ('ready' if account.get('confirmed') else 'needs_account')
 return clean(dict(schema_version='transition_workspace_v1',generated_at=today().isoformat(),strategy=policy,label=cfg['label'],signal_date=signal_date,trade_date=trade_date,status=status,account=account,market=market,equity=equity,target_fraction=cfg['fraction'],per_stock_budget=budget,sells=sells,holds=holds,buys=buy,can_open_new=can_buy,issues=issues,rules=RULES,execution_time='下一交易日收盘',planned_t={'signal_date':signal_date},snapshot_kind='实际持仓的次日条件计划；不会自动记账或下单'))

def verified_calendar(root):
 from quant.data.tushare_fetcher import TushareDataFetcher
 now=today();fetcher=TushareDataFetcher(cache_dir=Path(root)/'data/cache/transition_sources')
 f=fetcher.get_trade_calendar(start_date='20150101',end_date=f'{now.year}1231',exchange='SSE',is_open='1')
 if f.empty:raise ValueError('交易所交易日历不可用，不能推算次日')
 cal=pd.DatetimeIndex(pd.to_datetime(f.cal_date.astype(str))).sort_values().unique()
 cutoff=pd.Timestamp(now.date())
 if (now.hour,now.minute)<(15,15):cutoff-=pd.Timedelta(days=1)
 past=cal[cal<=cutoff]
 if not len(past):raise ValueError('交易日历没有覆盖最新收盘日')
 signal=past[-1];future=cal[cal>signal]
 if not len(future):raise ValueError('交易日历未覆盖下一交易日')
 return cal,iso(signal),iso(future[0])

def _canonical_daily(store,symbols,end):
 needed=['ts_code','trade_date','open','high','low','close','pre_close','amount','vol','volume','turnover']
 if store.config.backend in {'mysql','sql'}:
  probe=store.read_market_range('daily',start_date=end,end_date=end,symbols=symbols)
  if probe.empty:raise ValueError(f'日线未更新到{end}，请先更新行情')
  return store.read_market_range('daily',start_date='20150101',end_date=end,symbols=symbols,columns=[c for c in needed if c in probe.columns])
 # Explicit file backend only: intersect monthly schemas so the documented
 # amount/turnover alias is available without selecting nonexistent columns.
 import pyarrow.parquet as pq
 parts=[]
 for path in sorted((store.config.root/'daily_partitioned').glob('year_month=*/data.parquet')):
  month=path.parent.name.split('=')[-1]
  if not '201501'<=month<=end[:6]:continue
  cols=[c for c in needed if c in pq.read_schema(path).names];f=pd.read_parquet(path,columns=cols)
  f=f[f.ts_code.isin(symbols)&f.trade_date.astype(str).str.replace('-','',regex=False).le(end)]
  if len(f):parts.append(f)
 if not parts:raise ValueError('本地日线源没有可用的策略行情')
 return pd.concat(parts,ignore_index=True).drop_duplicates(['ts_code','trade_date'],keep='last')

def load_inputs(root,refresh=False):
 from quant.data.market_data_store import MarketDataStore,MarketDataStoreConfig
 from quant.data.tushare_fetcher import TushareDataFetcher
 from quant.infrastructure.publication import current_publication
 root=Path(root);cal,signal,target=verified_calendar(root);store=MarketDataStore(MarketDataStoreConfig.from_env(root=root/'data/raw'));cache=root/'data/cache/transition_workspace';cache.mkdir(parents=True,exist_ok=True)
 latest=store.latest_dataset_trade_date('daily')
 if latest is None or pd.Timestamp(latest)<pd.Timestamp(signal):raise ValueError(f'策略行情未更新到{signal}；请更新行情后再生成买卖计划')
 revision=store.dataset_revision('daily');files=[]
 if store.config.backend not in {'mysql','sql'}:files=[(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in sorted((store.config.root/'daily_partitioned').glob('year_month=*/data.parquet'))]
 view=current_publication();generation=getattr(view,'generation',None)
 contract=_SOURCE_CONTRACT
 key=hashlib.sha256(json.dumps([signal,revision,files,generation,contract],default=str).encode()).hexdigest()[:20]
 summary_path=cache/f'{key}.json';history_path=cache/f'{key}.parquet'
 if summary_path.exists() and history_path.exists() and not refresh:
  data=json.loads(summary_path.read_text());return data['summary'],pd.read_parquet(history_path),data['market'],cal,signal,target
 fetcher=TushareDataFetcher(cache_dir=root/'data/cache/transition_sources')
 # Keep the seven full index histories but refresh only a short overlap.
 def index(code):
  path=cache/f'index_{code}.parquet';old=pd.read_parquet(path) if path.exists() else pd.DataFrame()
  if len(old) and pd.to_datetime(old.date.astype(str)).max()>=pd.Timestamp(signal) and not refresh:return code,old
  start='20140101' if old.empty else (pd.to_datetime(old.date.astype(str)).max()-pd.Timedelta(days=20)).strftime('%Y%m%d')
  new=(fetcher.pro.index_daily(ts_code=code,start_date=start,end_date=signal.replace('-','')).rename(columns={'ts_code':'symbol','trade_date':'date'}) if refresh else fetcher.get_index_daily(code,start_date=start,end_date=signal.replace('-','')));f=pd.concat([old,new],ignore_index=True).drop_duplicates('date',keep='last');atomic_write_parquet(f,path,index=False);return code,f
 from concurrent.futures import ThreadPoolExecutor
 with ThreadPoolExecutor(max_workers=2) as pool:indices=dict(pool.map(index,INDICES))
 market=market_state(indices,signal)
 names_path=cache/f'names_{signal}.parquet'
 if names_path.exists() and not refresh:names=pd.read_parquet(names_path)
 else:
  names=fetcher.pro.stock_basic(exchange='',list_status='L',fields='ts_code,name');atomic_write_parquet(names,names_path,index=False)
 if names.empty:raise ValueError('当前股票名称与风险名单不可用')
 lookup=names.set_index('ts_code').name.to_dict()
 symbols=[s for s in store.list_symbols('daily') if (s.startswith('3') and s.endswith('.SZ')) or (s.startswith('688') and s.endswith('.SH'))]
 if not symbols:raise ValueError('行情源没有创业板/科创板股票')
 raw=_canonical_daily(store,symbols,signal.replace('-',''));summaries=[];histories=[]
 for symbol,g in raw.groupby('ts_code',sort=True):
  info,h=symbol_features(g,str(symbol));name=str(lookup.get(symbol,''));info['name']=name or str(symbol)
  info['eligible']=bool(info['eligible'] and name and not any(x in name.upper() for x in ['ST','退市','退整','终止上市']) and info['date']==signal)
  summaries.append(info);histories.append(h)
 history=pd.concat(histories,ignore_index=True) if histories else pd.DataFrame(columns=['symbol','date','close','continuous_close','factor'])
 atomic_write_parquet(history,history_path,index=False);atomic_write_json(clean(dict(summary=summaries,market=market,source_date=signal,revision=revision)),summary_path)
 return summaries,history,market,cal,signal,target

def workspace(root,policy='return',refresh=False):
 account=get_account(root,policy)
 try:
  with _DATA_LOCK:
   summary,history,market,cal,signal,target=load_inputs(root,refresh)
  return build_plan(policy,account,summary,history,market,cal,signal,target)
 except Exception as exc:
  # No historical snapshot or hardcoded research candidate is substituted
  # for a missing current source. Holdings remain intact.
  return dict(schema_version='transition_workspace_v1',generated_at=today().isoformat(),strategy=policy,label=POLICIES[policy]['label'],status='blocked',signal_date=None,trade_date=None,account=account,market={'state':'unavailable','label':'数据待更新','indices':[],'counts':{}},sells=[],holds=[],buys=[],can_open_new=False,issues=[str(exc)],rules=RULES,planned_t={'signal_date':None})


_SOURCE_CONTRACT=hashlib.sha256((inspect.getsource(symbol_features)+inspect.getsource(market_state)+inspect.getsource(load_inputs)+inspect.getsource(resolve_amount_kcny)).encode()).hexdigest()
