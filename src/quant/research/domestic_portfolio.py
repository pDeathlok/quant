"""Domestic portfolio experts, causal labels, and one shared cash ledger."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from quant.research.byd_allocation import state_features,OnlineAllocator,AllocationConfig
from quant.research.byd_mwu import ROOT,_RidgeExpert
from quant.research.byd_wear import WearConfig,stamp_rate,transfer_rate

CODES=('002594.SZ','510300.SH','510500.SH','518880.SH','511010.SH','512880.SH','510880.SH')
SOURCE=ROOT/'reports/research/domestic_portfolio_20261013/source'
START='2014-08-01'


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    data: dict[str,np.ndarray]
    features: np.ndarray
    feature_names: tuple[str,...]
    dividends: list[dict]
    coverage: list[dict]


def load_panel(source: Path=SOURCE, end_date: str=None) -> Panel:
    cal=pd.read_parquet(source/'calendar.parquet');cal=cal[(cal.is_open==1)&(cal.cal_date>='20110630')]
    dates=pd.DatetimeIndex(pd.to_datetime(cal.cal_date)).sort_values()
    if end_date is not None:dates=dates[dates<=pd.Timestamp(end_date)]
    keys=('open','high','low','close','volume','adjusted','adj_open','adj_low','tradable','eligible','dividend','bonus','split','up_limit','down_limit')
    data={k:np.zeros((len(dates),len(CODES))) for k in keys};data['split'][:]=1
    dividends=[];coverage=[];signals=[]
    for j,code in enumerate(CODES):
        if j==0:
            f=pd.read_parquet(ROOT/'reports/research/byd_mwu_20261012/market_snapshot.parquet').set_index('date').reindex(dates)
            factor=(f.close/f.pre_close).fillna(1).cumprod()/f.close
            volume=f.volume_shares
            f['adj_factor']=factor;f['split']=f['split'].fillna(1)
            for k,v in [('dividend','cash_dividend'),('bonus','bonus_shares')]:data[k][:,j]=f[v].fillna(0)
            data['split'][:,j]=f['split'].to_numpy()
        else:
            raw=pd.read_parquet(source/f'{code.replace(".","_")}.parquet');raw['date']=pd.to_datetime(raw.trade_date)
            f=raw.set_index('date').reindex(dates)
            volume=f.vol*100
            f['split']=1.
            if code=='510500.SH' and pd.Timestamp('2015-04-14') in dates:f.loc[pd.Timestamp('2015-04-14'),'split']=.28032483
            d=pd.read_parquet(source/f'{code.replace(".","_")}_dividends.parquet')
            if len(d):
                d=d[d.div_proc.eq('实施')&d.ex_date.notna()].drop_duplicates(['ex_date','div_cash'])
                for row in d.itertuples():
                    ex=pd.Timestamp(row.ex_date);pay=pd.Timestamp(row.pay_date) if pd.notna(row.pay_date) else ex
                    if ex in dates:dividends.append({'asset':j,'ex_index':int(dates.get_loc(ex)),
                        'pay_index':int(dates.searchsorted(pay)),'per_share':float(row.div_cash),
                        'ex_date':str(ex.date()),'pay_date':str(pay.date())})
        valid=f.close.notna()&f.open.gt(0)&volume.gt(0)
        data['tradable'][:,j]=valid.to_numpy()
        data['eligible'][:,j]=(valid.astype(int).cumsum()>=252).to_numpy()
        close=f.close.copy()
        # Mark holdings on a nonquoted split day at the mechanically adjusted price.
        for i in range(len(f)):
            if pd.isna(close.iloc[i]) and i and pd.notna(close.iloc[i-1]):
                close.iloc[i]=close.iloc[i-1]/float(f['split'].iloc[i])
        close=close.fillna(0)
        for k in ('open','high','low'):
            data[k][:,j]=f[k].fillna(close).to_numpy()
        data['close'][:,j]=close.to_numpy();data['volume'][:,j]=volume.fillna(0).to_numpy()
        data['split'][:,j]=f['split'].to_numpy()
        # Factor becomes known on the split day even though trading is suspended.
        factors=f.adj_factor.ffill()
        if j and code=='510500.SH' and pd.Timestamp('2015-04-14') in dates:factors.loc['2015-04-14']=factors.loc[:'2015-04-13'].iloc[-1]*.28032483
        for k,raw_key in [('adjusted','close'),('adj_open','open'),('adj_low','low')]:
            data[k][:,j]=data[raw_key][:,j]*factors.fillna(0).to_numpy()
        if j==0:
            data['up_limit'][:,j]=f.up_limit.fillna(close*1.1).to_numpy()
            data['down_limit'][:,j]=f.down_limit.fillna(close*.9).to_numpy()
        else:
            reference=f.pre_close.fillna(close)
            data['up_limit'][:,j]=np.round(reference.to_numpy()*1.1,3)
            data['down_limit'][:,j]=np.round(reference.to_numpy()*.9,3)
        s=pd.DataFrame({'date':dates,'close':data['adjusted'][:,j],
            'open':data['adj_open'][:,j],'high':data['high'][:,j]*factors.fillna(0).to_numpy(),
            'low':data['adj_low'][:,j],'volume':data['volume'][:,j]/f['split'].cumprod().to_numpy()})
        s.loc[s.close.le(0),['close','open','high','low']]=np.nan
        signals.append(s)
        inside=valid[valid.index>=valid[valid].index[0]] if valid.any() else valid.iloc[:0]
        coverage.append({'code':code,'first_quote':str(valid[valid].index[0].date()) if valid.any() else None,'last_quote':str(valid[valid].index[-1].date()) if valid.any() else None,
                         'missing_calendar_quotes_after_listing':int((~inside).sum()),'observed_quotes':int(valid.sum())})
    index=pd.DataFrame({'trade_date':dates.strftime('%Y%m%d'),'close':data['adjusted'][:,1]})
    features=[state_features(s,index) for s in signals]
    return Panel(dates,data,np.stack([f.to_numpy() for f in features],axis=1),tuple(features[0].columns),dividends,coverage)


def capped_weights(proposal: np.ndarray, equity_cap: float=.70) -> np.ndarray:
    w=np.maximum(np.nan_to_num(proposal),0).copy()
    w=np.minimum(w,np.array([.10,.35,.35,.60,.70,.15,.35]))
    equities=np.array([0,1,2,5,6]);total=w[equities].sum()
    if total>equity_cap:w[equities]*=equity_cap/total
    if w.sum()>1:w/=w.sum()
    return w


def matured_labels(panel: Panel, horizon: int, downside: bool=False) -> np.ndarray:
    n,k=panel.data['close'].shape;labels=np.full((n,k),np.nan)
    for t in range(n-horizon-1):
        enter=panel.data['adj_open'][t+1];exit=panel.data['adj_open'][t+horizon+1]
        valid=(enter>0)&(exit>0)&panel.data['eligible'][t].astype(bool)&panel.data['tradable'][t+1].astype(bool)&panel.data['tradable'][t+horizon+1].astype(bool)
        value=exit/np.maximum(enter,1e-12)-1-.0016
        value[0]-=.0005 if panel.dates[t]>=pd.Timestamp('2023-08-28') else .001
        if downside:
            low=panel.data['adj_low'][t+1:t+horizon+1].min(axis=0)
            value-=.5*np.maximum(1-low/np.maximum(enter,1e-12),0)
        labels[t,valid]=np.clip(value[valid],-.5,.5)
    return labels


def fit_models(panel: Panel) -> tuple[dict[str,np.ndarray],list[dict]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    n,k=panel.data['close'].shape
    specs=[('ridge_5',5,False),('ridge_21',21,False),('ridge_63',63,False),('ridge_downside21',21,True),('tree_downside21',21,True)]
    outputs={name:np.zeros((n,k)) for name,_,_ in specs};audit=[]
    for name,h,down in specs:
        y=matured_labels(panel,h,down);model=None
        for i in range(600,n):
            if model is None or (i-600)%21==0:
                end=i-h-1;begin=max(252,end-1259)
                x=panel.features[begin:end+1].reshape(-1,panel.features.shape[2]);label=y[begin:end+1].reshape(-1)
                usable=np.isfinite(label);x=x[usable];label=label[usable]
                if len(label)<600:continue
                model=HistGradientBoostingRegressor(max_iter=40,max_depth=3,max_leaf_nodes=8,min_samples_leaf=80,learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=20261013) if name.startswith('tree') else _RidgeExpert()
                model.fit(x,label)
                audit.append({'model':name,'fit_date':str(panel.dates[i].date()),'feature_end':str(panel.dates[end].date()),
                              'label_end':str(panel.dates[end+h+1].date()),'samples':len(label),'features':len(panel.feature_names)})
            if model is not None:outputs[name][i]=model.predict(panel.features[i]) / h
        print(f'Model {name}: mature-label rolling fits complete',flush=True)
    return outputs,audit


def expert_targets(panel: Panel, forecasts: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
    n,k=panel.data['close'].shape
    idx={name:i for i,name in enumerate(panel.feature_names)}
    f=panel.features;eligible=panel.data['eligible'].astype(bool)
    experts={name:np.zeros((n,k)) for name in ['balanced','growth','gold_bond','cash','invvol']}
    for lookback in (63,120,250):
        for top in (2,3):experts[f'mom{lookback}_top{top}']=np.zeros((n,k))
    for name in forecasts:experts[name]=np.zeros((n,k))
    # Decisions weekly; static baselines can also be held until monthly checks by execution.
    for t in range(n):
        for name,w in [('balanced',[.05,.15,.10,.25,.30,.05,.10]),('growth',[.10,.20,.15,.25,.15,.05,.10]),('gold_bond',[0,0,0,.5,.5,0,0])]:
            experts[name][t]=capped_weights(np.array(w)*eligible[t])
        vol=np.maximum(f[t,:,idx['vol63']]*np.sqrt(252),.02)
        raw=eligible[t]/vol;raw[0]*=.25;experts['invvol'][t]=capped_weights(raw/raw.sum()) if raw.sum() else 0
        for lookback in (63,120,250):
            for top in (2,3):
                key=f'mom{lookback}_top{top}'
                scores=f[t,:,idx[f'ret_{lookback}']]/vol
                good=eligible[t]&(f[t,:,idx[f'ret_{lookback}']]>0)&(f[t,:,idx['distance_120']]>0)
                selected=np.flatnonzero(good);selected=selected[np.argsort(scores[selected])[-top:]]
                raw=np.zeros(k);raw[selected]=1/top
                w=capped_weights(raw)
                # Put capped/unallocated capital into bonds up to their cap; remainder cash.
                if eligible[t,4]:w[4]+=min(.70-w[4],1-w.sum())
                experts[key][t]=w
        for name,predictions in forecasts.items():
            prediction=predictions[t]
            good=eligible[t]&(prediction>.0001)
            score=np.where(good,prediction/(vol**2),0)
            selected=np.argsort(score)[-3:];raw=np.zeros(k)
            if score[selected].sum()>0:raw[selected]=score[selected]/score[selected].sum()
            w=capped_weights(raw)
            if eligible[t,4]:w[4]+=min(.70-w[4],1-w.sum())
            experts[name][t]=w
    return experts


def simulate(panel: Panel, targets: np.ndarray, wear: WearConfig=WearConfig(), deadband: float=.02,
             start: str=START, frequency: int=5,
             basket_groups: tuple[tuple[int,...],...]=(), basket_limit: float=.20) -> tuple[pd.DataFrame,pd.DataFrame]:
    if targets.shape!=panel.data['close'].shape or (targets<0).any() or (targets.sum(axis=1)>1+1e-9).any():raise ValueError('Invalid portfolio targets')
    if basket_groups:
        if not 0<basket_limit<=1 or sorted(a for group in basket_groups for a in group)!=list(range(len(CODES))):raise ValueError('Invalid basket groups')
        if any((targets[:,group].sum(axis=1)>basket_limit+1e-9).any() for group in basket_groups):raise ValueError('Basket target exceeds limit')
    data=panel.data;n,k=targets.shape;first=int(panel.dates.searchsorted(pd.Timestamp(start)))
    shares=np.zeros(k);cash=wear.initial_cash;receivable=0.;equity=wear.initial_cash;payment={};rows=[];orders=[]
    events={}
    for e in panel.dividends:events.setdefault(e['ex_index'],[]).append(e)
    sigma=pd.DataFrame(data['adjusted']).replace(0,np.nan).pct_change(fill_method=None).rolling(20).std().fillna(0).to_numpy()
    for i in range(first,n):
        previous=equity;j=i-1
        paid=payment.pop(i,0.);cash+=paid;receivable-=paid
        for e in events.get(i,[]):
            value=shares[e['asset']]*e['per_share'];receivable+=value
            payment[e['pay_index']]=payment.get(e['pay_index'],0.)+value
            if e['pay_index']==i:
                cash+=value;receivable-=value;payment.pop(i,None)
        cash+=shares[0]*(data['dividend'][i,0]*(1-wear.dividend_tax_reserve)-data['bonus'][i,0]*wear.dividend_tax_reserve)
        shares*=data['split'][i]
        # Exchange account rounds fractional ETF shares down after the official reverse split.
        if data['split'][i,2]!=1:shares[2]=np.floor(shares[2])
        op=data['open'][i];eq_open=cash+receivable+np.dot(shares,op)
        w=targets[j].copy();current=shares*op/eq_open
        repair=np.zeros(k,dtype=bool)
        for group in basket_groups:
            if current[list(group)].sum()>basket_limit+1e-9:repair[list(group)]=True
        scheduled=(i-first)%frequency==0
        if repair.any():
            if not scheduled:w=current.copy()
            for group in basket_groups:
                a=list(group)
                if repair[a].any() and w[a].sum()>basket_limit-.005:w[a]*=(basket_limit-.005)/w[a].sum()
        change=np.zeros(k);cost=0.;turnover=0.;requested=0.;filled=0.;capacity_days=False
        if scheduled or repair.any():
            desired=np.floor(w*eq_open/np.maximum(op,1e-12)/100+1e-9)*100
            desired[w==0]=0
            no_trade=(np.abs(w-current)<=deadband)&~repair
            desired[no_trade]=shares[no_trade]
            desired[~data['tradable'][i].astype(bool)]=shares[~data['tradable'][i].astype(bool)]
            raw=desired-shares;requested=float(np.abs(raw).sum())
            for a in sorted(range(k),key=lambda a:raw[a]>=0):
                q=raw[a];cap=np.floor(data['volume'][j,a]*wear.participation/100)*100
                if abs(q)>cap:capacity_days=True
                q=np.sign(q)*min(abs(q),cap)
                if q>0:q=np.floor(q/100+1e-9)*100
                if q<0 and abs(q)<shares[a]:q=-np.floor(abs(q)/100+1e-9)*100
                if not q:continue
                up,low=data['up_limit'][i,a],data['down_limit'][i,a]
                tick=.01 if a==0 else .001
                if (q>0 and op[a]>=up-tick/2) or (q<0 and op[a]<=low+tick/2):continue
                stamp=stamp_rate(int(panel.dates[i].strftime('%Y%m%d')),wear) if a==0 and q<0 else 0.
                transfer=transfer_rate(int(panel.dates[i].strftime('%Y%m%d')),wear) if a==0 else 0.
                while q:
                    impact=wear.impact_eta*sigma[j,a]*np.sqrt(abs(q)/max(data['volume'][j,a],1))
                    price=np.clip(op[a]*(1+np.sign(q)*(wear.spread+impact)),low,up)
                    value=abs(q)*price;fee=max(value*wear.commission,wear.minimum_commission)+value*(stamp+transfer)
                    if q<0 or value+fee<=cash+1e-8:break
                    affordable=max(0,min(cash/(price*(1+wear.commission+transfer)),(cash-wear.minimum_commission)/(price*(1+transfer))))
                    reduced=np.floor(affordable/100)*100
                    q=min(q-100,reduced)
                if not q:continue
                cash-=q*price+fee;shares[a]+=q;change[a]=q
                drag=fee+abs(q)*abs(price-op[a]);cost+=drag;turnover+=abs(q)*op[a]/eq_open;filled+=abs(q)
                orders.append({'date':panel.dates[i],'signal_date':panel.dates[j],'code':CODES[a],'shares':q,
                    'price':price,'commission':max(value*wear.commission,wear.minimum_commission),'stamp':value*stamp,
                    'transfer':value*transfer,'spread_impact':abs(q)*abs(price-op[a]),'cost':drag})
        equity=cash+receivable+np.dot(shares,data['close'][i])
        if cash < -1e-6 or (shares < -1e-8).any():raise RuntimeError('Shared cash/shares invalid')
        row={'date':panel.dates[i],'signal_date':panel.dates[j],'return':equity/previous-1,'equity':equity,
             'cash':cash,'receivable':receivable,'position':np.dot(shares,data['close'][i])/equity,'turnover':turnover,
             'cost':cost,'orders':int(np.count_nonzero(change)),'requested_shares':requested,'filled_shares':filled,'capacity_limited':capacity_days}
        row.update({f'position_{c}':shares[a]*data['close'][i,a]/equity for a,c in enumerate(CODES)})
        row.update({f'shares_{c}':shares[a] for a,c in enumerate(CODES)})
        if basket_groups:
            actual=shares*data['close'][i]/equity
            row['basket_repair_requested']=bool(repair.any())
            row['basket_max_weight']=max(actual[list(group)].sum() for group in basket_groups)
            row['basket_excess']=max(0.,row['basket_max_weight']-basket_limit)
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(orders)


def portfolio_metrics(frame: pd.DataFrame) -> dict:
    r=frame['return'];wealth=np.r_[1.,np.cumprod(1+r.to_numpy())];dd=wealth/np.maximum.accumulate(wealth)-1
    years=(frame.date.iloc[-1]-frame.signal_date.iloc[0]).days/365.25
    vol=float(r.std()*np.sqrt(252));annual=float((wealth[-1])**(1/years)-1)
    return {'total_return':float(wealth[-1]-1),'cagr':annual,'volatility':vol,'max_drawdown':float(dd.min()),
            'sharpe':float(r.mean()*252/vol) if vol else 0.,'average_position':float(frame.position.mean()),
            'cost':float(frame.cost.sum()),'orders':int(frame.orders.sum()),'capacity_days':int(frame.capacity_limited.sum())}


def mix_targets(panel: Panel, experts: dict[str,np.ndarray], rewards: np.ndarray,
                config: AllocationConfig, start: str=START) -> tuple[np.ndarray,np.ndarray]:
    names=list(experts);bank=np.stack([experts[n] for n in names],axis=1)
    learner=OnlineAllocator(len(names),config);weights=np.full((len(panel.dates),len(names)),1/len(names))
    first=int(panel.dates.searchsorted(pd.Timestamp(start)))
    feature=panel.features[:,1,panel.feature_names.index('distance_120')]
    context=np.where(feature>.03,2,np.where(feature<-.03,0,1))
    for i in range(first,len(panel.dates)):
        learner.observe(rewards[i],int(context[i-1]));weights[i]=learner.weights(int(context[i]))
    return np.einsum('ne,nea->na',weights,bank),weights
