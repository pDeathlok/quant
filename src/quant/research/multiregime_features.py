"""Causal daily features with strictly completed-week context."""
import numpy as np
import pandas as pd
FAMILIES=('reclaim20','breakout20','trend_pullback','weekly_confirm','liquidity_anchor','vegas','oversold_recovery')
BASE=('mom3','mom10','mom20','mom60','ret1','vol20','dist20','dist60','dist250','slope20','slope60','pulse','close_pos','body','weekly_gap','weekly_slope','weekly_macd','weekly_boll','ema_gap','range20','log_amount','board_growth')
def causal_features(bars):
 b=bars.sort_values(['symbol','date']).reset_index(drop=True).copy();g=b.groupby('symbol',sort=False);c=b.continuous_close
 for n in (1,3,10,20,60):b['ret1' if n==1 else f'mom{n}']=g.continuous_close.pct_change(n,fill_method=None)
 for n in (60,250):b[f'ma{n}']=g.continuous_close.transform(lambda x:x.rolling(n).mean())
 for n in (20,60,250):b[f'dist{n}']=c/b[f'ma{n}']-1
 for n in (20,60):b[f'slope{n}']=b[f'ma{n}']/b.groupby('symbol')[f'ma{n}'].shift(5)-1
 b['vol20']=b.groupby('symbol').ret1.transform(lambda x:x.rolling(20).std())
 amount=b.groupby('symbol').amount_cny
 b['pulse']=b.amount_cny/amount.transform(lambda x:x.shift().rolling(5).mean())
 b['median_amount20']=amount.transform(lambda x:x.rolling(20).median());b['log_amount']=np.log(b.median_amount20.where(b.median_amount20>0))
 b['close_pos']=(b.close-b.low)/(b.high-b.low).replace(0,np.nan);b['body']=b.close/b.open-1
 high=b.groupby('symbol').continuous_close.transform(lambda x:x.shift().rolling(20).max())
 low=b.groupby('symbol').continuous_close.transform(lambda x:x.rolling(20).min());b['range20']=high/low-1
 e144=g.continuous_close.transform(lambda x:x.ewm(span=144,adjust=False,min_periods=144).mean());e169=g.continuous_close.transform(lambda x:x.ewm(span=169,adjust=False,min_periods=169).mean())
 b['ema_gap']=e144/e169-1
 b['week']=b.date.dt.to_period('W-SUN').dt.start_time
 w=b.groupby(['symbol','week'],sort=True).continuous_close.last().reset_index();wg=w.groupby('symbol',sort=False).continuous_close
 w['e10']=wg.transform(lambda x:x.ewm(span=10,adjust=False,min_periods=26).mean());w['e26']=wg.transform(lambda x:x.ewm(span=26,adjust=False,min_periods=26).mean())
 w['weekly_gap']=w.e10/w.e26-1;w['weekly_slope']=w.e26/w.groupby('symbol').e26.shift(4)-1
 dif=wg.transform(lambda x:x.ewm(span=12,adjust=False,min_periods=26).mean())-w.e26
 w['dif']=dif;w['weekly_macd']=dif-w.groupby('symbol').dif.transform(lambda x:x.ewm(span=9,adjust=False,min_periods=9).mean())
 mid=wg.transform(lambda x:x.rolling(20).mean());w['weekly_boll']=w.continuous_close/mid-1
 cols=['weekly_gap','weekly_slope','weekly_macd','weekly_boll']
 # Current week's final bar may be in the future; all derived weekly fields
 # are shifted before they are joined to any day of that week.
 w[cols]=w.groupby('symbol')[cols].shift(1)
 b=b.merge(w[['symbol','week',*cols]],on=['symbol','week'],how='left',validate='many_to_one')
 c=b.continuous_close;g=b.groupby('symbol',sort=False)
 b['reclaim20']=(c>b.ma20)&(g.continuous_close.shift()<=g.ma20.shift())&b.pulse.ge(1.3)
 b['breakout20']=(c>high.to_numpy())&b.pulse.ge(1.3)
 trend=(c>b.ma60)&b.slope60.gt(0)&b.slope20.gt(0)
 b['trend_pullback']=trend&b.dist20.between(-.03,.03)&b.mom3.lt(0)&b.ret1.gt(0)
 b['weekly_confirm']=b.weekly_gap.gt(0)&b.weekly_slope.gt(0)&b.weekly_macd.gt(0)&b.body.gt(0)&b.close_pos.ge(.6)&b.pulse.ge(1)
 anchor=(b.pulse.ge(2.5)).astype(int).groupby(b.symbol).transform(lambda x:x.shift().rolling(10).max())
 b['liquidity_anchor']=anchor.eq(1)&b.pulse.between(.5,1.3)&b.body.gt(0)&b.mom3.gt(0)&trend
 b['vegas']=b.ema_gap.gt(0)&(c>e144.to_numpy())&b.mom3.gt(0)&b.dist20.between(-.02,.08)
 b['oversold_recovery']=b.dist20.lt(-.06)&b.mom10.lt(-.08)&b.ret1.gt(.01)&b.close_pos.ge(.6)
 b['board_growth']=(b.symbol.str.startswith('3')|b.symbol.str.startswith('688')).astype(float)
 b['board']=np.where(b.board_growth.eq(1),'growth',np.where(b.symbol.str.endswith('.BJ'),'bse','main'))
 b['observed_bars']=g.cumcount()+1
 return b
