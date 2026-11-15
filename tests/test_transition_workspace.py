import numpy as np,pandas as pd,pytest
from quant.application.workspaces.transition import build_plan,market_state,INDICES,save_account,get_account,symbol_features,workspace
from quant.research.multiregime_features import causal_features
CAL=pd.bdate_range('2025-01-01',periods=100)
def fixture(policy='return',prices=None,entry=10,signal=25,cash=900000,confirmed=True):
 prices=np.full(100,100.) if prices is None else np.asarray(prices,float)
 history=pd.DataFrame(dict(symbol='300001.SZ',date=CAL,close=prices,continuous_close=prices,factor=1.))
 row=dict(symbol='300001.SZ',name='测试公司',date=str(CAL[signal].date()),close=prices[signal],eligible=True,vol20=.01,amount=100e6)
 account=dict(revision=1,confirmed=confirmed,cash=cash,positions=[dict(symbol='300001.SZ',entry_date=str(CAL[entry].date()),quantity=1000,cost=100)])
 market=dict(state='transition',label='过渡阶段',issues=[],counts={},indices=[])
 return policy,account,[row],history,market,CAL,str(CAL[signal].date()),str(CAL[signal+1].date())
def test_time_exit_counts_exchange_sessions_and_not_calendar_days():
 args=list(fixture(entry=5,signal=24));p=build_plan(*args)
 assert p['sells'][0]['deadline']==str(CAL[25].date())
 assert p['sells'][0]['sell_quantity']==1000

def test_new_entry_waits_for_entry_close_and_is_not_same_day_sell():
 p=build_plan(*fixture(entry=26,signal=25))
 assert not p['sells'] and 'T+1' in p['holds'][0]['reason']

def test_trailing_trigger_survives_price_recovery_and_future_quotes_do_not_leak():
 prices=np.full(100,100.);prices[20]=120;prices[21]=110;prices[22:]=125
 args=list(fixture(prices=prices,entry=10,signal=25));p=build_plan(*args)
 assert p['sells'][0]['trigger_date']==str(CAL[21].date())
 assert '未成交' in p['sells'][0]['reason']
 args[3].loc[args[3].date>CAL[25],'continuous_close']=10000
 later=build_plan(*args);later.pop("generated_at");p.pop("generated_at");assert later==p

def test_win_stop_and_profit_have_separate_explicit_reasons():
 prices=np.full(100,100.);prices[25]=104
 p=build_plan(*fixture('win',prices));assert '止盈' in p['sells'][0]['reason']
 prices[25]=94
 p=build_plan(*fixture('win',prices));assert '止损' in p['sells'][0]['reason']

@pytest.mark.parametrize('state,mode', [
 ('bull', 'research'), ('pullback', 'research'), ('transition', 'conditional_entry'),
 ('rebound', 'research'), ('bear', 'cash'), ('unavailable', 'blocked'),
])
def test_market_state_routes_entry_without_replacing_holding_rules(state, mode):
 args=list(fixture())
 args[4]['state']=state
 args[2].append({**args[2][0], 'symbol': '300002.SZ'})
 plan=build_plan(*args)
 assert plan['market_strategy']['mode']==mode
 assert {profile['state'] for profile in plan['strategy_states']}=={'bull','bear','pullback','rebound','transition'}
 assert not plan['sells'] and plan['holds'][0]['action']=='HOLD'
 if mode=='conditional_entry':
  assert plan['can_open_new'] and any(row['action']=='BUY' for row in plan['buys'])
 else:
  assert not plan['can_open_new'] and plan['buys']==[] and plan['per_stock_budget']==0

def test_unavailable_data_keeps_the_market_strategy_catalog(tmp_path, monkeypatch):
 def fail(*args):
  raise ValueError('行情待更新')
 monkeypatch.setattr('quant.application.workspaces.transition.load_inputs',fail)
 plan=workspace(tmp_path)
 assert plan['status']=='blocked'
 assert plan['market_strategy']['mode']=='blocked'
 assert len(plan['strategy_states'])==5

def test_missing_holding_quote_blocks_buys_without_discarding_position():
 args=list(fixture());args[2][0]['date']=str(CAL[24].date());p=build_plan(*args)
 assert p['status']=='blocked' and p['holds'][0]['action']=='CHECK' and not p['can_open_new']

def test_no_phantom_sell_proceeds_or_duplicate_buys():
 args=list(fixture(entry=0,cash=0));p=build_plan(*args)
 assert p['sells'] and all(x['estimated_quantity']==0 for x in p['buys'])
 assert '释放' in p['issues'][-1]

def test_unconfirmed_account_never_generates_buy_instruction():
 args=list(fixture(confirmed=False));args[1]['positions']=[];p=build_plan(*args)
 assert p['status']=='needs_account' and all(x['action']!='BUY' for x in p['buys'])

def test_half_allocation_keeps_empty_slots_as_cash():
 args=list(fixture('win',cash=1000000));args[1]['positions']=[];p=build_plan(*args)
 assert p['per_stock_budget']==100000 and p['buys'][0]['estimated_quantity']==900

def test_accounts_are_durable_separate_and_revision_protected(tmp_path):
 body=dict(revision=0,cash=800000,positions=[dict(symbol='300001',entry_date=str(CAL[10].date()),quantity=2000,cost=100)])
 saved=save_account(tmp_path,'return',body,CAL)
 assert saved['revision']==1 and get_account(tmp_path,'return')['positions'][0]['symbol']=='300001.SZ'
 assert get_account(tmp_path,'win')['confirmed'] is False
 with pytest.raises(ValueError,match='其他页面'):save_account(tmp_path,'return',body,CAL)
 with pytest.raises(ValueError):save_account(tmp_path,'return',dict(revision=1,cash=float('nan'),positions=[]),CAL)

def test_calendar_weekend_cannot_be_recorded_as_entry(tmp_path):
 with pytest.raises(ValueError,match='交易日'):
  save_account(tmp_path,'return',dict(revision=0,cash=1,positions=[dict(symbol='300001',entry_date='2025-01-04',quantity=100,cost=1)]),CAL)

def test_market_requires_all_seven_indices_and_full_year_warmup():
 dates=pd.bdate_range('2024-01-01',periods=260);price=np.r_[np.full(200,100.),np.full(40,150.),np.full(19,110.),130.]
 data={code:pd.DataFrame(dict(date=dates,close=price)) for code in INDICES}
 p=market_state(data,str(dates[-1].date()));assert p['state']=='transition' and p['counts']=={20:7,60:0,250:7}
 data.pop('000001.SH');assert market_state(data,str(dates[-1].date()))['state']=='unavailable'

def test_live_feature_vegas_contract_matches_research_definition():
 rng=np.random.default_rng(1);n=400;c=20*np.exp(np.cumsum(rng.normal(.002,.01,n)));d=pd.bdate_range('2024-01-01',periods=n)
 f=pd.DataFrame(dict(ts_code='300001.SZ',trade_date=d.strftime('%Y%m%d'),open=c,high=c*1.01,low=c*.99,close=c,pre_close=np.r_[c[0],c[:-1]],vol=100000,amount=100000))
 live,h=symbol_features(f,'300001.SZ');b=f.assign(symbol='300001.SZ',date=d,continuous_close=c,amount_cny=1e8,ma20=pd.Series(c).rolling(20).mean());research=causal_features(b).iloc[-1]
 assert live['eligible']==bool(research.vegas)
 assert live['vol20']==float(np.float32(research.vol20))
