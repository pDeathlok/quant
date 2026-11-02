from fastapi.testclient import TestClient
from quant.webapp.app import app
from quant.webapp import api

def test_new_plan_api_variant_and_no_legacy_holding(monkeypatch):
 calls=[]
 monkeypatch.setattr(api,'get_transition_daily_strategy',lambda strategy,refresh:calls.append((strategy,refresh)) or {'strategy':strategy,'status':'needs_account'})
 c=TestClient(app)
 assert c.get('/api/transition/plan?strategy=win').json()['strategy']=='win'
 assert calls==[('win',False)]
 assert c.get('/api/transition/plan?strategy=invalid').status_code==422

def test_account_validates_finite_values_and_optimistic_revision(monkeypatch):
 def save(strategy,body):
  if body['revision']==0:raise ValueError('持仓记录已被其他页面更新')
  return dict(strategy=strategy,**body)
 monkeypatch.setattr(api,'save_transition_account',save);c=TestClient(app)
 assert c.put('/api/transition/account/return',json={'revision':0,'cash':1000000,'positions':[]}).status_code==409
 assert c.put('/api/transition/account/win',json={'revision':1,'cash':100,'positions':[]}).json()['strategy']=='win'
 assert c.put('/api/transition/account/win',json={'revision':1,'cash':-1,'positions':[]}).status_code==422
 assert c.put('/api/transition/account/win',json={'revision':1,'cash':100,'positions':[dict(symbol='002594',entry_date='2026-10-30',quantity=100,cost=10)]}).status_code==422
