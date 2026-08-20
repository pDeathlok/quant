#!/usr/bin/env python3
"""Render a completed Deep-company dataset as a self-contained sortable HTML page."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT
        / "reports/good_company_deep_20260809/company_evaluations_deep_final.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/good_company_deep_20260809/good_company_dashboard.html",
    )
    parser.add_argument(
        "--expected",
        type=int,
        default=112,
        help="Required number of completed Deep companies.",
    )
    return parser.parse_args()


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>好公司研究台｜112家公司Deep研究</title>
<style>
:root{--ink:#0b1736;--muted:#667085;--line:#d9e0e7;--line2:#edf0f3;--canvas:#f5f7f8;--paper:#fff;--teal:#087e7a;--teal2:#e8f5f3;--red:#d9273e;--redbg:#fff0f2;--green:#168342;--greenbg:#edf8f1;--amber:#a86100;--amberbg:#fff6e5;--blue:#2459a8;--shadow:0 18px 45px rgba(11,23,54,.14)}
*{box-sizing:border-box}html{background:var(--canvas);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;font-size:14px}body{margin:0;min-width:320px}body.drawer-open{overflow:hidden}.app{min-height:100vh}.shell{max-width:1680px;margin:auto;padding:22px 24px 40px}.topbar{display:flex;align-items:flex-start;gap:28px;justify-content:space-between;margin-bottom:18px}.brand h1{font-size:30px;line-height:1.08;margin:0 0 7px;letter-spacing:-.7px}.brand p{color:var(--muted);margin:0}.nav{display:flex;gap:8px;align-self:center}.nav button{background:transparent;border:0;border-bottom:2px solid transparent;padding:10px 18px;font-size:15px;color:#344054;cursor:pointer}.nav button.active{color:var(--teal);border-color:var(--teal);font-weight:700}.cutoff{font-size:13px;color:#344054;border:1px solid var(--line);background:var(--paper);padding:9px 12px;white-space:nowrap}.kpis{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--paper);margin-bottom:14px}.kpi{padding:15px 18px;text-align:center;position:relative}.kpi+.kpi:before{content:"";position:absolute;left:0;top:14px;bottom:14px;border-left:1px solid var(--line)}.kpi span{display:block;color:var(--muted);font-size:12px;margin-bottom:4px}.kpi strong{font-size:27px;letter-spacing:-.5px}.kpi.danger strong{color:var(--red)}.filters{display:grid;grid-template-columns:minmax(190px,1.4fr) repeat(6,minmax(120px,1fr)) auto;gap:8px;padding:12px;border:1px solid var(--line);background:var(--paper);margin-bottom:14px}.control{min-width:0}.control label{display:block;color:var(--muted);font-size:11px;margin:0 0 5px}.control input,.control select{width:100%;height:36px;border:1px solid #cfd7df;background:#fff;color:var(--ink);padding:0 10px;border-radius:3px;outline:none}.control input:focus,.control select:focus{border-color:var(--teal);box-shadow:0 0 0 2px rgba(8,126,122,.10)}.check{display:flex;gap:8px;align-items:center;padding-top:17px;white-space:nowrap;color:#344054}.actions{display:flex;gap:7px;align-items:flex-end}.btn{height:36px;border:1px solid var(--line);background:#fff;color:var(--ink);padding:0 12px;border-radius:3px;cursor:pointer;font-weight:600}.btn:hover{border-color:var(--teal);color:var(--teal)}.btn.primary{background:var(--teal);border-color:var(--teal);color:white}.statusline{display:flex;align-items:center;justify-content:space-between;margin:8px 1px 10px;color:var(--muted);font-size:12px}.statusline strong{color:var(--ink)}.tablebox{border:1px solid var(--line);background:var(--paper);overflow:auto;max-height:calc(100vh - 330px);min-height:430px}.research-table{border-collapse:separate;border-spacing:0;width:100%;min-width:1450px}.research-table thead{position:sticky;top:0;z-index:3;background:#f7f8fa}.research-table th,.research-table td{border-bottom:1px solid var(--line2);border-right:1px solid var(--line2);padding:10px 9px;text-align:left;vertical-align:middle;white-space:nowrap}.research-table th{font-size:12px;color:#344054;font-weight:700}.research-table th:last-child,.research-table td:last-child{border-right:0}.research-table tbody tr{cursor:pointer}.research-table tbody tr:hover{background:#f5fbfa}.research-table tbody tr.selected{background:#eef7ff}.research-table .num{text-align:right;font-variant-numeric:tabular-nums}.sort{border:0;background:transparent;padding:0;color:inherit;font:inherit;font-weight:inherit;cursor:pointer}.sort .arrow{color:#98a2b3;margin-left:3px}.sort.active .arrow{color:var(--teal)}.star{border:0;background:transparent;font-size:21px;color:#98a2b3;cursor:pointer;padding:0 3px}.star.on{color:#e5a300}.company{min-width:130px}.company strong{display:block;font-size:14px}.company small{color:var(--muted);font-variant-numeric:tabular-nums}.badge{display:inline-block;border:1px solid #cbd5df;padding:3px 7px;border-radius:99px;font-size:11px;background:#fff}.badge.good{color:var(--teal);border-color:#84c7c3;background:var(--teal2)}.badge.great{color:#7b3fb5;border-color:#c8a7e4;background:#f7f0fc}.badge.potential{color:var(--amber);border-color:#e8c379;background:var(--amberbg)}.badge.ordinary{color:#667085}.gqs{font-size:15px;font-weight:800;color:var(--teal)}.modules{display:flex;align-items:flex-end;gap:2px;height:18px}.modules i{display:block;width:7px;background:var(--teal);min-height:3px}.positive{color:var(--red);font-weight:700}.negative{color:var(--green);font-weight:700}.na{color:#98a2b3}.tech{font-size:12px}.tech.strong{color:var(--red)}.tech.weak{color:var(--green)}.rowlinks{display:flex;align-items:center;gap:8px}.rowlinks a,.quick-links a{color:var(--blue);font-weight:700;text-decoration:none}.rowlinks a:hover,.quick-links a:hover{text-decoration:underline}.viewlink{border:0;background:transparent;color:var(--blue);font-weight:700;cursor:pointer;padding:0}.pagination{display:flex;align-items:center;justify-content:space-between;margin-top:12px}.pager{display:flex;gap:5px}.pager button{height:32px;min-width:32px;border:1px solid var(--line);background:white;cursor:pointer}.pager button.active{border-color:var(--teal);color:var(--teal);font-weight:800}.note{margin-top:14px;color:var(--muted);font-size:12px;line-height:1.7}.backdrop{position:fixed;inset:0;background:rgba(11,23,54,.25);z-index:20;display:none}.backdrop.open{display:block}.drawer{position:fixed;right:0;top:0;bottom:0;width:min(520px,100vw);background:#fff;z-index:21;box-shadow:var(--shadow);transform:translateX(102%);transition:transform .22s ease;display:flex;flex-direction:column}.drawer.open{transform:none}.drawer-head{padding:18px 20px 14px;border-bottom:1px solid var(--line);position:relative}.drawer-head h2{margin:0 40px 3px 0;font-size:25px}.drawer-head .code{color:var(--muted)}.close{position:absolute;right:15px;top:14px;border:0;background:transparent;font-size:28px;cursor:pointer;color:#475467}.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}.quick-links{display:flex;gap:16px;margin-top:12px;font-size:12px}.drawer-body{padding:18px 20px 94px;overflow:auto}.headline-score{display:grid;grid-template-columns:1fr 1fr;border:1px solid var(--line);margin-bottom:16px}.headline-score div{padding:12px 14px}.headline-score div+div{border-left:1px solid var(--line)}.headline-score span{display:block;color:var(--muted);font-size:11px}.headline-score strong{font-size:25px;color:var(--teal)}.section{border-top:1px solid var(--line);padding-top:15px;margin-top:16px}.section:first-child{border-top:0;margin-top:0}.section h3{font-size:14px;margin:0 0 11px}.score-row{display:grid;grid-template-columns:125px 1fr 64px;align-items:center;gap:8px;margin:9px 0}.score-row .bar{height:7px;background:#e9edf0;overflow:hidden}.score-row .bar i{display:block;height:100%;background:var(--teal)}.score-row b{text-align:right;font-variant-numeric:tabular-nums}.scenario-grid{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line)}.scenario{padding:11px;text-align:center}.scenario+.scenario{border-left:1px solid var(--line)}.scenario h4{margin:0 0 8px;font-size:12px}.scenario .target{font-size:19px;font-weight:800}.scenario small{display:block;color:var(--muted);margin:4px 0}.scenario .space{font-size:14px}.text-list{margin:0;padding-left:20px;color:#344054;line-height:1.65}.text-list li+li{margin-top:6px}.counter{border-left:3px solid var(--red);background:var(--redbg);padding:11px 12px;line-height:1.6;color:#47202a}.sources{list-style:none;padding:0;margin:0}.sources li{padding:9px 0;border-bottom:1px solid var(--line2)}.sources a{color:var(--blue);text-decoration:none;word-break:break-all}.sources small{display:block;color:var(--muted);margin-top:3px}.drawer-actions{position:absolute;bottom:0;left:0;right:0;background:rgba(255,255,255,.96);border-top:1px solid var(--line);padding:12px 20px;display:grid;grid-template-columns:1fr 1fr;gap:8px}.empty{padding:70px 20px;text-align:center;color:var(--muted)}
@media(max-width:1200px){.filters{grid-template-columns:repeat(4,1fr)}.topbar{flex-wrap:wrap}.nav{order:3;width:100%;border-top:1px solid var(--line)}.tablebox{max-height:none}}
@media(max-width:700px){.shell{padding:14px 10px 80px}.brand h1{font-size:24px}.cutoff{font-size:11px}.nav{overflow:auto}.nav button{padding:9px 12px}.kpis{grid-template-columns:repeat(2,1fr)}.kpi:nth-child(3):before{display:none}.kpi:nth-child(n+3){border-top:1px solid var(--line)}.filters{grid-template-columns:1fr 1fr}.control.search{grid-column:1/-1}.actions{grid-column:1/-1}.actions .btn{flex:1}.statusline{align-items:flex-start;gap:10px}.tablebox{min-height:350px}.drawer-head{padding-top:14px}.drawer-body{padding:15px 15px 88px}.score-row{grid-template-columns:105px 1fr 58px}.scenario-grid{grid-template-columns:1fr}.scenario+.scenario{border-left:0;border-top:1px solid var(--line)}.drawer-actions{padding:10px 14px}.note{font-size:11px}}
</style>
</head>
<body>
<div class="app"><main class="shell">
  <header class="topbar">
    <div class="brand"><h1>好公司研究台</h1><p>112 家逐公司 Deep 研究：质量、估值与风险一页筛选</p></div>
    <nav class="nav"><button class="active">公司池</button><button id="methodBtn">评分方法</button><button id="auditBtn">数据审计</button></nav>
    <div class="cutoff">研究更新至 <strong id="cutoffText"></strong></div>
  </header>
  <section class="kpis">
    <div class="kpi"><span>已完成 Deep</span><strong id="kpiTotal">—</strong></div>
    <div class="kpi"><span>估值可用</span><strong id="kpiValuation">—</strong></div>
    <div class="kpi"><span>平均 GQS-R</span><strong id="kpiGqs">—</strong></div>
    <div class="kpi danger"><span>卓越复利候选</span><strong id="kpiGreat">—</strong></div>
  </section>
  <section class="filters" aria-label="公司筛选">
    <div class="control search"><label for="search">公司 / 代码</label><input id="search" placeholder="输入名称或证券代码"></div>
    <div class="control"><label for="industry">行业</label><select id="industry"><option value="">全部行业</option></select></div>
    <div class="control"><label for="classification">分类</label><select id="classification"><option value="">全部分类</option></select></div>
    <div class="control"><label for="dimensionSort">按维度排序</label><select id="dimensionSort"><option value="">保持当前排序</option><option value="a_customer_business">A 客户价值</option><option value="b_scarcity_moat">B 稀缺性</option><option value="c_growth_reinvestment">C 成长</option><option value="d_returns_profitability">D 回报</option><option value="e_cash_accounting">E 现金</option><option value="f_resilience_risk">F 韧性</option><option value="g_governance_allocation">G 治理</option></select></div>
    <div class="control"><label for="minGqs">最低 GQS-F</label><input id="minGqs" type="number" min="0" max="100" step="1" placeholder="0"></div>
    <div class="control"><label for="minUpside">最低中性空间 %</label><input id="minUpside" type="number" step="5" placeholder="不限"></div>
    <div class="control"><label for="minCoverage">最低证据覆盖 %</label><input id="minCoverage" type="number" min="0" max="100" step="5" placeholder="0"></div>
    <div class="control"><label for="valuationStatus">估值状态</label><select id="valuationStatus"><option value="">全部</option><option value="available">可用</option><option value="unavailable">不可用</option></select></div>
    <div class="actions"><button class="btn" id="resetBtn">清空</button><button class="btn primary" id="exportBtn">导出 CSV</button></div>
    <label class="check"><input id="watchOnly" type="checkbox"> 仅看自选</label>
  </section>
  <div class="statusline"><div>筛选结果 <strong id="resultCount">0</strong> 家｜排序：<strong id="sortLabel">GQS-F ↓</strong></div><div>质量分、估值空间、证据覆盖与量价状态彼此独立</div></div>
  <section class="tablebox"><table class="research-table">
    <thead><tr>
      <th>自选</th><th><button class="sort" data-sort="name">公司 / 代码 <span class="arrow">↕</span></button></th>
      <th><button class="sort" data-sort="industry">行业 <span class="arrow">↕</span></button></th><th>分类</th>
      <th class="num"><button class="sort" data-sort="gqs_r">GQS-R <span class="arrow">↕</span></button></th>
      <th class="num"><button class="sort active" data-sort="gqs_f">GQS-F <span class="arrow">↓</span></button></th>
      <th>A–G</th><th class="num"><button class="sort" data-sort="current_price">当前价 <span class="arrow">↕</span></button></th>
      <th class="num"><button class="sort" data-sort="bear_upside">悲观空间 <span class="arrow">↕</span></button></th>
      <th class="num"><button class="sort" data-sort="base_upside">中性空间 <span class="arrow">↕</span></button></th>
      <th class="num"><button class="sort" data-sort="bull_upside">乐观空间 <span class="arrow">↕</span></button></th>
      <th class="num"><button class="sort" data-sort="coverage">证据覆盖 <span class="arrow">↕</span></button></th><th>技术状态</th><th>研究入口</th>
    </tr></thead><tbody id="tableBody"></tbody>
  </table><div id="empty" class="empty" hidden>没有符合条件的公司。请降低筛选门槛。</div></section>
  <div class="pagination"><div id="pageInfo">—</div><div class="pager" id="pager"></div><label>每页 <select id="pageSize"><option>20</option><option selected>25</option><option>50</option><option>100</option></select> 家</label></div>
  <p class="note">注：GQS 衡量公司质量，不包含估值和价格。三情景目标价是条件式估算；未通过价格、财报、预测样本、股本和行业方法门槛时显示“不可用”。红色为正收益、绿色为负收益，符合 A 股阅读习惯。本页面仅供研究与教育用途。</p>
</main></div>
<div class="backdrop" id="backdrop"></div>
<aside class="drawer" id="drawer" aria-hidden="true">
  <div class="drawer-head"><button class="close" id="closeDrawer" aria-label="关闭">×</button><h2 id="drawerName">—</h2><span class="code" id="drawerCode"></span><div class="chips" id="drawerChips"></div><div class="quick-links" id="drawerLinks"></div></div>
  <div class="drawer-body" id="drawerBody"></div>
  <div class="drawer-actions"><button class="btn" id="drawerWatch">☆ 加入自选</button><button class="btn primary" id="copySummary">复制摘要</button></div>
</aside>
<script id="companyData" type="application/json">__COMPANY_DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('companyData').textContent);
const $=id=>document.getElementById(id);const state={page:1,size:25,sort:'gqs_f',dir:'desc',selected:null};
const modules=[['a_customer_business',10,'A'],['b_scarcity_moat',20,'B'],['c_growth_reinvestment',10,'C'],['d_returns_profitability',20,'D'],['e_cash_accounting',15,'E'],['f_resilience_risk',10,'F'],['g_governance_allocation',15,'G']];
const moduleNames={A:'客户价值',B:'稀缺性',C:'成长',D:'回报',E:'现金',F:'韧性',G:'治理'};
function loadWatch(){try{const value=JSON.parse(localStorage.getItem('goodCompanyWatchlist')||'[]');return new Set(Array.isArray(value)?value:[]);}catch{return new Set();}}
function saveWatch(){try{localStorage.setItem('goodCompanyWatchlist',JSON.stringify([...watch]));}catch{}}
let watch=loadWatch();
const safe=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const num=(v,d=1)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toFixed(d);
const pct=(v,d=1)=>v==null||!Number.isFinite(Number(v))?'—':`${v>=0?'+':''}${(Number(v)*100).toFixed(d)}%`;
const pctPlain=(v,d=1)=>v==null||!Number.isFinite(Number(v))?'—':`${(Number(v)*100).toFixed(d)}%`;
const pctClass=v=>v==null?'na':v>=0?'positive':'negative';
const clsBadge=v=>v==='卓越复利候选'?'great':v==='优质公司'?'good':v==='潜力公司'?'potential':'ordinary';
const techClass=v=>['强势确认','建设性偏强'].includes(v)?'strong':['弱势未确认','谨慎'].includes(v)?'weak':'';
const reportUrl=code=>`individual_reports/${encodeURIComponent(code)}.md`;
const xueqiuUrl=code=>{const [symbol,exchange]=code.split('.');return `https://xueqiu.com/S/${exchange}${symbol}`;};
function flat(item){const v=item.valuation;return {item,code:item.identity.ts_code,name:item.identity.name,industry:item.identity.industry,broad:item.identity.broad_industry,classification:item.gqs.classification,gqs_r:item.gqs.gqs_r,gqs_f:item.gqs.gqs_f,a_customer_business:item.gqs.a_customer_business,b_scarcity_moat:item.gqs.b_scarcity_moat,c_growth_reinvestment:item.gqs.c_growth_reinvestment,d_returns_profitability:item.gqs.d_returns_profitability,e_cash_accounting:item.gqs.e_cash_accounting,f_resilience_risk:item.gqs.f_resilience_risk,g_governance_allocation:item.gqs.g_governance_allocation,coverage:item.gqs.coverage_ratio,current_price:item.market.current_price,bear_upside:v.bear?.price_upside??null,base_upside:v.base?.price_upside??null,bull_upside:v.bull?.price_upside??null,valuation_status:v.status,technical:item.market.technical_state};}
const ROWS=DATA.map(flat);
function init(){const cutoff=DATA.map(x=>x.cutoff.analysis_cutoff).filter(Boolean).sort().at(-1)||'—';$('cutoffText').textContent=cutoff.replace('T',' ').replace('+08:00','');$('kpiTotal').textContent=DATA.length;$('kpiValuation').textContent=DATA.filter(x=>x.valuation.status==='available').length;$('kpiGqs').textContent=num(DATA.reduce((s,x)=>s+x.gqs.gqs_r,0)/DATA.length,1);$('kpiGreat').textContent=DATA.filter(x=>x.gqs.classification==='卓越复利候选').length;[...new Set(ROWS.map(x=>x.broad))].sort().forEach(x=>$('industry').insertAdjacentHTML('beforeend',`<option>${safe(x)}</option>`));[...new Set(ROWS.map(x=>x.classification))].sort().forEach(x=>$('classification').insertAdjacentHTML('beforeend',`<option>${safe(x)}</option>`));bind();render();}
function filtered(){const q=$('search').value.trim().toLowerCase(),ind=$('industry').value,cl=$('classification').value,minG=Number($('minGqs').value||0),up=$('minUpside').value===''?null:Number($('minUpside').value)/100,cov=Number($('minCoverage').value||0)/100,vs=$('valuationStatus').value,wo=$('watchOnly').checked;return ROWS.filter(x=>(!q||x.name.toLowerCase().includes(q)||x.code.toLowerCase().includes(q))&&(!ind||x.broad===ind)&&(!cl||x.classification===cl)&&(x.gqs_f??-1)>=minG&&(up==null||(x.base_upside!=null&&x.base_upside>=up))&&(x.coverage??0)>=cov&&(!vs||x.valuation_status===vs)&&(!wo||watch.has(x.code)));}
function sortRows(rows){return rows.sort((a,b)=>{let av=a[state.sort],bv=b[state.sort];if(av==null&&bv==null)return 0;if(av==null)return 1;if(bv==null)return -1;if(typeof av==='string')return av.localeCompare(bv,'zh-CN')*(state.dir==='asc'?1:-1);return (av-bv)*(state.dir==='asc'?1:-1);});}
function moduleBars(item){const detail=modules.map(([k,max,l])=>`${l} ${num(item.gqs[k])}/${max}`).join(' · ');return `<div class="modules" title="${detail}">${modules.map(([k,max])=>`<i style="height:${Math.max(3,(item.gqs[k]??0)/max*18)}px"></i>`).join('')}</div>`;}
function render(){const rows=sortRows(filtered());$('resultCount').textContent=rows.length;const pages=Math.max(1,Math.ceil(rows.length/state.size));state.page=Math.min(state.page,pages);const start=(state.page-1)*state.size,pageRows=rows.slice(start,start+state.size);$('tableBody').innerHTML=pageRows.map(x=>`<tr data-code="${x.code}" class="${state.selected===x.code?'selected':''}"><td><button class="star ${watch.has(x.code)?'on':''}" data-star="${x.code}" title="加入自选">${watch.has(x.code)?'★':'☆'}</button></td><td class="company"><strong>${safe(x.name)}</strong><small>${x.code}</small></td><td>${safe(x.industry)}</td><td><span class="badge ${clsBadge(x.classification)}">${safe(x.classification)}</span></td><td class="num gqs">${num(x.gqs_r)}</td><td class="num gqs">${num(x.gqs_f)}</td><td>${moduleBars(x.item)}</td><td class="num">${num(x.current_price,2)}</td><td class="num ${pctClass(x.bear_upside)}">${pct(x.bear_upside)}</td><td class="num ${pctClass(x.base_upside)}">${pct(x.base_upside)}</td><td class="num ${pctClass(x.bull_upside)}">${pct(x.bull_upside)}</td><td class="num">${pctPlain(x.coverage)}</td><td><span class="tech ${techClass(x.technical)}">${safe(x.technical||'数据不足')}</span></td><td><div class="rowlinks"><a data-external href="${reportUrl(x.code)}" target="_blank" rel="noreferrer" aria-label="打开${safe(x.name)}个股报告">MD</a><a data-external href="${xueqiuUrl(x.code)}" target="_blank" rel="noreferrer" aria-label="打开${safe(x.name)}雪球页面">雪球</a><button class="viewlink" data-view="${x.code}">查看</button></div></td></tr>`).join('');$('empty').hidden=pageRows.length>0;$('pageInfo').textContent=`${rows.length?start+1:0}–${Math.min(start+state.size,rows.length)} / 共 ${rows.length} 家`;$('pager').innerHTML=pagerHtml(pages);$('sortLabel').textContent=`${sortName(state.sort)} ${state.dir==='asc'?'↑':'↓'}`;document.querySelectorAll('.sort').forEach(b=>{b.classList.toggle('active',b.dataset.sort===state.sort);b.querySelector('.arrow').textContent=b.dataset.sort===state.sort?(state.dir==='asc'?'↑':'↓'):'↕';});}
function pagerHtml(pages){const nums=new Set([1,pages,state.page-1,state.page,state.page+1]);return `<button data-page="${Math.max(1,state.page-1)}">‹</button>${[...nums].filter(x=>x>=1&&x<=pages).sort((a,b)=>a-b).map((x,i,a)=>`${i&&x-a[i-1]>1?'<span>…</span>':''}<button class="${x===state.page?'active':''}" data-page="${x}">${x}</button>`).join('')}<button data-page="${Math.min(pages,state.page+1)}">›</button>`;}
function sortName(k){return ({name:'公司',industry:'行业',gqs_r:'GQS-R',gqs_f:'GQS-F',a_customer_business:'A 客户价值',b_scarcity_moat:'B 稀缺性',c_growth_reinvestment:'C 成长',d_returns_profitability:'D 回报',e_cash_accounting:'E 现金',f_resilience_risk:'F 韧性',g_governance_allocation:'G 治理',current_price:'当前价',bear_upside:'悲观空间',base_upside:'中性空间',bull_upside:'乐观空间',coverage:'证据覆盖'})[k]||k;}
function bind(){['search','industry','classification','minGqs','minUpside','minCoverage','valuationStatus','watchOnly'].forEach(id=>$(id).addEventListener(id==='search'||id.startsWith('min')?'input':'change',()=>{state.page=1;render();}));$('dimensionSort').addEventListener('change',e=>{if(!e.target.value)return;state.sort=e.target.value;state.dir='desc';state.page=1;render();});$('pageSize').addEventListener('change',e=>{state.size=Number(e.target.value);state.page=1;render();});$('resetBtn').onclick=()=>{['search','industry','classification','dimensionSort','minGqs','minUpside','minCoverage','valuationStatus'].forEach(id=>$(id).value='');$('watchOnly').checked=false;state.sort='gqs_f';state.dir='desc';state.page=1;render();};document.addEventListener('click',e=>{const external=e.target.closest('[data-external]'),sort=e.target.closest('[data-sort]'),star=e.target.closest('[data-star]'),view=e.target.closest('[data-view]'),page=e.target.closest('[data-page]'),tr=e.target.closest('tr[data-code]');if(external){e.stopPropagation();return;}if(sort){const k=sort.dataset.sort;state.dir=state.sort===k&&state.dir==='desc'?'asc':'desc';state.sort=k;render();}else if(star){e.stopPropagation();toggleWatch(star.dataset.star);}else if(view){e.stopPropagation();openDrawer(view.dataset.view);}else if(page){state.page=Number(page.dataset.page);render();}else if(tr)openDrawer(tr.dataset.code);});document.addEventListener('keydown',e=>{if(e.key==='Escape'&&state.selected)closeDrawer();});$('closeDrawer').onclick=closeDrawer;$('backdrop').onclick=closeDrawer;$('drawerWatch').onclick=()=>state.selected&&toggleWatch(state.selected);$('exportBtn').onclick=exportCsv;$('copySummary').onclick=copySummary;$('methodBtn').onclick=()=>alert('GQS共100分：A客户价值10、B稀缺性20、C成长10、D回报20、E现金15、F韧性10、G治理15。估值和价格不进入GQS。');$('auditBtn').onclick=()=>alert(`数据审计：${DATA.length}家公司；估值可用${DATA.filter(x=>x.valuation.status==='available').length}家；估值不可用${DATA.filter(x=>x.valuation.status!=='available').length}家。详情中的来源与局限可逐家查看。`);}
function toggleWatch(code){watch.has(code)?watch.delete(code):watch.add(code);saveWatch();render();if(state.selected===code)updateDrawerWatch();}
function updateDrawerWatch(){const on=watch.has(state.selected);$('drawerWatch').textContent=on?'★ 已加入自选':'☆ 加入自选';}
function openDrawer(code){const x=ROWS.find(r=>r.code===code);if(!x)return;state.selected=code;render();const i=x.item;$('drawerName').textContent=i.identity.name;$('drawerCode').textContent=i.identity.ts_code;$('drawerChips').innerHTML=`<span class="badge ${clsBadge(i.gqs.classification)}">${safe(i.gqs.classification)}</span><span class="badge">${safe(i.identity.broad_industry)}</span><span class="badge good">证据 ${pctPlain(i.gqs.coverage_ratio)}</span>`;$('drawerLinks').innerHTML=`<a data-external href="${reportUrl(code)}" target="_blank" rel="noreferrer">打开个股报告 MD ↗</a><a data-external href="${xueqiuUrl(code)}" target="_blank" rel="noreferrer">打开雪球 ↗</a>`;$('drawerBody').innerHTML=detailHtml(i);$('drawer').classList.add('open');$('backdrop').classList.add('open');document.body.classList.add('drawer-open');$('drawer').setAttribute('aria-hidden','false');updateDrawerWatch();$('closeDrawer').focus();}
function closeDrawer(){$('drawer').classList.remove('open');$('backdrop').classList.remove('open');document.body.classList.remove('drawer-open');$('drawer').setAttribute('aria-hidden','true');state.selected=null;render();}
function detailHtml(i){return `<div class="headline-score"><div><span>GQS-R 历史验证</span><strong>${num(i.gqs.gqs_r)}</strong> /100</div><div><span>GQS-F 前瞻调整后</span><strong>${num(i.gqs.gqs_f)}</strong> /100</div></div><section class="section"><h3>七维评分（得分 / 满分）</h3>${modules.map(([k,max,l])=>`<div class="score-row"><span>${l} ${moduleNames[l]}</span><div class="bar"><i style="width:${(i.gqs[k]??0)/max*100}%"></i></div><b>${num(i.gqs[k])} / ${max}</b></div>`).join('')}</section><section class="section"><h3>12个月三情景｜${safe(i.valuation.method_primary)}</h3>${scenarioHtml(i)}</section><section class="section"><h3>核心逻辑</h3><ul class="text-list">${i.research.thesis_pillars.map(x=>`<li>${safe(x)}</li>`).join('')}</ul></section><section class="section"><h3>最强反方</h3><div class="counter">${safe(i.research.strongest_counterargument)}</div></section><section class="section"><h3>证伪条件</h3><ul class="text-list">${i.research.falsifiers.map(x=>`<li>${safe(x)}</li>`).join('')}</ul></section><section class="section"><h3>监测清单</h3><ul class="text-list">${i.research.monitoring.map(x=>`<li>${safe(x)}</li>`).join('')}</ul></section><section class="section"><h3>来源与局限</h3><ul class="sources">${i.evidence.sources.map(source=>{const target=source.url||source.path||'';const link=source.url?`<a href="${safe(source.url)}" target="_blank" rel="noreferrer">${safe(source.label)}</a>`:`<span>${safe(source.label)}</span>`;return `<li>${link}<small>${safe(target)}｜可得：${safe(source.available_at||'未标注')}</small></li>`;}).join('')}</ul><ul class="text-list">${i.gqs.score_limitations.map(x=>`<li>${safe(x)}</li>`).join('')}</ul>${i.evidence.data_conflicts.length?`<div class="counter">数据冲突：${i.evidence.data_conflicts.map(x=>safe(x.conflict+'；处理：'+x.resolution)).join('<br>')}</div>`:''}</section>`;}
function scenarioHtml(i){const v=i.valuation;if(v.status!=='available')return `<div class="counter">目标价不可用：${safe(v.missing_reasons.join('；'))}</div>`;return `<div class="scenario-grid">${[['bear','悲观'],['base','中性'],['bull','乐观']].map(([k,l])=>{const s=v[k];return `<div class="scenario"><h4>${l}</h4><div class="target">${num(s.target_price,2)} 元</div><small>价格空间</small><div class="space ${pctClass(s.price_upside)}">${pct(s.price_upside)}</div><small>含股息 ${pct(s.total_return)}</small></div>`;}).join('')}</div><p class="note">当前价 ${num(i.market.current_price,2)} 元｜预测基准 ${safe(v.forecast_basis)}｜${safe(v.method_crosscheck||'无交叉检验')}</p>`;}
function exportCsv(){const rows=sortRows(filtered());const headers=['代码','公司','行业','分类','GQS-R','GQS-F','证据覆盖','当前价','悲观空间','中性空间','乐观空间','估值状态','技术状态','个股MD','雪球链接'];const body=rows.map(x=>[x.code,x.name,x.industry,x.classification,x.gqs_r,x.gqs_f,x.coverage,x.current_price,x.bear_upside,x.base_upside,x.bull_upside,x.valuation_status,x.technical,new URL(reportUrl(x.code),location.href).href,xueqiuUrl(x.code)]);const csv=[headers,...body].map(row=>row.map(v=>`"${String(v??'').replaceAll('"','""')}"`).join(',')).join('\n');window.__lastExportRowCount=rows.length;const blob=new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);const stamp=(DATA.map(x=>x.cutoff.analysis_cutoff).filter(Boolean).sort().at(-1)||'').slice(0,10).replaceAll('-','');a.download=`好公司筛选结果_${stamp||'latest'}.csv`;a.click();URL.revokeObjectURL(a.href);$('exportBtn').textContent=`已导出 ${rows.length} 家`;setTimeout(()=>$('exportBtn').textContent='导出 CSV',1200);}
async function copySummary(){const i=DATA.find(x=>x.identity.ts_code===state.selected);if(!i)return;const text=`${i.identity.name}（${i.identity.ts_code}）\n${i.research.summary}\n最强反方：${i.research.strongest_counterargument}`;try{await navigator.clipboard.writeText(text);}catch{const area=document.createElement('textarea');area.value=text;area.style.position='fixed';area.style.opacity='0';document.body.appendChild(area);area.select();document.execCommand('copy');area.remove();}$('copySummary').textContent='已复制';setTimeout(()=>$('copySummary').textContent='复制摘要',1200);}
init();
</script>
</body></html>'''


def main() -> None:
    args = parse_args()
    validator = PROJECT_ROOT / "scripts/research/validate_priority_deep_dataset.py"
    validation = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--data",
            str(args.input),
            "--expected",
            str(args.expected),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if validation.returncode:
        raise SystemExit(
            "priority Deep dataset validation failed before HTML generation:\n"
            + validation.stdout
            + validation.stderr
        )
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if len(data) != args.expected:
        raise SystemExit(f"expected {args.expected} companies, got {len(data)}")
    report_dir = PROJECT_ROOT / "reports/good_company_deep_20260809/individual_reports"
    missing_reports = [
        item["identity"]["ts_code"]
        for item in data
        if not (report_dir / f'{item["identity"]["ts_code"]}.md').is_file()
    ]
    if missing_reports:
        raise SystemExit(f"missing individual reports: {', '.join(missing_reports)}")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("</script", "<\\/script")
    html = HTML_TEMPLATE.replace("__COMPANY_DATA__", payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "companies": len(data), "bytes": args.output.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
