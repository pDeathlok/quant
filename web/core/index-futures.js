import { escapeHtml as esc } from "./formatters.js";

let payload = null, pending = null, bound = false, error = "", refreshing = false, displayedExpired = false;
const view = { mode: "rolling_qvix", profile: "stress", window: "sessions_244", year: "2026", phases: false };
const profiles = { stress: { fee: .0006, margin: .25, label: "压力 · 6bps / 25%保证金" }, basic: { fee: .0003, margin: .15, label: "基础 · 3bps / 15%保证金" } };
const policyNames = {base:"基础入场",aux55:"辅助胜率 ≥55%",confidence60:"方向置信度 ≥60%",opposing_exit:"对向信号退出"};
const pc = v => Number.isFinite(v) ? `${(v*100).toFixed(2)}%` : "—";
const money = v => Number.isFinite(v) ? `${(v/10000).toFixed(2)}万元` : "—";
const score = v => Number.isFinite(v) ? (v*100).toFixed(1) : "—";
const date = v => String(v || "").replace(/^(\d{4})(\d{2})(\d{2})$/, "$1-$2-$3");
const median = values => { const a=[...values].filter(Number.isFinite).sort((a,b)=>a-b);return a.length ? (a[Math.floor(a.length/2)]+a[Math.floor((a.length-1)/2)])/2:null; };
const cell = (v,loss=false) => `<span class="${v>0?(loss?"if-negative":"if-positive"):v<0?"if-negative":""}">${pc(v)}</span>`;
const options = (items,selected) => items.map(([v,label])=>`<option value="${esc(v)}" ${v===selected?"selected":""}>${esc(label)}</option>`).join("");
const records = (mode=view.mode) => payload.records.filter(r=>r.startup_mode===mode&&r.fee===profiles[view.profile].fee&&r.margin===profiles[view.profile].margin);
const table = (head,body,label) => `<div class="table-wrap if-table" tabindex="0" role="region" aria-label="${esc(label)}"><table><thead><tr>${head.map(x=>`<th>${x}</th>`).join("")}</tr></thead><tbody>${body}</tbody></table></div>`;
const badge = (pass,label) => `<span class="if-gate ${pass?"if-pass":"if-fail"}">${pass?"✓":"×"} ${esc(label)}</span>`;

function openingPanel() {
  const s=payload.signal, current=s?.profiles?.[view.profile]?.[view.mode];
  if(!current) return `<section class="panel if-opening"><h3>当前开仓检查</h3><p class="if-warning">信号尚未生成，无法判断开仓。请更新行情与评分。</p><button type="button" data-if-refresh ${refreshing?"disabled":""}>${refreshing?"更新中…":"更新行情与评分"}</button></section>`;
  const expired=Date.now()>=Date.parse(s.valid_until), stale=expired||current.status==="stale";
  const conditional=!stale&&current.status==="conditional";
  const eligible=current.candidate;
  const title=stale?"信号已过期 · 暂不能判断开仓":conditional?"条件满足 · 待开盘价确认":"当前不开仓";
  const reason=stale?"请更新行情与评分，旧候选仅作回看。":!current.entry_due?`下一交易日 ${date(s.execution_date)} 不在全局入场日内；下一检查日 ${date(current.next_check)}。`:!eligible?"没有同时通过方向、盈利概率、收益风险和资金检查的标的。":"这是下一交易日的收盘后预案，实际价格与账户权益仍需确认。";
  const allBlocked=!current.candidates.some(r=>r.passes);
  const fallback=current.candidates.some(r=>r.fallback!=="none");
  return `<section class="panel if-opening" id="ifOpening"><div class="panel-head"><div><span class="if-kicker">当前开仓检查 · 新账户试算</span><h3>${title}</h3></div><button class="secondary-button" type="button" data-if-refresh ${refreshing?"disabled":""}>${refreshing?"更新行情与评分中…":"更新行情与评分"}</button></div>
    <p class="if-note">${esc(reason)}${!stale&&!current.entry_due&&allBlocked?" 当前候选评分也全部未过线。":""}</p>
    <div class="if-signal-facts"><div><span>信号收盘日</span><strong>${date(s.signal_date)}</strong></div><div><span>计划执行日</span><strong>${date(s.execution_date)}</strong></div><div><span>开仓标的</span><strong>${conditional&&eligible?`${esc(eligible.contract)} · ${eligible.side==="long"?"做多":"做空"}`:"无"}</strong></div><div><span>当前入场政策</span><strong>${esc(policyNames[current.policy]||current.policy)}</strong></div></div>
    <p class="if-note">50万元可用权益 · 当前空仓 · 最多一手；未连接真实账户。下表按信号日收盘价估算，包含保证金、开仓摩擦与风险储备。${fallback?"<strong>QVIX连续特征不足，已回退原85因子模型。</strong>":"使用QVIX增强模型。"}</p>
    ${table(["合约 / 方向","方向评分","5日盈利概率","收益风险分","最低权益 / 缺口","检查结果"],current.candidates.map(r=>`<tr><td><strong>${esc(r.contract)}</strong><br>${r.side==="long"?"做多":"做空"} · ${{1:"牛市",0:"过渡","-1":"熊市"}[r.state]}</td><td>${score(r.confidence)} / 100</td><td>${pc(r.probability)}</td><td>${cell(r.utility)}</td><td>${money(r.minimum_equity)}<br><small>缺口 ${money(r.capital_gap)}</small></td><td>${r.passes?"评分与资金通过":"未通过："+Object.entries(r.gates).filter(([,ok])=>!ok).map(([k])=>({direction:"方向",probability:"胜率",utility:"收益风险",capital:"资金"}[k])).join("、")}</td></tr>`).join(""),"当前开仓候选")}
    <details class="if-continuous"><summary>查看逐项开仓逻辑、资金计算与退出规则</summary><div class="if-candidate-grid">${current.candidates.map(r=>`<article><h4>${esc(r.contract)} · ${r.side==="long"?"做多":"做空"}</h4><div class="if-gates">${badge(!stale,"数据有效")}${badge(current.entry_due,"入场日")}${badge(r.gates.direction,"方向")}${badge(r.gates.probability,"盈利概率")}${badge(r.gates.utility,"收益风险")}${badge(r.gates.capital,"资金")}</div><p>多头分 ${score(r.long_score)} / 空头分 ${score(r.short_score)}；取较高方向，要求 &gt;45分且多空差 ≥10分。${current.policy==="confidence60"?"本政策另要求 ≥60分。":current.policy==="aux55"?"本政策另要求辅助盈利概率 ≥55%。":""}</p><p>5日盈利概率 ${pc(r.probability)}，门槛 ≥55%；预计净收益 ${pc(r.pred_net)} − 0.5 × 预计不利波动 ${pc(r.pred_risk)} = ${pc(r.utility)}，要求 &gt;0。收益风险分以合约名义本金为分母，并非账户收益承诺。</p><p>参考价 ${r.price.toFixed(1)}；一手保证金 ${money(r.margin_cash)}，含风险储备最低权益 ${money(r.minimum_equity)}。按50万元计算，其他条件不变时最高可承受开仓价约 ${r.max_entry_price.toFixed(1)}。</p><p>持有上限 ${r.holding_limit} 个交易日；5日续持预测连续两次为负、资金保护或到期条件先触发先退出。${current.policy==="opposing_exit"?"另有短期对向概率 ≥35%退出。":""}</p><p>共同QVIX原始日期 ${date(r.qvix_common_date)||"缺失"}，自身 ${date(r.qvix_own_date)||"缺失"}；${r.fallback==="none"?"增强模型":"回退原模型（单日有值不代表连续特征完整）"}。</p></article>`).join("")}</div><p class="if-note">盈利概率为模型输出，未经新样本校准；方向评分用于排序，不等同于该笔交易胜率。资金门槛沿用研究中的至少5%或3倍日波动压力损失，并预留20%权益；开盘跳空、实际费率或保证金变化都会改变结果。</p></details>
    ${view.mode.startsWith("rolling")?`<details class="if-continuous"><summary>查看本次启动的策略选择评分</summary><p class="if-note">按50万元空仓启动，用过去36个月已完成的122/244日多起点窗口选策，验证窗口最晚至 ${date(s.history_end)}。这不是已有持仓账户的当前政策；持续账户须用自己的权益与既有政策重验。</p>${table(["政策","稳健评分","收益下分位","收益中位","回撤90分位","资金阻挡比例"],Object.entries(current.selection).map(([name,r])=>`<tr class="${name===current.policy?"if-selected-row":""}"><td>${esc(policyNames[name]||name)}${name===current.policy?" · 选中":""}</td><td>${pc(r.score)}</td><td>${pc(r.q25)}</td><td>${pc(r.median)}</td><td>${pc(r.dd90)}</td><td>${pc(r.block)}</td></tr>`).join(""),"启动选策评分")}<p class="if-note">评分 = 0.5 ×（收益25分位＋收益中位－回撤90分位），122/244日分别统计后等权，近期半衰期12个月。政策还需满足足够交易、资金阻挡和历史强平约束；已有账户的季度切换另检查切换成本与优势。</p></details>`:""}
    <p class="if-note">QVIX源更新：${s.qvix_refresh?.status==="updated"?"本次已获取": "本次源不可用，使用有日期的缓存；连续特征缺失则回退原模型"}。</p>
    <p class="if-note">模型季度 ${esc(s.model.quarter)} · 训练成熟标签早于 ${date(s.model.cutoff)} · 本次生成 ${esc(s.generated_at.slice(0,19).replace("T"," "))}（北京时间）。有效期至 ${date(s.execution_date)} 开盘前，届时须重新核验。</p></section>`;
}

function equityChart(curve) {
  if(!curve?.length)return "";
  const values=curve.map(r=>r.equity/500000-1), lo=Math.min(0,...values)-.03, hi=Math.max(0,...values)+.03;
  const x=i=>62+i*670/Math.max(1,curve.length-1), y=v=>175-(v-lo)/(hi-lo)*145;
  const minDD=Math.min(-.01,...curve.map(r=>r.drawdown));
  return `<svg class="if-chart if-equity-chart" viewBox="0 0 800 310" role="img" aria-label="2025启动持续账户累计收益和回撤曲线"><title>2025启动持续账户累计收益和回撤</title>${[lo+(hi-lo)*.2,lo+(hi-lo)*.5,lo+(hi-lo)*.8].map(v=>`<line class="if-gridline" x1="60" x2="740" y1="${y(v)}" y2="${y(v)}"/><text x="52" y="${y(v)+4}" text-anchor="end">${pc(v)}</text>`).join("")}<polyline class="if-selected-line" points="${curve.map((r,i)=>`${x(i)},${y(r.equity/500000-1)}`).join(" ")}"/><text x="62" y="15">累计收益 · 同一账户</text><line class="if-gridline" x1="60" x2="740" y1="220" y2="220"/><text x="62" y="205">回撤 · 高点至当前</text><polyline class="if-drawdown-line" points="${curve.map((r,i)=>`${x(i)},${220+r.drawdown/minDD*50}`).join(" ")}"/><text x="52" y="274" text-anchor="end">${pc(minDD)}</text>${curve.filter((r,i)=>i%Math.max(1,Math.floor(curve.length/5))===0).map(r=>`<text x="${x(curve.indexOf(r))}" y="300" text-anchor="middle">${date(r.date).slice(0,7)}</text>`).join("")}</svg>`;
}

function recentPanel() {
  const r=payload.recent[view.mode][view.profile];
  return `<section class="panel" id="ifRecent"><div class="panel-head"><h3>近期表现与回撤</h3><span>同一账户 · 2025年初50万元启动</span></div><div class="if-metrics if-inline-metrics"><article><span>累计收益</span><strong>${cell(r.total_return)}</strong></article><article><span>最大回撤</span><strong>${cell(-r.max_drawdown,true)}</strong></article><article><span>当前回撤</span><strong>${cell(-r.current_drawdown,true)}</strong></article><article><span>期末权益</span><strong>${money(r.equity)}</strong></article></div>${equityChart(r.curve)}
    ${table(["截至研究末日","区间收益","区间最大回撤","净盈亏","空仓比例"],r.windows.map(w=>`<tr><td>近${w.sessions}个交易日</td><td>${cell(w.total_return)}</td><td>${cell(-w.max_drawdown,true)}</td><td>${money(w.net_pnl)}</td><td>${pc(w.flat_ratio)}</td></tr>`).join(""),"近期表现")}
    <p class="if-note">表现截至 ${payload.data_end}，与上方最新信号日期分开。区间收益以该区间实际起始权益为分母，区间回撤包含期初权益；当前回撤相对持续账户历史高点。所有收益已扣研究费用。</p>
    <details class="if-continuous"><summary>查看最近8笔交易</summary>${table(["入场","退出 / 估值","合约（末段）","方向","净盈亏","持有交易日"],r.trades.map(t=>`<tr><td>${date(t.entry)}</td><td>${date(t.exit)}${t.end_mark?" · 截止估值":""}</td><td>${esc(t.code)}</td><td>${t.side==="long"?"多":"空"}</td><td>${money(t.pnl)}</td><td>${t.days}</td></tr>`).join(""),"近期交易")}<p class="if-note">合约栏为交易末段合约，跨换月仍是一笔策略交易；截止估值包含假设平仓费用，不表示真实成交。</p></details></section>`;
}

function render() {
  const root=document.querySelector("#indexFuturesContent");if(!root)return;
  if(!payload){root.innerHTML=`<section class="panel"><p role="status">${error?esc(error):"正在载入股指战法…"}</p>${error?'<button type="button" data-if-reload>重新载入</button>':""}</section>`;return;}
  displayedExpired=Boolean(payload.signal && Date.now()>=Date.parse(payload.signal.valid_until));
  const all=records(), equal=all.filter(r=>r.window===view.window&&r.start_kind==="monthly");
  const annual=all.filter(r=>r.annual&&r.window==="single_year").sort((a,b)=>a.start.localeCompare(b.start));
  const cumulative=new Map(all.filter(r=>r.annual&&r.window==="to_latest").map(r=>[r.start,r]));
  const starts=all.filter(r=>r.window==="to_latest"&&(view.phases||r.start_kind==="monthly")&&(view.year==="all"||r.start.startsWith(view.year))).sort((a,b)=>a.start.localeCompare(b.start));
  const totals=all.filter(r=>r.window==="to_latest"), worst=totals.reduce((a,b)=>a.total_return<b.total_return?a:b), worstDD=totals.reduce((a,b)=>a.max_drawdown<b.max_drawdown?a:b);
  const continuous=payload.continuous_2025.filter(r=>r.startup_mode===view.mode&&r.fee===profiles[view.profile].fee&&r.margin===profiles[view.profile].margin);
  const n=view.window==="sessions_244"?244:122;
  root.innerHTML=`<section class="if-strategy-cards" aria-label="保留策略">${Object.entries(payload.modes).map(([mode,label])=>`<button type="button" data-if-mode="${mode}" aria-pressed="${mode===view.mode}" class="panel if-strategy-card ${mode===view.mode?"if-active-card":""}"><span class="if-kicker">${mode.startsWith("fixed")?"偏收益":"偏稳定"}</span><strong>${esc(label)}</strong><span>${esc(payload.descriptions[mode])}</span></button>`).join("")}</section>
    <section class="if-controls panel" aria-label="股指回测筛选"><label>策略<select id="ifMode">${options(Object.entries(payload.modes),view.mode)}</select></label><label>费用与保证金<select id="ifProfile">${options(Object.entries(profiles).map(([k,v])=>[k,v.label]),view.profile)}</select></label><label>等长窗口<select id="ifWindow">${options([["sessions_244","244个交易日"],["sessions_122","122个交易日"]],view.window)}</select></label><p>50万元 · 最多一手 · 多空均可<br>模型季度重训 · 全局5日入场日历</p></section>
    ${error?`<p class="if-warning" role="alert">${esc(error)}；保留上次成功结果，请核对日期。</p>`:""}${openingPanel()}
    ${recentPanel()}
    <section aria-labelledby="ifEqualTitle"><div class="panel-head if-section-head"><h3 id="ifEqualTitle">等长窗口 · ${n}个交易日</h3><span>${equal.length}个月度起点 · 不同启动时间</span></div><div class="if-metrics">${[["收益中位",median(equal.map(r=>r.total_return)),false],["最差收益",Math.min(...equal.map(r=>r.total_return)),false],["最大回撤",-Math.min(...equal.map(r=>r.max_drawdown)),true],["空仓中位",median(equal.map(r=>r.flat_ratio)),null]].map(([label,value,loss])=>`<article class="panel"><span>${label}</span><strong>${loss===null?pc(value):cell(value,loss)}</strong></article>`).join("")}</div><p class="if-note">最差收益和最大回撤可能来自不同账户。窗口重叠，不是独立样本；2026尚无完整244日新起点。</p></section>
    <section class="panel"><div class="panel-head"><h3>逐年表现</h3><span>历史截至 ${payload.data_end}</span></div>${table(["启动年","独立单年收益","单年回撤","该年启动累计至研究末日","累计回撤"],annual.map(r=>{const c=cumulative.get(r.start);return `<tr><td>${r.start.slice(0,4)}${r.start.startsWith("2026")?"（部分年度）":""}</td><td>${cell(r.total_return)}</td><td>${cell(-r.max_drawdown,true)}</td><td>${cell(c?.total_return)}</td><td>${cell(c?-c.max_drawdown:null,true)}</td></tr>`;}).join(""),"逐年收益回撤")}<p class="if-note">独立单年每年重置50万元；累计从对应年初持续至研究末日，中间不重置。账户按开盘账本记账。</p><details class="if-continuous"><summary>查看2025启动持续账户的逐年表现</summary>${table(["记账年","年度收益","年度回撤"],continuous.map(r=>`<tr><td>${r.year}</td><td>${cell(r.total_return)}</td><td>${cell(-r.max_drawdown,true)}</td></tr>`).join(""),"持续账户年度表现")}<p class="if-note">进入下一年不重置资金，收益分母为各年实际期初权益。</p></details></section>
    <section class="panel"><div class="panel-head"><h3>两种保留策略 · 同一${n}日窗口</h3></div>${table(["策略","收益中位","最差收益","最大回撤","空仓中位"],Object.entries(payload.modes).map(([mode,label])=>{const rows=records(mode).filter(r=>r.window===view.window&&r.start_kind==="monthly");return `<tr class="${mode===view.mode?"if-selected-row":""}"><td>${esc(label)}</td><td>${cell(median(rows.map(r=>r.total_return)))}</td><td>${cell(Math.min(...rows.map(r=>r.total_return)))}</td><td>${cell(-Math.min(...rows.map(r=>r.max_drawdown)),true)}</td><td>${pc(median(rows.map(r=>r.flat_ratio)))}</td></tr>`;}).join(""),"策略对照")}<p class="if-note">按完整等长窗口的收益与回撤取舍保留，没有仅按2026挑冠军。“稳定候选”仅指当前历史压力情景相对较好，仍有亏损起点，基础情景也并非所有指标领先。</p></section>
    <section class="panel"><div class="panel-head if-start-head"><h3>最差起点与启动时间</h3><label>启动年份<select id="ifYear">${options([["all","全部年份"],...['2022','2023','2024','2025','2026'].map(x=>[x,x])],view.year)}</select></label></div><p class="if-note">全部月度与相邻日起点中，最差累计收益 ${pc(worst.total_return)}（${date(worst.start)}），最大回撤 ${pc(-worstDD.max_drawdown)}（${date(worstDD.start)}）。各起点运行至 ${payload.data_end}，区间长度不同。</p><label class="if-check"><input id="ifPhases" type="checkbox" ${view.phases?"checked":""}>包含年初相邻交易日起点</label>${table(["起点","累计收益","最大回撤","空仓比例","交易笔数"],starts.map(r=>`<tr><td>${date(r.start)}${r.start_kind==="phase"?" · 相邻日":""}</td><td>${cell(r.total_return)}</td><td>${cell(-r.max_drawdown,true)}</td><td>${pc(r.flat_ratio)}</td><td>${r.trades}</td></tr>`).join(""),"月度及相邻日起点结果")}</section>
    <details class="panel if-details"><summary>完整战法、研究边界与原方案资料</summary><div class="if-detail-body"><p>两种策略均使用期货、ETF、相对指数、牛熊及交互因子，加入QVIX历史分位、变化和隐含/实际波动差。共同QVIX缺失按原规则回退，不沿用旧值。模型每季度用过去36个月成熟样本重训，多空分开，近期半衰期12个月。</p><p>牛：指数收盘价 &gt;MA200＋ATR20，且MA200较20日前上升；熊反之；其他为过渡。允许有收益证据的逆势交易。入场时锁定顺势20日、过渡10日、逆势5日上限，保留原5日续持退出及资金保护。</p><p>固定策略保留 base 政策；滚动策略在基础、辅助概率、较高方向置信度、对向退出四种政策间季度重选，用实际权益做历史验证，并考虑切换成本。当前开仓区展示新账户启动试算，不跟踪用户真实持仓、续持或补资。</p><p>3/6bps为研究交易摩擦假设，不是券商报价。历史多轮研究已使用，不能称为新的OOT；一手容量导致起点、资金与费用变化影响交易路径，日线回测不能证明盘中不会强平。</p><p>保留两个研究候选供查看，不改变已存在的研究文件或真实账户。自建隐波尚未全面胜出，留在以下资料中。</p></div><div class="if-links"><a href="/data/index-futures/comparison.csv" download>两种策略数据 CSV ↗</a><a href="/data/index-futures/strategy-research.md" download>原状态期限方案 ↗</a><a href="/data/index-futures/panic-research.md" download>QVIX完整研究 ↗</a><a href="/data/index-futures/option-iv-research.md" download>自建隐波研究 ↗</a></div></details>`;
  if(document.querySelector("#futuresPage")?.classList.contains("active"))document.querySelector("#generatedAt").textContent=`股指信号 ${date(payload.signal?.signal_date)||"未生成"} · 回测 ${payload.data_end}`;
}

async function refreshSignals() {
  if(refreshing)return;refreshing=true;error="";render();
  try {
    let response=await fetch('/api/index-futures/refresh',{method:'POST'});
    if(!response.ok)throw new Error(`更新服务不可用（${response.status}）`);
    let status=await response.json();
    for(let attempt=0;status.status==="running"&&attempt<180;attempt++){
      await new Promise(resolve=>setTimeout(resolve,2000));
      response=await fetch('/api/index-futures/refresh/status',{cache:'no-store'});
      if(!response.ok)throw new Error(`无法读取更新状态（${response.status}）`);
      status=await response.json();
    }
    if(status.status!=="complete")throw new Error(status.message||"更新尚未完成，请稍后重新载入");
    await showIndexFuturesPage(true);
  }catch(err){error=err.message||"信号更新失败";}
  finally{refreshing=false;render();}
}

export function showIndexFuturesPage(reload=false) {
  const root=document.querySelector("#indexFuturesContent");if(!root)return Promise.resolve();
  if(!bound){
    bound=true;
    root.addEventListener("change",event=>{
      const {id,value,checked}=event.target;
      const valid={ifMode:Object.keys(payload?.modes||{}),ifProfile:Object.keys(profiles),ifWindow:['sessions_122','sessions_244'],ifYear:['all','2022','2023','2024','2025','2026']};
      if(id==='ifPhases')view.phases=checked;
      else if(valid[id]?.includes(value))view[({ifMode:'mode',ifProfile:'profile',ifWindow:'window',ifYear:'year'})[id]]=value;
      else return;
      render();document.getElementById(id)?.focus({preventScroll:true});
    });
    root.addEventListener("click",event=>{
      const card=event.target.closest('[data-if-mode]');
      if(card&&payload?.modes[card.dataset.ifMode]){view.mode=card.dataset.ifMode;render();root.querySelector(`[data-if-mode="${view.mode}"]`)?.focus({preventScroll:true});}
      if(event.target.closest('[data-if-reload]'))showIndexFuturesPage(true);
      if(event.target.closest('[data-if-refresh]'))refreshSignals();
    });
    const expireIfNeeded=()=>{if(payload?.signal && !displayedExpired && Date.now()>=Date.parse(payload.signal.valid_until))render();};
    document.addEventListener('visibilitychange',()=>{if(!document.hidden)expireIfNeeded();});
    setInterval(expireIfNeeded,60000);
  }
  if(pending)return pending;
  error='';
  const loadResearch=async()=>{
    if(payload&&!reload)return payload;
    const response=await fetch('/data/index-futures/research.json',{cache:'no-store'});
    if(!response.ok)throw new Error(`研究数据暂不可用（${response.status}）`);
    const next=await response.json();
    if(next.version!==2||!Array.isArray(next.records)||!next.records.length||!next.recent||!next.modes)throw new Error('研究数据格式不完整');
    return next;
  };
  pending=loadResearch().then(async next=>{
    payload=next;if(!payload.modes[view.mode])view.mode=payload.default_mode;
    const response=await fetch('/data/index-futures/signal.json',{cache:'no-store'});
    if(response.status===404)return; // Legacy research exports may embed the last signal.
    if(!response.ok)throw new Error(`开仓预案暂不可用（${response.status}）`);
    const signal=await response.json();
    if(signal.version!==1||!signal.signal_date||!signal.profiles)throw new Error('开仓预案格式不完整');
    payload={...payload,signal};
  }).catch(err=>{error=err.message||'研究结果载入失败';}).finally(()=>{pending=null;render();});
  render();return pending;
}
