import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

p = Path(__file__).parent
b = json.loads((p / 'template.json').read_text())
inputs = json.loads((p / 'valuation_input.json').read_text())
results = json.loads((p / 'valuation_result.json').read_text())
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
b['analysis_cutoff'] = '2026-11-03T09:40:23+08:00'
b['conclusion'] = {
    'stance': 'neutral_cautious', 'confidence': 'medium_low',
    'summary': 'Revenue growth has not translated into profit growth. Base conditional value CNY15.5 versus CNY14.17 close implies 9.4% upside with material margin and dilution risk.'
}
pillars = [
    ('P1', 'Customer certification and process capabilities create barriers but do not guarantee pricing power.', 'Suspension revenue, utilization and gross margin', 'Two quarters with gross margin below 24% and falling suspension revenue'),
    ('P2', 'German asset acquisition expands revenue; sustainable profit conversion remains unproven.', 'German operating profit, management expense ratio and deliveries', 'Two quarters with parent net margin below 8% despite revenue growth'),
    ('P3', 'OCF improvement is offset by capex and working-capital needs.', 'OCF less capex, commercial bills and inventory', 'Persistently negative annual simplified FCF alongside further financed expansion')
]
b['thesis'] = {
    'pillars': [{'id': i, 'statement': s, 'status': 'new', 'monitoring': m, 'falsifier': f} for i, s, m, f in pillars],
    'strongest_counterargument': 'Start-up and integration costs may be temporary; utilization gains could release operating leverage.',
    'falsifiers': [x[3] for x in pillars]
}
stories = {
    'bear': 'Pricing pressure and low utilization persist; integration expenses remain elevated; convertible remains debt.',
    'base': 'Suspension stabilizes and German integration expenses normalize; no conversion or downward reset assumed.',
    'bull': 'Deliveries, utilization, expense ratio and cash collection improve together; full conversion at initial CNY18.48 assumed.'
}
cashflows = {'bear': [2.1, 3.2, -1.1], 'base': [3.2, 3.0, 0.2], 'bull': [4.3, 2.8, 1.5]}
invalidators = {
    'bear': 'Two consecutive quarters of gross margin above 28% and falling expense ratios.',
    'base': 'Two consecutive quarters of parent net margin below 8%, or a conversion-price reset.',
    'bull': '2027 H1 gross margin remains below 26% with no German operating improvement.'
}
for k in ('bear', 'base', 'bull'):
    b['scenarios'][k] = {
        **inputs['scenarios'][k], **results['scenarios'][k],
        'narrative': stories[k], 'falsifier': invalidators[k],
        'target_date': '2027-11-03', 'probability': None,
        'cashflows_CNY_100million': dict(zip(('OCF', 'capex', 'simplified_FCF'), cashflows[k])),
        'dividend_assumption': 'Not forecast; total return not calculated.'
    }
b['monitoring'] = [{'pillar_id': i, 'item': m, 'frequency': 'Quarterly and material announcements', 'invalidation': f} for i, s, m, f in pillars]
b['monitoring'] += [
    {'item': 'Convertible reset decisions, actual conversion price and share count', 'next_check': '2026-11-06'},
    {'item': 'Delayed capital-project acceptance and utilization', 'next_check': '2026-12-31'},
    {'item': 'Verified robot customers, orders, revenue and collections', 'frequency': 'Contract and IR announcements'},
    {'item': 'Related-party transactions, pledges, sales by shareholders, regulatory and litigation updates', 'frequency': 'Event-driven'}
]
sources = [
    ('annual', 'FY2025 audited annual report', '2026-04-03', '2026-04-04', 'https://disc.static.szse.cn/download/disc/disk03/finalpage/2026-04-03/ffd4be59-1a69-4db5-bdaf-ecbcb7b2c859.PDF'),
    ('h1', '2026 H1 report', '2026-08-27', '2026-08-28', 'https://static.cninfo.com.cn/finalpage/2026-08-27/1225512040.PDF'),
    ('q3', '2026 Q3 and nine-month report', '2026-10-24', '2026-10-25', 'https://static.cninfo.com.cn/finalpage/2026-10-24/1225619897.PDF'),
    ('prospectus', 'Convertible prospectus summary', '2026-10-13', '2026-10-14', 'https://static.cninfo.com.cn/finalpage/2026-10-13/1225602400.PDF'),
    ('downward', 'Potential conversion-price reset notice', '2026-10-30', '2026-10-31', 'https://static.cninfo.com.cn/finalpage/2026-10-30/1225639056.PDF'),
    ('listing', 'Convertible listing announcement', '2026-10-31', '2026-11-01', 'https://static.cninfo.com.cn/finalpage/2026-10-31/1225649914.PDF')
]
b['evidence_ledger'] = [{
    'id': i, 'title': title, 'published_at': d,
    'available_at': a + 'T00:00:00+08:00',
    'availability_policy': 'Date-only source conservatively admitted at next calendar-day midnight.',
    'accessed_at': now, 'accessed_at_definition': 'Verification completed at; not first public availability.',
    'url': url, 'classification': 'fact', 'verified': True
} for i, title, d, a, url in sources]
b['evidence_ledger'].append({
    'id': 'market', 'period': '2026-11-02', 'price': 14.17,
    'source': 'Project canonical Tushare daily; cross-checked with Choice MCP.',
    'available_at': '2026-11-02T16:00:00+08:00', 'classification': 'fact', 'verified': True
})
b['data_snapshot'] = {
    'currency': 'CNY', 'current_price': 14.17,
    'price_as_of': '2026-11-02T15:00:00+08:00',
    'price_available_at': '2026-11-02T16:00:00+08:00',
    'price_basis': 'unadjusted', 'total_shares': 270854208,
    'treasury_shares': 1240100, 'market_cap_CNY': 14.17 * 270854208,
    'technical_snapshot': json.loads((p / 'technicals.json').read_text()),
    'benchmark': '000300.SH, synchronized 250 complete daily bars; no sector benchmark.',
    'mcp_execution': {
        'provider': 'Eastmoney Miaoxiang/Choice', 'server': 'mx-ds-mcp',
        'runtime_loaded': True, 'attempted': True,
        'actual_call_count': 4, 'successful_call_count': 3, 'empty_result_count': 1,
        'tools': [
            {'name': 'mx_ashare_finance_data', 'request_summary': 'Identity, latest financials and 2026-11-02 market valuation.'},
            {'name': 'mx_finance_search_notice', 'request_summary': 'Annual, interim, Q3 reports and material events.'},
            {'name': 'mx_finance_search_notice', 'request_summary': 'Post-quarter convertible and risk notices; empty result.'},
            {'name': 'mx_ashare_finance_data', 'request_summary': 'Same-date 001380, 300611, 000887 valuation and consensus metadata.'}
        ],
        'adopted_fields': ['Close 14.17', 'PE TTM 16.4', 'PB 1.864', 'Share count', 'Financial discovery', 'Rough peer valuation references'],
        'cross_checked_fields': ['Identity', 'Close', 'Financial statements', 'Share count', 'Convertible issuance and terms'],
        'excluded_fields': ['Intraday quotes after research cutoff', 'FY1 consensus without clear fiscal-year mapping or sample count'],
        'status': 'partial_success', 'failure_class': 'one_empty_notice_response_and_incomplete_consensus_metadata',
        'fallback': 'Canonical project prices and official CNINFO/SZSE filings; official announcement query resolved post-quarter events.'
    },
    'data_gaps': ['Sector index', 'Robot-specific orders and operating data', 'Qualified 2027 consensus', 'Future financing amortization, free cash and conversion path', 'No exhaustive regulatory/judicial database review']
}
b['report_markdown'] = (p / 'report.md').read_text()
(p / 'research_bundle.json').write_text(json.dumps(b, ensure_ascii=False, indent=2))
print('Research bundle prepared.')
