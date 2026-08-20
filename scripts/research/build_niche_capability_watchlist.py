#!/usr/bin/env python3
"""Overlay business-level scarcity hypotheses on the GQS stage-one pool.

The output is a discovery watchlist, not a completed scarcity score.  Every
company must already be present in the reproducible stage-one candidate pool;
only rows with a cited annual report or official company source are marked as
initially verified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RECORDS = """
工业装备与自动化|豪迈科技|轮胎模具、燃气轮机大型零部件的工艺与交付能力
工业装备与自动化|柏楚电子|激光切割运动控制系统、切割头与工业软件生态
工业装备与自动化|美亚光电|色选设备、口腔影像设备的光机电算法一体化
工业装备与自动化|博实股份|固体物料后处理智能装备与工厂运维服务
工业装备与自动化|杰瑞股份|油气田高端装备、天然气处理与全球化服务网络
工业装备与自动化|杭氧股份|大型空分设备与工业气体运营的一体化能力
工业装备与自动化|浙江鼎力|电动化、模块化高空作业平台及全球租赁客户认证
工业装备与自动化|恒立液压|高压油缸、液压泵阀的规模制造与主机厂认证
工业装备与自动化|杭叉集团|工业车辆电动化、海外渠道与供应链效率
工业装备与自动化|杰克科技|工业缝制设备、服装生产数字化解决方案
工业装备与自动化|大豪科技|缝制及针纺设备电脑控制系统
工业装备与自动化|宏发股份|电磁继电器、高压直流继电器的材料与制造平台
工业装备与自动化|国电南瑞|电网调度、继电保护与自动化系统的长期装机基础
工业装备与自动化|思进智能|多工位高速冷成形装备与工艺数据库
工业装备与自动化|长盛轴承|自润滑轴承材料配方、制造与客户认证
工业装备与自动化|凌霄泵业|不锈钢泵、卫浴泵等小型泵的制造与渠道
工业装备与自动化|巨星科技|手工具产品矩阵、跨境渠道和品牌运营
工业装备与自动化|亚星锚链|船用锚链、海洋系泊链的大规格制造认证
工业装备与自动化|铁科轨道|高铁扣件与轨道工程材料的认证和项目经验
工业装备与自动化|华明装备|电力变压器有载分接开关及存量运维
工业装备与自动化|纽威股份|工业阀门的多品种制造、认证和全球客户覆盖
工业装备与自动化|信测标准|汽车、电子等领域检测认证与实验室网络
工业装备与自动化|中国汽研|汽车测试评价、公告认证与试验场能力
电子、半导体与软件|法拉电子|薄膜电容器的材料、工艺、客户认证与规模制造
电子、半导体与软件|天孚通信|高速光器件精密制造、封装平台与大客户协同
电子、半导体与软件|亿联网络|企业通信终端、协作设备与全球渠道
电子、半导体与软件|澜起科技|服务器内存接口及互连芯片的标准、研发和客户壁垒
电子、半导体与软件|瑞芯微|端侧SoC平台、算法生态与长期客户适配
电子、半导体与软件|芯动联科|高性能MEMS惯性传感器与高可靠场景认证
电子、半导体与软件|金山办公|办公软件文档格式、用户习惯与生态迁移成本
电子、半导体与软件|同花顺|金融数据、交易终端与高频用户网络
电子、半导体与软件|海康威视|视频感知、机器视觉、软硬件渠道和工程化能力
电子、半导体与软件|生益科技|覆铜板材料配方、客户认证与规模供应
电子、半导体与软件|沪电股份|高端通信及服务器PCB的良率和大客户认证
电子、半导体与软件|中航光电|高可靠连接器、军工与工业客户认证
电子、半导体与软件|力鼎光电|光学镜头的设计、精密制造和客户协同
电子、半导体与软件|联瑞新材|电子级硅微粉的粒径控制、提纯与客户验证
电子、半导体与软件|菲利华|高性能石英玻璃材料与半导体、航空认证
电子、半导体与软件|国瓷材料|电子陶瓷粉体到精密陶瓷制品的平台化能力
电子、半导体与软件|鼎龙股份|CMP抛光垫、打印耗材等材料的配方与量产
电子、半导体与软件|安集科技|集成电路CMP抛光液及功能湿电子化学品
电子、半导体与软件|华峰测控|模拟及混合信号半导体测试机平台
电子、半导体与软件|中微公司|刻蚀、薄膜沉积设备的工艺验证与客户协同
医疗与生命科学|艾德生物|肿瘤伴随诊断试剂、药企合作和注册证组合
医疗与生命科学|惠泰医疗|电生理与血管介入耗材、术式适配和临床渠道
医疗与生命科学|迈瑞医疗|医疗设备平台、全球渠道和售后服务网络
医疗与生命科学|新产业|化学发光仪器试剂一体化与海外装机
医疗与生命科学|我武生物|过敏原诊断与脱敏治疗产品、医生教育
医疗与生命科学|瑞普生物|动物疫苗、药品研发制造与养殖客户服务
医疗与生命科学|马应龙|肛肠用药品牌、医院终端与产品延展
医疗与生命科学|华特达因|儿童用药品牌和儿科医生渠道
先进材料|蓝晓科技|吸附分离树脂材料、应用工艺包与项目经验
先进材料|久立特材|高端不锈钢及特种合金管材、能源客户认证
先进材料|中信特钢|特钢品种体系、规模制造与高端客户认证
先进材料|中国巨石|玻纤成本、规模、配方与全球供应网络
先进材料|光威复材|碳纤维及预浸料一体化与高可靠客户认证
先进材料|中简科技|高性能碳纤维及航空航天验证
先进材料|银河磁体|粘结钕铁硼材料与精密小型磁体制造
先进材料|新莱福|吸附功能材料、电子陶瓷与复合材料小品类平台
先进材料|沃顿科技|复合反渗透膜材料、膜元件和工程应用
消费品与专业渠道|公牛集团|民用电工品牌、线下终端密度和产品延展
消费品与专业渠道|三花智控|制冷控制元件及汽车热管理的规模与客户认证
消费品与专业渠道|伟星股份|钮扣、拉链的柔性快反、品牌客户和海外产能
消费品与专业渠道|安克创新|消费电子产品定义、跨境品牌和渠道运营
消费品与专业渠道|明月镜片|镜片材料、品牌零售教育与渠道覆盖
消费品与专业渠道|共创草坪|人造草坪研发制造、认证与海外客户覆盖
消费品与专业渠道|东鹏饮料|能量饮料品牌、冰柜终端和区域复制
消费品与专业渠道|珀莱雅|多品牌产品开发、内容营销和渠道运营
消费品与专业渠道|海天味业|调味品品牌、经销网络和规模成本
消费品与专业渠道|海大集团|饲料配方、养殖服务与渠道密度
消费品与专业渠道|安井食品|速冻食品渠道、供应链与产品迭代
消费品与专业渠道|中宠股份|宠物食品制造、海外客户与自主品牌
平台与专业服务|焦点科技|跨境B2B平台、供应商数据与会员网络
平台与专业服务|分众传媒|楼宇媒体点位网络和广告客户覆盖
平台与专业服务|嘉友国际|跨境资源物流节点、口岸运营与项目组织
"""


EXCEPTION_RECORDS = """
工业装备与自动化|中控技术|流程工业控制系统、工业软件与长期装机数据
工业装备与自动化|安徽合力|工业车辆全系列制造、渠道与规模成本
工业装备与自动化|川仪股份|流程工业仪器仪表的产品谱系与行业客户基础
工业装备与自动化|苏试试验|环境可靠性试验设备、检测服务和标准参与
电子、半导体与软件|北方华创|半导体装备多产品平台与晶圆厂工艺验证
电子、半导体与软件|盛美上海|半导体清洗、电镀等设备的差异化工艺平台
医疗与生命科学|开立医疗|超声、内镜设备的影像算法与临床渠道
医疗与生命科学|澳华内镜|软性内镜主机、镜体和临床术式适配
医疗与生命科学|联影医疗|高端医学影像设备、软件和全球服务体系
消费品与专业渠道|森麒麟|高端轮胎智能制造、海外渠道与认证
消费品与专业渠道|石头科技|清洁机器人算法、产品定义与全球品牌
消费品与专业渠道|赛轮轮胎|轮胎技术、全球产能布局与渠道
"""


SOURCES = {
    "豪迈科技": ("年报初核", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12041754&stockid=002595"),
    "柏楚电子": ("年报初核", "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12074797&stockid=688188"),
    "国瓷材料": ("年报初核", "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12120185"),
    "惠泰医疗": ("年报初核", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12015886&stockid=688617"),
    "伟星股份": ("年报初核", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12102650&stockid=002003"),
    "法拉电子": ("官网初核", "https://www.faratronic.com/"),
    "宏发股份": ("年报初核", "https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12054908&stockid=600885"),
    "浙江鼎力": ("年报初核", "https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/listedinfo/announcement/c/new/2026-04-17/603338_20260417_JXVS.pdf"),
    "艾德生物": ("官网初核", "https://www.amoydx.com/about.html"),
    "天孚通信": ("年报初核", "https://money.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?id=12066361&stockid=300394"),
    "苏试试验": ("年报初核", "https://pdf.dfcfw.com/pdf/H2_AN202603261820773057_1.pdf"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--screened-universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def hypotheses() -> pd.DataFrame:
    rows = []
    for line in RECORDS.strip().splitlines():
        category, name, capability = line.split("|", maxsplit=2)
        rows.append(
            {"capability_category": category, "name": name, "scarcity_hypothesis": capability}
        )
    return pd.DataFrame(rows)


def exception_hypotheses() -> pd.DataFrame:
    rows = []
    for line in EXCEPTION_RECORDS.strip().splitlines():
        category, name, capability = line.split("|", maxsplit=2)
        rows.append(
            {"capability_category": category, "name": name, "scarcity_hypothesis": capability}
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates)
    screened_universe = pd.read_csv(args.screened_universe)
    watchlist = hypotheses().merge(candidates, on="name", how="left", validate="one_to_one")
    missing = watchlist[watchlist["ts_code"].isna()]["name"].tolist()
    if missing:
        raise SystemExit(f"Not in stage-one candidate pool: {', '.join(missing)}")
    watchlist[["evidence_status", "scarcity_source_url"]] = watchlist["name"].apply(
        lambda name: pd.Series(SOURCES.get(name, ("待逐家核验", "")))
    )
    watchlist["scarcity_review_priority"] = watchlist["evidence_status"].map(
        {"年报初核": "第一批", "官网初核": "第一批"}
    ).fillna("第二批")
    watchlist = watchlist.sort_values(
        ["scarcity_review_priority", "capability_category", "stage1_proxy_score"],
        ascending=[True, True, False],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    keep = [
        "scarcity_review_priority", "capability_category", "ts_code", "name",
        "industry", "scarcity_hypothesis", "evidence_status", "scarcity_source_url",
        "stage1_proxy_score", "within_industry_percentile", "roe_mean_5y",
        "cashflow_quality_3y", "fina_debt_to_assets", "forecast_eps_growth_2027",
        "forecast_2026_report_count",
    ]
    watchlist[keep].to_csv(args.output_dir / "niche_capability_watchlist.csv", index=False)

    exceptions = exception_hypotheses().merge(
        screened_universe, on="name", how="left", validate="one_to_one"
    )
    missing_exceptions = exceptions[exceptions["ts_code"].isna()]["name"].tolist()
    if missing_exceptions:
        raise SystemExit(
            f"Not in screened universe: {', '.join(missing_exceptions)}"
        )
    if exceptions["candidate"].fillna(False).any():
        invalid = exceptions.loc[exceptions["candidate"].fillna(False), "name"].tolist()
        raise SystemExit(f"Exception companies unexpectedly passed: {', '.join(invalid)}")
    exceptions[["evidence_status", "scarcity_source_url"]] = exceptions["name"].apply(
        lambda name: pd.Series(SOURCES.get(name, ("待逐家核验", "")))
    )
    exception_keep = [
        column for column in keep if column != "scarcity_review_priority"
    ]
    exceptions[["candidate", *exception_keep, "candidate_fail_reasons"]].to_csv(
        args.output_dir / "niche_capability_exceptions.csv", index=False
    )

    lines = [
        "# A股业务级稀缺能力观察池",
        "",
        "> 这是在阶段一财务候选池之上的发现层：入选表示“财务质量通过初筛且存在值得核验的稀缺能力线索”，不表示已完成稀缺性评分或完整GQS。",
        "",
        f"候选数：{len(watchlist)}｜已用年报/官网初核：{watchlist['scarcity_source_url'].ne('').sum()}｜待逐家核验：{watchlist['scarcity_source_url'].eq('').sum()}",
        "",
        "证据状态说明：`年报/官网初核`只确认该能力线索有一手材料支持；仍需核验市场份额口径、持续时间、竞争反证和利润池归属。",
        "",
    ]
    for category, frame in watchlist.groupby("capability_category", sort=False):
        lines.extend(
            [
                f"## {category}",
                "",
                "| 代码 | 公司 | 稀缺能力线索 | 初筛分 | 行业内百分位 | 5年ROE | 证据 |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for row in frame.itertuples(index=False):
            evidence = row.evidence_status
            if row.scarcity_source_url:
                evidence = f"[{evidence}]({row.scarcity_source_url})"
            lines.append(
                f"| {row.ts_code} | {row.name} | {row.scarcity_hypothesis} | "
                f"{row.stage1_proxy_score:.1f} | {row.within_industry_percentile * 100:.1f}% | "
                f"{row.roe_mean_5y:.1f}% | {evidence} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 稀缺性例外观察：未进入财务候选池",
            "",
            "> 下列公司具有业务稀缺性线索，但本期行业内财务质量代理没有进入前20%，因此不能与上面的75家公司等同。保留它们是为了防止财务筛选漏掉正处于投入期、周期错位或财务转化尚未完成的公司。",
            "",
            "| 代码 | 公司 | 稀缺能力线索 | 初筛分 | 行业内百分位 | 未入池原因 | 证据 |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    reason_labels = {
        "coverage": "覆盖率不足",
        "industry_percentile": "行业百分位不足",
        "profit_history": "盈利历史不足",
        "nonpositive_roe": "ROE非正",
    }
    for row in exceptions.sort_values("within_industry_percentile", ascending=False).itertuples(index=False):
        reasons = "、".join(
            reason_labels.get(reason, reason)
            for reason in row.candidate_fail_reasons.strip(";").split(";")
            if reason
        )
        evidence = row.evidence_status
        if row.scarcity_source_url:
            evidence = f"[{evidence}]({row.scarcity_source_url})"
        lines.append(
            f"| {row.ts_code} | {row.name} | {row.scarcity_hypothesis} | "
            f"{row.stage1_proxy_score:.1f} | {row.within_industry_percentile * 100:.1f}% | "
            f"{reasons} | {evidence} |"
        )
    lines.append("")
    lines.extend(
        [
            "## 后续核验顺序",
            "",
            "1. 一手材料确认产品边界、市场份额口径及领先持续时间。",
            "2. 用客户认证周期、替换成本、良率/工艺诀窍、渠道密度或网络效应解释稀缺性的来源。",
            "3. 检查稀缺能力是否真正落到毛利率、ROIC、现金流和每股价值，而非只有技术称号。",
            "4. 搜索降价、客户自研、技术替代、单一客户依赖和海外贸易限制等反证。",
        ]
    )
    (args.output_dir / "niche_capability_watchlist.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
