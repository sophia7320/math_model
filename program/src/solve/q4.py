"""C 题 问题四：实时波动电价下的重算（Q2 层 / Q3 层）。

口径：
- 电价 G（题面主口径）：未来电价已由附件 4 给出，决策直接使用对应日价格；
  H（仅历史价格）是额外的信息受限扩展，用滚动费用标定 v（W=7）+ β=0.1 动态修正。
- 负荷由目标日前历史数据预测（κ·EWMA h=5）；光伏用附件 3 的 0:00 官方预报，
  2 日视野中的次日用 asof 历史自适应预报（无前视）。
- 执行：逐槽因果（core.causal.exec_segment_causal，free 策略，不读未来实际值）。
- 储能结构：日循环 vs 2 日滚动（跨日连续，末端自由），按费用择优。

# ===========================================================================
# 费用口径（结算一律用附件 4 真实电价 p4）
#     Q2 层： C = Σ p4·x + 5·Σ p4·e
#     Q3 层： C = Σ [ p4·x_adj + 0.5·p4·|x_plan − x_adj| ] + 5·Σ p4·e
# 决策价（H）： P̂ = v1·P4(D−1) + v2·P4(D−7) + v3·P̄_typ（见 models/price.py）
# 结构：daily = 日循环 plan_day；2day = plan_two_day（48h 滚动、末端自由）
# ===========================================================================

实现分层：年度流程在 solve.flows.q4_year，结果表在 solve.io.excel，
电价模型在 solve.models.price；本文件只做编排、校验与出图。

运行（program/ 下）：
    uv run python -m solve.q4 --probe        # 结构对比探针
    uv run python -m solve.q4 --result4-2    # 生成 result4-2.xlsx（Q2 层）
    uv run python -m solve.q4 --result4-3    # 生成 result4-3.xlsx（Q3 层）
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import program as pm
from solve import consistency as cs
from solve.common import E_MAX, E_MIN, ETA, N_DAY, REPORT_START, ROOT
from solve.core.lp import plan_horizon  # noqa: F401  （公开 API：供探针与测试引用）
from solve.data.attachments import load_q4_data
from solve.flows.q4_year import (
    run_year,
    run_year_q3,
    simulate_day_q3,  # noqa: F401  （公开 API：守卫测试校验默认对冲开关）
    total_of,
    total_of_q3,
)
from solve.io.checks import verify_records
from solve.io.excel import (
    verify_result4_2,
    verify_result4_3,
    write_result4_2,
    write_result4_3,
)
from solve.io.report import record
from solve.models.price import price_forecast_d, price_forecast_next, rolling_v_seq  # noqa: F401


# ---------------------------------------------------------------------------
# 图表 / 报告
# ---------------------------------------------------------------------------
def make_figures(data, recs_h, recs_g, s_h, s_g, daily) -> None:
    """Q2 层论文图：H/G 费用对比 + 逐日紧急购电。"""
    pm.bar_group(
        ["计划购电费", "紧急购电费"],
        {"H·历史预测": [s_h["plan"] / 1e4, s_h["emerg"] / 1e4],
         "G·题面已知电价": [s_g["plan"] / 1e4, s_g["emerg"] / 1e4]},
        ylabel="334 天费用 / 万元", save="Q4_H与G费用对比",
    )
    days = list(range(REPORT_START, N_DAY))
    e_h = np.array([float(recs_h[D]["e"].sum()) for D in days])
    e_g = np.array([float(recs_g[D]["e"].sum()) for D in days])
    fig, ax = pm.line(
        np.arange(len(days)), [e_h, e_g],
        labels=["H·历史预测", "G·题面已知电价"],
        xlabel="日期（2025-02-01 起）", ylabel="紧急购电量 / kWh",
    )
    pm.save_fig(fig, "Q4_逐日紧急购电",
                data=pd.DataFrame({"日期": [data["dates"][D] for D in days],
                                   "H紧急量/kWh": e_h, "G紧急量/kWh": e_g}))


# ---------------------------------------------------------------------------
# Q2 层主流程
# ---------------------------------------------------------------------------
def run_result42() -> None:
    """生成 result4-2（H·2 日滚动为主口径；G 同结构对照）。"""
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q4")
    t0 = time.time()
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    recs_h = run_year(data, p4, p_typ, vseq, price_mode="H", storage="2day")
    recs_g = run_year(data, p4, p_typ, vseq, price_mode="G", storage="2day")
    s_h, s_g = total_of(recs_h), total_of(recs_g)
    chk = verify_records(recs_g, data, x_key="x")
    print(f"问题四 Q2 层：H {s_h['total']/1e4:.1f} 万（计划 {s_h['plan']/1e4:.1f} + "
          f"紧急 {s_h['emerg']/1e4:.1f}）；G {s_g['total']/1e4:.1f} 万；"
          f"电价信息价值 {100*(s_h['total']-s_g['total'])/s_h['total']:.2f}%（{time.time()-t0:.0f}s）")

    p = write_result4_2(data, p4, recs_g)
    excel = verify_result4_2(p, data, p4, recs_g)
    log.info("result4-2.xlsx -> {}", p)

    days = list(range(REPORT_START, N_DAY))
    daily = pd.DataFrame({
        "日期": [data["dates"][D] for D in days],
        "计划购电费/元": [recs_g[D]["plan_cost"] for D in days],
        "紧急购电量/kWh": [float(recs_g[D]["e"].sum()) for D in days],
        "紧急购电费/元": [recs_g[D]["emerg"] for D in days],
        "总费用/元": [recs_g[D]["total"] for D in days],
    })
    pm.save_outputs(daily, "q4_result42_daily")
    make_figures(data, recs_h, recs_g, s_h, s_g, daily)

    record(
        "问题四 结果：Q2 层（result4-2，2025-02-01 ~ 12-31）",
        {
            "题面口径 G·总费用/万元": round(s_g["total"] / 1e4, 1),
            "G·计划购电费/万元": round(s_g["plan"] / 1e4, 1),
            "G·紧急购电费/万元": round(s_g["emerg"] / 1e4, 1),
            "G·紧急购电量/kWh": float(s_g["emerg_kwh"]),
            "扩展口径 H·历史价格预测/万元": round(s_h["total"] / 1e4, 1),
            "电价信息价值（H 比 G 高）": f"{100 * (s_h['total'] - s_g['total']) / s_h['total']:.2f}%",
            "储能结构": "2 日滚动（跨日连续）",
            **excel,
        },
        note=(
            "题面口径 G：附件 4 已给未来电价，0:00 用历史负荷预测 + 附件 3 的 0:00 光伏预报；"
            "采用 2 日滚动计划、逐槽因果执行并按附件 4 结算。H 口径不读取目标日价格，"
            "仅作为信息受限扩展，不写入正式结果表。"
            "图 figures/Q4_H与G费用对比.pdf、Q4_逐日紧急购电.pdf；"
            "逐日表 code/outputs/q4_result42_daily.csv；结构探针 code/outputs/q4_structure_probe.csv。"
        ),
    )
    record("问题四 约束与一致性校验", chk,
           note="功率平衡、SOC 递推、区间与充放互斥全部回代；残差应为浮点误差量级。")


# ---------------------------------------------------------------------------
# Q3 层：0:00 计划 + 6/12/18 调整（+6/12 对冲），结算用附件 4 真实价
# ---------------------------------------------------------------------------
def probe_q3() -> None:
    """Q4-3 结构探针：价格口径 × 储能结构 × 调整/对冲。"""
    pm.init(seed=42, root=str(ROOT))
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    cfgs = [
        ("H", "daily", (), False, 1.0, 1.0),
        ("H", "daily", (6, 12, 18), False, 1.0, 1.0),
        ("H", "daily", (6, 12, 18), True, 1.0, 1.0),
        ("H", "daily", (6, 12, 18), True, 0.7, 0.7),
        ("H", "2day", (6, 12, 18), True, 0.7, 0.7),
        ("G", "2day", (6, 12, 18), True, 0.7, 0.7),
    ]
    rows = []
    for mode, storage, adj, hedge, lam, adj_lam in cfgs:
        t0 = time.time()
        recs = run_year_q3(data, p4, p_typ, vseq, price_mode=mode,
                           storage=storage, adj_hours=adj, hedge=hedge,
                           lam=lam, adj_lam=adj_lam)
        s = total_of_q3(recs)
        tag = (f"{mode}·{storage}·{'三点' if adj else '无调整'}"
               f"{'+对冲' if hedge else ''}{'·组合' if lam < 0.9 else '·官方'}")
        print(f"  [{tag:<16}] 总 {s['total']/1e4:7.1f} 万（买入 {s['buy']/1e4:.1f} + "
              f"偏差 {s['dev']/1e4:.1f} + 紧急 {s['emerg']/1e4:.1f}）"
              f" 调整量 {s['adj_kwh']:,.0f} kWh（{time.time()-t0:.0f}s）")
        rows.append({"配置": tag, "计划费": s["plan"], "买入": s["buy"],
                     "偏差费": s["dev"], "紧急": s["emerg"], "总费用": s["total"]})
    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q4_q3_probe.csv",
                              index=False, encoding="utf-8-sig")


def make_figures_q3(data, recs_g, recs_h, s_g, s_h, daily_df) -> None:
    """Q3 层论文图：G/H 费用构成对比 + 题面口径 G 的逐日紧急购电。"""
    # 1) 同一正式策略下的电价信息口径对比
    pm.bar_group(
        ["买入费", "偏差费", "紧急购电费"],
        {"G·题面已知电价": [s_g["buy"] / 1e4, s_g["dev"] / 1e4, s_g["emerg"] / 1e4],
         "H·历史价格预测": [s_h["buy"] / 1e4, s_h["dev"] / 1e4, s_h["emerg"] / 1e4]},
        ylabel="334 天费用 / 万元", save="Q4_调整层策略对比",
    )
    # 2) 题面口径 G 的逐日紧急购电
    fig, ax = pm.line(
        np.arange(len(daily_df)), daily_df["紧急购电量/kWh"],
        xlabel="日期（2025-02-01 起）", ylabel="紧急购电量 / kWh",
    )
    pm.save_fig(fig, "Q4_调整层逐日紧急购电",
                data=daily_df[["日期", "紧急购电量/kWh"]])


def run_result43() -> None:
    """生成 result4-3：G·2 日·三点调整·组合预测·无对冲；H 作扩展对照。"""
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q4")
    t0 = time.time()
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    kw = dict(storage="2day", adj_hours=(6, 12, 18), hedge=cs.Q3_USE_HEDGE,
              n_scen=cs.N_SCEN, seed=7, lam=cs.Q3_LAM, adj_lam=cs.Q3_ADJ_LAM)
    recs_h = run_year_q3(data, p4, p_typ, vseq, price_mode="H", **kw)
    recs_g = run_year_q3(data, p4, p_typ, vseq, price_mode="G", **kw)
    s_h, s_g = total_of_q3(recs_h), total_of_q3(recs_g)
    chk = verify_records(recs_g, data, x_key="x_final", scenario=True)
    print(f"问题四 Q3 层：H {s_h['total']/1e4:.1f} 万（买入 {s_h['buy']/1e4:.1f} + "
          f"偏差 {s_h['dev']/1e4:.1f} + 紧急 {s_h['emerg']/1e4:.1f}）；G {s_g['total']/1e4:.1f} 万"
          f"（{time.time()-t0:.0f}s）")

    p = write_result4_3(data, p4, recs_g)
    excel = verify_result4_3(p, data, p4, recs_g)
    log.info("result4-3.xlsx -> {}", p)

    days = list(range(REPORT_START, N_DAY))
    daily = pd.DataFrame({
        "日期": [data["dates"][D] for D in days],
        "计划购电费/元": [recs_g[D]["plan_cost"] for D in days],
        "买入/元": [recs_g[D]["buy"] for D in days],
        "偏差费/元": [recs_g[D]["dev"] for D in days],
        "紧急购电费/元": [recs_g[D]["emerg"] for D in days],
        "总费用/元": [recs_g[D]["total"] for D in days],
        "紧急购电量/kWh": [float(recs_g[D]["e"].sum()) for D in days],
        "调整量/kWh": [float(np.abs(recs_g[D]["x_final"] - recs_g[D]["x_plan"]).sum())
                    for D in days],
    })
    pm.save_outputs(daily, "q4_result43_daily")
    make_figures_q3(data, recs_g, recs_h, s_g, s_h, daily)

    record(
        "问题四 结果：Q3 层（result4-3，2025-02-01 ~ 12-31）",
        {
            "题面口径 G·总费用/万元": round(s_g["total"] / 1e4, 1),
            "G·买入/万元": round(s_g["buy"] / 1e4, 1),
            "G·偏差费/万元": round(s_g["dev"] / 1e4, 1),
            "G·紧急购电费/万元": round(s_g["emerg"] / 1e4, 1),
            "G·紧急购电量/kWh": float(s_g["emerg_kwh"]),
            "G·调整量/kWh": float(s_g["adj_kwh"]),
            "扩展口径 H·历史价格预测/万元": round(s_h["total"] / 1e4, 1),
            "电价信息价值（H 比 G 高）": f"{100 * (s_h['total'] - s_g['total']) / s_h['total']:.2f}%",
            "储能结构": "2 日滚动（跨日连续）",
            **excel,
        },
        note=(
            "题面口径 G：0:00 计划使用附件 4 已知电价、历史负荷预测及组合光伏预测；"
            "6/12/18 更新光伏并调整；正式方案关闭场景对冲，逐槽因果执行；"
            "采用 2 日滚动跨日储能并按附件 4 结算。H 不读取目标日价格，仅作扩展对照。"
            "图 figures/Q4_调整层策略对比.pdf、Q4_调整层逐日紧急购电.pdf；"
            "逐日表 code/outputs/q4_result43_daily.csv；探针表 code/outputs/q4_q3_probe.csv。"
        ),
    )
    record("问题四 Q3 层约束与校验", chk,
           note="功率平衡、SOC 递推、区间与充放互斥全部回代通过；正式方案未启用场景池。")


def fig_weights() -> None:
    """电价预测权重演化图：滚动费用标定 v* + 动态修正 β=0.1（2025-02-01 起）。"""
    pm.init(seed=42, root=str(ROOT))
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    days = list(range(REPORT_START, N_DAY))
    df = pd.DataFrame({
        "日期": [data["dates"][D] for D in days],
        "v1(D−1)": vseq[days, 0], "v2(D−7)": vseq[days, 1], "v3(典型日)": vseq[days, 2],
    })
    fig, ax = pm.line(
        np.arange(len(days)), [vseq[days, 0], vseq[days, 1], vseq[days, 2]],
        labels=["$v_1$·P(D−1)", "$v_2$·P(D−7)", "$v_3$·典型日"],
        xlabel="日期（2025-02-01 起）", ylabel="电价预测权重",
    )
    pm.save_fig(fig, "Q4_电价权重演化", data=df)
    print("已生成 figures/Q4_电价权重演化.pdf")


# ---------------------------------------------------------------------------
# 结构探针
# ---------------------------------------------------------------------------
def probe() -> None:
    """Q4-2 结构探针：价格口径 × 储能结构（daily vs 2day）。"""
    pm.init(seed=42, root=str(ROOT))
    t0 = time.time()
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    print(f"数据就绪（{time.time()-t0:.0f}s），v 序列 D=31..364："
          f"v1 {vseq[31:,0].mean():.2f} / v2 {vseq[31:,1].mean():.2f} / v3 {vseq[31:,2].mean():.2f}")
    rows = []
    for price_mode in ("H", "G"):
        for storage in ("daily", "2day"):
            t1 = time.time()
            recs = run_year(data, p4, p_typ, vseq, price_mode=price_mode, storage=storage)
            s = total_of(recs)
            print(f"  [{price_mode} · {storage:5}] 总 {s['total']/1e4:7.1f} 万"
                  f"（计划 {s['plan']/1e4:.1f} + 紧急 {s['emerg']/1e4:.1f}）"
                  f" 紧急量 {s['emerg_kwh']:,.0f} kWh（{time.time()-t1:.0f}s）")
            rows.append({"价格口径": price_mode, "储能结构": storage, **s})
    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q4_structure_probe.csv",
                              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="储能/价格口径结构对比探针")
    ap.add_argument("--probe-q3", action="store_true", help="Q3 层结构探针")
    ap.add_argument("--result4-2", action="store_true", help="生成 result4-2.xlsx（Q2 层）")
    ap.add_argument("--result4-3", action="store_true", help="生成 result4-3.xlsx（Q3 层）")
    ap.add_argument("--fig-weights", action="store_true", help="生成电价权重演化图")
    args = ap.parse_args()
    if args.probe:
        probe()
    elif args.probe_q3:
        probe_q3()
    elif args.result4_2:
        run_result42()
    elif args.result4_3:
        run_result43()
    elif args.fig_weights:
        fig_weights()
    else:
        ap.print_help()
