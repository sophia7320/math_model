"""C 题 问题二：日前计划 + 紧急购电（口径 D）+ 年度成本概率分布。

口径（论文假设，常量区可调）：
- 计划：每天 0:00 依据附件 3 的 0:00 光伏预报（负荷视为已知）解 LP，最小化计划购电费；
  储能日循环 E(0:00)=E(24:00)=6000 kWh；
- 执行：计划购电量已承诺（take-or-pay），储能逐槽因果执行（core.causal.exec_segment_causal，
  不读取未来实际值；core.lp.exec_day 事后 LP 仅作前视下界），缺口按 5 倍电价紧急购电；
- 各日相互独立（储能日循环）；结果自 2025-02-01 起报送。

概率扩展：以附件 3 预报误差的"整日标准化误差块"重采样，蒙特卡洛评估年度总成本分布
（误差模型参数见 models/errors.py 与 C题_预报误差分析.md）。

本模块保留既有入口与公开 API（兼容门面）：
    uv run python -m solve.q2     # 全年滚动 + 100 年蒙特卡洛 + 对冲对照

分层结构：核心 LP/执行在 solve.core，数据在 solve.data，年度流程在 solve.flows，
结果表与校验在 solve.io；本文件只做编排（run_q2）与出图（make_figures）。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import program as pm
from solve.common import (
    DATA_C,
    E0,
    E_MAX,
    E_MIN,
    EMERG_MULT,
    EPS_TH,
    EPS_THROUGHPUT,
    ETA,
    N_DAY,
    P_MAX_E,
    REPORT_START,
    RESULTS_DIR,
    ROOT,
    T,
)
from solve.core.causal import (
    CAUSAL_POLICY_VERSION,
    exec_day_causal,
    exec_segment_causal,
)
from solve.core.lp import exec_day, plan_day
from solve.core.slots import _events, _fmt_time, _hour_to_slots
from solve.data.attachments import load_all
from solve.flows.q2_year import hedge_day, run_deterministic, run_hedge, run_mc
from solve.io.checks import verify
from solve.io.excel import write_result2
from solve.io.figures import month_axis


def make_figures(dates, recs, mc_costs, plan_cost_report):
    """Q2 论文图：逐日紧急购电量 + 年度总费用分布。

    上图：每日紧急购电量（kWh），按月份刻度；作图数据同步 figure_data/*.csv。
    下图：年度总购电费直方图（计划费 + 蒙特卡洛紧急费），标均值与 P95。
    """
    import matplotlib.pyplot as plt

    rep = list(range(REPORT_START, N_DAY))
    emerg_daily = np.array([recs[d]["emerg_kwh"] for d in rep])
    fig, ax = pm.line(np.arange(len(rep)), emerg_daily,
                      xlabel="日期", ylabel="紧急购电量 / kWh")
    month_axis(ax, [dates[d] for d in rep])
    pm.save_fig(fig, "Q2_逐日紧急购电",
                data=pd.DataFrame({"日期": [dates[d] for d in rep],
                                   "紧急购电量_kWh": emerg_daily}))

    if mc_costs is not None and len(mc_costs):
        total = plan_cost_report + mc_costs
        fig2, ax2 = plt.subplots(figsize=(7, 4.3))
        ax2.hist(total, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax2.axvline(total.mean(), color="#C44E52", ls="--",
                    label=f"均值 {total.mean():,.0f} 元")
        ax2.axvline(np.percentile(total, 95), color="#55A868", ls=":",
                    label=f"P95 {np.percentile(total, 95):,.0f} 元")
        ax2.set_xlabel("年度总购电费 / 元")
        ax2.set_ylabel("频数")
        ax2.legend()
        pm.save_fig(fig2, "Q2_总费用分布",
                    data=pd.DataFrame({"年度总购电费_元": total}))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_q2(n_mc: int = 100, seed: int = 42, hedge: bool = True) -> dict:
    """口径 D 主流程：滚动 → MC → 对冲对照 → 校验 → 记录 → 出图 → result2。"""
    pm.init(root=str(ROOT))
    log = pm.get_logger("q2")
    data = load_all()

    t0 = time.time()
    x_plans, recs = run_deterministic(data)
    log.info("确定性滚动完成（365 天，用时 {:.1f}s）", time.time() - t0)

    rep = list(range(REPORT_START, N_DAY))
    plan_cost = sum(recs[d]["plan_cost"] for d in rep)
    emerg_cost = sum(recs[d]["emerg_cost"] for d in rep)
    emerg_kwh = sum(recs[d]["emerg_kwh"] for d in rep)
    spill_kwh = sum(recs[d]["spill_kwh"] for d in rep)
    n_days = sum(1 for d in rep if recs[d]["emerg_kwh"] > 1e-3)
    max_day = max(rep, key=lambda d: recs[d]["emerg_kwh"])

    pm.record_result(
        "问题二（口径D）确定性结果（2025-02-01 ~ 12-31）",
        {
            "计划购电量/kWh": float(x_plans[rep].sum()),
            "计划购电费/元": float(plan_cost),
            "紧急购电量/kWh": float(emerg_kwh),
            "紧急购电费/元": float(emerg_cost),
            "发生紧急购电的天数": n_days,
            "单日最大紧急购电量/kWh": float(recs[max_day]["emerg_kwh"]),
            "全程弃电量/kWh": float(spill_kwh),
            "日初=日末储电量/kWh": float(recs[REPORT_START]["E_start"]),
            "日内储电量最小值/kWh": float(min(recs[d]["E"].min() for d in rep)),
            "日内储电量最大值/kWh": float(max(recs[d]["E"].max() for d in rep)),
        },
        note=(
            "口径 D：0:00 用附件 3 的 0:00 光伏预报制定计划（负荷视为已知），储能日循环；"
            "计划购电 take-or-pay，实际按附件 2 执行，缺口按 5 倍交易时刻电价紧急购电；"
            "储能日循环（各日独立），结果自 2 月 1 日报送；图中位置：figures/Q2_逐日紧急购电.pdf。"
        ),
    )

    checks = verify(data, x_plans, recs)
    pm.record_result("问题二 约束与一致性校验", checks,
                     note="残差为数值误差量级即可；储电量须在 [1200, 10800] kWh 内。")

    daily = pd.DataFrame({
        "日期": [data["dates"][d] for d in rep],
        "计划购电量/kWh": [float(x_plans[d].sum()) for d in rep],
        "计划购电费/元": [recs[d]["plan_cost"] for d in rep],
        "紧急购电量/kWh": [recs[d]["emerg_kwh"] for d in rep],
        "紧急购电费/元": [recs[d]["emerg_cost"] for d in rep],
        "弃电量/kWh": [recs[d]["spill_kwh"] for d in rep],
        "日初储电量/kWh": [recs[d]["E_start"] for d in rep],
        "日末储电量/kWh": [recs[d]["E_end"] for d in rep],
    })
    pm.save_outputs(daily, "q2_daily_summary")

    out = write_result2(data["dates"], data["price"], x_plans, recs)
    log.info("result2.xlsx -> {}", out)

    mc_costs = None
    if n_mc > 0:
        t1 = time.time()
        mc_costs, _mc_kwhs = run_mc(data, x_plans, recs[REPORT_START]["E_start"],
                                    n_years=n_mc, seed=seed)
        log.info("蒙特卡洛完成（{} 年，用时 {:.1f}s）", n_mc, time.time() - t1)
        total = plan_cost + mc_costs
        p95 = float(np.percentile(total, 95))
        pm.record_result(
            "问题二 年度总成本分布（蒙特卡洛）",
            {
                "模拟年数": n_mc,
                "年度总成本均值/元": float(total.mean()),
                "年度总成本标准差/元": float(total.std()),
                "P5/元": float(np.percentile(total, 5)),
                "P50/元": float(np.percentile(total, 50)),
                "P95/元": p95,
                "CVaR95（尾部均值）/元": float(total[total >= p95].mean()),
                "年度紧急费用均值/元": float(mc_costs.mean()),
                "年度紧急费用P95/元": float(np.percentile(mc_costs, 95)),
            },
            note=(
                "固定计划、仅重采样附件 3 预报误差（整日标准化误差块，保留日内相关与左尾）；"
                "计划费固定，分布差异来自紧急购电；图 figures/Q2_总费用分布.pdf。"
            ),
        )
        pm.save_outputs(
            pd.DataFrame({"计划费_元": plan_cost, "紧急费_元": mc_costs,
                          "总费用_元": plan_cost + mc_costs}),
            "q2_mc_annual_costs",
        )

        if hedge:
            import matplotlib.pyplot as plt

            t2 = time.time()
            x_hedge = run_hedge(data, n_scen=20, seed=7)
            log.info("对冲计划求解完成（用时 {:.1f}s）", time.time() - t2)
            plan_cost_h = float(sum(float(data["price"] @ x_hedge[d]) for d in rep))
            mc_h, _ = run_mc(data, x_hedge, recs[REPORT_START]["E_start"],
                             n_years=n_mc, seed=seed)
            total_h = plan_cost_h + mc_h
            p95h = float(np.percentile(total_h, 95))
            pm.record_result(
                "问题二 对冲计划 vs 朴素计划（蒙特卡洛对比）",
                {
                    "朴素：计划购电费/元": float(plan_cost),
                    "朴素：紧急费均值/元": float(mc_costs.mean()),
                    "朴素：总成本均值/元": float(total.mean()),
                    "朴素：总成本P95/元": p95,
                    "对冲：计划购电费/元": plan_cost_h,
                    "对冲：紧急费均值/元": float(mc_h.mean()),
                    "对冲：总成本均值/元": float(total_h.mean()),
                    "对冲：总成本P95/元": p95h,
                    "期望费用下降/元": float(total.mean() - total_h.mean()),
                    "期望费用下降/%": float(100 * (total.mean() - total_h.mean()) / total.mean()),
                    "P95 下降/元": float(p95 - p95h),
                },
                note="对冲计划 = 两阶段场景 LP（每日 20 个误差情景）最小化期望总费用；两套计划用同一种子评估。",
            )
            pm.save_outputs(
                pd.DataFrame({"朴素_总费用_元": total, "对冲_总费用_元": total_h,
                              "朴素_紧急费_元": mc_costs, "对冲_紧急费_元": mc_h}),
                "Q2_对冲费用分布",
            )
            fig3, ax3 = plt.subplots(figsize=(7, 4.3))
            ax3.hist(total, bins=25, alpha=0.55, color="#4C72B0", label="朴素计划")
            ax3.hist(total_h, bins=25, alpha=0.55, color="#C44E52", label="对冲计划")
            ax3.set_xlabel("年度总购电费 / 元")
            ax3.set_ylabel("频数")
            ax3.legend()
            pm.save_fig(fig3, "Q2_对冲费用分布",
                        data=pd.DataFrame({"朴素_总费用_元": total,
                                           "对冲_总费用_元": total_h}))

    make_figures(data["dates"], recs, mc_costs, float(plan_cost))
    log.info("问题二完成：计划购电费 {:.0f} 元，紧急购电费 {:.0f} 元（{} 天有缺口）",
             plan_cost, emerg_cost, n_days)
    return {"plan_cost": plan_cost, "emerg_cost": emerg_cost,
            "emerg_kwh": emerg_kwh, "x_plans": x_plans, "recs": recs,
            "mc_costs": mc_costs}


if __name__ == "__main__":
    run_q2()
