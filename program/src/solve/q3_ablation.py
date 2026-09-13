"""C 题 问题三 关键消融（因果执行口径）：对冲 / 组合 / 信息退化的单独贡献。

主方案 = 组合（λ=0.7）+ 三点调整 + 6/12 无前视对冲（统一口径 v1.2，free 执行器）。
本脚本复算主方案并补充三个消融配置：
- 官方·三点+对冲：对冲在官方口径下的贡献；
- 组合·三点·无对冲：组合在无对冲下的贡献（与主方案差 = 对冲贡献）；
- 信息退化：计划用组合、调整层仅用官方 → 验证"调整层必须同口径"。

运行（program/ 下）：uv run python -m solve.q3_ablation
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import program as pm
from solve import consistency as cs
from solve import q3_proto as qp
from solve.common import ROOT
from solve.io.report import record

# 统一口径：EWMA 权重由 load_extended() 提供；情景数取唯一参数源。
N_SCEN = cs.N_SCEN
SEED = 7
LAM = 0.7
DAYS = list(range(31, 365))


def run_cfg(tag, fn, *, lam, adj_lam, hedge, data):
    t0 = time.time()
    tot = plan = em = 0.0
    e_kwh = 0.0
    E = float(qp.E0)  # 跨日 SOC 连续（与官方主程序一致）
    for D in DAYS:
        kw = {"adj_lam": adj_lam, "e_start": E}
        if hedge:
            kw.update({"n_scen": N_SCEN, "seed": SEED})
        r = fn(data, D, lam, **kw)
        tot += r["total"]
        plan += r["plan_cost"]
        em += r["emerg"]
        e_kwh += float(r["e"].sum())
        E = float(r["E_end"])
    print(f"  {tag:<22} 总 {tot/1e4:7.1f} 万（计划 {plan/1e4:.1f} + 紧急 {em/1e4:.1f}）"
          f" 紧急量 {e_kwh:,.0f} kWh（{time.time()-t0:.0f}s）")
    return {"配置": tag, "总费用/万元": round(tot / 1e4, 1),
            "计划费/万元": round(plan / 1e4, 1), "紧急费/万元": round(em / 1e4, 1)}


def main():
    pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()  # 含统一 EWMA 权重（EWMA_WU）
    print("Q3 消融（因果口径，334 天）：")
    rows = [
        run_cfg("官方·三点+对冲", qp.simulate_day_rt_hedge,
                lam=1.0, adj_lam=1.0, hedge=True, data=data),
        run_cfg("组合·三点·无对冲", qp.simulate_day_rt,
                lam=LAM, adj_lam=LAM, hedge=False, data=data),
        run_cfg("主方案（组合+对冲）", qp.simulate_day_rt_hedge,
                lam=LAM, adj_lam=LAM, hedge=True, data=data),
        run_cfg("信息退化（调整层仅官方）", qp.simulate_day_rt_hedge,
                lam=LAM, adj_lam=1.0, hedge=True, data=data),
    ]
    df = pd.DataFrame(rows)
    pm.save_outputs(df, "q3_ablation")
    record(
        "问题三 消融实验（因果口径，2025-02-01 ~ 12-31）",
        df,
        note=(
            "因果执行口径下的单项消融（跨日 SOC 连续，与官方主程序一致）："
            "官方·三点+对冲（无组合）；组合·三点·无对冲；"
            "主方案（组合 λ=0.7 + 三点 + 6/12 对冲）——应与官方 result3 同值；"
            "信息退化（计划组合、调整层仅官方）。用于论文中“对冲/组合/同口径”的贡献分解。"
            "对照：无调整官方 1455.7 万、三点官方 1358.9 万（RESULTS_REPORT 主方案章节）。"
        ),
    )
    print("已保存 code/outputs/q3_ablation.csv")


if __name__ == "__main__":
    main()
