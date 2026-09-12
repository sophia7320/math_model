"""Q2 口径 E 权重平滑回测：新旧权重混合（β）对年度费用的影响。

平滑：权重_D = β·权重_new_D + (1−β)·权重_{D−1}（凸组合，负荷 w 与光伏 u 可分别控制）
对比基准（因果口径）：W=7 网格（官方 result2，1467.4 万）、W=1+梯度（1515.4 万）。

运行（program/ 下）：uv run python -m solve.experiments.q2e_smooth
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve.common import ROOT
from solve.models.adaptive import AdaptiveWeightModel
from solve.models.weights import softmax


def smoothed_from_table(model, W, beta, smooth_load=True, smooth_pv=True):
    """平滑"网格选择"权重序列（无前视）。"""
    grid = model.grid
    out = []
    prev_w = prev_u = None
    for d in range(q2.REPORT_START, q2.N_DAY):
        lo = max(model.START, d - W)
        i, j = np.unravel_index(np.argmin(model.C[lo:d].mean(axis=0)),
                                (len(grid), len(grid)))
        w_new, u_new = grid[i].copy(), grid[j].copy()
        if prev_w is None:
            prev_w, prev_u = w_new, u_new
        else:
            if smooth_load:
                w_new = beta * w_new + (1 - beta) * prev_w
            if smooth_pv:
                u_new = beta * u_new + (1 - beta) * prev_u
            prev_w, prev_u = w_new, u_new
        out.append((w_new.copy(), u_new.copy()))
    return out


def smoothed_from_th(model, beta, smooth_load=True, smooth_pv=True):
    """平滑 W=1 梯度精化权重（TH）序列（无前视）。"""
    out = []
    prev_w = prev_u = None
    for d in range(q2.REPORT_START, q2.N_DAY):
        theta = model.TH[d - 1]
        w_new, u_new = softmax(theta[:3]), softmax(theta[3:])
        if prev_w is None:
            prev_w, prev_u = w_new, u_new
        else:
            if smooth_load:
                w_new = beta * w_new + (1 - beta) * prev_w
            if smooth_pv:
                u_new = beta * u_new + (1 - beta) * prev_u
            prev_w, prev_u = w_new, u_new
        out.append((w_new.copy(), u_new.copy()))
    return out


def cost_of(model, weights):
    df = model.simulate(weights)
    plan = float(df["计划购电费/元"].sum())
    em = float(df["紧急购电费/元"].sum())
    return plan, em


def main():
    pm.init(seed=42, root=str(ROOT))
    model = AdaptiveWeightModel().load().build_table()
    rows = []
    t0 = time.time()

    plan17, em17 = cost_of(model, model.weights(W=7, use_gd=False))
    plan11, em11 = cost_of(model, model.weights(W=1, use_gd=True))
    base7 = plan17 + em17
    print(f"基准 W=7 网格      : {base7 / 1e4:7.1f} 万（计划 {plan17 / 1e4:.1f} + 紧急 {em17 / 1e4:.1f}）")
    print(f"基准 W=1 网格+梯度 : {(plan11 + em11) / 1e4:7.1f} 万（计划 {plan11 / 1e4:.1f} + 紧急 {em11 / 1e4:.1f}）")
    rows += [{"配置": "基准 W=7", "总费用/万元": round(base7 / 1e4, 1)},
             {"配置": "基准 W=1+梯度", "总费用/万元": round((plan11 + em11) / 1e4, 1)}]

    print("\n网格权重平滑（W=7 基准）：")
    for beta in (0.1, 0.2, 0.3, 0.5):
        for only_pv in (False, True):
            ws = smoothed_from_table(model, 7, beta, smooth_load=not only_pv)
            plan, em = cost_of(model, ws)
            tot = plan + em
            tag = f"W=7 平滑 β={beta}（{'仅光伏' if only_pv else '负荷+光伏'}）"
            print(f"  {tag:<28} {tot / 1e4:7.1f} 万"
                  f"（计划 {plan / 1e4:.1f} + 紧急 {em / 1e4:.1f}，vs W=7 {100 * (tot / base7 - 1):+.2f}%）")
            rows.append({"配置": tag, "总费用/万元": round(tot / 1e4, 1)})

    print("\n梯度权重平滑（W=1+梯度 基准）：")
    for beta in (0.1, 0.3):
        ws = smoothed_from_th(model, beta)
        plan, em = cost_of(model, ws)
        tot = plan + em
        tag = f"W=1+梯度 平滑 β={beta}"
        print(f"  {tag:<28} {tot / 1e4:7.1f} 万"
              f"（计划 {plan / 1e4:.1f} + 紧急 {em / 1e4:.1f}）")
        rows.append({"配置": tag, "总费用/万元": round(tot / 1e4, 1)})

    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q2e_smooth_compare.csv",
                              index=False, encoding="utf-8-sig")
    print(f"\n已保存 code/outputs/q2e_smooth_compare.csv（用时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
