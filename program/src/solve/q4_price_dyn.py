"""Q4 电价 v 参数的动态修正（新旧混合平滑）验证。

流程：
1. 复用 q4_price_fit 的逐日费用表（daily_costs）做滚动费用标定 → v*_D 序列（无前视）
2. 动态修正：v_D = β·v*_D + (1−β)·v_{D−1}（β∈{0.1,0.3,0.5,0.7}）
3. 用平滑后的连续 v 序列重新模拟全年（跨日连续储能），对比 334 天费用

运行（program/ 下）：uv run python -m solve.q4_price_dyn
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve.common import DATA_C, E0, ROOT, T

DAY_START = q2.REPORT_START
N_DAY = q2.N_DAY


def forecast_price(D, v, p4, p_typ):
    return v[0] * p4[D - 1] + v[1] * p4[D - 7] + v[2] * p_typ


def run_year_vseq(p4, p_typ, data, v_seq):
    """按给定 v 序列模拟全年（v_seq: {day: v}；缺省天用纯典型日），返回 334 天费用。"""
    load = data["load"]; pv_act = data["pv_act"]; fc0 = data["fc0"]
    E = E0
    tot = 0.0
    for D in range(N_DAY):
        v = v_seq.get(D, (0.0, 0.0, 1.0))
        p_hat = forecast_price(D, v, p4, p_typ) if D >= 7 else p_typ
        x, _E_plan, _ = q2.plan_day(p_hat, load[D] / 6.0,
                                    q2._hour_to_slots(fc0[D]) / 6.0, E, eps=1e-3)
        ex = q2.exec_day_causal(p4[D], load[D] / 6.0, pv_act[D] / 6.0, x, E)
        E = float(ex["E"][-1])
        if D >= DAY_START:
            tot += float(p4[D] @ x + q2.EMERG_MULT * (p4[D] @ ex["e"]))
    return tot


def main():
    pm.init(seed=42, root=str(ROOT))
    data = q2.load_all()
    p_typ = data["price"]
    df4 = pm.read_table(DATA_C / "附件4.xlsx")
    p4 = df4.iloc[:, 1:145].to_numpy(float)

    z = np.load(ROOT / "code" / "outputs" / "q4_price_fit_daily.npz")
    daily_costs = z["daily_costs"]
    grid = [tuple(v) for v in z["grid"]]

    # 滚动费用标定 → v*_D
    v_star = {}
    for D in range(DAY_START, N_DAY):
        i0 = max(DAY_START, D - 7)
        j = int(np.argmin(daily_costs[i0:D].mean(axis=0)))
        v_star[D] = grid[j]
    print(f"v* 逐日选择：v1 均值 {np.mean([v[0] for v in v_star.values()]):.2f}、"
          f"v2 {np.mean([v[1] for v in v_star.values()]):.2f}、"
          f"v3 {np.mean([v[2] for v in v_star.values()]):.2f}")

    t0 = time.time()
    base = run_year_vseq(p4, p_typ, data, v_star)
    print(f"滚动费用标定（不平滑）: {base / 1e4:.1f} 万元（{time.time() - t0:.0f}s）")

    rows = [{"配置": "滚动·不平滑", "总费用/万元": round(base / 1e4, 1)}]
    for beta in (0.1, 0.3, 0.5, 0.7):
        v_sm = {}
        prev = None
        for D in range(DAY_START, N_DAY):
            v_new = np.array(v_star[D])
            prev = v_new if prev is None else beta * v_new + (1 - beta) * prev
            v_sm[D] = tuple(prev)
        tot = run_year_vseq(p4, p_typ, data, v_sm)
        tag = f"动态修正 β={beta}"
        print(f"{tag}: {tot / 1e4:.1f} 万元（vs 不平滑 {100 * (tot / base - 1):+.2f}%）"
              f"  ({time.time() - t0:.0f}s)")
        rows.append({"配置": tag, "总费用/万元": round(tot / 1e4, 1)})

    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q4_price_dyn.csv",
                              index=False, encoding="utf-8-sig")
    print("已保存 code/outputs/q4_price_dyn.csv")


if __name__ == "__main__":
    main()
