"""Q3 原型第八批：历史预测权重的参数平滑（新旧参数混合）+ 动态 λ。

- 参数平滑：u_D = β·u_new + (1−β)·u_{D−1}（β=1 为不平滑）
- 评估基座：实时执行 + 全开调整 + 调整层组合

运行（program/ 下）：uv run python -m solve.experiments.q3_proto_smooth
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from solve import q3_proto as qp
from solve.common import ROOT

DAYS = list(range(31, 365))
LAM_SUB = np.array([0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])


def daily_costs(data, fn, lam, **kw):
    return np.array([fn(data, D, lam, **kw)["total"] for D in DAYS])


def main():
    qp.pm.init(seed=42, root=str(ROOT))
    base = qp.load_extended()
    rows = []

    print("① 参数平滑 beta 扫描（实时模式，固定 λ=0.5，调整层组合）：")
    tot_by_beta = {}
    for beta in (0.1, 0.3, 0.5, 0.7, 1.0):
        d2 = dict(base)
        d2["U_SMOOTH"] = qp.make_smooth_u(base, beta)
        t0 = time.time()
        tot = daily_costs(d2, qp.simulate_day_rt, 0.5, adj_lam=0.5).sum()
        tot_by_beta[beta] = tot
        rows.append({"实验": "beta", "参数": beta, "总费用/万元": round(tot / 1e4, 1)})
        print(f"  beta={beta:.1f}  {tot / 1e4:8.1f} 万  ({time.time() - t0:.0f}s)")
    beta_star = min(tot_by_beta, key=tot_by_beta.get)
    print(f"  -> 最优 beta*={beta_star}")

    print("\n② λ 网格 + 动态 λ（实时，无对冲）：")
    tabs = {}
    for beta in (1.0, beta_star):
        d2 = dict(base)
        d2["U_SMOOTH"] = qp.make_smooth_u(base, beta)
        t0 = time.time()
        tab = np.zeros((len(DAYS), len(LAM_SUB)))
        for i, D in enumerate(DAYS):
            for j, lam in enumerate(LAM_SUB):
                tab[i, j] = qp.simulate_day_rt(d2, D, float(lam), adj_lam=float(lam))["total"]
        tabs[beta] = tab
        print(f"  beta={beta} 网格完成（{time.time() - t0:.0f}s）")

    def dyn(tab, W=30, lo=0.2, hi=0.8):
        cost = np.zeros(len(DAYS))
        for i in range(len(DAYS)):
            i0 = max(0, i - W)
            score = tab[i0:i].mean(axis=0) if i > i0 else tab[:1].mean(axis=0)
            valid = (LAM_SUB >= lo - 1e-9) & (LAM_SUB <= hi + 1e-9)
            li = int(np.argmin(np.where(valid, score, np.inf)))
            cost[i] = tab[i, li]
        return cost

    for beta in (1.0, beta_star):
        tab = tabs[beta]
        ev = tab[7:]
        d_cost = dyn(tab)[7:]
        print(f"\n  beta={beta}:")
        for j, lam in enumerate(LAM_SUB):
            print(f"    固定 λ={lam:.1f}: {ev.sum(axis=0)[j] / 1e4:8.1f} 万")
        print(f"    动态 W=30 限[0.2,0.8]: {d_cost.sum() / 1e4:8.1f} 万")
        print(f"    离线最优（参考）:       {ev.min(axis=1).sum() / 1e4:8.1f} 万")
        rows.append({"实验": "lambda", "参数": f"beta={beta} 动态",
                     "总费用/万元": round(d_cost.sum() / 1e4, 1)})

    print("\n③ 对冲模式对比（λ=0.7）：")
    for beta in (1.0, beta_star):
        d2 = dict(base)
        d2["U_SMOOTH"] = qp.make_smooth_u(base, beta)
        t0 = time.time()
        tot = sum(qp.simulate_day_rt_hedge(d2, D, 0.7, adj_lam=0.7, n_scen=10, seed=7)["total"]
                  for D in DAYS)
        rows.append({"实验": "hedge", "参数": f"beta={beta}", "总费用/万元": round(tot / 1e4, 1)})
        print(f"  beta={beta}: {tot / 1e4:8.1f} 万  ({time.time() - t0:.0f}s)")

    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q3_proto_smooth.csv",
                              index=False, encoding="utf-8-sig")
    print("已保存 code/outputs/q3_proto_smooth.csv")


if __name__ == "__main__":
    main()
