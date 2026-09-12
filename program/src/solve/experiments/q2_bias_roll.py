# -*- coding: utf-8 -*-
"""Q2 偏差策略的滚动标定验证（无前视）：每天用过去 W 天选最优 δ。

运行（program/ 下）：uv run python -m solve.experiments.q2_bias_roll
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve.experiments import q2_bias as qb
from solve.common import ROOT

DELTAS = [0, 50, 100, 150, 200, 250, 300, 350, 400, 500]


def main():
    pm.init(seed=42, root=str(ROOT))
    data = q2.load_all()
    rep = np.arange(q2.REPORT_START, q2.N_DAY)
    tab = np.zeros((q2.N_DAY, len(DELTAS)))
    for j, dl in enumerate(DELTAS):
        r = qb.simulate_det(data, float(dl))
        tab[:, j] = r["daily"]

    base = tab[rep, 0].sum()
    i300 = DELTAS.index(300)
    fixed300 = tab[rep, i300].sum()
    oracle = tab[rep].min(axis=1).sum()

    rows = [{"策略": "基准 δ=0", "总费用/万元": round(base / 1e4, 1)}]
    print(f"基准 δ=0         {base / 1e4:7.1f} 万")
    print(f"固定 δ=300       {fixed300 / 1e4:7.1f} 万（确定性事后最优附近的代表值）")
    print(f"离线最优（参考）  {oracle / 1e4:7.1f} 万\n")
    rows.append({"策略": "固定 δ=300", "总费用/万元": round(fixed300 / 1e4, 1)})

    for W in (14, 30, 60):
        tot = 0.0
        picks = []
        for d in rep:
            i0 = max(0, d - W)
            sc = tab[i0:d].mean(axis=0)
            j = int(np.argmin(sc))
            tot += tab[d, j]
            picks.append(DELTAS[j])
        picks = np.array(picks)
        print(f"滚动 W={W:2d}        {tot / 1e4:7.1f} 万   平均 δ={picks.mean():5.0f} kW  "
              f"（选择分布：{dict(zip(*np.unique(picks, return_counts=True)))}）")
        rows.append({"策略": f"滚动 W={W}", "总费用/万元": round(tot / 1e4, 1)})

    rows.append({"策略": "离线最优（参考）", "总费用/万元": round(oracle / 1e4, 1)})
    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q2_bias_rolling.csv",
                              index=False, encoding="utf-8-sig")
    print("\n已保存 code/outputs/q2_bias_rolling.csv")


if __name__ == "__main__":
    main()
