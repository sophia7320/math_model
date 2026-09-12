"""Q3 原型第六批：保守裕度扫描 + 调整层场景对冲。

运行（program/ 下）：uv run python -m solve.q3_proto_final
"""
from __future__ import annotations

import time

import pandas as pd

from solve import q3_proto as qp
from solve.common import ROOT

DAYS = list(range(31, 365))


def total(data, fn, lam, **kw):
    return float(sum(fn(data, D, lam, **kw)["total"] for D in DAYS))


def main():
    qp.pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()
    rows = []

    print("① 保守裕度扫描（λ=0.5，全开，调整层组合）：")
    for m in (0.0, 0.005, 0.01, 0.02, 0.03):
        t0 = time.time()
        tot = total(data, qp.simulate_day_rt, 0.5, adj_lam=0.5, margin=m)
        rows.append({"实验": "margin", "参数": f"{m:.3f}", "总费用/万元": round(tot / 1e4, 1)})
        print(f"  margin={m:.3f}  {tot / 1e4:8.1f} 万  ({time.time() - t0:.0f}s)")

    print("\n② 调整层场景对冲（全开）：")
    for lam in (0.5, 1.0):
        arg = {"adj_lam": (lam if lam < 1 else None)}
        t0 = time.time()
        base = total(data, qp.simulate_day_rt, lam, **arg)
        t1 = time.time()
        hedge = total(data, qp.simulate_day_rt_hedge, lam, n_scen=10, seed=7, **arg)
        rows += [{"实验": "hedge", "参数": f"lam={lam} base", "总费用/万元": round(base / 1e4, 1)},
                 {"实验": "hedge", "参数": f"lam={lam} hedge", "总费用/万元": round(hedge / 1e4, 1)}]
        print(f"  λ={lam}: 基准 {base / 1e4:8.1f} 万（{t1 - t0:.0f}s） → 对冲 {hedge / 1e4:8.1f} 万"
              f"（{time.time() - t1:.0f}s）  差 {(hedge - base) / 1e4:+.1f} 万")

    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q3_proto_final.csv",
                              index=False, encoding="utf-8-sig")
    print("已保存 code/outputs/q3_proto_final.csv")


if __name__ == "__main__":
    main()
