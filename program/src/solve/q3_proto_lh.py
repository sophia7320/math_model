"""Q3 原型第七批：对冲模式下组合权重扫描 + 调整点确认。

运行（program/ 下）：uv run python -m solve.q3_proto_lh
"""
from __future__ import annotations

import time

import pandas as pd

from solve import q3_proto as qp
from solve.common import ROOT

DAYS = list(range(31, 365))


def run(data, lam, adj_lam, adj_hours=(6, 12, 18)):
    outs = [qp.simulate_day_rt_hedge(data, D, lam, adj_hours=adj_hours,
                                     adj_lam=adj_lam, n_scen=10, seed=7) for D in DAYS]
    return {
        "total": sum(o["total"] for o in outs),
        "plan": sum(o["plan_cost"] for o in outs),
        "adj": sum(o["adjust_net"] for o in outs),
        "emerg": sum(o["emerg"] for o in outs),
    }


def main():
    qp.pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()
    rows = []

    print("对冲模式 · 组合权重扫描（全开调整，6/12 对冲）：")
    for lam in (0.3, 0.5, 0.7, 1.0):
        t0 = time.time()
        r = run(data, lam, None if lam == 1.0 else lam)
        rows.append({"配置": f"λ={lam} 全开对冲", "总费用/万元": round(r["total"] / 1e4, 1),
                     "计划/万元": round(r["plan"] / 1e4, 1),
                     "调整/万元": round(r["adj"] / 1e4, 1),
                     "紧急/万元": round(r["emerg"] / 1e4, 1)})
        print(f"  λ={lam:<4} 总 {r['total'] / 1e4:8.1f} 万 "
              f"（计划 {r['plan'] / 1e4:.1f} + 调整 {r['adj'] / 1e4:+.1f} + 紧急 {r['emerg'] / 1e4:.1f}）"
              f"  {time.time() - t0:.0f}s")

    print("\n对冲模式 · 调整点确认（λ=1）：")
    for hs, name in [((12, 18), "12+18"), ((6, 12, 18), "全开")]:
        t0 = time.time()
        r = run(data, 1.0, None, adj_hours=hs)
        rows.append({"配置": f"λ=1 {name} 对冲", "总费用/万元": round(r["total"] / 1e4, 1),
                     "计划/万元": round(r["plan"] / 1e4, 1),
                     "调整/万元": round(r["adj"] / 1e4, 1),
                     "紧急/万元": round(r["emerg"] / 1e4, 1)})
        print(f"  {name:<6} 总 {r['total'] / 1e4:8.1f} 万 "
              f"（计划 {r['plan'] / 1e4:.1f} + 调整 {r['adj'] / 1e4:+.1f} + 紧急 {r['emerg'] / 1e4:.1f}）"
              f"  {time.time() - t0:.0f}s")

    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q3_proto_lambda_hedge.csv",
                              index=False, encoding="utf-8-sig")
    print("已保存 code/outputs/q3_proto_lambda_hedge.csv")


if __name__ == "__main__":
    main()
