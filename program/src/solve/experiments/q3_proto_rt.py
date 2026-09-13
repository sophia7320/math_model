"""Q3 原型第四批：实时执行模式对比与调整点稳健性。

实时执行 = 每段先用实际功率执行到下一调整时刻（得到真实储电量），调整 LP 以该
储电量为初值（利用已实现信息），段执行终端跟踪最近一次规划轨迹。

运行（program/ 下）：uv run python -m solve.experiments.q3_proto_rt
"""
from __future__ import annotations

import time

import pandas as pd

from solve import q3_proto as qp
from solve.common import ROOT

DAY_START, N_DAYS = 31, 334
DAYS = list(range(DAY_START, DAY_START + N_DAYS))


def run(data, fn, lam, **kw):
    outs = [fn(data, D, lam, **kw) for D in DAYS]
    return {
        "total": sum(o["total"] for o in outs),
        "plan": sum(o["plan_cost"] for o in outs),
        "adj": sum(o["adjust_net"] for o in outs),
        "emerg": sum(o["emerg"] for o in outs),
    }


def main():
    qp.pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()

    print("模式对比（λ=1，三点调整，334 天）：")
    for name, fn in (("事后执行（全天一次执行 LP）", qp.simulate_day),
                     ("实时执行（实际重放+分段）", qp.simulate_day_rt)):
        t0 = time.time()
        r = run(data, fn, 1.0)
        print(f"  {name}  总 {r['total'] / 1e4:8.1f} 万  "
              f"（计划 {r['plan'] / 1e4:.1f} + 调整 {r['adj'] / 1e4:+.1f} "
              f"+ 紧急 {r['emerg'] / 1e4:.1f}）  {time.time() - t0:.0f}s")

    print("\n实时执行模式 · 调整点消融：")
    combos = [(), (6,), (12,), (18,), (6, 12), (6, 18), (12, 18), (6, 12, 18)]
    rows = []
    for lam in (1.0, 0.5):
        for c in combos:
            t0 = time.time()
            r = run(data, qp.simulate_day_rt, lam, adj_hours=c)
            rows.append({"λ": lam, "调整点": str(c),
                         "总费用/万元": round(r["total"] / 1e4, 1),
                         "紧急费/万元": round(r["emerg"] / 1e4, 1)})
            print(f"  λ={lam}  {str(c):<12} 总 {r['total'] / 1e4:8.1f} 万  "
                  f"紧急 {r['emerg'] / 1e4:5.1f} 万  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "code" / "outputs" / "q3_proto_rt.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 code/outputs/q3_proto_rt.csv")


if __name__ == "__main__":
    main()
