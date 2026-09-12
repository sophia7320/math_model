"""Q3 原型第五批：调整层组合预报修正（消除"信息退化"）。

发现：0:00 计划用组合预测（较准），调整层却用纯官方预报（较不准）→ 调整会把更准的
计划改坏（实时模式下 λ=0.5 的"仅 6:00"反而比无调整贵）。本实验让调整层也用同一组合权重。

运行（program/ 下）：uv run python -m solve.experiments.q3_proto_adjmix
"""
from __future__ import annotations

import time

import pandas as pd

from solve import q3_proto as qp
from solve.common import ROOT

DAYS = list(range(31, 365))


def total(data, lam, **kw):
    return float(sum(qp.simulate_day_rt(data, D, lam, **kw)["total"] for D in DAYS))


def main():
    qp.pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()
    jobs = []
    for lam, combos in [(0.5, [("无调整", ()), ("全开", (6, 12, 18)), ("12+18", (12, 18)), ("仅6", (6,))]),
                        (0.3, [("全开", (6, 12, 18))])]:
        for name, c in combos:
            jobs.append((lam, name, c))

    rows = []
    for lam, name, c in jobs:
        for adj_lam, tag in ((None, "官方"), (lam, "组合")):
            if not c and tag == "组合":
                continue
            t0 = time.time()
            tot = total(data, lam, adj_hours=c, adj_lam=adj_lam)
            rows.append({"λ": lam, "调整点": name, "调整层": tag,
                         "总费用/万元": round(tot / 1e4, 1)})
            print(f"  λ={lam} {name:<6} 调整层={tag}  {tot / 1e4:8.1f} 万  ({time.time() - t0:.0f}s)")
    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q3_proto_adjmix.csv",
                              index=False, encoding="utf-8-sig")
    print("已保存 code/outputs/q3_proto_adjmix.csv")


if __name__ == "__main__":
    main()
