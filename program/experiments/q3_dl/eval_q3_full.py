# -*- coding: utf-8 -*-
"""Q3 DL 试验：全年（滚动重训 244 天）预测与费用对比。

运行（program/ 下）：uv run python experiments/q3_dl/eval_q3_full.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import numpy as np

import solve.q2 as q2
import solve.q3_proto as qp
from common import load_raw

OUT = _DIR / "outputs"


def metrics(pred, act):
    e = np.abs(pred - act)
    m = act >= 500
    return (float(e.mean()), float(e[m].mean()) if m.any() else float("nan"),
            float(np.sqrt(((pred - act) ** 2).mean())))


def run(ext, ov, lam, days):
    d2 = dict(ext)
    if ov is not None:
        d2["PV_OVERRIDE"] = ov
    tot = plan = adj = em = 0.0
    for d in days:
        r = qp.simulate_day_rt(d2, int(d), lam=lam)
        tot += r["total"]; plan += r["plan_cost"]
        adj += r["adjust_net"]; em += r["emerg"]
    return tot, plan, adj, em


def main():
    z = np.load(OUT / "dl_pred_full.npz")
    dl_days = z["days"]
    dl_pred = z["pred"]
    print(f"DL 全年预测：{len(dl_days)} 天（{dl_days[0]} ~ {dl_days[-1]}），"
          f"各块 val MAE {np.round(z['val_mae'], 1)}")

    data = load_raw()
    act = data["pv_act"][dl_days]
    ext = qp.load_extended()
    EVAL_DAYS = [int(d) for d in dl_days]

    fc0 = np.stack([q2._hour_to_slots(data["fc0"][int(d)]) for d in dl_days])
    hist = np.stack([qp.hist_forecast(ext, int(d)) for d in dl_days])
    combo = 0.5 * fc0 + 0.5 * hist

    print("\n===== 全年预测误差（244 天） =====")
    for name, p in (("官方 0:00", fc0), ("历史口E", hist), ("组合 0.5", combo), ("DL（滚动）", dl_pred)):
        m_all, m_day, rmse = metrics(p, act)
        print(f"  {name:<12} 全天 MAE {m_all:7.1f}  光照 {m_day:7.1f}  RMSE {rmse:7.1f} kW")

    months = np.array([int(data["dates"][int(d)][5:7]) for d in dl_days])
    print("\n月度 MAE / kW（全天）:")
    print("  月份   官方   历史E   DL")
    for mo in sorted(set(months.tolist())):
        mm = months == mo
        print(f"  {mo:>4}  {np.abs(fc0[mm] - act[mm]).mean():6.1f}  "
              f"{np.abs(hist[mm] - act[mm]).mean():6.1f}  {np.abs(dl_pred[mm] - act[mm]).mean():6.1f}")

    ov = {int(d): dl_pred[i] for i, d in enumerate(dl_days)}
    configs = [("官方 0:00", None, 1.0), ("历史口E", None, 0.0), ("组合 0.5", None, 0.5),
               ("DL", ov, 0.0), ("官方+DL λ=0.3", ov, 0.3), ("官方+DL λ=0.5", ov, 0.5)]
    rows = []
    print(f"\n===== 全年费用（244 天，实时执行 + 全开调整） =====")
    print(f"  {'配置':<16}{'总费用/万元':>12}{'计划/万元':>10}{'调整/万元':>10}{'紧急/万元':>10}")
    for name, o, lam in configs:
        tot, plan, adj, em = run(ext, o, lam, EVAL_DAYS)
        rows.append({"配置": name, "总费用/万元": round(tot / 1e4, 2),
                     "计划费/万元": round(plan / 1e4, 2),
                     "调整/万元": round(adj / 1e4, 2), "紧急/万元": round(em / 1e4, 2)})
        print(f"  {name:<16}{tot / 1e4:>12.2f}{plan / 1e4:>10.2f}{adj / 1e4:>10.2f}{em / 1e4:>10.2f}")
    base = rows[0]["总费用/万元"]
    print("\n相对官方 0:00：")
    for r in rows[1:]:
        print(f"  {r['配置']:<16}{100 * (r['总费用/万元'] / base - 1):+.3f}%")

    with open(OUT / "q3_cost_full.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"已保存 {OUT / 'q3_cost_full.json'}")


if __name__ == "__main__":
    main()
