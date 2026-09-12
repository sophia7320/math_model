# -*- coding: utf-8 -*-
"""全年（244 天）三源组合预测：官方 + 历史 + DL，无前视滚动权重。

每 30 天用最近 30 天重估三源权重（网格 0.05），评估后 30 天。
"""
from __future__ import annotations

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


def main():
    z = np.load(OUT / "dl_pred_full.npz")
    dl_days, dl_pred = z["days"], z["pred"]
    data = load_raw()
    act = data["pv_act"][dl_days]
    ext = qp.load_extended()
    fc0 = np.stack([q2._hour_to_slots(data["fc0"][int(d)]) for d in dl_days])
    hist = np.stack([qp.hist_forecast(ext, int(d)) for d in dl_days])

    n = len(dl_days)
    combo3 = np.zeros_like(dl_pred)
    weights = []
    step = 30
    for t in range(0, n, step):
        i0 = max(0, t - step)
        j0, j1 = t, min(n, t + step)
        # 拟合（用 [i0, t)，初始段用 [0, t+step) 的前半？首段无历史 → 用等权 0.5/0.25/0.25 之外的做法：
        fit = slice(i0, j0) if j0 > i0 else slice(0, j1)
        best = None
        for w1 in np.arange(0, 1.0001, 0.05):
            for w2 in np.arange(0, 1.0001 - w1 + 1e-9, 0.05):
                w3 = 1 - w1 - w2
                mae = np.abs(w1 * fc0[fit] + w2 * hist[fit] + w3 * dl_pred[fit] - act[fit]).mean()
                if best is None or mae < best[0]:
                    best = (mae, w1, w2, w3)
        _, w1, w2, w3 = best
        combo3[j0:j1] = w1 * fc0[j0:j1] + w2 * hist[j0:j1] + w3 * dl_pred[j0:j1]
        weights.append((int(dl_days[j0]), round(w1, 2), round(w2, 2), round(w3, 2)))

    print("三源组合滚动权重（日期, 官方, 历史, DL）：")
    for w in weights:
        print("  ", w)

    def m(p):
        return float(np.abs(p - act).mean())

    print(f"\n全年 MAE：官方 {m(fc0):.1f}  历史 {m(hist):.1f}  固定0.5二源 {m(0.5 * fc0 + 0.5 * hist):.1f}  "
          f"DL {m(dl_pred):.1f}  滚动三源 {m(combo3):.1f} kW")

    # 费用评估（三源组合 vs 二源组合）
    ov = {int(d): combo3[i] for i, d in enumerate(dl_days)}
    d2 = dict(ext)
    d2["PV_OVERRIDE"] = ov
    tot = plan = adj = em = 0.0
    for d in dl_days:
        r = qp.simulate_day_rt(d2, int(d), lam=0.0)
        tot += r["total"]; plan += r["plan_cost"]
        adj += r["adjust_net"]; em += r["emerg"]
    print(f"\n三源组合费用（244 天）：总 {tot / 1e4:.2f} 万 "
          f"（计划 {plan / 1e4:.2f} + 调整 {adj / 1e4:.2f} + 紧急 {em / 1e4:.2f}）")
    print("对照：二源组合 0.5 = 1012.22 万；官方 = 1020.27 万")


if __name__ == "__main__":
    main()
