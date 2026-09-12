"""约束回代校验：功率平衡、储能动态、安全区间与变量非负。

# ===========================================================================
# 回代公式（逐日、逐槽最大残差）
#     功率平衡： P_t + x_t + d_t + e_t − L_t − c_t − s_t = 0
#     储能动态： E_t − E_{t-1} − η·c_t + d_t/η = 0
#     安全区间： E_min ≤ E_t ≤ E_max（1200 ≤ E ≤ 10800 kWh）
#     跨日衔接： E_end(d) = E_start(d+1)
# 残差应为浮点误差量级（~1e-12）；非零越界数（e/s < 0）必须为 0。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np

from solve.common import ETA, N_DAY


def verify(data: dict, x_plans, recs) -> dict:
    """Q2 全年逐日约束回代（口径 D/E 共用）。

    返回：平衡残差、储能动态残差、储电量最值与跨日衔接误差、负值计数。
    """
    load, pv_act = data["load"], data["pv_act"]
    bal, dyn = 0.0, 0.0
    Emin, Emax = 1e18, -1e18
    for d in range(N_DAY):
        r = recs[d]
        res = (pv_act[d] / 6.0 + r["x"] + r["d"] + r["e"]
               - load[d] / 6.0 - r["c"] - r["s"])
        bal = max(bal, float(np.abs(res).max()))
        E = np.concatenate([[r["E_start"]], r["E"]])
        dyn = max(dyn, float(np.abs(np.diff(E) - (ETA * r["c"] - r["d"] / ETA)).max()))
        Emin = min(Emin, float(r["E"].min()))
        Emax = max(Emax, float(r["E"].max()))
    cont = max(abs(recs[d]["E_end"] - recs[d + 1]["E_start"]) for d in range(N_DAY - 1))
    return {
        "功率平衡最大残差/kWh": bal,
        "储能动态最大残差/kWh": dyn,
        "储电量最小值/kWh": Emin,
        "储电量最大值/kWh": Emax,
        "跨日储电量衔接最大误差/kWh": float(cont),
        "紧急购电量负值数": int(sum((recs[d]["e"] < -1e-6).sum() for d in range(N_DAY))),
        "弃电负值数": int(sum((recs[d]["s"] < -1e-6).sum() for d in range(N_DAY))),
    }
