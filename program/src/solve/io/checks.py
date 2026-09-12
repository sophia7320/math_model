"""约束回代校验：功率平衡、储能动态、安全区间与变量非负。

# ===========================================================================
# 回代公式（逐日、逐槽最大残差）
#     功率平衡： P_t + x_t + d_t + e_t − L_t − c_t − s_t = 0
#     储能动态： E_t − E_{t-1} − η·c_t + d_t/η = 0
#     安全区间： E_min ≤ E_t ≤ E_max（1200 ≤ E ≤ 10800 kWh）
#     充放互斥： 同槽 (c_t>0 & d_t>0) 次数应为 0
#     跨日衔接： E_end(d) = E_start(d+1)（Q2 日循环口径）
#     场景无前视： scenario_max_day < D（Q3/Q4 对冲场景池）
# 残差应为浮点误差量级（~1e-12）；负值/越界计数必须为 0。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np

from solve.common import E_MAX, E_MIN, ETA, N_DAY, REPORT_START


def _scan_records(recs, load, pv_act, *, x_key="x", start=0, end=None,
                  cross_day=False) -> dict:
    """逐日扫描公共指标（供 Q2/Q4 两种输出格式复用）。

    recs : 序列；元素含 D（可选，缺省用下标）、x_key、c/d/s/e/E、E_start；
           跨日检查需要 E_end。
    start: 起算日；q2 口径 D 传 0（1 月预热也回代），Q4 传 REPORT_START。
    返回 dict：bal/dyn/emin/emax/overlap/cross/neg_e/neg_s/future_viol。
    """
    end = len(recs) if end is None else end
    bal = dyn = 0.0
    emin, emax = np.inf, -np.inf
    overlap = 0
    neg_e = neg_s = 0
    future_viol = 0
    cross = 0.0
    prev = None
    for i in range(end):
        r = recs[i]
        D = int(r["D"]) if isinstance(r, dict) and "D" in r else i
        if isinstance(r, dict) and r.get("scenario_max_day", -1) >= D and D >= start:
            future_viol += 1
        if D < start:
            prev = None
            continue
        x = r[x_key]
        # ① 功率平衡残差
        res = pv_act[D] / 6.0 + x + r["d"] + r["e"] - load[D] / 6.0 - r["c"] - r["s"]
        bal = max(bal, float(np.abs(res).max()))
        # ② 储能动态残差
        E = np.concatenate([[r["E_start"]], r["E"]])
        dyn = max(dyn, float(np.abs(np.diff(E) - (ETA * r["c"] - r["d"] / ETA)).max()))
        emin = min(emin, float(r["E"].min()))
        emax = max(emax, float(r["E"].max()))
        # ③ 同槽充放计数与负值计数
        overlap += int(np.sum((r["c"] > 1e-7) & (r["d"] > 1e-7)))
        neg_e += int(np.sum(r["e"] < -1e-6))
        neg_s += int(np.sum(r["s"] < -1e-6))
        # ④ 跨日衔接（仅当记录含 E_end）
        if cross_day and prev is not None and "E_end" in prev:
            cross = max(cross, abs(float(prev["E_end"]) - float(r["E_start"])))
        prev = r
    return {
        "bal": bal, "dyn": dyn, "emin": emin, "emax": emax,
        "overlap": overlap, "cross": float(cross),
        "neg_e": neg_e, "neg_s": neg_s, "future_viol": future_viol,
    }


def verify(data: dict, x_plans, recs) -> dict:
    """Q2 全年逐日约束回代（口径 D/E 共用）。

    返回：平衡残差、储能动态残差、储电量最值与跨日衔接误差、负值计数。
    """
    s = _scan_records(recs, data["load"], data["pv_act"], x_key="x",
                      start=0, end=N_DAY, cross_day=True)
    return {
        "功率平衡最大残差/kWh": s["bal"],
        "储能动态最大残差/kWh": s["dyn"],
        "储电量最小值/kWh": s["emin"],
        "储电量最大值/kWh": s["emax"],
        "跨日储电量衔接最大误差/kWh": s["cross"],
        "紧急购电量负值数": s["neg_e"],
        "弃电负值数": s["neg_s"],
    }


def verify_records(recs, data: dict, *, x_key="x", scenario=False) -> dict:
    """Q4 通用约束回代（Q2 层 x/x 或 Q3 层 x_final/x_plan）。

    返回：平衡/动态残差、储电量最值、区间检查、同槽充放次数；
    scenario=True 时附加"场景池前视违规天数"。
    """
    s = _scan_records(recs, data["load"], data["pv_act"], x_key=x_key,
                      start=REPORT_START, end=len(recs))
    out = {
        "功率平衡最大残差/kWh": s["bal"],
        "储能动态最大残差/kWh": s["dyn"],
        "储电量最小值/kWh": s["emin"],
        "储电量最大值/kWh": s["emax"],
        "约束区间检查": bool(s["emin"] >= E_MIN - 1e-6 and s["emax"] <= E_MAX + 1e-6),
        "同槽同时充放次数": s["overlap"],
    }
    if scenario:
        out["场景池前视违规天数"] = s["future_viol"]
    return out
