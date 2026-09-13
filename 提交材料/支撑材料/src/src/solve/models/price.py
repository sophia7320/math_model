"""Q4 电价预测模型：三源凸组合 + 滚动费用标定 + 动态修正。

# ===========================================================================
# 三源预测（对附件 4 的实时波动电价）：
#     P̂(D,t) = v1·P4(D−1,t) + v2·P4(D−7,t) + v3·P̄_typ(t)
#     其中 P4 为附件 4 逐槽电价，P̄_typ 为附件 1 典型日电价（逐槽均值），
#     v = (v1,v2,v3) 为凸权重（softmax 参数化或网格）。
#
# 滚动标定（无前视）：v*_D = argmin_{v∈网格} (1/|H_D|)·Σ_{k∈H_D} C_k(v)
#     H_D = [max(31, D−7), D)，C_k(v) 为日 k 按预测价计划、按真实价结算的费用；
# 动态修正（参数平滑）：v_D = β·v*_D + (1−β)·v_{D−1}，β = 0.1。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np

from solve.common import N_DAY, REPORT_START, ROOT

BETA = 0.1              # 电价权重的动态修正（新旧混合）
W_V = 7                 # 滚动标定窗口（天）


def forecast_price(D, v, p4, p_typ):
    """三源预测（kW 同量纲，元/kWh）；D 需要 ≥7。

    P̂ = v1·P4(D−1) + v2·P4(D−7) + v3·P̄_typ。
    """
    return (v[0] * p4[D - 1] + v[1] * p4[D - 7] + v[2] * p_typ)


def price_forecast_d(D: int, v: np.ndarray, p4: np.ndarray, p_typ: np.ndarray) -> np.ndarray:
    """目标日 D 的电价预测（D≥7）；D<7 退化为典型日。"""
    return forecast_price(D, v, p4, p_typ) if D >= 7 else p_typ


def price_forecast_next(D: int, v: np.ndarray, p4: np.ndarray, p_typ: np.ndarray) -> np.ndarray:
    """day D+1 的预测（仅用 day D 可得信息）：P(D) 未知，用 P(D−1) 替代。

    q4 年度流程专用（末端夹取到 N_DAY−1）。
    """
    d1 = min(D + 1, N_DAY - 1)
    return v[0] * p4[d1 - 2] + v[1] * p4[d1 - 7] + v[2] * p_typ


def forecast_price_next_asof(D, v, p4, p_typ):
    """在 D 日 0:00 预测 D+1 电价，不读取 P(D)（电价参数拟合/动态修正用，末端不夹取）。

    D<7 退化为典型日；P̂(D+1) = v1·P4(D−1) + v2·P4(D−6) + v3·P̄_typ。
    """
    if D < 7:
        return p_typ
    return v[0] * p4[D - 1] + v[1] * p4[D - 6] + v[2] * p_typ


def rolling_v_star() -> np.ndarray:
    """逐日电价三源权重 v*：窗口 W=7 费用标定的 argmin（不平滑，无前视）。

    v*_D = argmin_{v∈网格} (1/|H_D|)·Σ_{k∈H_D} C_k(v)，H_D = [max(31, D−7), D)。
    返回 (N_DAY, 3)；D < REPORT_START 的位置为占位零向量（调用方不取）。
    依赖 code/outputs/q4_price_fit_daily.npz（由 solve.q4_price_fit 生成）。
    """
    z = np.load(ROOT / "code" / "outputs" / "q4_price_fit_daily.npz")
    daily_costs = z["daily_costs"]
    grid = [tuple(v) for v in z["grid"]]
    seq = np.zeros((N_DAY, 3))
    for D in range(REPORT_START, N_DAY):
        hist = daily_costs[max(REPORT_START, D - W_V):D]
        # 预热首日窗口为空：退回纯典型日 (0,0,1)（与旧实现一致）
        seq[D] = (np.array(grid[int(np.argmin(hist.mean(axis=0)))])
                  if len(hist) else np.array([0.0, 0.0, 1.0]))
    return seq


def rolling_v_seq() -> np.ndarray:
    """逐日电价三源权重：滚动费用标定（W=7）+ 动态修正 β=0.1（无前视）。

    v_D = β·v*_D + (1−β)·v_{D−1}；D<31 预热用纯典型日 (0,0,1)。
    """
    v_star = rolling_v_star()
    seq = np.tile(np.array([0.0, 0.0, 1.0]), (N_DAY, 1))
    prev = None
    for D in range(REPORT_START, N_DAY):
        v_new = v_star[D]
        prev = v_new if prev is None else BETA * v_new + (1.0 - BETA) * prev
        seq[D] = prev
    return seq
