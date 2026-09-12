"""附件 3 预报误差模型（相对 (时刻,月) 期望出力；见 C题_预报误差分析.md）。

系统偏差 μ(k) = 0.0050 − 0.00128k；误差尺度 σ(k) = 0.0006 + 0.0084k。
"""
from __future__ import annotations

import numpy as np


# ===========================================================================
# 误差模型（相对 (时刻,月) 期望出力的乘性误差）
#
#     μ(k) = 0.0050 − 0.00128·k                  （系统偏差，k = 1..24）
#     σ(k) = 0.0006 + 0.0084·k                   （误差尺度）
#
# 标准化误差日块：
#     z_dk = [ (F_dk − A_dk)/T_mk − μ(k) ] / σ(k)     （仅 T_mk > 100 kW 的槽）
# 情景还原（蒙特卡洛/对冲采样）：
#     P̂_dk = F_dk + T_mk·( μ(k) + σ(k)·z_dk )，再裁剪 ≥ 0
#
# 其中 F 为附件 3 的 0:00 预报、A 为附件 2 实际、T_mk 为典型日 (时刻,月) 均值。
# 保留整日误差块（含日内相关与左尾）是 Q2 概率扩展的关键口径。
# ===========================================================================
def mu(k):
    """系统偏差 μ(k) = 0.0050 − 0.00128k。"""
    return 0.0050 - 0.00128 * np.asarray(k, dtype=float)


def sigma(k):
    """误差尺度 σ(k) = 0.0006 + 0.0084k。"""
    return 0.0006 + 0.0084 * np.asarray(k, dtype=float)


def scenario_hourly(fc0_row, month, z_row, typ_hm) -> np.ndarray:
    """由标准化误差块生成情景小时序列（裁剪到 ≥0）。

    P̂_k = F_k − T_k·(μ_k + σ_k·z_k)，其中 k = 1..24（整点口径）。
    注意 err = 预报 − 实际，因此实际情景 = 预报 − err。
    """
    ks = np.arange(1, 25)
    typ = typ_hm[1:25, month]
    err = typ * (mu(ks) + sigma(ks) * z_row)
    return np.clip(fc0_row - err, 0.0, None)


def scenario_from_residual(fc0_row, residual_row) -> np.ndarray:
    """由原始残差块生成小时实际情景；残差定义为"预报 − 实际"。"""
    return np.clip(np.asarray(fc0_row, float) - np.asarray(residual_row, float),
                   0.0, None)


# 兼容别名（旧调用点）
_mu = mu
_sigma = sigma
_scenario_hourly = scenario_hourly
