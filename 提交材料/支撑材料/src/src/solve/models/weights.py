"""凸权重工具：softmax 参数化、单纯形网格与权重序列平滑。"""
from __future__ import annotations

import numpy as np

from solve.common import N_DAY


# ===========================================================================
# 一、凸权重参数化
#     softmax:  w_i = exp(θ_i − max θ) / Σ_j exp(θ_j − max θ)
#               （减最大值仅用于数值稳定，不改变结果；输出非负且 Σw = 1）
#     单纯形网格（步长 h = 1/n）：
#               { (i/n, j/n, (n−i−j)/n) : i,j ≥ 0, i+j ≤ n }
#               组数 = (n+1)(n+2)/2
# ===========================================================================
def softmax(theta) -> np.ndarray:
    """logit 向量 → 凸权重（非负、和=1）。"""
    e = np.exp(theta - theta.max())
    return e / e.sum()


def simplex_grid(step: float) -> np.ndarray:
    """3 维凸权重网格（非负、和=1），组数 (n+1)(n+2)/2。"""
    n = int(round(1.0 / step))
    return np.array(
        [
            (i / n, j / n, (n - i - j) / n)
            for i in range(n + 1)
            for j in range(n + 1 - i)
        ]
    )


# ===========================================================================
# 二、权重序列平滑（参数混合）
#     u_D = β·u_new(D) + (1−β)·u_{D−1}
#     u_new(D) = softmax(TH[D−1][3:6])   （Q2E 在标定日 D−1 为次日给出的权重）
#     凸组合保持非负、和=1；β=1 退化为不平滑。
# ===========================================================================
def make_smooth_u(data: dict, beta: float) -> np.ndarray:
    """参数平滑（新旧混合）：u_D = β·u_new + (1−β)·u_{D−1}。

    u_new[D] = softmax(TH[D−1][3:6])（Q2E 在标定日 D−1 为次日给出的权重）；
    凸组合自动保持权重非负且和=1；β=1 退化为不平滑。
    """
    TH = data["TH"]
    U = np.zeros((N_DAY, 3))
    prev = None
    for D in range(N_DAY):
        j = max(D - 1, 13)
        unew = softmax(TH[j, 3:6])
        prev = unew if prev is None else beta * unew + (1.0 - beta) * prev
        U[D] = prev
    return U


# 兼容别名（旧代码以 _softmax 调用）
_softmax = softmax
