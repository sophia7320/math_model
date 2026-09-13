"""时间-槽位映射与紧急购电事件（相位敏感的公共实现）。

关键口径：``hour_to_slots`` 只适用于 0:00 发布的预报（24 整点 → 144 槽）；
6:00/12:00/18:00 发布的预报必须用 ``fc_slots`` 做相位对齐，
否则整体错位 6/12/18 小时（费用翻倍级错误）。
"""
from __future__ import annotations

import numpy as np

from solve.common import T


def hour_to_slots(y24: np.ndarray) -> np.ndarray:
    """24 个整点值 → 144 个 10 分钟槽（线性插值到槽中心）。

    插值形式：以 (0, 0), (1, y1), ..., (24, y24) 为节点做分段线性插值，
    在槽中心 τ_k = (k + 0.5)/6 处取值，k = 0..143。
    """
    fp = np.concatenate([[0.0], np.asarray(y24, dtype=float)])
    centers = (np.arange(T) + 0.5) / 6.0
    return np.interp(centers, np.arange(0, 25, dtype=float), fp)


def fc_slots(fc24: np.ndarray, publish: int) -> np.ndarray:
    """附件 3 预报行 → 144 槽（kW），相位对齐（预报 k 小时 = 发布时刻 + k 时）。

    构造 0..24 整点的值序列 hv（发布时刻及以前用第一个预报值填充），
    再线性插值到 144 个槽中心。调整 LP 只使用 [publish*6, 144) 段。

    相位规则（关键）：
        hv[0..publish] = y[0]                     （发布时刻已过/当前，用首值）
        hv[publish + k] = y[k-1],  k = 1..24−publish
    即预报第 k 个值对应"发布时刻 + k 小时"，与 hour_to_slots 的 0:00 口径区分。
    """
    y = np.asarray(fc24, dtype=float)
    hv = np.empty(25)
    hv[: publish + 1] = y[0]
    hv[publish + 1: 25] = y[: 24 - publish]
    centers = (np.arange(T) + 0.5) / 6.0
    return np.interp(centers, np.arange(25, dtype=float), hv)


def fmt_time(minute: int) -> str:
    """分钟数 → 时间标签（1440 → 0:00+1）。"""
    if minute >= 1440:
        return "0:00+1"
    return f"{minute // 60}:{minute % 60:02d}"


def emergency_events(e_slots: np.ndarray, tol: float = 1e-3):
    """逐槽紧急购电合并为事件区间：[(起槽, 止槽, kWh), ...]。

    合并规则：相邻槽 e_t > tol 时归为同一事件，区间电量 = Σ e_t。
    时间标签转换见 :func:`fmt_time`（起槽 ×10 分钟 → "H:MM"）。
    """
    out = []
    t = 0
    while t < T:
        if e_slots[t] > tol:
            j = t
            while j + 1 < T and e_slots[j + 1] > tol:
                j += 1
            out.append((t, j, float(e_slots[t:j + 1].sum())))
            t = j + 1
        else:
            t += 1
    return out


# 兼容别名（旧脚本按私有名调用）
_hour_to_slots = hour_to_slots
_fmt_time = fmt_time
_events = emergency_events
