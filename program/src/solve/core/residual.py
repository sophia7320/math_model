"""无前视残差池与"最新可用预报"（Q3/Q4 对冲与场景采样共用）。"""
from __future__ import annotations

import numpy as np

from solve.common import T
from solve.core.slots import fc_slots, hour_to_slots


def latest_forecast(data: dict, D: int) -> np.ndarray:
    """当天"最新可用预报"（144 槽 kW）：按段取 0:00/6:00/12:00/18:00 发布值。"""
    v = np.empty(T)
    v[:36] = hour_to_slots(data["fc0"][D])[:36]
    v[36:72] = fc_slots(data["fc6"][D], 6)[36:72]
    v[72:108] = fc_slots(data["fc12"][D], 12)[72:108]
    v[108:] = fc_slots(data["fc18"][D], 18)[108:]
    return v


def causal_residual_pool(data: dict, D: int, min_same_month: int = 14,
                         lookback: int = 90) -> np.ndarray:
    """构造目标日 D 的历史残差池，严格保证所有日序号小于 D（无前视）。

    同月历史达到 ``min_same_month`` 天时优先使用；新月初样本不足时回退到
    最近 ``lookback`` 天。从第 15 天起池化，避免历史预报 d-14 索引越界。
    """
    if D <= 14:
        raise ValueError("残差池至少需要 14 天预热数据")
    months = np.asarray(data["months"])
    past = np.arange(max(14, D - lookback), D, dtype=int)
    same = past[months[past] == months[D]]
    pool = same if len(same) >= min_same_month else past
    if not len(pool) or int(pool.max()) >= D:
        raise RuntimeError(f"残差池因果性校验失败：D={D}, pool={pool.tolist()}")
    return pool
