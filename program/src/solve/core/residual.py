"""无前视残差池与"最新可用预报"（Q3/Q4 对冲与场景采样共用）。"""
from __future__ import annotations

import numpy as np

from solve import consistency as cs
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


def forecast_at_publish(data: dict, D: int, publish: int) -> np.ndarray:
    """返回历史日 ``D`` 在指定发布时刻可得到的整条当日预报（144 槽，kW）。

    场景残差必须与当前决策的信息集一致。例如 6:00 调整只能使用历史 6:00
    发布预报的误差，不能把历史 12:00/18:00 的更新拼接进剩余时段。
    """
    if publish == 0:
        return hour_to_slots(data["fc0"][D])
    if publish not in (6, 12, 18):
        raise ValueError(f"不支持的预报发布时刻：{publish}")
    return fc_slots(data[f"fc{publish}"][D], publish)


def causal_residual_pool(data: dict, D: int, min_same_month: int = 14,
                         lookback: int = 90) -> np.ndarray:
    """构造目标日 D 的历史残差池，严格保证所有日序号小于 D（无前视）。

    统一实现见 ``consistency.scenario_indices``（同月优先、不足回看）。
    """
    if D <= 14:
        raise ValueError("残差池至少需要 14 天预热数据")
    return cs.scenario_indices(np.asarray(data["months"]), D,
                               pool_min_same_month=min_same_month, lookback=lookback)
