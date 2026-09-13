"""求解内核：LP 模型、因果执行器、槽位映射与无前视残差池（不含文件 I/O）。

分层约定：
    core  ←  models  ←  flows  ←  顶层入口（q1~q4、q2_tune 等）
core 只依赖 ``solve.common``（物理常量与路径）。
"""
from solve.core.causal import (
    CAUSAL_POLICY_VERSION,
    exec_day_causal,
    exec_segment_causal,
    run_exec,
)
from solve.core.lp import (
    adjust_day,
    adjust_day_hedge,
    exec_day,
    exec_segment_hindsight,
    plan_day,
    plan_horizon,
)
from solve.core.residual import causal_residual_pool, forecast_at_publish, latest_forecast
from solve.core.slots import (
    _events,
    _fmt_time,
    _hour_to_slots,
    emergency_events,
    fc_slots,
    fmt_time,
    hour_to_slots,
)

__all__ = [
    "CAUSAL_POLICY_VERSION", "exec_day_causal", "exec_segment_causal", "run_exec",
    "plan_day", "plan_horizon", "exec_day", "exec_segment_hindsight",
    "adjust_day", "adjust_day_hedge",
    "causal_residual_pool", "latest_forecast", "forecast_at_publish",
    "hour_to_slots", "fc_slots", "fmt_time", "emergency_events",
    "_hour_to_slots", "_fmt_time", "_events",
]
