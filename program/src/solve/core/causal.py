"""逐槽因果执行器：每槽只读取当前已实现的负荷与光伏，不使用未来实际值。

正式口径（2026-09-12 修订）为 ``exec_segment_causal`` / ``exec_day_causal``；
事后 LP（``solve.core.lp.exec_day`` 等）只能作为理论下界对照。
修改执行策略时必须同步更新 ``CAUSAL_POLICY_VERSION``（参与 Q2E 缓存键）。
"""
from __future__ import annotations

import numpy as np

from solve.common import E_MAX, E_MIN, ETA, P_MAX_E

CAUSAL_POLICY_VERSION = "greedy-reachable-v1"   # 因果执行策略版本（参与缓存键）


# ===========================================================================
# 逐槽因果执行算法（贪心 + 终端可达性投影）
#
# 记槽 i 的净需求  net_i = L_i − P_i − x_i            （>0 缺口，<0 富余）
# 单槽功率极限（折算为 kWh/槽）：
#     最大充电量  up   = η·P̄
#     最大放电量  down = P̄/η
# 自然平衡所需的 SOC 变化：
#     Δ_balance = −net_i/η        （缺口 → 放电，E 下降）
#     Δ_balance = −η·net_i        （富余 → 充电，E 上升）
#
# 末段目标与执行策略（EXEC_POLICY）：
#   - e_end 给定（"target"）：把下一时刻 SOC 投影到仍能到达 e_end 的区间
#         lo_i = max(E_min, E_now − down, e_end − remain·up)
#         hi_i = min(E_max, E_now + up, e_end + remain·down)
#   - e_end=None（"free"，统一口径）或 no_purchase_charge=True：
#         仅受容量/功率约束 lo_i = max(E_min, E_now−down), hi_i = min(E_max, E_now+up)
#     free 策略不迫使"紧急购电充能/扣留放电"，与 dayend 同值且更简洁
#     （见 reports/执行器段末目标实验.md）。
#     E_{i+1} = clip(E_now + Δ_balance, lo_i, hi_i)      remain = m−i−1
#
# 由 E_{i+1} 反推充放量：
#     Δ ≥ 0:  c_i = Δ/η            （充电）
#     Δ < 0:  d_i = −Δ·η           （放电）
# 功率平衡残差转为紧急购电或弃光：
#     gap_i = L_i + c_i − P_i − x_i − d_i
#     gap_i ≥ 0 → e_i = gap_i（紧急购电）；gap_i < 0 → s_i = −gap_i（弃光）
# ===========================================================================
def exec_segment_causal(load_kwh, pv_kwh, x_commit, e_start, e_end,
                        no_purchase_charge: bool = False):
    """逐槽因果执行：每槽只使用当前已实现的负荷和光伏，不读取未来实际值。

    策略：先用储能吸收当前富余或填补当前缺口，再把下一时刻 SOC 投影到
    "仍能在剩余时段到达 e_end"的可达区间；无法被储能覆盖的缺口记为紧急购电，
    富余且无法消纳的部分记为弃电。

    可选策略开关（2026-09-13 增设，供"段末目标机制"对照实验）：
    - ``e_end=None``：完全自由末端（仅受容量/功率约束，不做可达性投影）；
    - ``no_purchase_charge=True``：禁止"为充能而紧急购电"——SOC 不得被抬升到
      自然平衡点之上，段末目标允许偏离（偏差在返回值中报告）。
    参数均为长度 m 的 kWh/槽序列；返回 c/d/s/E/e（另附诊断量）。
    """
    load_kwh = np.asarray(load_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    x_commit = np.asarray(x_commit, dtype=float)
    if not (load_kwh.ndim == pv_kwh.ndim == x_commit.ndim == 1):
        raise ValueError("因果执行输入必须是一维序列")
    if not (len(load_kwh) == len(pv_kwh) == len(x_commit)):
        raise ValueError("因果执行的负荷、光伏与购电序列必须等长")
    if not (E_MIN <= e_start <= E_MAX):
        raise ValueError("因果执行的初始 SOC 超出安全区间")
    if e_end is not None and not (E_MIN <= e_end <= E_MAX):
        raise ValueError("因果执行的末端 SOC 超出安全区间")

    m = len(load_kwh)
    c = np.zeros(m)
    d = np.zeros(m)
    spill = np.zeros(m)
    E = np.zeros(m)
    emerg = np.zeros(m)
    e_now = float(e_start)
    max_up = ETA * P_MAX_E          # 单槽最大充电量（kWh）
    max_down = P_MAX_E / ETA        # 单槽最大放电量（kWh）
    tol = 1e-8

    for i in range(m):
        remain = m - i - 1
        if e_end is None or no_purchase_charge:
            # 自由末端（或禁止购电充能）：仅受容量与功率约束
            lo = max(E_MIN, e_now - max_down)
            hi = min(E_MAX, e_now + max_up)
        else:
            # E_next 既要能由当前 SOC 一步到达，也要能在剩余槽回到 e_end
            lo = max(E_MIN, e_now - max_down, float(e_end) - remain * max_up)
            hi = min(E_MAX, e_now + max_up, float(e_end) + remain * max_down)
        if lo > hi + tol:
            raise RuntimeError(
                f"终端 SOC 不可达：i={i}, interval=[{lo:.6f}, {hi:.6f}]"
            )

        net = load_kwh[i] - pv_kwh[i] - x_commit[i]
        # 使当前槽供需平衡的自然 SOC 变化；若会破坏终端可达性则投影
        delta_balance = -net / ETA if net >= 0.0 else -ETA * net
        e_next = float(np.clip(e_now + delta_balance, lo, hi))
        delta = e_next - e_now
        if delta >= 0.0:
            c[i] = delta / ETA
        else:
            d[i] = -delta * ETA

        gap = load_kwh[i] + c[i] - pv_kwh[i] - x_commit[i] - d[i]
        if gap >= 0.0:
            emerg[i] = gap
        else:
            spill[i] = -gap
        E[i] = e_next
        e_now = e_next

    term_err = abs(E[-1] - float(e_end)) if e_end is not None else 0.0
    if e_end is not None and m and not no_purchase_charge and term_err > 1e-6:
        raise RuntimeError(f"因果执行末端 SOC 偏差 {E[-1] - e_end:.3e} kWh")
    return {"c": c, "d": d, "s": spill, "E": E, "e": emerg,
            "terminal_error_kwh": float(term_err)}


def exec_day_causal(price, load_kwh, pv_kwh, x_plan, e_start, e_end=None):
    """旧版（日循环）因果执行包装：e_end=None 时终端取 e_start。

    正式口径（``consistency.EXEC_POLICY="free"``）已改为无段末硬目标：
    直接调用 ``exec_segment_causal(..., e_end=None)``；本函数仅供历史对照。
    """
    del price
    return exec_segment_causal(
        load_kwh, pv_kwh, x_plan, e_start, e_start if e_end is None else e_end
    )


def run_exec(policy: str, load, pv, x, e0, e1, last: bool = False):
    """按执行策略调用逐槽因果执行器（段末目标机制对照，consistency 规范扩展）。

    policy:
      "target" 段末目标硬跟踪（原口径，历史对照）；
      "free"   无段末目标（仅容量/功率约束；统一口径 EXEC_POLICY）；
      "dayend" 仅最后一段跟踪段末目标，中间段自由（与 free 同值，见探针）。
    """
    if policy == "free" or (policy == "dayend" and not last):
        return exec_segment_causal(load, pv, x, e0, None)
    return exec_segment_causal(load, pv, x, e0, e1)
