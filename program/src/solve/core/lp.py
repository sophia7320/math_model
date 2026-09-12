"""LP 求解器：日前/多日计划、事后执行下界、Q3 调整与场景对冲。

变量块口径（每个时段 t 一类变量，按块顺序拼接）：
- 计划/多日计划 ``plan_day``/``plan_horizon``：
      [ x 购电 | c 充电 | d 放电 | s 弃光 | E 储电量 ]
- 事后执行 ``exec_day``/``exec_segment_hindsight``：
      [ c 充电 | d 放电 | s 弃光 | E 储电量 | e 紧急购电 ]
- 调整 ``adjust_day``/``adjust_day_hedge``：
      [ adj 购电 | c | d | s | E | y 结算基准 ]

统一实现：
- ``plan_lp`` 是计划/多日计划 LP 的唯一构造（由 ``plan_day``（H=T 日循环）
  与 ``plan_horizon``（H 可变、跨日）两个薄包装复用）；
- ``_exec_lp`` 是事后执行 LP 的唯一构造（``exec_day`` 段长 T、日循环，
  ``exec_segment_hindsight`` 任意段与终端）。

HiGHS 对矩阵顺序敏感：本模块的矩阵构造在重构期间由金标测试锁定数值
（plan_day ≡ plan_horizon(H=T) ≡ 旧 q1.build_lp(eps=0)；
 exec_day ≡ exec_segment_hindsight(t0=0, e_end=e_start)）。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

import program as pm
from solve.common import (
    E0,
    E_MAX,
    E_MIN,
    EMERG_MULT,
    EPS_TH,
    EPS_THROUGHPUT,
    ETA,
    P_MAX_E,
    T,
)


# ===========================================================================
# 一、计划 LP（日前 / 多日滚动）
#
#     min  Σ_t p_t·x_t + ε·Σ_t (c_t + d_t)            （购电费 + 充放唯一化罚项）
#     s.t. P_t + x_t + d_t = L_t + c_t + s_t          （功率平衡，∀t）
#          E_t = E_{t-1} + η·c_t − d_t/η              （储能动态，∀t）
#          0 ≤ x_t,  0 ≤ c_t,d_t ≤ P̄,  0 ≤ s_t ≤ P_t  （边界；s 为弃光松弛）
#          E_min ≤ E_t ≤ E_max                        （储电量安全区间）
#          E_0 = e_start,  E_{H-1} = e_terminal       （跨日/日循环终端条件）
#
#     ε=0 时与"纯购电费最小"完全一致（计划 LP）；ε>0 在等费用平面上
#     唯一化充放方案，供口径 E 标定与 Q3/Q4 使用。
# ===========================================================================
def plan_lp(price_h, load_h, pv_h, e_start, e_terminal, eps=0.0):
    """统一计划 LP：返回完整解字典与 OptResult。

    解字典键：x 购电、c 充、d 放、s 弃光、E 储电量（均为长度 H 数组）。
    """
    H = len(price_h)
    n = 5 * H
    c_obj = np.zeros(n)
    # 目标函数：仅购电量 x 按电价计费；c/d 加 ε 罚项抑制无意义充放
    c_obj[0:H] = price_h
    c_obj[H:3 * H] = eps

    # ---- 等式约束行布局：[0, H) 功率平衡；[H, 2H) 储能动态；第 2H 行终端条件 ----
    A_eq = lil_matrix((2 * H + 1, n))
    b_eq = np.zeros(2 * H + 1)
    for t in range(H):
        # 功率平衡：+x_t + d_t − c_t − s_t = L_t − P_t
        A_eq[t, 0 * H + t] = 1.0
        A_eq[t, 1 * H + t] = -1.0
        A_eq[t, 2 * H + t] = 1.0
        A_eq[t, 3 * H + t] = -1.0
        b_eq[t] = load_h[t] - pv_h[t]
    for t in range(H):
        # 储能动态：E_t − E_{t-1} − η·c_t + d_t/η = 0（t=0 时右端为 e_start）
        r = H + t
        A_eq[r, 4 * H + t] = 1.0
        if t:
            A_eq[r, 4 * H + t - 1] = -1.0
        A_eq[r, 1 * H + t] = -ETA
        A_eq[r, 2 * H + t] = 1.0 / ETA
        b_eq[r] = e_start if t == 0 else 0.0
    # 终端条件：E_{H-1} = e_terminal（日循环或跨日循环）
    A_eq[2 * H, 4 * H + H - 1] = 1.0
    b_eq[2 * H] = e_terminal

    # 变量边界：x∈[0,∞)，c,d∈[0,P̄]，s∈[0,该槽可用光伏]，E∈[E_min,E_max]
    bounds = (
        [(0.0, None)] * H
        + [(0.0, P_MAX_E)] * H
        + [(0.0, P_MAX_E)] * H
        + [(0.0, float(v)) for v in pv_h]
        + [(E_MIN, E_MAX)] * H
    )
    res = pm.optimize.solve_lp(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"计划 LP 失败：{res.message}")
    sol = {
        "x": res.x[0:H],
        "c": res.x[H:2 * H],
        "d": res.x[2 * H:3 * H],
        "s": res.x[3 * H:4 * H],
        "E": res.x[4 * H:5 * H],
    }
    return sol, res


def plan_day(price, load_kwh, pv_kwh, e_start, eps=0.0):
    """当日计划 LP：min Σp·x + eps·Σ(c+d)；储能循环 E(0)=E(24)=e_start。

    eps > 0 为极小正则项，用于在等价位内唯一化购电/充放方案（口径 E 标定）；
    默认 0 与原有口径完全一致。返回 (x, E, res)。
    """
    sol, res = plan_lp(price, load_kwh, pv_kwh, e_start, e_start, eps)
    return sol["x"], sol["E"], res


def plan_horizon(price_h, load_h, pv_h, e_start, e_terminal, eps=1e-3):
    """多日计划 LP：min Σp·x + eps·Σ(c+d)；储能跨日连续 E(0)=e_start、E(H)=e_terminal。"""
    sol, _res = plan_lp(price_h, load_h, pv_h, e_start, e_terminal, eps)
    return sol["x"], sol["E"]


# ===========================================================================
# 二、事后执行下界 LP（不得用于正式结果）
#
#     min  ε·Σ(c_t + d_t) + Σ_t 5·p_t·e_t
#     s.t. P_t + d_t + e_t = L_t + c_t + s_t + x_t   （x 已承诺 take-or-pay）
#          E_t = E_{t-1} + η·c_t − d_t/η
#          E_end = e_end                              （日循环或段末终端）
#
#     仅作"全天/整段信息已知"的理论下界；正式口径为逐槽因果执行（core.causal）。
# ===========================================================================
def _exec_lp(price, load_kwh, pv_kwh, x_seg, e_start, e_end, t0, eps):
    """统一事后执行 LP：对段 [t0, t0+m) 重优化储能；返回 c/d/s/E/e 字典。"""
    m = len(x_seg)
    n = 5 * m

    def I(b, i):
        return b * m + i

    c_obj = np.zeros(n)
    # 目标：c/d 唯一化罚项；紧急购电按 5 倍电价惩罚
    c_obj[0:m] = eps
    c_obj[m:2 * m] = eps
    c_obj[4 * m:5 * m] = EMERG_MULT * price[t0:t0 + m]

    rows = 2 * m + 1
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    for i in range(m):
        # 供需平衡：−c_i + d_i − s_i + e_i = L − P − x_seg（缺口由 e 或放电补齐）
        A_eq[i, I(0, i)] = -1.0
        A_eq[i, I(1, i)] = 1.0
        A_eq[i, I(2, i)] = -1.0
        A_eq[i, I(4, i)] = 1.0
        b_eq[i] = load_kwh[t0 + i] - pv_kwh[t0 + i] - x_seg[i]

        # 储能动态：E_i − E_{i-1} − η·c_i + d_i/η = 0
        r = m + i
        A_eq[r, I(3, i)] = 1.0
        if i:
            A_eq[r, I(3, i - 1)] = -1.0
        A_eq[r, I(0, i)] = -ETA
        A_eq[r, I(1, i)] = 1.0 / ETA
        b_eq[r] = e_start if i == 0 else 0.0
    # 段末终端：E_{m-1} = e_end
    A_eq[2 * m, I(3, m - 1)] = 1.0
    b_eq[2 * m] = e_end

    bounds = (
        [(0.0, P_MAX_E)] * m
        + [(0.0, P_MAX_E)] * m
        + [(0.0, None)] * m
        + [(E_MIN, E_MAX)] * m
        + [(0.0, None)] * m
    )
    res = pm.optimize.solve_lp(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"执行 LP 失败（t0={t0}）：{res.message}")
    return {
        "c": res.x[0:m], "d": res.x[m:2 * m], "s": res.x[2 * m:3 * m],
        "E": res.x[3 * m:4 * m], "e": res.x[4 * m:5 * m],
    }


def exec_day(price, load_kwh, pv_kwh, x_plan, e_start):
    """事后执行下界：x 已承诺，使用全天实际曲线重优化储能。

    本函数保留给历史对照和理论下界，**不得**标记为"实时执行"；
    正式因果口径使用 :func:`solve.core.causal.exec_day_causal` /
    :func:`solve.core.causal.exec_segment_causal`。
    """
    return _exec_lp(price, load_kwh, pv_kwh, x_plan, e_start, e_start, 0,
                    EPS_THROUGHPUT)


def exec_segment_hindsight(price, load_kwh, pv_kwh, x_seg, e_start, e_end, t0, eps=EPS_TH):
    """事后段执行下界：使用整段实际功率重优化储能。

    仅用于理论下界对照，**不得**进入实时口径；正式因果执行使用
    :func:`solve.core.causal.exec_segment_causal`（逐槽只读当前已实现值）。
    """
    return _exec_lp(price, load_kwh, pv_kwh, x_seg, e_start, e_end, t0, eps)


# ===========================================================================
# 三、Q3 调整 LP
#
#     分段结算规则（题目）：调低部分按 50%、调高部分按 150% 结算，
#     即  费用 = p·adj + 0.5·p·|x_plan − adj|
#     LP 线性化（用 y 作分段下包络，目标含 −p·y）：
#         y_t ≤ x_plan_t,  y_t ≤ adj_t,   y_t ≥ 0
#         Σ [1.5·p_t·adj_t − p_t·y_t] + ε·Σ(c_t + d_t)
#     （最优时 y_t = min(x_plan_t, adj_t)，与原分段费用等价）
#
#     变量块：[ adj | c | d | s | E | y ]
#     约束：  P_t + adj_t + d_t = L_t + c_t + s_t
#            E_t = E_{t-1} + η·c_t − d_t/η
#            s_t ≤ 预测光伏（弃光不能超过可用光伏）
#            E_t0-1 = e_init，E_{T-1} = e_terminal
# ===========================================================================
def adjust_day(price, load_kwh, pv_fc_kwh, x_plan, e_init, t0, eps=EPS_TH, e_terminal=E0):
    """Q3 调整 LP：对 [t0, T) 重优化购电与储能，偏差费线性化（新预报视为已知）。

    e_terminal：段末（当天 24:00）目标储电量，默认 E0（日循环）；跨日模式传 2 日计划的当日末值。

    变量块（各 m = T−t0 个）：0=adj 购电, 1=c 充, 2=d 放, 3=s 弃光, 4=E 储电量, 5=y 结算基准
    目标：min Σ_{t≥t0} [1.5·p·adj − p·y] + ε·Σ(c+d)
          其中 y = min(x_plan, adj)（系数为负 → 约束 y≤x_plan, y≤adj 下自动取等）
    返回 [t0, T) 的 adj 序列。
    """
    m = T - t0
    n = 6 * m

    def I(b, i):
        return b * m + i

    c_obj = np.zeros(n)
    # 目标系数：adj 记 1.5p，y 记 −p（等价于 p·adj + 0.5p|x_plan−adj|），c/d 罚项
    c_obj[0 * m:1 * m] = 1.5 * price[t0:]
    c_obj[1 * m:2 * m] = eps
    c_obj[2 * m:3 * m] = eps
    c_obj[5 * m:6 * m] = -price[t0:]

    # ---- 等式约束：[0,m) 平衡；[m,2m) 储能动态；第 2m 行终端 ----
    rows = 2 * m + 1
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    for i in range(m):
        t = t0 + i
        A_eq[i, I(0, i)] = 1.0
        A_eq[i, I(1, i)] = -1.0
        A_eq[i, I(2, i)] = 1.0
        A_eq[i, I(3, i)] = -1.0
        b_eq[i] = load_kwh[t] - pv_fc_kwh[t]

        r = m + i
        A_eq[r, I(4, i)] = 1.0
        if i:
            A_eq[r, I(4, i - 1)] = -1.0
        A_eq[r, I(1, i)] = -ETA
        A_eq[r, I(2, i)] = 1.0 / ETA
        b_eq[r] = e_init if i == 0 else 0.0
    A_eq[2 * m, I(4, m - 1)] = 1.0
    b_eq[2 * m] = e_terminal

    # ---- 不等式：y_i ≤ x_plan_i（上次承诺）；y_i − adj_i ≤ 0 ----
    A_ub = lil_matrix((2 * m, n))
    b_ub = np.zeros(2 * m)
    for i in range(m):
        t = t0 + i
        A_ub[i, I(5, i)] = 1.0
        b_ub[i] = x_plan[t]
        A_ub[m + i, I(5, i)] = 1.0
        A_ub[m + i, I(0, i)] = -1.0
    bounds = (
        [(0.0, None)] * m
        + [(0.0, P_MAX_E)] * m
        + [(0.0, P_MAX_E)] * m
        + [(0.0, float(pv_fc_kwh[t0 + i])) for i in range(m)]   # 弃光 ≤ 预测光伏
        + [(E_MIN, E_MAX)] * m
        + [(0.0, None)] * m
    )
    res = pm.optimize.solve_lp(
        c_obj, A_ub=csr_matrix(A_ub), b_ub=b_ub,
        A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds,
    )
    if not res.success:
        raise RuntimeError(f"调整 LP 失败（t0={t0}）：{res.message}")
    return res.x[0:m], res.x[4 * m:5 * m]


# ===========================================================================
# 四、Q3 调整 LP（场景对冲版，两阶段随机规划）
#
#     第一阶段（共享）：adj 为所有情景共用的购电决策；
#     第二阶段（每情景独立）：c_s/d_s/s_s/E_s/e_s。
#
#     min Σ[1.5p·adj − p·y] + (1/S)·Σ_s [ ε(c_s+d_s) + 5p·e_s ]
#     s.t. 每情景 s：功率平衡、储能动态、终端 E_{T-1}=e_terminal
#          y ≤ x_plan, y ≤ adj（第一阶段结算基准）
#
#     变量块顺序：[ adj | y | (c_s,d_s,s_s,E_s,e_s) for s=1..S ]
# ===========================================================================
def adjust_day_hedge(price, load_kwh, pv_scens, x_plan, e_init, t0, eps=EPS_TH, e_terminal=E0):
    """调整 LP（场景对冲）：adj 为第一阶段共享决策；每场景储能/紧急独立。

    e_terminal：段末（当天 24:00）目标储电量（各场景同值），默认 E0。

    pv_scens : list，第 0 个为名义预测（用于返回基准储能轨迹），其余为扰动场景；
               每个元素是长度 m = T−t0 的 kWh/槽数组。
    目标：min Σ[1.5p·adj − p·y] + (1/S)·Σ_s[ε(c_s+d_s) + 5p·e_s]
    返回 (adj, 名义储能轨迹 E_nom)。
    """
    m = T - t0
    S = len(pv_scens)
    n = 2 * m + S * 5 * m

    def A(i):
        return i

    def Y(i):
        return m + i

    def base(s):
        return 2 * m + s * 5 * m

    # 目标：第一阶段 adj/y；第二阶段各情景的罚项与期望紧急费（1/S 缩放）
    c_obj = np.zeros(n)
    c_obj[0:m] = 1.5 * price[t0:]
    c_obj[m:2 * m] = -price[t0:]
    for s in range(S):
        b0 = base(s)
        c_obj[b0:b0 + m] = eps / S
        c_obj[b0 + m:b0 + 2 * m] = eps / S
        c_obj[b0 + 4 * m:b0 + 5 * m] = EMERG_MULT * price[t0:] / S

    # ---- 等式约束：每情景 (2m+1) 行（平衡 m + 动态 m + 终端 1）----
    rows = 2 * m * S + S
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    r = 0
    for s in range(S):
        b0 = base(s)
        for i in range(m):
            # 情景 s 的功率平衡：adj + d_s − c_s − s_s + e_s = L − P_s
            A_eq[r, A(i)] = 1.0
            A_eq[r, b0 + i] = -1.0
            A_eq[r, b0 + m + i] = 1.0
            A_eq[r, b0 + 2 * m + i] = -1.0
            A_eq[r, b0 + 4 * m + i] = 1.0
            b_eq[r] = load_kwh[t0 + i] - pv_scens[s][i]
            r += 1
        for i in range(m):
            # 情景 s 的储能动态
            A_eq[r, b0 + 3 * m + i] = 1.0
            if i:
                A_eq[r, b0 + 3 * m + i - 1] = -1.0
            A_eq[r, b0 + i] = -ETA
            A_eq[r, b0 + m + i] = 1.0 / ETA
            b_eq[r] = e_init if i == 0 else 0.0
            r += 1
        # 情景 s 的段末终端
        A_eq[r, b0 + 3 * m + m - 1] = 1.0
        b_eq[r] = e_terminal
        r += 1

    # ---- 不等式：y ≤ x_plan；y − adj ≤ 0（第一阶段结算）----
    A_ub = lil_matrix((2 * m, n))
    b_ub = np.zeros(2 * m)
    for i in range(m):
        A_ub[i, Y(i)] = 1.0
        b_ub[i] = x_plan[t0 + i]
        A_ub[m + i, Y(i)] = 1.0
        A_ub[m + i, A(i)] = -1.0

    bounds = [(0.0, None)] * m + [(0.0, None)] * m
    for _ in range(S):
        bounds += ([(0.0, P_MAX_E)] * m + [(0.0, P_MAX_E)] * m
                   + [(0.0, None)] * m + [(E_MIN, E_MAX)] * m + [(0.0, None)] * m)
    res = pm.optimize.solve_lp(
        c_obj, A_ub=csr_matrix(A_ub), b_ub=b_ub,
        A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds,
    )
    if not res.success:
        raise RuntimeError(f"对冲调整 LP 失败（t0={t0}）：{res.message}")
    E_nom = res.x[base(0) + 3 * m: base(0) + 4 * m]
    return res.x[0:m], E_nom
