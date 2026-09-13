"""Q2 年度流程：确定性滚动、蒙特卡洛、两阶段对冲计划（统一口径 v1.1）。

# ===========================================================================
# 1) 确定性滚动
#     每日 d：κ·历史负荷预测 + 0:00 官方光伏预报，做 48 小时滚动计划
#     （plan_horizon 视野末端完全自由），只执行首日；逐槽因果执行（free）。
#     费用：计划费 = Σ p·x，紧急费 = 5·Σ p·e；日末 SOC 跨日传递。
# 2) 蒙特卡洛（固定计划，离线风险评价，不参与在线决策）
#     对 d ∈ [31, 365) 从同月整日原始残差块 R0（预报−实际）重采样；
#     计划费固定，分布差异完全来自紧急购电。
# 3) 两阶段对冲计划（期望费用最小）
#     min Σ p·x + (1/S)·Σ_s [ 5p·e_s + ε(c_s+d_s) ]
#     s.t. x 为所有情景共用第一阶段决策；每情景 s 的储能/紧急独立、
#          段末回到 e_terminal；场景池严格取目标日之前（同月 ≥14 天优先，
#          否则回看 90 天；Q2 离线评估用同月全年样本，在线对冲无前视）。
# ===========================================================================
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import csr_matrix

import program as pm
from solve.common import (
    E0,
    E_MAX,
    E_MIN,
    EMERG_MULT,
    EPS_THROUGHPUT,
    ETA,
    N_DAY,
    P_MAX_E,
    REPORT_START,
    T,
)
from solve.core.causal import exec_segment_causal
from solve.core.lp import plan_horizon
from solve.core.slots import hour_to_slots
from solve.models.errors import scenario_from_residual


def run_deterministic(data: dict):
    """2025 全年确定性滚动（1 月预热）：按统一预测计划、按实际因果执行。"""
    price = data["price"]
    load, pv_act, fc0 = data["load"], data["pv_act"], data["fc0"]
    E = E0
    x_plans = np.zeros((N_DAY, T))
    recs = []
    for d in range(N_DAY):
        pv_fc = hour_to_slots(fc0[d]) / 6.0          # kW → kWh/时段
        load_kwh = load[d] / 6.0
        # 48 小时滚动：次日无对应 0:00 官方预报，使用典型日作保守占位；
        # 当前日末 SOC 由两日优化决定并跨日传递，规划窗口末端完全自由。
        xh, _Eh, _res = plan_horizon(
            np.tile(price, 2),
            np.concatenate([load_kwh, data["load_typ"] / 6.0]),
            np.concatenate([pv_fc, data["pv_typ"] / 6.0]),
            E, None,
        )
        x = xh[:T]
        # 统一执行策略（consistency.EXEC_POLICY="free"）：无段末硬目标
        ex = exec_segment_causal(load_kwh, pv_act[d] / 6.0, x, E, None)
        recs.append({
            "x": x, "plan_cost": float(price @ x),
            "c": ex["c"], "d": ex["d"], "s": ex["s"], "e": ex["e"], "E": ex["E"],
            "emerg_kwh": float(ex["e"].sum()),
            "emerg_cost": float(EMERG_MULT * (price @ ex["e"])),
            "E_start": float(E), "E_end": float(ex["E"][-1]),
            "spill_kwh": float(ex["s"].sum()),
        })
        x_plans[d] = x
        E = float(ex["E"][-1])
    return x_plans, recs


def run_mc(data, x_plans, e_start_feb, e_targets, n_years=100, seed=42):
    """固定计划，对 2.1–12.31 重采样同月整日原始残差块，评估年度紧急费用分布。"""
    price = data["price"]
    load, fc0, residuals, months = (
        data["load"], data["fc0"], data["R0"], data["months"],
    )
    rng = np.random.default_rng(seed)
    emerg_costs = np.zeros(n_years)
    emerg_kwhs = np.zeros(n_years)
    t0 = time.time()
    for rep in range(n_years):
        E = float(e_start_feb)
        for d in range(REPORT_START, N_DAY):
            pool = np.flatnonzero(months == months[d])
            zi = int(rng.choice(pool))
            pv_s = hour_to_slots(scenario_from_residual(fc0[d], residuals[zi])) / 6.0
            # 统一执行策略（free）：无段末硬目标
            ex = exec_segment_causal(load[d] / 6.0, pv_s, x_plans[d], E, None)
            emerg_kwhs[rep] += float(ex["e"].sum())
            emerg_costs[rep] += float(EMERG_MULT * (price @ ex["e"]))
            E = float(ex["E"][-1])
        if (rep + 1) % 10 == 0:
            print(f"  MC {rep + 1}/{n_years} 完成，用时 {time.time() - t0:.1f}s")
    return emerg_costs, emerg_kwhs


def hedge_day(price, load_kwh, fc0_row, residual_pool, e_start, e_terminal,
              n_scen=20, rng=None):
    """两阶段场景 LP：计划 x 所有场景共用；每场景储能再调度 + 紧急购电。

    等概率情景下目标为期望费用：
        min  Σp·x + (1/S)·Σ_s [5p·e_s + ε·(c_s + d_s)]

    计划成本只发生一次，场景补救成本必须按概率 ``1/S`` 加权；否则情景数会
    人为改变风险项相对计划成本的权重。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    picks = rng.integers(0, len(residual_pool), size=n_scen)
    pv_scen = [
        hour_to_slots(scenario_from_residual(fc0_row, residual_pool[i])) / 6.0
        for i in picks
    ]

    # 变量块：0=x（第一阶段，T），其后每情景 5T（c/d/s/E/e）
    n = T + n_scen * 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = price
    for s in range(n_scen):
        base = T + s * 5 * T
        c_obj[base:base + T] = EPS_THROUGHPUT / n_scen
        c_obj[base + T:base + 2 * T] = EPS_THROUGHPUT / n_scen
        c_obj[base + 4 * T:base + 5 * T] = EMERG_MULT * price / n_scen

    # 等式约束按 COO 三元组拼接：每情景 (2T+1) 行
    rows = n_scen * (2 * T + 1)
    rr, cc, vv = [], [], []
    for s in range(n_scen):
        r0 = s * (2 * T + 1)
        base = T + s * 5 * T
        t = np.arange(T)
        # 功率平衡：x + d_s − c_s − s_s + e_s = L − P_s
        rr += [r0 + t, r0 + t, r0 + t, r0 + t, r0 + t]
        cc += [t, base + t, base + T + t, base + 2 * T + t, base + 4 * T + t]
        vv += [np.ones(T), -np.ones(T), np.ones(T), -np.ones(T), np.ones(T)]
        # 储能动态：E_s,t − E_s,t−1 − η·c + d/η = 0
        rr += [r0 + T + t, r0 + T + t[1:], r0 + T + t, r0 + T + t]
        cc += [base + 3 * T + t, base + 3 * T + t[:-1], base + t, base + T + t]
        vv += [np.ones(T), -np.ones(T - 1), -ETA * np.ones(T), (1.0 / ETA) * np.ones(T)]
        # 情景末 SOC 回到段末目标（各场景同值）
        rr.append(np.array([r0 + 2 * T]))
        cc.append(np.array([base + 3 * T + T - 1]))
        vv.append(np.array([1.0]))
    A_eq = csr_matrix(
        (np.concatenate(vv), (np.concatenate(rr), np.concatenate(cc))),
        shape=(rows, n),
    )
    b_eq = np.zeros(rows)
    for s in range(n_scen):
        r0 = s * (2 * T + 1)
        b_eq[r0:r0 + T] = load_kwh - pv_scen[s]
        b_eq[r0 + T] = e_start
        b_eq[r0 + 2 * T] = e_terminal

    bounds = [(0.0, None)] * T
    for _ in range(n_scen):
        bounds += [(0.0, P_MAX_E)] * T + [(0.0, P_MAX_E)] * T
        bounds += [(0.0, None)] * T + [(E_MIN, E_MAX)] * T + [(0.0, None)] * T
    res = pm.optimize.solve_lp(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"对冲 LP 失败：{res.message}")
    return res.x[0:T], res


def run_hedge(data: dict, e_starts, e_targets, n_scen: int = 20, seed: int = 7):
    """对 2.1–12.31 逐日求解无前视对冲计划（期望费用最小）。

    目标日 d 的场景池只取其前 90 天（同月样本 ≥14 天时优先同月）的
    整日残差块采样，严格无前视。
    """
    price, load, fc0 = data["price"], data["load"], data["fc0"]
    months, residuals = data["months"], data["R0"]
    rng = np.random.default_rng(seed)
    xh = np.zeros((N_DAY, T))
    t0 = time.time()
    for d in range(REPORT_START, N_DAY):
        past = np.arange(max(0, d - 90), d, dtype=int)
        same = past[months[past] == months[d]]
        pool_days = same if len(same) >= 14 else past
        residual_hist = residuals[pool_days]
        if not len(residual_hist) or int(pool_days.max()) >= d:
            raise RuntimeError(f"对冲场景池为空：d={d}")
        xh[d], _ = hedge_day(
            price, load[d] / 6.0, fc0[d], residual_hist,
            float(e_starts[d]), float(e_targets[d]),
            n_scen=n_scen, rng=rng,
        )
        if (d - REPORT_START + 1) % 50 == 0:
            print(f"  hedge {d - REPORT_START + 1}/334，用时 {time.time() - t0:.0f}s")
    return xh
