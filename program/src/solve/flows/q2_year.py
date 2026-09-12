"""Q2 年度流程：确定性滚动、蒙特卡洛、两阶段对冲计划。

# ===========================================================================
# 1) 确定性滚动（口径 D/E 共用）
#     每日 d：以 0:00 预报做出计划 LP → 逐槽因果执行（用实际曲线）
#     费用：计划费 = Σ p·x（take-or-pay），紧急费 = 5·Σ p·e
# 2) 蒙特卡洛（固定计划）
#     对 d ∈ [31, 365) 从整日标准化误差块 Z 中重采样（Z[zi] 直接套用），
#     得到年度紧急费用分布；计划费固定，分布差异完全来自紧急购电。
# 3) 两阶段对冲计划（期望费用最小）
#     min Σ p·x + Σ_s [ 5p·e_s + ε(c_s+d_s) ]
#     s.t. x 为所有情景共用第一阶段决策；每情景 s 的储能/紧急独立
#          （场景取自目标日之前 90 天，无前视；Q2 口径为全窗口重采样）
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
from solve.core.causal import exec_day_causal
from solve.core.lp import plan_day
from solve.core.slots import hour_to_slots
from solve.models.errors import scenario_hourly


def run_deterministic(data: dict):
    """2025 全年确定性滚动（1 月预热）：按预报计划、按实际因果执行。"""
    price = data["price"]
    load, pv_act, fc0 = data["load"], data["pv_act"], data["fc0"]
    E = E0
    x_plans = np.zeros((N_DAY, T))
    recs = []
    for d in range(N_DAY):
        pv_fc = hour_to_slots(fc0[d]) / 6.0          # kW → kWh/时段
        load_kwh = load[d] / 6.0
        x, _E_plan, _ = plan_day(price, load_kwh, pv_fc, E)
        ex = exec_day_causal(price, load_kwh, pv_act[d] / 6.0, x, E)
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


def run_mc(data, x_plans, e_start_feb, n_years=100, seed=42):
    """固定计划，对 2.1–12.31 重采样整日误差块，评估年度紧急费用分布。"""
    price = data["price"]
    load, fc0, Z, months, typ_hm = (
        data["load"], data["fc0"], data["Z"], data["months"], data["typ_hm"],
    )
    rng = np.random.default_rng(seed)
    emerg_costs = np.zeros(n_years)
    emerg_kwhs = np.zeros(n_years)
    t0 = time.time()
    for rep in range(n_years):
        E = float(e_start_feb)
        for d in range(REPORT_START, N_DAY):
            zi = int(rng.integers(0, N_DAY))
            pv_s = hour_to_slots(scenario_hourly(fc0[d], months[d], Z[zi], typ_hm)) / 6.0
            ex = exec_day_causal(price, load[d] / 6.0, pv_s, x_plans[d], E)
            emerg_kwhs[rep] += float(ex["e"].sum())
            emerg_costs[rep] += float(EMERG_MULT * (price @ ex["e"]))
            E = float(ex["E"][-1])
        if (rep + 1) % 10 == 0:
            print(f"  MC {rep + 1}/{n_years} 完成，用时 {time.time() - t0:.1f}s")
    return emerg_costs, emerg_kwhs


def hedge_day(price, load_kwh, fc0_row, month, z_pool, typ_hm, e_start,
              n_scen=20, rng=None):
    """两阶段场景 LP：计划 x 所有场景共用；每场景储能再调度 + 紧急购电。

    min  Σp·x + Σ_s [5p·e_s + ε·(c_s + d_s)]
    """
    if rng is None:
        rng = np.random.default_rng(0)
    picks = rng.integers(0, len(z_pool), size=n_scen)
    pv_scen = [
        hour_to_slots(scenario_hourly(fc0_row, month, z_pool[i], typ_hm)) / 6.0
        for i in picks
    ]

    # 变量块：0=x（第一阶段，T），其后每情景 5T（c/d/s/E/e）
    n = T + n_scen * 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = price
    for s in range(n_scen):
        base = T + s * 5 * T
        c_obj[base:base + T] = EPS_THROUGHPUT
        c_obj[base + T:base + 2 * T] = EPS_THROUGHPUT
        c_obj[base + 4 * T:base + 5 * T] = EMERG_MULT * price

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
        # 情景末 SOC 回到日初
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
        b_eq[r0 + 2 * T] = e_start

    bounds = [(0.0, None)] * T
    for _ in range(n_scen):
        bounds += [(0.0, P_MAX_E)] * T + [(0.0, P_MAX_E)] * T
        bounds += [(0.0, None)] * T + [(E_MIN, E_MAX)] * T + [(0.0, None)] * T
    res = pm.optimize.solve_lp(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"对冲 LP 失败：{res.message}")
    return res.x[0:T], res


def run_hedge(data: dict, n_scen: int = 20, seed: int = 7):
    """对 2.1–12.31 逐日求解无前视对冲计划（期望费用最小）。

    目标日 d 的误差场景只从其前 90 天（不足则用全部已有日）的残差块采样。
    """
    price, load, fc0 = data["price"], data["load"], data["fc0"]
    months, Z, typ_hm = data["months"], data["Z"], data["typ_hm"]
    rng = np.random.default_rng(seed)
    xh = np.zeros((N_DAY, T))
    t0 = time.time()
    for d in range(REPORT_START, N_DAY):
        z_hist = Z[max(0, d - 90):d]
        if not len(z_hist):
            raise RuntimeError(f"对冲场景池为空：d={d}")
        xh[d], _ = hedge_day(price, load[d] / 6.0, fc0[d], months[d], z_hist, typ_hm,
                             E0, n_scen=n_scen, rng=rng)
        if (d - REPORT_START + 1) % 50 == 0:
            print(f"  hedge {d - REPORT_START + 1}/334，用时 {time.time() - t0:.0f}s")
    return xh
