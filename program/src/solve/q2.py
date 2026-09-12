"""C 题 问题二：日前计划 + 紧急购电（口径 D）+ 年度成本概率分布。

口径（论文假设，常量区可调）：
- 计划：每天 0:00 依据附件 3 的 0:00 光伏预报（负荷视为已知）解 LP，最小化计划购电费；
  储能日循环 E(0:00)=E(24:00)=6000 kWh；
- 执行：计划购电量已承诺（take-or-pay），储能逐槽因果执行（`exec_segment_causal`，
  不读取未来实际值；`exec_day` 事后 LP 仅作前视下界），缺口按 5 倍电价紧急购电；
- 各日相互独立（储能日循环）；结果自 2025-02-01 起报送。

概率扩展：以附件 3 的“整日原始残差块（预报−实际）”重采样，蒙特卡洛评估年度
总成本分布；对冲决策的场景池严格限于目标日之前，离线风险评估则使用同月全年样本。
"""
from __future__ import annotations

import time

import numpy as np
import openpyxl
import pandas as pd
import program as pm
from scipy.sparse import csr_matrix, lil_matrix

from solve.common import (
    DATA_C,
    E0,
    E_MAX,
    E_MIN,
    ETA,
    P_MAX_E,
    RESULTS_DIR,
    ROOT,
    T,
)

EPS_THROUGHPUT = 1e-3   # 执行 LP：抑制无意义充放的小罚项（元/kWh）
EMERG_MULT = 5.0        # 紧急购电价倍数
REPORT_START = 31       # 2025-02-01 的日序号（0 基）
N_DAY = 365
CAUSAL_POLICY_VERSION = "greedy-reachable-v1"   # 因果执行策略版本（参与缓存键）


# ---- 预报误差模型（相对 (时刻,月) 期望出力；见 C题_预报误差分析.md）----
def _mu(k):
    """系统偏差 μ(k) = 0.0050 − 0.00128k。"""
    return 0.0050 - 0.00128 * np.asarray(k, dtype=float)


def _sigma(k):
    """误差尺度 σ(k) = 0.0006 + 0.0084k。"""
    return 0.0006 + 0.0084 * np.asarray(k, dtype=float)


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_all() -> dict:
    """读取附件 1/2/3，返回数组与辅助量。"""
    sheets = pm.read_sheets(DATA_C / "附件2.xlsx")
    ldf = sheets["小区负载"]
    pdf = sheets["光伏发电实际功率"]
    dates = pd.to_datetime(ldf.iloc[:, 0]).dt.strftime("%Y-%m-%d").tolist()
    load = ldf.iloc[:, 1:145].to_numpy(float)
    pv_act = pdf.iloc[:, 1:145].to_numpy(float)

    a1 = pm.read_table(DATA_C / "附件1.xlsx")
    price = a1["电价"].to_numpy(float)
    load_typ = a1["小区负载"].to_numpy(float)
    pv_typ = a1["光伏发电预测功率"].to_numpy(float)

    fc3 = pm.read_table(DATA_C / "附件3.xlsx")
    fc0 = np.zeros((N_DAY, 24))
    day = -1
    for _, r in fc3.iterrows():
        cell = r["日期"]
        if isinstance(cell, str) and cell.strip():
            day += 1
        if int(str(r["预报时刻"]).split(":")[0]) == 0:
            fc0[day] = r.iloc[2:26].astype(float).to_numpy()

    months = np.array([int(x[5:7]) for x in dates])
    typ_hm = np.zeros((25, 13))
    for k in range(1, 25):
        col = pv_act[:, 6 * k - 1]
        for m in range(1, 13):
            typ_hm[k, m] = col[months == m].mean()

    # 标准化误差日块 Z[d, k]（0:00 预报，k=1..24）
    Z = np.zeros((N_DAY, 24))
    R0 = np.zeros((N_DAY, 24))  # 原始残差：预报 − 实际（用于严格因果块重采样）
    ks = np.arange(1, 25)
    for i in range(N_DAY):
        m = months[i]
        typ = typ_hm[1:25, m]
        err = fc0[i] - pv_act[i, 6 * ks - 1]
        R0[i] = err
        mask = typ > 100
        zz = np.zeros(24)
        zz[mask] = (err[mask] / typ[mask] - _mu(ks[mask])) / _sigma(ks[mask])
        Z[i] = zz

    return {
        "dates": dates, "load": load, "pv_act": pv_act, "price": price,
        "fc0": fc0, "months": months, "typ_hm": typ_hm, "Z": Z, "R0": R0,
        "load_typ": load_typ, "pv_typ": pv_typ,
    }


def _hour_to_slots(y24: np.ndarray) -> np.ndarray:
    """24 个整点值 → 144 个 10 分钟槽（线性插值到槽中心）。"""
    fp = np.concatenate([[0.0], np.asarray(y24, dtype=float)])
    centers = (np.arange(T) + 0.5) / 6.0
    return np.interp(centers, np.arange(0, 25, dtype=float), fp)


def _scenario_hourly(fc0_row, month, z_row, typ_hm) -> np.ndarray:
    """由标准化误差块生成情景小时序列（裁剪到 ≥0）。"""
    ks = np.arange(1, 25)
    typ = typ_hm[1:25, month]
    err = typ * (_mu(ks) + _sigma(ks) * z_row)
    # err = 预报 − 实际，因此实际情景 = 预报 − err。
    return np.clip(fc0_row - err, 0.0, None)


def _scenario_from_residual(fc0_row, residual_row) -> np.ndarray:
    """由原始残差块生成小时实际情景；残差定义为“预报 − 实际”。"""
    return np.clip(np.asarray(fc0_row, float) - np.asarray(residual_row, float), 0.0, None)


# ---------------------------------------------------------------------------
# LP：计划 / 执行
# ---------------------------------------------------------------------------
def _ix(b: int, t: int) -> int:
    return b * T + t


def plan_day(price, load_kwh, pv_kwh, e_start, eps=0.0):
    """单日循环计划 LP（仅保留作历史对照）：E(0)=E(24)=e_start。

    eps > 0 为极小正则项，用于在等价位内唯一化购电/充放方案（口径 E 标定）；
    默认 0 与原有口径完全一致。返回 (x, E, res)。
    """
    n = 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = price
    c_obj[T:3 * T] = eps
    A_eq = lil_matrix((2 * T + 1, n))
    b_eq = np.zeros(2 * T + 1)
    for t in range(T):
        A_eq[t, _ix(0, t)] = 1
        A_eq[t, _ix(1, t)] = -1
        A_eq[t, _ix(2, t)] = 1
        A_eq[t, _ix(3, t)] = -1
        b_eq[t] = load_kwh[t] - pv_kwh[t]
    for t in range(T):
        r = T + t
        A_eq[r, _ix(4, t)] = 1
        if t:
            A_eq[r, _ix(4, t - 1)] = -1
        A_eq[r, _ix(1, t)] = -ETA
        A_eq[r, _ix(2, t)] = 1.0 / ETA
        b_eq[r] = e_start if t == 0 else 0.0
    A_eq[2 * T, _ix(4, T - 1)] = 1
    b_eq[2 * T] = e_start
    bounds = (
        [(0.0, None)] * T
        + [(0.0, P_MAX_E)] * T
        + [(0.0, P_MAX_E)] * T
        + [(0.0, float(v)) for v in pv_kwh]
        + [(E_MIN, E_MAX)] * T
    )
    res = pm.optimize.solve_lp(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"计划 LP 失败：{res.message}")
    return res.x[0:T], res.x[4 * T:5 * T], res


def plan_horizon(price_h, load_h, pv_h, e_start, e_terminal=E0, eps=0.0):
    """有限视野计划 LP；正式滚动策略使用 2 日（288 槽）视野。

    只约束视野最远端 ``E(H)=e_terminal``；``e_terminal=None`` 时末端自由
    （供滚动脚本的结构对照与自由末端口径）。中间每个自然日的末端 SOC 都是
    优化变量。每日只执行前 144 槽，次日以真实末端 SOC 重新求解。
    变量块：0=x 购电 1=c 充电 2=d 放电 3=s 弃光 4=E。返回 (x, E, res)。
    """
    price_h = np.asarray(price_h, dtype=float)
    load_h = np.asarray(load_h, dtype=float)
    pv_h = np.asarray(pv_h, dtype=float)
    H = len(price_h)
    if len(load_h) != H or len(pv_h) != H:
        raise ValueError("price/load/pv 的滚动视野长度必须一致")
    n = 5 * H
    c_obj = np.zeros(n)
    c_obj[0:H] = price_h
    c_obj[H:3 * H] = eps
    n_rows = 2 * H + (1 if e_terminal is not None else 0)
    A_eq = lil_matrix((n_rows, n))
    b_eq = np.zeros(n_rows)
    for t in range(H):
        A_eq[t, t] = 1.0
        A_eq[t, H + t] = -1.0
        A_eq[t, 2 * H + t] = 1.0
        A_eq[t, 3 * H + t] = -1.0
        b_eq[t] = load_h[t] - pv_h[t]
        r = H + t
        A_eq[r, 4 * H + t] = 1.0
        if t:
            A_eq[r, 4 * H + t - 1] = -1.0
        A_eq[r, H + t] = -ETA
        A_eq[r, 2 * H + t] = 1.0 / ETA
        b_eq[r] = e_start if t == 0 else 0.0
    if e_terminal is not None:
        A_eq[2 * H, 5 * H - 1] = 1.0
        b_eq[2 * H] = float(e_terminal)
    bounds = (
        [(0.0, None)] * H
        + [(0.0, P_MAX_E)] * H
        + [(0.0, P_MAX_E)] * H
        + [(0.0, float(v)) for v in pv_h]
        + [(E_MIN, E_MAX)] * H
    )
    res = pm.optimize.solve_lp(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"滚动视野计划 LP 失败：{res.message}")
    return res.x[0:H], res.x[4 * H:5 * H], res


def exec_day(price, load_kwh, pv_kwh, x_plan, e_start):
    """事后执行下界：x 已承诺，使用全天实际曲线重优化储能。

    本函数保留给历史对照和理论下界，**不得**标记为"实时执行"；
    正式因果口径使用 :func:`exec_day_causal` / :func:`exec_segment_causal`。

    变量块：0=c 充电 1=d 放电 2=s 弃电 3=E 储电量 4=e 紧急购电。
    """
    n = 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = EPS_THROUGHPUT
    c_obj[T:2 * T] = EPS_THROUGHPUT
    c_obj[4 * T:5 * T] = EMERG_MULT * price
    A_eq = lil_matrix((2 * T + 1, n))
    b_eq = np.zeros(2 * T + 1)
    for t in range(T):
        A_eq[t, _ix(0, t)] = -1
        A_eq[t, _ix(1, t)] = 1
        A_eq[t, _ix(2, t)] = -1
        A_eq[t, _ix(4, t)] = 1
        b_eq[t] = load_kwh[t] - pv_kwh[t] - x_plan[t]
    for t in range(T):
        r = T + t
        A_eq[r, _ix(3, t)] = 1
        if t:
            A_eq[r, _ix(3, t - 1)] = -1
        A_eq[r, _ix(0, t)] = -ETA
        A_eq[r, _ix(1, t)] = 1.0 / ETA
        b_eq[r] = e_start if t == 0 else 0.0
    A_eq[2 * T, _ix(3, T - 1)] = 1          # 日循环：E(24:00) = E(0:00)
    b_eq[2 * T] = e_start
    bounds = (
        [(0.0, P_MAX_E)] * T
        + [(0.0, P_MAX_E)] * T
        + [(0.0, None)] * T
        + [(E_MIN, E_MAX)] * T
        + [(0.0, None)] * T
    )
    res = pm.optimize.solve_lp(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"执行 LP 失败：{res.message}")
    return {
        "c": res.x[0:T], "d": res.x[T:2 * T], "s": res.x[2 * T:3 * T],
        "E": res.x[3 * T:4 * T], "e": res.x[4 * T:5 * T],
    }


def exec_segment_causal(load_kwh, pv_kwh, x_commit, e_start, e_end):
    """逐槽因果执行：每槽只使用当前已实现的负荷和光伏，不读取未来实际值。

    策略：先用储能吸收当前富余或填补当前缺口，再把下一时刻 SOC 投影到
    "仍能在剩余时段到达 e_end"的可达区间；无法被储能覆盖的缺口记为紧急购电，
    富余且无法消纳的部分记为弃电。严格满足容量、功率与终端条件。

    参数均为长度 m 的 kWh/槽序列；返回 c/d/s/E/e 五条序列。
    说明：终端目标 e_end 需落在 m 步可达域内（真实段长 36 槽恒满足）。
    """
    load_kwh = np.asarray(load_kwh, dtype=float)
    pv_kwh = np.asarray(pv_kwh, dtype=float)
    x_commit = np.asarray(x_commit, dtype=float)
    if not (load_kwh.ndim == pv_kwh.ndim == x_commit.ndim == 1):
        raise ValueError("因果执行输入必须是一维序列")
    if not (len(load_kwh) == len(pv_kwh) == len(x_commit)):
        raise ValueError("因果执行的负荷、光伏与购电序列必须等长")
    if not (E_MIN <= e_start <= E_MAX and E_MIN <= e_end <= E_MAX):
        raise ValueError("因果执行的始末 SOC 超出安全区间")

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

    if m and abs(E[-1] - e_end) > 1e-6:
        raise RuntimeError(f"因果执行末端 SOC 偏差 {E[-1] - e_end:.3e} kWh")
    return {"c": c, "d": d, "s": spill, "E": E, "e": emerg}


def exec_day_causal(price, load_kwh, pv_kwh, x_plan, e_start, e_end=None):
    """正式因果执行口径：不利用未来实际值，缺口按 5 倍价计费（价格仅在外层汇总）。"""
    del price
    return exec_segment_causal(
        load_kwh, pv_kwh, x_plan, e_start, e_start if e_end is None else e_end
    )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _fmt_time(minute: int) -> str:
    """分钟数 → 时间标签（1440 → 0:00+1）。"""
    if minute >= 1440:
        return "0:00+1"
    return f"{minute // 60}:{minute % 60:02d}"


def _events(e_slots: np.ndarray, tol: float = 1e-3):
    """逐槽紧急购电合并为事件区间：[(起槽, 止槽, kWh), ...]。"""
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


# ---------------------------------------------------------------------------
# 确定性模拟（2025 全年滚动，1 月预热）
# ---------------------------------------------------------------------------
def run_deterministic(data: dict):
    price = data["price"]
    load, pv_act, fc0 = data["load"], data["pv_act"], data["fc0"]
    E = E0
    x_plans = np.zeros((N_DAY, T))
    recs = []
    for d in range(N_DAY):
        pv_fc = _hour_to_slots(fc0[d]) / 6.0          # kW → kWh/时段
        load_kwh = load[d] / 6.0
        # 48 小时滚动：次日无对应 0:00 官方预报，使用典型日作保守占位；
        # 当前日末 SOC 由两日优化决定并跨日传递。
        xh, Eh, _ = plan_horizon(
            np.tile(price, 2),
            np.concatenate([load_kwh, data["load_typ"] / 6.0]),
            np.concatenate([pv_fc, data["pv_typ"] / 6.0]),
            E, E0,
        )
        x = xh[:T]
        e_day_end = float(Eh[T - 1])
        ex = exec_day_causal(
            price, load_kwh, pv_act[d] / 6.0, x, E, e_end=e_day_end
        )
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


# ---------------------------------------------------------------------------
# 蒙特卡洛：年度总成本分布
# ---------------------------------------------------------------------------
def run_mc(data, x_plans, e_start_feb, e_targets, n_years=100, seed=42):
    """固定计划，对 2.1–12.31 重采样整日误差块，评估年度紧急费用分布。"""
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
            pv_s = _hour_to_slots(_scenario_from_residual(fc0[d], residuals[zi])) / 6.0
            ex = exec_day_causal(
                price, load[d] / 6.0, pv_s, x_plans[d], E,
                e_end=float(e_targets[d]),
            )
            emerg_kwhs[rep] += float(ex["e"].sum())
            emerg_costs[rep] += float(EMERG_MULT * (price @ ex["e"]))
            E = float(ex["E"][-1])
        if (rep + 1) % 10 == 0:
            print(f"  MC {rep + 1}/{n_years} 完成，用时 {time.time() - t0:.1f}s")
    return emerg_costs, emerg_kwhs


# ---------------------------------------------------------------------------
# 对冲计划（两阶段场景 LP：期望费用最小化）
# ---------------------------------------------------------------------------
def hedge_day(price, load_kwh, fc0_row, residual_pool, e_start,
              e_terminal,
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
        _hour_to_slots(_scenario_from_residual(fc0_row, residual_pool[i])) / 6.0
        for i in picks
    ]

    n = T + n_scen * 5 * T
    c_obj = np.zeros(n)
    c_obj[0:T] = price
    for s in range(n_scen):
        base = T + s * 5 * T
        c_obj[base:base + T] = EPS_THROUGHPUT / n_scen
        c_obj[base + T:base + 2 * T] = EPS_THROUGHPUT / n_scen
        c_obj[base + 4 * T:base + 5 * T] = EMERG_MULT * price / n_scen

    rows = n_scen * (2 * T + 1)
    rr, cc, vv = [], [], []
    for s in range(n_scen):
        r0 = s * (2 * T + 1)
        base = T + s * 5 * T
        t = np.arange(T)
        rr += [r0 + t, r0 + t, r0 + t, r0 + t, r0 + t]
        cc += [t, base + t, base + T + t, base + 2 * T + t, base + 4 * T + t]
        vv += [np.ones(T), -np.ones(T), np.ones(T), -np.ones(T), np.ones(T)]
        rr += [r0 + T + t, r0 + T + t[1:], r0 + T + t, r0 + T + t]
        cc += [base + 3 * T + t, base + 3 * T + t[:-1], base + t, base + T + t]
        vv += [np.ones(T), -np.ones(T - 1), -ETA * np.ones(T), (1.0 / ETA) * np.ones(T)]
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

    目标日 d 的误差场景只从其前 90 天（不足则用全部已有日）的残差块采样。
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
def verify(data: dict, x_plans, recs) -> dict:
    load, pv_act = data["load"], data["pv_act"]
    bal, dyn = 0.0, 0.0
    Emin, Emax = 1e18, -1e18
    for d in range(N_DAY):
        r = recs[d]
        res = (pv_act[d] / 6.0 + r["x"] + r["d"] + r["e"]
               - load[d] / 6.0 - r["c"] - r["s"])
        bal = max(bal, float(np.abs(res).max()))
        E = np.concatenate([[r["E_start"]], r["E"]])
        dyn = max(dyn, float(np.abs(np.diff(E) - (ETA * r["c"] - r["d"] / ETA)).max()))
        Emin = min(Emin, float(r["E"].min()))
        Emax = max(Emax, float(r["E"].max()))
    cont = max(abs(recs[d]["E_end"] - recs[d + 1]["E_start"]) for d in range(N_DAY - 1))
    return {
        "功率平衡最大残差/kWh": bal,
        "储能动态最大残差/kWh": dyn,
        "储电量最小值/kWh": Emin,
        "储电量最大值/kWh": Emax,
        "跨日储电量衔接最大误差/kWh": float(cont),
        "紧急购电量负值数": int(sum((recs[d]["e"] < -1e-6).sum() for d in range(N_DAY))),
        "弃电负值数": int(sum((recs[d]["s"] < -1e-6).sum() for d in range(N_DAY))),
    }


def write_result2(dates, price, x_plans, recs, out_name="result2.xlsx"):
    """按附件 5 模板生成 result2.xlsx（计划购电量 / 充放电量 / 紧急购电量）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result2.xlsx")
    rep = list(range(REPORT_START, N_DAY))

    ws = wb["计划购电量"]
    for i, d in enumerate(rep):
        row = 2 + i
        for t in range(T):
            ws.cell(row=row, column=2 + t, value=float(x_plans[d, t]))
        ws.cell(row=row, column=146, value=float(x_plans[d].sum()))
        ws.cell(row=row, column=147, value=float(price @ x_plans[d]))

    ws = wb["充放电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for d in rep:
        r = recs[d]
        for b in range(6):
            rr = row + b
            if b == 0:
                ws.cell(row=rr, column=1, value=pd.Timestamp(dates[d]).to_pydatetime())
            ws.cell(row=rr, column=2, value=f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(row=rr, column=3, value=float(r["c"][24 * b:24 * (b + 1)].sum()))
            ws.cell(row=rr, column=4, value=float(r["d"][24 * b:24 * (b + 1)].sum()))
            if b == 0:
                ws.cell(row=rr, column=5, value="00:00")
                ws.cell(row=rr, column=6, value=r["E_start"])
            if b == 1:
                ws.cell(row=rr, column=5, value="24:00")
                ws.cell(row=rr, column=6, value=r["E_end"])
        row += 6

    ws = wb["紧急购电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for d in rep:
        for j, (i0, i1, kwh) in enumerate(_events(recs[d]["e"])):
            if j == 0:
                ws.cell(row=row, column=1, value=pd.Timestamp(dates[d]).to_pydatetime())
            ws.cell(row=row, column=2,
                    value=f"{_fmt_time(i0 * 10)}-{_fmt_time((i1 + 1) * 10)}")
            ws.cell(row=row, column=3, value=round(kwh, 4))
            row += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / out_name
    wb.save(out)
    wb.close()
    return out


def make_figures(dates, recs, mc_costs, plan_cost_report):
    import matplotlib.pyplot as plt

    rep = list(range(REPORT_START, N_DAY))
    emerg_daily = np.array([recs[d]["emerg_kwh"] for d in rep])
    fig, ax = pm.line(np.arange(len(rep)), emerg_daily,
                      xlabel="日期", ylabel="紧急购电量 / kWh")
    month_starts = [i for i, d in enumerate(rep) if dates[d].endswith("-01")]
    ax.set_xticks(month_starts)
    ax.set_xticklabels([dates[rep[i]][5:7] + "月" for i in month_starts])
    pm.save_fig(fig, "Q2_逐日紧急购电",
                data=pd.DataFrame({"日期": [dates[d] for d in rep],
                                   "紧急购电量_kWh": emerg_daily}))

    if mc_costs is not None and len(mc_costs):
        total = plan_cost_report + mc_costs
        fig2, ax2 = plt.subplots(figsize=(7, 4.3))
        ax2.hist(total, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
        ax2.axvline(total.mean(), color="#C44E52", ls="--",
                    label=f"均值 {total.mean():,.0f} 元")
        ax2.axvline(np.percentile(total, 95), color="#55A868", ls=":",
                    label=f"P95 {np.percentile(total, 95):,.0f} 元")
        ax2.set_xlabel("年度总购电费 / 元")
        ax2.set_ylabel("频数")
        ax2.legend()
        pm.save_fig(fig2, "Q2_总费用分布",
                    data=pd.DataFrame({"年度总购电费_元": total}))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_q2(n_mc: int = 100, seed: int = 42, hedge: bool = True,
           write_result2_file: bool = False) -> dict:
    pm.init(root=str(ROOT))
    log = pm.get_logger("q2")
    data = load_all()

    t0 = time.time()
    x_plans, recs = run_deterministic(data)
    log.info("确定性滚动完成（365 天，用时 {:.1f}s）", time.time() - t0)

    rep = list(range(REPORT_START, N_DAY))
    plan_cost = sum(recs[d]["plan_cost"] for d in rep)
    emerg_cost = sum(recs[d]["emerg_cost"] for d in rep)
    emerg_kwh = sum(recs[d]["emerg_kwh"] for d in rep)
    spill_kwh = sum(recs[d]["spill_kwh"] for d in rep)
    n_days = sum(1 for d in rep if recs[d]["emerg_kwh"] > 1e-3)
    max_day = max(rep, key=lambda d: recs[d]["emerg_kwh"])

    pm.record_result(
        "问题二（口径D）确定性结果（2025-02-01 ~ 12-31）",
        {
            "计划购电量/kWh": float(x_plans[rep].sum()),
            "计划购电费/元": float(plan_cost),
            "紧急购电量/kWh": float(emerg_kwh),
            "紧急购电费/元": float(emerg_cost),
            "发生紧急购电的天数": n_days,
            "单日最大紧急购电量/kWh": float(recs[max_day]["emerg_kwh"]),
            "全程弃电量/kWh": float(spill_kwh),
            "日初=日末储电量/kWh": float(recs[REPORT_START]["E_start"]),
            "日内储电量最小值/kWh": float(min(recs[d]["E"].min() for d in rep)),
            "日内储电量最大值/kWh": float(max(recs[d]["E"].max() for d in rep)),
        },
        note=(
            "口径 D：0:00 用附件 3 的 0:00 光伏预报制定计划（负荷视为已知），储能日循环；"
            "计划购电 take-or-pay，实际按附件 2 执行，缺口按 5 倍交易时刻电价紧急购电；"
            "储能日循环（各日独立），结果自 2 月 1 日报送；图中位置：figures/Q2_逐日紧急购电.pdf。"
        ),
    )

    checks = verify(data, x_plans, recs)
    pm.record_result("问题二 约束与一致性校验", checks,
                     note="残差为数值误差量级即可；储电量须在 [1200, 10800] kWh 内。")

    daily = pd.DataFrame({
        "日期": [data["dates"][d] for d in rep],
        "计划购电量/kWh": [float(x_plans[d].sum()) for d in rep],
        "计划购电费/元": [recs[d]["plan_cost"] for d in rep],
        "紧急购电量/kWh": [recs[d]["emerg_kwh"] for d in rep],
        "紧急购电费/元": [recs[d]["emerg_cost"] for d in rep],
        "弃电量/kWh": [recs[d]["spill_kwh"] for d in rep],
        "日初储电量/kWh": [recs[d]["E_start"] for d in rep],
        "日末储电量/kWh": [recs[d]["E_end"] for d in rep],
    })
    pm.save_outputs(daily, "q2_daily_summary")

    if write_result2_file:
        # 口径 D 仅为使用附件 3 的额外信息对照；正式 result2 由 q2_tune 生成。
        out = write_result2(
            data["dates"], data["price"], x_plans, recs,
            out_name="result2_D_backup.xlsx",
        )
        log.info("口径 D 对照结果 -> {}", out)

    mc_costs = None
    if n_mc > 0:
        t1 = time.time()
        e_targets = np.array([r["E_end"] for r in recs], dtype=float)
        e_starts = np.array([r["E_start"] for r in recs], dtype=float)
        mc_costs, _mc_kwhs = run_mc(data, x_plans, recs[REPORT_START]["E_start"], e_targets,
                                    n_years=n_mc, seed=seed)
        log.info("蒙特卡洛完成（{} 年，用时 {:.1f}s）", n_mc, time.time() - t1)
        total = plan_cost + mc_costs
        p95 = float(np.percentile(total, 95))
        pm.record_result(
            "问题二 年度总成本分布（蒙特卡洛）",
            {
                "模拟年数": n_mc,
                "年度总成本均值/元": float(total.mean()),
                "年度总成本标准差/元": float(total.std()),
                "P5/元": float(np.percentile(total, 5)),
                "P50/元": float(np.percentile(total, 50)),
                "P95/元": p95,
                "CVaR95（尾部均值）/元": float(total[total >= p95].mean()),
                "年度紧急费用均值/元": float(mc_costs.mean()),
                "年度紧急费用P95/元": float(np.percentile(mc_costs, 95)),
            },
            note=(
                "固定计划、离线重采样附件 3 同月整日原始残差块（预报−实际，保留日内相关）；"
                "计划费固定，分布差异来自紧急购电；此分布只作事后风险评价，不参与在线决策；"
                "图 figures/Q2_总费用分布.pdf。"
            ),
        )
        pm.save_outputs(
            pd.DataFrame({"计划费_元": plan_cost, "紧急费_元": mc_costs,
                          "总费用_元": plan_cost + mc_costs}),
            "q2_mc_annual_costs",
        )

        if hedge:
            import matplotlib.pyplot as plt

            t2 = time.time()
            x_hedge = run_hedge(data, e_starts, e_targets, n_scen=20, seed=7)
            log.info("对冲计划求解完成（用时 {:.1f}s）", time.time() - t2)
            plan_cost_h = float(sum(float(data["price"] @ x_hedge[d]) for d in rep))
            mc_h, _ = run_mc(data, x_hedge, recs[REPORT_START]["E_start"], e_targets,
                             n_years=n_mc, seed=seed)
            total_h = plan_cost_h + mc_h
            p95h = float(np.percentile(total_h, 95))
            pm.record_result(
                "问题二 对冲计划 vs 朴素计划（蒙特卡洛对比）",
                {
                    "朴素：计划购电费/元": float(plan_cost),
                    "朴素：紧急费均值/元": float(mc_costs.mean()),
                    "朴素：总成本均值/元": float(total.mean()),
                    "朴素：总成本P95/元": p95,
                    "对冲：计划购电费/元": plan_cost_h,
                    "对冲：紧急费均值/元": float(mc_h.mean()),
                    "对冲：总成本均值/元": float(total_h.mean()),
                    "对冲：总成本P95/元": p95h,
                    "期望费用下降/元": float(total.mean() - total_h.mean()),
                    "期望费用下降/%": float(100 * (total.mean() - total_h.mean()) / total.mean()),
                    "P95 下降/元": float(p95 - p95h),
                },
                note=("对冲计划 = 两阶段场景 LP（每日 20 个等概率原始残差情景）最小化期望总费用；"
                      "目标日场景只取此前 90 日、同月样本不少于 14 日时优先同月；"
                      "两套计划用同一种子作离线评估。"),
            )
            pm.save_outputs(
                pd.DataFrame({"朴素_总费用_元": total, "对冲_总费用_元": total_h,
                              "朴素_紧急费_元": mc_costs, "对冲_紧急费_元": mc_h}),
                "Q2_对冲费用分布",
            )
            fig3, ax3 = plt.subplots(figsize=(7, 4.3))
            ax3.hist(total, bins=25, alpha=0.55, color="#4C72B0", label="朴素计划")
            ax3.hist(total_h, bins=25, alpha=0.55, color="#C44E52", label="对冲计划")
            ax3.set_xlabel("年度总购电费 / 元")
            ax3.set_ylabel("频数")
            ax3.legend()
            pm.save_fig(fig3, "Q2_对冲费用分布",
                        data=pd.DataFrame({"朴素_总费用_元": total,
                                           "对冲_总费用_元": total_h}))

    make_figures(data["dates"], recs, mc_costs, float(plan_cost))
    log.info("问题二完成：计划购电费 {:.0f} 元，紧急购电费 {:.0f} 元（{} 天有缺口）",
             plan_cost, emerg_cost, n_days)
    return {"plan_cost": plan_cost, "emerg_cost": emerg_cost,
            "emerg_kwh": emerg_kwh, "x_plans": x_plans, "recs": recs,
            "mc_costs": mc_costs}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=100, help="蒙特卡洛年数；0 表示跳过")
    ap.add_argument("--no-hedge", action="store_true", help="跳过两阶段场景对冲")
    ap.add_argument("--write-result2-d", action="store_true",
                    help="把口径 D 对照写为 result2_D_backup.xlsx（不覆盖正式 result2）")
    args = ap.parse_args()
    run_q2(n_mc=args.mc, hedge=not args.no_hedge,
           write_result2_file=args.write_result2_d)
