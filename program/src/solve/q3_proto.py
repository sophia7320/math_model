"""C 题 问题三原型：官方预报 + 历史预测组合、滚动成本标定。

口径（原型，写进后续正式化假设）：
- 0:00 计划：光伏预测 = λ·官方 f0 + (1−λ)·历史预测（口径 E 加权；u 来自 Q2E 缓存）
- 6/12/18 调整：用附件 3 最新预报对未执行时段重优化（偏差按 50%/150% 结算）
- 执行：购电承诺 take-or-pay，储能再调度（日循环），缺口 5 倍价紧急购电
- 结算：p·min(plan,adj) + 0.5p·(plan−adj)⁺ + 1.5p·(adj−plan)⁺ + 5p·e

对比策略：
- A：λ=1（纯官方）
- B：λ=0.5（固定组合）
- C：滚动窗口 W 天费用最优 λ（动态组合）
- D：λ=0（纯历史）
- P：完美下界（0:00 用实际光伏做计划，不调整）

运行（program/ 下）：
    uv run python -m solve.q3_proto --start 52 --days 67
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

import program as pm
from solve import consistency as cs
from solve import q2
from solve.common import DATA_C, E0, E_MAX, E_MIN, ETA, P_MAX_E, ROOT, T

LAMBDA_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)   # 11 个组合权重
W_WINDOW = 7            # 滚动标定窗口（天）
EPS_TH = 1e-3           # LP 正则项（元/kWh）
EMERG_MULT = 5.0
DAY_TABLE_START = 52    # 建表起点（给滚动窗口预热）
DAY_EVAL_START = 59     # 评估起点 2025-03-01
DAY_END = 119           # 评估终点（不含）= 2025-04-30


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_extended() -> dict:
    """q2 基础数据 + 6/12/18 预报 + 典型日光伏 + Q2E 历史权重。"""
    data = q2.load_all()

    fc3 = pm.read_table(DATA_C / "附件3.xlsx")
    fc = {h: np.zeros((q2.N_DAY, 24)) for h in (6, 12, 18)}
    day = -1
    for _, r in fc3.iterrows():
        cell = r["日期"]
        if isinstance(cell, str) and cell.strip():
            day += 1
        h = int(str(r["预报时刻"]).split(":")[0])
        if h in fc:
            fc[h][day] = r.iloc[2:26].astype(float).to_numpy()
    data["fc6"], data["fc12"], data["fc18"] = fc[6], fc[12], fc[18]

    a1 = pm.read_table(DATA_C / "附件1.xlsx")
    pv_col = "光伏发电预测功率" if "光伏发电预测功率" in a1.columns else a1.columns[3]
    load_col = "小区负载" if "小区负载" in a1.columns else a1.columns[2]
    data["pv_typ"] = a1[pv_col].to_numpy(float)
    data["load_typ"] = a1[load_col].to_numpy(float)

    cache_dir = ROOT / "code" / "outputs" / "cache" / "q2e"
    files = sorted(cache_dir.glob("table_*.npz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    npz_path = None
    for p in files:
        try:
            with np.load(p) as z:
                tag = str(z["tag"]) if "tag" in z.files else "q2e"
        except Exception:
            continue
        if tag == "q2e":
            npz_path = p
            break
    if npz_path is None:
        npz_path = files[0]
    _z = np.load(npz_path)
    data["TH"] = _z["TH"]
    # 统一口径（consistency.py）：EWMA h=5 逐日权重（负荷/光伏），全项目共用；
    # 覆盖 [15, 365)——日序 <31 的权重仅用于场景池残差的名义预测（无前视）。
    from solve.q2_adaptive import simplex_grid

    data["EWMA_WU"] = cs.ewma_weights_from_table(
        _z["C"], simplex_grid(0.2), first_target=cs.WARM_FROM + 1)
    return data


def hist_forecast(data: dict, D: int) -> np.ndarray:
    """统一口径光伏预测（144 槽，kW）：三源凸加权，EWMA h=5 权重（无前视）。

    与 ``consistency.py`` 一致：P̂ = u1·P(D−1) + u2·P(D−2) + u3·典型日，裁剪非负；
    权重取 ``data["EWMA_WU"][D]``（旧数据若无该字段回退 U_SMOOTH/TH）。
    """
    if "PV_OVERRIDE" in data and D in data["PV_OVERRIDE"]:
        return np.clip(np.asarray(data["PV_OVERRIDE"][D], dtype=float), 0.0, None)
    if D < 2:
        return np.clip(np.asarray(data["pv_typ"], dtype=float), 0.0, None)
    if "EWMA_WU" in data and D in data["EWMA_WU"]:
        u = data["EWMA_WU"][D][1]
    elif "U_SMOOTH" in data:
        u = data["U_SMOOTH"][D]
    else:
        u = _softmax(data["TH"][D - 1, 3:6])
    pv, typ = data["pv_act"], data["pv_typ"]
    return np.clip(u[0] * pv[D - 1] + u[1] * pv[D - 2] + u[2] * typ, 0.0, None)


def hist_load_forecast(data: dict, D: int) -> np.ndarray:
    """统一口径负荷预测（144 槽，kW）：同星期日/两周前/典型日三源凸加权。

    权重取 ``data["EWMA_WU"][D]``（EWMA h=5，只用 D−1 及更早信息）；
    目标日实际负荷只用于执行回放与残差统计。
    """
    if D < 14:
        return np.clip(np.asarray(data["load_typ"], dtype=float), 0.0, None)
    if "EWMA_WU" in data and D in data["EWMA_WU"]:
        w = data["EWMA_WU"][D][0]
    else:
        w = _softmax(data["TH"][D - 1, :3])
    load, typ = data["load"], data["load_typ"]
    d7, d14 = max(0, D - 7), max(0, D - 14)
    return np.clip(w[0] * load[d7] + w[1] * load[d14] + w[2] * typ, 0.0, None)


def hist_forecast_asof(data: dict, target_D: int, asof_D: int) -> np.ndarray:
    """在 ``asof_D`` 日 0:00 预测目标日光伏，禁止读取当日及以后实际值。

    两日滚动时 ``target_D=asof_D+1``，附件 3 尚无次日 0:00 预报，因此沿用
    当日已冻结的 EWMA 权重与截至 ``asof_D-1`` 的最近两条实际日曲线。
    """
    if target_D == asof_D:
        return hist_forecast(data, target_D)
    if target_D < asof_D or target_D > asof_D + 1:
        raise ValueError("历史光伏预测仅支持当前日或次日两日窗口")
    if asof_D < 2:
        return np.clip(np.asarray(data["pv_typ"], dtype=float), 0.0, None)
    if "EWMA_WU" in data and asof_D in data["EWMA_WU"]:
        u = data["EWMA_WU"][asof_D][1]
    elif "U_SMOOTH" in data:
        u = data["U_SMOOTH"][asof_D]
    else:
        u = _softmax(data["TH"][max(asof_D - 1, 13), 3:6])
    pv, typ = data["pv_act"], data["pv_typ"]
    d1, d2 = max(0, asof_D - 1), max(0, asof_D - 2)
    return np.clip(u[0] * pv[d1] + u[1] * pv[d2] + u[2] * typ, 0.0, None)


def hist_load_forecast_asof(data: dict, target_D: int, asof_D: int) -> np.ndarray:
    """在 ``asof_D`` 日 0:00 预测当前/次日负荷（两日窗口，无前视）。"""
    if target_D < asof_D or target_D > asof_D + 1:
        raise ValueError("历史负荷预测仅支持当前日或次日两日窗口")
    if asof_D < 14:
        return np.clip(np.asarray(data["load_typ"], dtype=float), 0.0, None)
    if "EWMA_WU" in data and asof_D in data["EWMA_WU"]:
        w = data["EWMA_WU"][asof_D][0]
    else:
        w = _softmax(data["TH"][asof_D - 1, :3])
    load, typ = data["load"], data["load_typ"]
    d7, d14 = max(0, target_D - 7), max(0, target_D - 14)
    if d7 >= asof_D or d14 >= asof_D:
        raise RuntimeError("两日负荷预测读取了尚未实现的实际日")
    return np.clip(w[0] * load[d7] + w[1] * load[d14] + w[2] * typ, 0.0, None)


def plan_two_day(data: dict, D: int, lam: float, e_start: float,
                 kappa: float = cs.KAPPA, margin: float = cs.MARGIN,
                 price2=None):
    """在 D 日 0:00 做 48 小时滚动计划，只执行首日（规划窗口末端完全自由）。

    统一口径（``consistency.py``）：负荷 = κ·L̂，光伏 = max(P̂ − m, 0)；
    ``price2`` 可传入 288 槽决策价格（Q4 用），默认附件 1 日价格重复两日。
    """
    price2 = np.tile(data["price"], 2) if price2 is None else np.asarray(price2, float)
    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = hist_forecast_asof(data, D, D)
    pv0 = cs.margin_pv(lam * f0 + (1.0 - lam) * ph, margin)
    pv1 = cs.margin_pv(hist_forecast_asof(data, D + 1, D), margin)
    load0 = cs.kappa_load(hist_load_forecast_asof(data, D, D), kappa)
    load1 = cs.kappa_load(hist_load_forecast_asof(data, D + 1, D), kappa)
    xh, Eh, res = q2.plan_horizon(
        price2, np.concatenate([load0, load1]) / 6.0,
        np.concatenate([pv0, pv1]) / 6.0, e_start, None, eps=EPS_TH,
    )
    return xh[:T], Eh[:T], res


def joint_residual_blocks(data: dict, pool, pub: int, adj_lam,
                          kappa: float = cs.KAPPA,
                          margin: float = cs.MARGIN):
    """历史池日的（Δ负荷, Δ光伏）联合整日残差块（kWh/槽，144 槽）。

    名义 = κ·负荷历史预测 与 max(组合光伏预测 − m, 0)（与当前决策同一构造，
    历史日使用其自身发布时刻的附件 3 预报）；残差 = 实际 − 名义。
    """
    RL = np.zeros((len(pool), T))
    RP = np.zeros((len(pool), T))
    for k, j in enumerate(pool):
        j = int(j)
        l_nom = cs.kappa_load(hist_load_forecast(data, j), kappa) / 6.0
        off_j = fc_slots(data[f"fc{pub}"][j], pub)
        pj = hist_forecast(data, j)
        src_j = off_j if adj_lam is None else adj_lam * off_j + (1.0 - adj_lam) * pj
        p_nom = cs.margin_pv(np.clip(src_j, 0.0, None), margin) / 6.0
        RL[k] = data["load"][j] / 6.0 - l_nom
        RP[k] = data["pv_act"][j] / 6.0 - p_nom
    return RL, RP


def run_exec(policy: str, load, pv, x, e0, e1, last: bool = False):
    """按执行策略调用逐槽因果执行器（段末目标机制对照，consistency 规范扩展）。

    policy:
      "target" 段末目标硬跟踪（原口径）；
      "free"   无段末目标（仅容量/功率约束；禁止购电充能自动满足）；
      "dayend" 仅最后一段跟踪段末目标，中间段自由。
    """
    if policy == "free" or (policy == "dayend" and not last):
        return q2.exec_segment_causal(load, pv, x, e0, None)
    return q2.exec_segment_causal(load, pv, x, e0, e1)


def make_smooth_u(data: dict, beta: float) -> np.ndarray:
    """参数平滑（新旧混合）：u_D = β·u_new + (1−β)·u_{D−1}。

    u_new[D] = softmax(TH[D−1][3:6])（Q2E 在标定日 D−1 为次日给出的权重）；
    凸组合自动保持权重非负且和=1；β=1 退化为不平滑。
    """
    TH = data["TH"]
    U = np.zeros((q2.N_DAY, 3))
    prev = None
    for D in range(q2.N_DAY):
        j = max(D - 1, 13)
        unew = _softmax(TH[j, 3:6])
        prev = unew if prev is None else beta * unew + (1.0 - beta) * prev
        U[D] = prev
    return U


def fc_slots(fc24: np.ndarray, publish: int) -> np.ndarray:
    """附件 3 预报行 → 144 槽（kW），相位对齐（预报 k 小时 = 发布时刻 + k 时）。

    构造 0..24 整点的值序列 hv（发布时刻及以前用第一个预报值填充），
    再线性插值到 144 个槽中心。调整 LP 只使用 [publish*6, 144) 段。
    """
    y = np.asarray(fc24, dtype=float)
    hv = np.empty(25)
    hv[: publish + 1] = y[0]
    hv[publish + 1: 25] = y[: 24 - publish]
    centers = (np.arange(T) + 0.5) / 6.0
    return np.interp(centers, np.arange(25, dtype=float), hv)


# ---------------------------------------------------------------------------
# LP：调整
# ---------------------------------------------------------------------------
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
    c_obj[0 * m:1 * m] = 1.5 * price[t0:]
    c_obj[1 * m:2 * m] = eps
    c_obj[2 * m:3 * m] = eps
    c_obj[5 * m:6 * m] = -price[t0:]

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


# ---------------------------------------------------------------------------
# 单日 Q3 模拟
# ---------------------------------------------------------------------------
def simulate_day(data: dict, D: int, lam: float, adj_hours=(6, 12, 18),
                 settle: str = "final", use_new_fc: bool = True,
                 adj_lam: float | None = None) -> dict:
    """单日 Q3 模拟。

    adj_hours : 启用哪些调整时刻（6/12/18 的子集）；空元组 = 无调整（Q2 口径）。
    settle    : "final"：计划与最终调整值结算一次；
                "sequential"：每次调整分别与上一次结算（逐次计费）。
    use_new_fc: 调整时是否使用新预报；False 时沿用 0:00 预报（用于价值分解）。
    adj_lam   : 调整层的组合权重（None = 纯官方新预报；0~1 = 与历史预测组合）。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    load_fc_kwh = hist_load_forecast(data, D) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = (lam * f0 + (1.0 - lam) * ph) / 6.0

    x_plan, E_plan, _ = q2.plan_day(price, load_fc_kwh, pv_plan, E0, eps=EPS_TH)

    x_final = x_plan.copy()
    x_hist = {0: x_plan.copy()}
    for t0, fc, pub in ((36, data["fc6"][D], 6), (72, data["fc12"][D], 12), (108, data["fc18"][D], 18)):
        if pub not in adj_hours:
            continue
        if use_new_fc:
            src = fc_slots(fc, pub)
            if adj_lam is not None:
                src = adj_lam * src + (1.0 - adj_lam) * ph
        else:
            src = f0
        pv_new = np.clip(src, 0.0, None) / 6.0
        x_adj, _E_adj = adjust_day(
            price, load_fc_kwh, pv_new, x_plan, float(E_plan[t0 - 1]), t0
        )
        x_final[t0:] = x_adj
        x_hist[pub] = x_final.copy()

    ex = q2.exec_day(price, load_kwh, pv_act_kwh, x_final, E0)
    e = ex["e"]

    # 统一结算：调低部分 50% 违约、调高部分 150% ⇒ 费用 = p·adj + 0.5p·|plan−adj|
    buy = float(price @ x_final)
    if settle == "final":
        dev = float((0.5 * price * np.abs(x_final - x_plan)).sum())
    elif settle == "sequential":
        dev = 0.0
        base_seq = x_plan.copy()
        for pub in sorted(k for k in x_hist if k > 0):
            cur = x_hist[pub]
            seg = slice(pub * 6, T)
            dev += float((0.5 * price[seg] * np.abs(cur[seg] - base_seq[seg])).sum())
            base_seq[seg] = cur[seg]
    else:
        raise ValueError(f"未知结算口径：{settle}")
    emerg = float((EMERG_MULT * price * e).sum())
    plan_cost = float(price @ x_plan)
    total = buy + dev + emerg
    return {
        "total": total,
        "plan_cost": plan_cost,
        "adjust_net": total - plan_cost - emerg,
        "emerg": emerg,
        "adj_abs_kwh": float(np.abs(x_final - x_plan).sum()),
        "x_plan": x_plan, "x_final": x_final, "e": e, "E": ex["E"],
    }


def exec_segment_hindsight(price, load_kwh, pv_kwh, x_seg, e_start, e_end, t0, eps=EPS_TH):
    """事后段执行下界：使用整段实际功率重优化储能。

    仅用于理论下界对照，**不得**进入实时口径；正式因果执行使用
    :func:`q2.exec_segment_causal`（逐槽只读当前已实现值）。

    变量块：0=c 充 1=d 放 2=s 弃 3=E 储电量 4=e 紧急；返回 c/d/s/E/e 字典。
    """
    m = len(x_seg)
    n = 5 * m

    def I(b, i):
        return b * m + i

    c_obj = np.zeros(n)
    c_obj[0:m] = eps
    c_obj[m:2 * m] = eps
    c_obj[4 * m:5 * m] = EMERG_MULT * price[t0:t0 + m]

    rows = 2 * m + 1
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    for i in range(m):
        A_eq[i, I(0, i)] = -1.0
        A_eq[i, I(1, i)] = 1.0
        A_eq[i, I(2, i)] = -1.0
        A_eq[i, I(4, i)] = 1.0
        b_eq[i] = load_kwh[t0 + i] - pv_kwh[t0 + i] - x_seg[i]

        r = m + i
        A_eq[r, I(3, i)] = 1.0
        if i:
            A_eq[r, I(3, i - 1)] = -1.0
        A_eq[r, I(0, i)] = -ETA
        A_eq[r, I(1, i)] = 1.0 / ETA
        b_eq[r] = e_start if i == 0 else 0.0
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
        raise RuntimeError(f"段执行 LP 失败（t0={t0}）：{res.message}")
    return {
        "c": res.x[0:m], "d": res.x[m:2 * m], "s": res.x[2 * m:3 * m],
        "E": res.x[3 * m:4 * m], "e": res.x[4 * m:5 * m],
    }


def simulate_day_rt(data: dict, D: int, lam: float, adj_hours=(6, 12, 18),
                    settle: str = "final", use_new_fc: bool = True,
                    adj_lam: float | None = None, kappa: float = cs.KAPPA,
                    margin: float = cs.MARGIN,
                    e_start: float = E0, exec_policy: str = cs.EXEC_POLICY) -> dict:
    """因果实时执行：调整使用已实现 SOC，每槽只读取当前实际值。

    与 ``simulate_day`` 的区别（后者为"事后执行"：全天一次执行 LP）：
    - 每段（0-6/6-12/12-18/18-24）用逐槽因果执行器 ``q2.exec_segment_causal``；
    - 调整 LP 以该实际储电量为初值，利用已实现信息；
    - 用终端可达性投影锁定最近一次规划的段末 SOC，避免短视排空。
    统一口径：负荷 = κ·L̂，光伏 = max(P̂ − m, 0)（``consistency.py``）。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    load_fc_kwh = cs.kappa_load(hist_load_forecast(data, D), kappa) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = cs.margin_pv(np.clip(lam * f0 + (1.0 - lam) * ph, 0.0, None),
                           margin) / 6.0

    x_plan, E_plan, _ = plan_two_day(data, D, lam, e_start,
                                     kappa=kappa, margin=margin)
    e_day_end = float(E_plan[-1])

    x_seq = x_plan.copy()
    E_target = E_plan.copy()
    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    E_exec = np.zeros(T)
    x_hist = {0: x_plan.copy()}
    E_now = e_start
    t = 0
    for pub in (6, 12, 18):
        t1 = pub * 6
        seg = run_exec(exec_policy, load_kwh[t:t1], pv_act_kwh[t:t1],
                       x_seq[t:t1], E_now,
                       float(np.clip(E_target[t1 - 1], E_MIN, E_MAX)))
        e_total[t:t1] = seg["e"]
        c_all[t:t1] = seg["c"]
        d_all[t:t1] = seg["d"]
        E_exec[t:t1] = seg["E"]
        E_now = float(seg["E"][-1])
        if pub in adj_hours:
            fc = data[f"fc{pub}"][D]
            if use_new_fc:
                src = fc_slots(fc, pub)
                if adj_lam is not None:
                    src = adj_lam * src + (1.0 - adj_lam) * ph
            else:
                src = f0
            pv_new = cs.margin_pv(np.clip(src, 0.0, None), margin) / 6.0
            x_adj, E_adj = adjust_day(
                price, load_fc_kwh, pv_new, x_plan, E_now, t1,
                e_terminal=e_day_end,
            )
            x_seq[t1:] = x_adj
            E_target[t1:] = E_adj
            x_hist[pub] = x_seq.copy()
        t = t1
    seg = run_exec(exec_policy, load_kwh[108:144], pv_act_kwh[108:144],
                   x_seq[108:144], E_now,
                   float(np.clip(E_target[143], E_MIN, E_MAX)), last=True)
    e_total[108:] = seg["e"]
    c_all[108:] = seg["c"]
    d_all[108:] = seg["d"]
    E_exec[108:] = seg["E"]

    x_final = x_seq.copy()
    buy = float(price @ x_final)
    if settle == "final":
        dev = float((0.5 * price * np.abs(x_final - x_plan)).sum())
    elif settle == "sequential":
        dev = 0.0
        base_seq = x_plan.copy()
        for pub in sorted(k for k in x_hist if k > 0):
            cur = x_hist[pub]
            seg = slice(pub * 6, T)
            dev += float((0.5 * price[seg] * np.abs(cur[seg] - base_seq[seg])).sum())
            base_seq[seg] = cur[seg]
    else:
        raise ValueError(f"未知结算口径：{settle}")
    emerg = float((EMERG_MULT * price * e_total).sum())
    plan_cost = float(price @ x_plan)
    total = buy + dev + emerg
    return {
        "total": total,
        "plan_cost": plan_cost,
        "adjust_net": total - plan_cost - emerg,
        "emerg": emerg,
        "adj_abs_kwh": float(np.abs(x_final - x_plan).sum()),
        "x_plan": x_plan, "x_final": x_final, "e": e_total, "E": E_target,
        "c": c_all, "d": d_all, "E_exec": E_exec,
        "E_start": float(e_start), "E_end": float(E_exec[-1]),
    }


def latest_forecast(data: dict, D: int) -> np.ndarray:
    """当天"最新可用预报"（144 槽 kW）：按段取 0:00/6:00/12:00/18:00 发布值。"""
    v = np.empty(T)
    v[:36] = q2._hour_to_slots(data["fc0"][D])[:36]
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
        return q2._hour_to_slots(data["fc0"][D])
    if publish not in (6, 12, 18):
        raise ValueError(f"不支持的预报发布时刻：{publish}")
    return fc_slots(data[f"fc{publish}"][D], publish)


def forecast_residual(data: dict, D: int, publish: int,
                      adj_lam: float | None) -> np.ndarray:
    """指定发布时刻的“预报 − 实际”残差（kWh/槽；正 = 高估）。"""
    official = forecast_at_publish(data, D, publish)
    if adj_lam is None:
        forecast = official
    else:
        forecast = adj_lam * official + (1.0 - adj_lam) * hist_forecast(data, D)
    return (forecast - data["pv_act"][D]) / 6.0


def causal_residual_pool(data: dict, D: int, min_same_month: int = 14,
                         lookback: int = 90) -> np.ndarray:
    """构造目标日 D 的历史残差池，严格保证所有日序号小于 D（无前视）。

    统一实现见 ``consistency.scenario_indices``（同月优先、不足回看）。
    """
    if D <= 14:
        raise ValueError("残差池至少需要 14 天预热数据")
    return cs.scenario_indices(np.asarray(data["months"]), D,
                               pool_min_same_month=min_same_month, lookback=lookback)


def adjust_day_hedge(price, load_kwh, pv_scens, x_plan, e_init, t0, eps=EPS_TH,
                     e_terminal=E0, load_scens=None):
    """调整 LP（场景对冲）：adj 为第一阶段共享决策；每场景储能/紧急独立。

    e_terminal：段末（当天 24:00）目标储电量（各场景同值），默认 E0。
    load_scens ：每场景负荷（list，长度 m；None 时所有场景共用 ``load_kwh``）；
    统一口径下与 ``pv_scens`` 一起由（Δ负荷, Δ光伏）联合残差块构造。

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

    c_obj = np.zeros(n)
    c_obj[0:m] = 1.5 * price[t0:]
    c_obj[m:2 * m] = -price[t0:]
    for s in range(S):
        b0 = base(s)
        c_obj[b0:b0 + m] = eps / S
        c_obj[b0 + m:b0 + 2 * m] = eps / S
        c_obj[b0 + 4 * m:b0 + 5 * m] = EMERG_MULT * price[t0:] / S

    rows = 2 * m * S + S
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    r = 0
    for s in range(S):
        b0 = base(s)
        for i in range(m):
            A_eq[r, A(i)] = 1.0
            A_eq[r, b0 + i] = -1.0
            A_eq[r, b0 + m + i] = 1.0
            A_eq[r, b0 + 2 * m + i] = -1.0
            A_eq[r, b0 + 4 * m + i] = 1.0
            b_eq[r] = ((load_kwh[t0 + i] if load_scens is None
                        else load_scens[s][i]) - pv_scens[s][i])
            r += 1
        for i in range(m):
            A_eq[r, b0 + 3 * m + i] = 1.0
            if i:
                A_eq[r, b0 + 3 * m + i - 1] = -1.0
            A_eq[r, b0 + i] = -ETA
            A_eq[r, b0 + m + i] = 1.0 / ETA
            b_eq[r] = e_init if i == 0 else 0.0
            r += 1
        A_eq[r, b0 + 3 * m + m - 1] = 1.0
        b_eq[r] = e_terminal
        r += 1

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


def simulate_day_rt_hedge(data: dict, D: int, lam: float, adj_hours=(6, 12, 18),
                          n_scen: int = cs.N_SCEN, seed: int = 0,
                          adj_lam: float | None = None, kappa: float = cs.KAPPA,
                          margin: float = cs.MARGIN,
                          settle: str = "final",
                          pool_min_month: int = cs.SCEN_MIN_SAME_MONTH,
                          pool_lookback: int = cs.SCEN_LOOKBACK,
                          e_start: float = E0,
                          exec_policy: str = cs.EXEC_POLICY) -> dict:
    """因果执行 + 无前视场景对冲（6:00/12:00 用对冲 LP，18:00 段光伏≈0 不对冲）。

    统一口径（``consistency.py``）：名义输入 = κ·L̂ 与 max(P̂ − m, 0)；
    场景 = 名义 + （Δ负荷, Δ光伏）同月整日联合残差块（池严格取目标日之前，
    随机流由 ``(seed, D, pub)`` 唯一确定）；执行层逐槽因果（q2.exec_segment_causal）。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    load_nom = cs.kappa_load(hist_load_forecast(data, D), kappa) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = cs.margin_pv(np.clip(lam * f0 + (1.0 - lam) * ph, 0.0, None),
                           margin) / 6.0
    x_plan, E_plan, _ = plan_two_day(data, D, lam, e_start,
                                     kappa=kappa, margin=margin)
    e_day_end = float(E_plan[-1])

    x_seq = x_plan.copy()
    E_target = E_plan.copy()
    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    s_all = np.zeros(T)
    E_exec = np.zeros(T)
    scenario_days: list[int] = []
    x_hist = {0: x_plan.copy()}
    E_now = e_start
    t = 0
    for pub in (6, 12, 18):
        t1 = pub * 6
        seg = run_exec(exec_policy, load_kwh[t:t1], pv_act_kwh[t:t1],
                       x_seq[t:t1], E_now,
                       float(np.clip(E_target[t1 - 1], E_MIN, E_MAX)))
        e_total[t:t1] = seg["e"]
        c_all[t:t1] = seg["c"]
        d_all[t:t1] = seg["d"]
        s_all[t:t1] = seg["s"]
        E_exec[t:t1] = seg["E"]
        E_now = float(seg["E"][-1])
        if pub in adj_hours:
            fc = data[f"fc{pub}"][D]
            off = fc_slots(fc, pub)
            src = off if adj_lam is None else adj_lam * off + (1.0 - adj_lam) * ph
            pv_nom_full = cs.margin_pv(np.clip(src, 0.0, None), margin) / 6.0
            if pub in (6, 12):
                pool = causal_residual_pool(data, D, min_same_month=pool_min_month,
                                            lookback=pool_lookback)
                rng = np.random.default_rng(np.random.SeedSequence([seed, D, pub]))
                sel = rng.choice(len(pool), size=max(0, n_scen - 1), replace=True)
                scenario_days.extend(int(pool[k]) for k in sel)
                RL, RP = joint_residual_blocks(data, pool, pub, adj_lam, kappa, margin)
                load_scens = [load_nom[t1:]] + [
                    np.clip(load_nom[t1:] + RL[k][t1:], 0.0, None) for k in sel]
                pv_scens = [pv_nom_full[t1:]] + [
                    np.clip(pv_nom_full[t1:] + RP[k][t1:], 0.0, None) for k in sel]
                x_adj, E_adj = adjust_day_hedge(
                    price, load_nom, pv_scens, x_plan, E_now, t1,
                    e_terminal=e_day_end, load_scens=load_scens,
                )
            else:
                x_adj, E_adj = adjust_day(
                    price, load_nom, pv_nom_full, x_plan, E_now, t1,
                    e_terminal=e_day_end,
                )
            x_seq[t1:] = x_adj
            E_target[t1:] = E_adj
            x_hist[pub] = x_seq.copy()
        t = t1
    seg = q2.exec_segment_causal(
        load_kwh[108:144], pv_act_kwh[108:144], x_seq[108:144],
        E_now, float(np.clip(E_target[143], E_MIN, E_MAX)),
    )
    e_total[108:] = seg["e"]
    c_all[108:] = seg["c"]
    d_all[108:] = seg["d"]
    s_all[108:] = seg["s"]
    E_exec[108:] = seg["E"]

    x_final = x_seq.copy()
    buy = float(price @ x_final)
    if settle == "final":
        dev = float((0.5 * price * np.abs(x_final - x_plan)).sum())
    else:
        dev = 0.0
        base_seq = x_plan.copy()
        for pub in sorted(k for k in x_hist if k > 0):
            cur = x_hist[pub]
            seg = slice(pub * 6, T)
            dev += float((0.5 * price[seg] * np.abs(cur[seg] - base_seq[seg])).sum())
            base_seq[seg] = cur[seg]
    emerg = float((EMERG_MULT * price * e_total).sum())
    plan_cost = float(price @ x_plan)
    total = buy + dev + emerg
    return {
        "total": total, "plan_cost": plan_cost,
        "buy_cost": buy, "deviation_cost": dev,
        "adjust_net": total - plan_cost - emerg, "emerg": emerg,
        "adj_abs_kwh": float(np.abs(x_final - x_plan).sum()),
        "x_plan": x_plan, "x_final": x_final, "e": e_total, "E": E_target,
        "c": c_all, "d": d_all, "s": s_all, "E_exec": E_exec,
        "scenario_max_day": max(scenario_days, default=-1),
        "scenario_count": len(scenario_days),
        "E_start": float(e_start), "E_end": float(E_exec[-1]),
    }


def perfect_day(data: dict, D: int) -> float:
    """完美下界：0:00 用实际光伏做计划且不调整（Q3 规则下不可实现，仅参考）。"""
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0
    x, _E, _ = q2.plan_day(price, load_kwh, pv_act_kwh, E0, eps=EPS_TH)
    ex = q2.exec_day(price, load_kwh, pv_act_kwh, x, E0)
    return float(price @ x + EMERG_MULT * price @ ex["e"])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=DAY_TABLE_START)
    ap.add_argument("--days", type=int, default=DAY_END - DAY_TABLE_START)
    ap.add_argument("--eval-start", type=int, default=DAY_EVAL_START)
    args = ap.parse_args()

    pm.init(seed=42, root=str(ROOT))
    t0 = time.time()
    data = load_extended()
    print(f"数据加载完成 {time.time() - t0:.1f}s，Q2E 权重示例 u = {np.round(_softmax(data['TH'][58, 3:6]), 3)}")

    days = list(range(args.start, args.start + args.days))
    nl = len(LAMBDA_GRID)
    table = np.zeros((len(days), nl))
    detail = []
    ts = time.time()
    for i, D in enumerate(days):
        rd = []
        for j, lam in enumerate(LAMBDA_GRID):
            r = simulate_day(data, D, float(lam))
            table[i, j] = r["total"]
            rd.append({k: r[k] for k in ("total", "plan_cost", "adjust_net", "emerg", "adj_abs_kwh")})
        detail.append(rd)
        if (i + 1) % 10 == 0 or i == len(days) - 1:
            print(f"  {i + 1}/{len(days)} 天完成，用时 {time.time() - ts:.0f}s")

    # 评估（滚动窗口在 table 行上，无前视）
    i_eval = max(0, args.eval_start - args.start)
    idx_l1, idx_half, idx_l0 = nl - 1, int(np.argmin(np.abs(LAMBDA_GRID - 0.5))), 0

    rows = []
    for i in range(i_eval, len(days)):
        i0 = max(0, i - W_WINDOW)
        li = int(np.argmin(table[i0:i].mean(axis=0))) if i > i0 else idx_half
        rows.append({
            "day": days[i],
            "date": data["dates"][days[i]],
            "A_official": table[i, idx_l1],
            "B_mix50": table[i, idx_half],
            "C_adaptive": table[i, li],
            "D_hist": table[i, idx_l0],
            "lam_star": LAMBDA_GRID[li],
            "perfect": perfect_day(data, days[i]),
        })
        if (i - i_eval + 1) % 20 == 0:
            print(f"  评估 {i - i_eval + 1}/{len(days) - i_eval} 天")

    df = pd.DataFrame(rows)
    n = len(df)
    print("\n===== Q3 原型结果（%d 天：%s ~ %s）=====" % (n, df["date"].iloc[0], df["date"].iloc[-1]))
    print(f"{'策略':<12}{'总费用/万元':>12}{'vs A':>9}{'日均/元':>11}")
    for name, col in [("A 纯官方", "A_official"), ("B 固定0.5", "B_mix50"),
                      ("C 动态", "C_adaptive"), ("D 纯历史", "D_hist"),
                      ("P 完美下界", "perfect")]:
        tot = df[col].sum()
        tag = "—" if col == "A_official" else f"{100 * (tot / df['A_official'].sum() - 1):+.2f}%"
        print(f"{name:<12}{tot / 1e4:>12.1f}{tag:>9}{tot / n:>11.0f}")

    print("\nλ* 取值分布：", dict(zip(*np.unique(df["lam_star"], return_counts=True))))
    print("λ* 均值：%.2f" % df["lam_star"].mean())

    # C 与 A 的费用分解对比（重新计算高 λ* 天的分解，取 C 策略选中 λ 的 detail）
    dec = []
    for i in range(i_eval, len(days)):
        i0 = max(0, i - W_WINDOW)
        li = int(np.argmin(table[i0:i].mean(axis=0))) if i > i0 else idx_half
        d_c = detail[i][li]
        d_a = detail[i][idx_l1]
        dec.append({"plan_C": d_c["plan_cost"], "adj_C": d_c["adjust_net"], "em_C": d_c["emerg"],
                    "plan_A": d_a["plan_cost"], "adj_A": d_a["adjust_net"], "em_A": d_a["emerg"]})
    dec = pd.DataFrame(dec)
    print("\n费用分解（元，60 天合计）：")
    print(f"  A：计划 {dec['plan_A'].sum() / 1e4:.1f} 万 + 调整净额 {dec['adj_A'].sum() / 1e4:.1f} 万 + 紧急 {dec['em_A'].sum() / 1e4:.1f} 万")
    print(f"  C：计划 {dec['plan_C'].sum() / 1e4:.1f} 万 + 调整净额 {dec['adj_C'].sum() / 1e4:.1f} 万 + 紧急 {dec['em_C'].sum() / 1e4:.1f} 万")

    out_dir = ROOT / "code" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table, index=[data["dates"][d] for d in days],
                 columns=[f"lam={v:.1f}" for v in LAMBDA_GRID]).to_csv(
        out_dir / "q3_proto_lambda_table.csv", encoding="utf-8-sig")
    df.to_csv(out_dir / "q3_proto_eval.csv", index=False, encoding="utf-8-sig")
    print(f"\n已保存：{out_dir / 'q3_proto_lambda_table.csv'}")
    print(f"总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
