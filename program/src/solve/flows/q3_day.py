"""Q3 单日流程：0:00 计划 + 6/12/18 调整（含场景对冲）+ 因果执行/结算。

# ===========================================================================
# 主方案（正式口径 = simulate_day_rt_hedge）
#   0:00 计划：光伏预测 = λ·官方 f0 + (1−λ)·历史口E（λ=0.7，历史权重平滑 β=0.1）
#              x_plan = plan_day(price, L/6, P̂/6, E0)
#   6/12/18 调整：用最新预报重优化未执行时段（6/12 叠加无前视场景对冲）
#              x_adj = adjust_day / adjust_day_hedge(...)
#   执行：逐槽因果（exec_segment_causal，段末 SOC 跟踪规划轨迹）
#   结算：费用 = Σ [ p·x_adj + 0.5·p·|x_plan − x_adj| ] + 5·Σ p·e
#
#   simulate_day（事后口径）与 q2.exec_day（事后 LP）仅作下界对照，禁入正式结果。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np

from solve.common import E0, E_MAX, E_MIN, EMERG_MULT, EPS_TH, T
from solve.core.causal import exec_segment_causal
from solve.core.lp import adjust_day, adjust_day_hedge, exec_day, plan_day
from solve.core.residual import causal_residual_pool, latest_forecast
from solve.core.slots import fc_slots, hour_to_slots
from solve.models.adaptive import hist_forecast


def simulate_day(data: dict, D: int, lam: float, adj_hours=(6, 12, 18),
                 settle: str = "final", use_new_fc: bool = True,
                 adj_lam: float | None = None) -> dict:
    """单日 Q3 模拟（事后执行口径，仅对照）。

    adj_hours : 启用哪些调整时刻（6/12/18 的子集）；空元组 = 无调整（Q2 口径）。
    settle    : "final"：计划与最终调整值结算一次；
                "sequential"：每次调整分别与上一次结算（逐次计费）。
    use_new_fc: 调整时是否使用新预报；False 时沿用 0:00 预报（用于价值分解）。
    adj_lam   : 调整层的组合权重（None = 纯官方新预报；0~1 = 与历史预测组合）。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = (lam * f0 + (1.0 - lam) * ph) / 6.0

    x_plan, E_plan, _ = plan_day(price, load_kwh, pv_plan, E0, eps=EPS_TH)

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
        x_adj, _E_adj = adjust_day(price, load_kwh, pv_new, x_plan, float(E_plan[t0 - 1]), t0)
        x_final[t0:] = x_adj
        x_hist[pub] = x_final.copy()

    ex = exec_day(price, load_kwh, pv_act_kwh, x_final, E0)
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


def simulate_day_rt(data: dict, D: int, lam: float, adj_hours=(6, 12, 18),
                    settle: str = "final", use_new_fc: bool = True,
                    adj_lam: float | None = None, margin: float = 0.0) -> dict:
    """因果实时执行：调整使用已实现 SOC，每槽只读取当前实际值。

    与 ``simulate_day`` 的区别（后者为"事后执行"：全天一次执行 LP）：
    - 每段（0-6/6-12/12-18/18-24）用逐槽因果执行器 ``exec_segment_causal``；
    - 调整 LP 以该实际储电量为初值，利用已实现信息；
    - 用终端可达性投影锁定最近一次规划的段末 SOC，避免短视排空。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = (lam * f0 + (1.0 - lam) * ph) / 6.0
    if margin:
        pv_plan = pv_plan * (1.0 - margin)      # 光伏预测保守打折（多买保险）

    x_plan, E_plan, _ = plan_day(price, load_kwh, pv_plan, E0, eps=EPS_TH)

    x_seq = x_plan.copy()
    E_target = E_plan.copy()
    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    E_exec = np.zeros(T)
    x_hist = {0: x_plan.copy()}
    E_now = E0
    t = 0
    for pub in (6, 12, 18):
        t1 = pub * 6
        seg = exec_segment_causal(
            load_kwh[t:t1], pv_act_kwh[t:t1], x_seq[t:t1],
            E_now, float(np.clip(E_target[t1 - 1], E_MIN, E_MAX)),
        )
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
            pv_new = np.clip(src, 0.0, None) / 6.0
            x_adj, E_adj = adjust_day(price, load_kwh, pv_new, x_plan, E_now, t1)
            x_seq[t1:] = x_adj
            E_target[t1:] = E_adj
            x_hist[pub] = x_seq.copy()
        t = t1
    seg = exec_segment_causal(
        load_kwh[108:144], pv_act_kwh[108:144], x_seq[108:144],
        E_now, float(np.clip(E_target[143], E_MIN, E_MAX)),
    )
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
    }


def forecast_residual(data: dict, D: int, adj_lam: float | None) -> np.ndarray:
    """按当日实际预报口径返回"预报 − 实际"残差（kWh/槽；正 = 高估）。"""
    official = latest_forecast(data, D)
    if adj_lam is None:
        forecast = official
    else:
        forecast = adj_lam * official + (1.0 - adj_lam) * hist_forecast(data, D)
    return (forecast - data["pv_act"][D]) / 6.0


def simulate_day_rt_hedge(data: dict, D: int, lam: float, adj_hours=(6, 12, 18),
                          n_scen: int = 10, seed: int = 0,
                          adj_lam: float | None = None, margin: float = 0.0,
                          settle: str = "final", pool_min_month: int = 14,
                          pool_lookback: int = 90) -> dict:
    """因果执行 + 无前视场景对冲（6:00/12:00 用对冲 LP，18:00 段光伏≈0 不对冲）。

    场景只从目标日之前的残差块（``causal_residual_pool``）采样，
    随机流由 ``(seed, D, pub)`` 唯一确定；执行层逐槽因果（``exec_segment_causal``）。
    ``pool_min_month`` / ``pool_lookback`` 为场景池参数（灵敏度用，默认与正式一致）。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = (lam * f0 + (1.0 - lam) * ph) / 6.0
    if margin:
        pv_plan = pv_plan * (1.0 - margin)
    x_plan, E_plan, _ = plan_day(price, load_kwh, pv_plan, E0, eps=EPS_TH)

    x_seq = x_plan.copy()
    E_target = E_plan.copy()
    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    E_exec = np.zeros(T)
    scenario_days: list[int] = []
    x_hist = {0: x_plan.copy()}
    E_now = E0
    t = 0
    for pub in (6, 12, 18):
        t1 = pub * 6
        seg = exec_segment_causal(
            load_kwh[t:t1], pv_act_kwh[t:t1], x_seq[t:t1],
            E_now, float(np.clip(E_target[t1 - 1], E_MIN, E_MAX)),
        )
        e_total[t:t1] = seg["e"]
        c_all[t:t1] = seg["c"]
        d_all[t:t1] = seg["d"]
        E_exec[t:t1] = seg["E"]
        E_now = float(seg["E"][-1])
        if pub in adj_hours:
            fc = data[f"fc{pub}"][D]
            off = fc_slots(fc, pub)
            src = off if adj_lam is None else adj_lam * off + (1.0 - adj_lam) * ph
            pv_nom_full = np.clip(src, 0.0, None) / 6.0
            if pub in (6, 12):
                # 无前视残差池 + 由 (seed, D, pub) 决定的随机流
                pool = causal_residual_pool(data, D, min_same_month=pool_min_month,
                                            lookback=pool_lookback)
                rng = np.random.default_rng(np.random.SeedSequence([seed, D, pub]))
                idxs = rng.choice(pool, size=max(0, n_scen - 1), replace=True)
                scenario_days.extend(int(ix) for ix in idxs)
                scens = [pv_nom_full[t1:]] + [
                    np.clip(
                        pv_nom_full[t1:] - forecast_residual(data, int(ix), adj_lam)[t1:],
                        0.0, None,
                    )
                    for ix in idxs
                ]
                x_adj, E_adj = adjust_day_hedge(price, load_kwh, scens, x_plan, E_now, t1)
            else:
                x_adj, E_adj = adjust_day(price, load_kwh, pv_nom_full, x_plan, E_now, t1)
            x_seq[t1:] = x_adj
            E_target[t1:] = E_adj
            x_hist[pub] = x_seq.copy()
        t = t1
    seg = exec_segment_causal(
        load_kwh[108:144], pv_act_kwh[108:144], x_seq[108:144],
        E_now, float(np.clip(E_target[143], E_MIN, E_MAX)),
    )
    e_total[108:] = seg["e"]
    c_all[108:] = seg["c"]
    d_all[108:] = seg["d"]
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
        "c": c_all, "d": d_all, "E_exec": E_exec,
        "scenario_max_day": max(scenario_days, default=-1),
        "scenario_count": len(scenario_days),
    }


def perfect_day(data: dict, D: int) -> float:
    """完美下界：0:00 用实际光伏做计划且不调整（Q3 规则下不可实现，仅参考）。"""
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0
    x, _E, _ = plan_day(price, load_kwh, pv_act_kwh, E0, eps=EPS_TH)
    ex = exec_day(price, load_kwh, pv_act_kwh, x, E0)
    return float(price @ x + EMERG_MULT * price @ ex["e"])
