"""Q3 单日流程：0:00 计划 + 6/12/18 调整（含场景对冲）+ 因果执行/结算。

# ===========================================================================
# 主方案（正式口径 = simulate_day_rt_hedge，统一 v1.1）
#   预测：负荷 = κ·三源 EWMA h=5 预测；光伏 = 0:00 组合预测（λ·官方 + (1−λ)·历史）
#         再折减 m；权重由 Q2E 成本表统一标定（consistency.py）。
#   计划：plan_two_day —— 48 小时滚动视野（末端完全自由），只执行首日；
#         日末 SOC 自由并跨日传递。
#   6/12/18 调整：用最新发布预报重优化未执行时段（6/12 叠加无前视场景对冲）；
#         场景 = （Δ负荷, Δ光伏）同月整日联合残差块，N_SCEN=40，严格取目标日之前。
#   执行：逐槽因果（exec_segment_causal，EXEC_POLICY="free" 无段末硬目标）
#   结算：费用 = Σ [ p·x_adj + 0.5·p·|x_plan − x_adj| ] + 5·Σ p·e
#
#   simulate_day（事后口径）与 exec_day（事后 LP）仅作下界对照，禁入正式结果。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np

from solve import consistency as cs
from solve.common import E0, E_MAX, E_MIN, EMERG_MULT, EPS_TH, T
from solve.core.causal import exec_segment_causal, run_exec
from solve.core.lp import adjust_day, adjust_day_hedge, exec_day, plan_day, plan_horizon
from solve.core.residual import causal_residual_pool, forecast_at_publish, latest_forecast
from solve.core.slots import fc_slots, hour_to_slots
from solve.models.adaptive import (
    hist_forecast,
    hist_forecast_asof,
    hist_load_forecast,
    hist_load_forecast_asof,
)


def plan_two_day(data: dict, D: int, lam: float, e_start: float,
                 kappa: float = cs.KAPPA, margin: float = cs.MARGIN,
                 price2=None):
    """在 D 日 0:00 做 48 小时滚动计划，只执行首日（规划窗口末端完全自由）。

    统一口径（``consistency.py``）：负荷 = κ·L̂，光伏 = max(P̂ − m, 0)；
    ``price2`` 可传入 288 槽决策价格（Q4 用），默认附件 1 日价格重复两日。
    返回 (x_day, E_day, res)。
    """
    price2 = np.tile(data["price"], 2) if price2 is None else np.asarray(price2, float)
    f0 = hour_to_slots(data["fc0"][D])
    ph = hist_forecast_asof(data, D, D)
    pv0 = cs.margin_pv(lam * f0 + (1.0 - lam) * ph, margin)
    pv1 = cs.margin_pv(hist_forecast_asof(data, D + 1, D), margin)
    load0 = cs.kappa_load(hist_load_forecast_asof(data, D, D), kappa)
    load1 = cs.kappa_load(hist_load_forecast_asof(data, D + 1, D), kappa)
    xh, Eh, res = plan_horizon(
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
    load_fc_kwh = hist_load_forecast(data, D) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = (lam * f0 + (1.0 - lam) * ph) / 6.0

    x_plan, E_plan, _ = plan_day(price, load_fc_kwh, pv_plan, E0, eps=EPS_TH)

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

    ex = exec_day(price, load_fc_kwh, pv_act_kwh, x_final, E0)
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
                    adj_lam: float | None = None, kappa: float = cs.KAPPA,
                    margin: float = cs.MARGIN,
                    e_start: float = E0, exec_policy: str = cs.EXEC_POLICY) -> dict:
    """因果实时执行：调整使用已实现 SOC，每槽只读取当前实际值。

    与 ``simulate_day`` 的区别（后者为"事后执行"：全天一次执行 LP）：
    - 每段（0-6/6-12/12-18/18-24）用逐槽因果执行器 ``exec_segment_causal``；
    - 调整 LP 以该实际储电量为初值，利用已实现信息；
    - 统一口径：负荷 = κ·L̂，光伏 = max(P̂ − m, 0)；48 小时滚动计划。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    load_fc_kwh = cs.kappa_load(hist_load_forecast(data, D), kappa) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = hour_to_slots(data["fc0"][D])
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


def forecast_residual(data: dict, D: int, publish: int,
                      adj_lam: float | None) -> np.ndarray:
    """指定发布时刻的"预报 − 实际"残差（kWh/槽；正 = 高估）。

    场景残差必须与当前决策的信息集一致：6:00 调整只能使用历史 6:00 发布预报
    的误差，不能把历史 12:00/18:00 的更新拼接进剩余时段。
    """
    official = forecast_at_publish(data, D, publish)
    if adj_lam is None:
        forecast = official
    else:
        forecast = adj_lam * official + (1.0 - adj_lam) * hist_forecast(data, D)
    return (forecast - data["pv_act"][D]) / 6.0


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
    随机流由 ``(seed, D, pub)`` 唯一确定）；执行层逐槽因果（free 策略）。
    """
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    load_nom = cs.kappa_load(hist_load_forecast(data, D), kappa) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0

    f0 = hour_to_slots(data["fc0"][D])
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
                # 无前视残差池 + 由 (seed, D, pub) 决定的随机流
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
    # v1.2 末段修复：末段也必须走 run_exec（free 时完全无段末硬目标；
    # v1.1 曾直接传 E_target[143]，使 free 实际退化为 dayend，见 reports/执行器段末目标实验.md）
    seg = run_exec(exec_policy, load_kwh[108:144], pv_act_kwh[108:144],
                   x_seq[108:144], E_now,
                   float(np.clip(E_target[143], E_MIN, E_MAX)), last=True)
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
    x, _E, _ = plan_day(price, load_kwh, pv_act_kwh, E0, eps=EPS_TH)
    ex = exec_day(price, load_kwh, pv_act_kwh, x, E0)
    return float(price @ x + EMERG_MULT * price @ ex["e"])
