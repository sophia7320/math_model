"""Q4 年度流程：统一口径下的 2 日滚动（Q2 层）与调整层（Q3 层）。

# ===========================================================================
# Q2 层（run_year）
#   决策价：G（题面主口径）= 附件 4 已知电价；H（扩展）= 三源预测（滚动 v + β）
#   预测：负荷 = κ·EWMA h=5 历史预测；光伏 = 附件 3 的 0:00 预报（次日用 asof 历史）
#   储能：daily = 日循环（plan_day）；2day = 48 小时滚动（plan_horizon，末端自由）
#   执行：逐槽因果（free）；结算用附件 4 真实电价：Σ p4·x + 5·p4·e
#
# Q3 层（simulate_day_q3，G 主口径）
#   0:00 计划 + 6/12/18 调整（6/12 叠加 40 情景联合残差对冲）；
#   结算 = Σ [ p4·x_adj + 0.5·p4·|x_plan − x_adj| ] + 5·Σ p4·e
# ===========================================================================
"""
from __future__ import annotations

import numpy as np

from solve import consistency as cs
from solve.common import (
    E0,
    E_MAX,
    E_MIN,
    EMERG_MULT,
    EPS_TH,
    N_DAY,
    REPORT_START,
    T,
)
from solve.core.causal import exec_segment_causal
from solve.core.lp import adjust_day, adjust_day_hedge, plan_day, plan_horizon
from solve.core.residual import causal_residual_pool
from solve.core.slots import fc_slots, hour_to_slots
from solve.flows.q3_day import forecast_residual, joint_residual_blocks, plan_two_day
from solve.models.adaptive import hist_forecast, hist_forecast_asof, hist_load_forecast_asof
from solve.models.price import price_forecast_d, price_forecast_next


# ===========================================================================
# Q4 Q2 层：全年滚动（计划/执行/结算）
# ===========================================================================
def run_year(data, p4, p_typ, vseq, price_mode="H", storage="2day",
             start=REPORT_START, end=N_DAY) -> list[dict]:
    """全年滚动模拟（统一口径：EWMA 预测 + κ/m；2-1 起 E0 起步）。

    price_mode = "G"（题面已知电价）/ "H"（历史价格预测）；
    storage = "daily"（日循环）或 "2day"（48 小时滚动、跨日连续、末端自由）。
    返回逐日记录（含 x/c/d/s/e/E 与费用，REPORT_START 起计费）。
    """
    load, pv_act = data["load"], data["pv_act"]
    E = E0
    # 按日序号对齐（D < start 的月份仅作预测预热，占位记录不计费）
    recs: list[dict] = [{"D": D} for D in range(N_DAY)]
    for D in range(start, end):
        # 决策价格：G = 当天真实价；H = 三源预测（次日另用 D+1 预测）
        if price_mode == "G":
            p_d = p4[D]
            p_d1 = p4[min(D + 1, N_DAY - 1)]
        else:
            v = vseq[D]
            p_d = price_forecast_d(D, v, p4, p_typ)
            p_d1 = price_forecast_next(D, v, p4, p_typ) if D + 1 < N_DAY else p_d
        load_act = load[D] / 6.0
        load_fc_d = cs.kappa_load(hist_load_forecast_asof(data, D, D)) / 6.0
        pv_fc_d = cs.margin_pv(hist_forecast_asof(data, D, D)) / 6.0

        if storage == "daily":
            # 日循环：E(0)=E(24)=当前 SOC
            x_day, _E_plan, _ = plan_day(p_d, load_fc_d, pv_fc_d, E, eps=EPS_TH)
        else:
            # 48 小时滚动：次日负荷/光伏用 asof 无前视预测，窗口末端完全自由
            load_fc_d1 = cs.kappa_load(hist_load_forecast_asof(data, D + 1, D)) / 6.0
            pv_fc_d1 = cs.margin_pv(hist_forecast_asof(data, D + 1, D)) / 6.0
            xh, _Eh, _ = plan_horizon(
                np.concatenate([p_d, p_d1]),
                np.concatenate([load_fc_d, load_fc_d1]),
                np.concatenate([pv_fc_d, pv_fc_d1]),
                E, None, eps=EPS_TH,
            )
            x_day = xh[:T]

        # 统一执行策略（free）：无段末硬目标
        ex = exec_segment_causal(load_act, pv_act[D] / 6.0, x_day, E, None)
        rec = {"D": D, "x": x_day, "c": ex["c"], "d": ex["d"], "s": ex["s"],
               "e": ex["e"], "E": ex["E"], "E_start": float(E)}
        if D >= REPORT_START:
            # 结算用附件 4 真实电价（与决策价无关）
            rec["plan_cost"] = float(p4[D] @ x_day)
            rec["emerg"] = float(EMERG_MULT * (p4[D] @ ex["e"]))
            rec["total"] = rec["plan_cost"] + rec["emerg"]
        recs[D] = rec
        E = float(ex["E"][-1])
    return recs


def total_of(recs) -> dict:
    """Q2 层费用汇总（仅计费日）：计划/紧急/合计与紧急电量。"""
    days = [r for r in recs if "total" in r]
    return {
        "plan": sum(r["plan_cost"] for r in days),
        "emerg": sum(r["emerg"] for r in days),
        "total": sum(r["total"] for r in days),
        "emerg_kwh": sum(r["e"].sum() for r in days),
        "days": len(days),
    }


# ===========================================================================
# Q4 Q3 层：0:00 计划 + 6/12/18 调整（+6/12 对冲），结算用附件 4 真实价
# ===========================================================================
def simulate_day_q3(data, p4, p_typ, vseq, D, price_mode="H", storage="daily",
                    adj_hours=(6, 12, 18), hedge=True, n_scen=cs.N_SCEN, seed=7,
                    lam=0.7, adj_lam=0.7, kappa=cs.KAPPA, margin=cs.MARGIN,
                    e_start=E0, eps=EPS_TH) -> dict:
    """Q4 Q3 层单日模拟（统一口径：EWMA 预测 + κ/m + 联合残差场景）。

    H 口径：决策价格 = 三源预测（滚动 v）；G：决策价格 = 当天真实价。
    storage='daily'：日循环（计划/调整末端回 E0）；'2day'：48 小时滚动、跨日连续。
    结算一律用附件 4 真实价：买入 + 偏差费（0.5p|x−x⁰|）+ 5 倍紧急。
    """
    p_dec = p4[D] if price_mode == "G" else price_forecast_d(D, vseq[D], p4, p_typ)
    load_act_kwh = data["load"][D] / 6.0
    load_nom = cs.kappa_load(hist_load_forecast_asof(data, D, D), kappa) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0
    f0 = hour_to_slots(data["fc0"][D])
    ph = hist_forecast(data, D)
    pv_plan = cs.margin_pv(np.clip(lam * f0 + (1.0 - lam) * ph, 0.0, None),
                           margin) / 6.0

    if storage == "daily":
        x_plan, E_plan, _ = plan_day(p_dec, load_nom, pv_plan, e_start, eps=eps)
        e_day_end = E0
    else:
        # 48 小时滚动：次日价格预测 + asof 历史预测，窗口末端自由
        D1 = min(D + 1, N_DAY - 1)
        p_dec1 = (p4[D1] if price_mode == "G"
                  else price_forecast_next(D, vseq[D], p4, p_typ))
        x_plan, E_plan, _ = plan_two_day(
            data, D, lam, e_start, kappa=kappa, margin=margin,
            price2=np.concatenate([p_dec, p_dec1]))
        e_day_end = float(E_plan[-1])

    x_seq = x_plan.copy()
    E_target = E_plan.copy()
    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    s_all = np.zeros(T)
    E_exec = np.zeros(T)
    scenario_days: list[int] = []
    E_now = e_start
    t = 0
    for pub in (6, 12, 18):
        t1 = pub * 6
        # 统一执行策略（free）：无段末硬目标
        seg = exec_segment_causal(
            load_act_kwh[t:t1], pv_act_kwh[t:t1], x_seq[t:t1], E_now, None,
        )
        e_total[t:t1] = seg["e"]
        c_all[t:t1] = seg["c"]
        d_all[t:t1] = seg["d"]
        s_all[t:t1] = seg["s"]
        E_exec[t:t1] = seg["E"]
        E_now = float(seg["E"][-1])
        if pub in adj_hours:
            off = fc_slots(data[f"fc{pub}"][D], pub)
            src = off if adj_lam is None else adj_lam * off + (1.0 - adj_lam) * ph
            pv_new = cs.margin_pv(np.clip(src, 0.0, None), margin) / 6.0
            if pub in (6, 12) and hedge and D > 14:
                # 无前视联合残差场景对冲（残差块仅取自目标日之前）
                pool = causal_residual_pool(data, D)
                rng = np.random.default_rng(np.random.SeedSequence([seed, D, pub]))
                sel = rng.choice(len(pool), size=max(0, n_scen - 1), replace=True)
                scenario_days.extend(int(pool[k]) for k in sel)
                RL, RP = joint_residual_blocks(data, pool, pub, adj_lam, kappa, margin)
                load_scens = [load_nom[t1:]] + [
                    np.clip(load_nom[t1:] + RL[k][t1:], 0.0, None) for k in sel]
                pv_scens = [pv_new[t1:]] + [
                    np.clip(pv_new[t1:] + RP[k][t1:], 0.0, None) for k in sel]
                x_adj, E_adj = adjust_day_hedge(
                    p_dec, load_nom, pv_scens, x_plan, E_now, t1,
                    e_terminal=e_day_end, load_scens=load_scens)
            else:
                x_adj, E_adj = adjust_day(
                    p_dec, load_nom, pv_new, x_plan, E_now, t1,
                    e_terminal=e_day_end)
            x_seq[t1:] = x_adj
            E_target[t1:] = E_adj
        t = t1
    seg = exec_segment_causal(
        load_act_kwh[108:144], pv_act_kwh[108:144], x_seq[108:144], E_now, None,
    )
    e_total[108:] = seg["e"]
    c_all[108:] = seg["c"]
    d_all[108:] = seg["d"]
    s_all[108:] = seg["s"]
    E_exec[108:] = seg["E"]

    # 结算（附件 4 真实价）：买入 + 0.5p|x_adj−x_plan| + 5p·紧急
    x_final = x_seq.copy()
    buy = float(p4[D] @ x_final)
    dev = float((0.5 * p4[D] * np.abs(x_final - x_plan)).sum())
    emerg = float(EMERG_MULT * (p4[D] @ e_total))
    return {
        "D": D, "x_plan": x_plan, "x_final": x_final,
        "c": c_all, "d": d_all, "s": s_all, "E": E_exec, "E_start": float(e_start),
        "e": e_total, "plan_cost": float(p4[D] @ x_plan),
        "buy": buy, "dev": dev, "emerg": emerg, "total": buy + dev + emerg,
        "E_end": float(E_exec[-1]),
        "scenario_max_day": max(scenario_days, default=-1),
    }


def run_year_q3(data, p4, p_typ, vseq, price_mode="H", storage="daily",
                adj_hours=(6, 12, 18), hedge=True, n_scen=cs.N_SCEN, seed=7,
                lam=0.7, adj_lam=0.7) -> list[dict]:
    """Q4 Q3 层全年滚动（自 REPORT_START 起）：daily 回 E0；2day 跨日传递 SOC。"""
    E = E0
    # 按日序号对齐（占位记录仅用于索引对齐，不计费）
    recs: list[dict] = [{"D": D} for D in range(N_DAY)]
    for D in range(REPORT_START, N_DAY):
        r = simulate_day_q3(
            data, p4, p_typ, vseq, D, price_mode=price_mode, storage=storage,
            adj_hours=adj_hours, hedge=hedge, n_scen=n_scen, seed=seed,
            lam=lam, adj_lam=adj_lam,
            e_start=(E0 if storage == "daily" else E),
        )
        recs[D] = r
        E = r["E_end"] if storage == "2day" else E0
    return recs


def total_of_q3(recs) -> dict:
    """Q3 层费用汇总（仅计费日）：计划/买入/偏差/紧急/合计、紧急与调整电量。"""
    days = [r for r in recs if r["D"] >= REPORT_START]
    return {
        "plan": sum(r["plan_cost"] for r in days),
        "buy": sum(r["buy"] for r in days),
        "dev": sum(r["dev"] for r in days),
        "emerg": sum(r["emerg"] for r in days),
        "total": sum(r["total"] for r in days),
        "emerg_kwh": sum(float(r["e"].sum()) for r in days),
        "adj_kwh": sum(float(np.abs(r["x_final"] - r["x_plan"]).sum()) for r in days),
        "days": len(days),
    }
