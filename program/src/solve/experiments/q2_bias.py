"""Q2 预测偏差实验：精度最优 vs 决策最优（分位数 / 费用标定 / 场景对冲）。

【历史专题·旧口径】基于"日循环 + 口径 D"结构（2026-09-12 前），未适配
2 日滚动 / 统一执行器口径；`q2.run_hedge` / `q2.run_mc` 的签名已在合流中变更，
如需复跑须先适配这两个调用。结论仅作历史引用（见 reports/ 专题文档）。

E1 三条路线（Q2 口径：0:00 计划 take-or-pay + 5 倍紧急）：
  基准   无偏预测（口径 D）
  偏差   光伏折扣 δ（确定性扫描 → 最优 δ*；理论分位数 δ_q = F_e⁻¹(0.8)）
  对冲   两阶段场景 LP（q2.run_hedge）
评估：确定性全年费用（2/1~12/31）+ 蒙特卡洛 50 年（共同随机数）。

运行（program/ 下）：uv run python -m solve.experiments.q2_bias
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve.common import E0, ROOT, T


def simulate_det(data: dict, delta_kw: float = 0.0) -> dict:
    """确定性全年模拟：计划用"0:00 预报 − δ"（光伏折扣），储能跨日连续。"""
    price = data["price"]
    E = E0
    x_all = np.zeros((q2.N_DAY, T))
    plan_cost = 0.0
    emerg_cost = 0.0
    E_feb = None
    daily = np.zeros(q2.N_DAY)
    for d in range(q2.N_DAY):
        pv_raw = q2._hour_to_slots(data["fc0"][d])
        pv_fc = np.clip(pv_raw - delta_kw, 0.0, None) / 6.0
        load_kwh = data["load"][d] / 6.0
        x, _E, _ = q2.plan_day(price, load_kwh, pv_fc, E)
        ex = q2.exec_day_causal(price, load_kwh, data["pv_act"][d] / 6.0, x, E)
        x_all[d] = x
        E = float(ex["E"][-1])
        if d >= q2.REPORT_START:
            c_plan = float(price @ x)
            c_em = float(q2.EMERG_MULT * (price @ ex["e"]))
            plan_cost += c_plan
            emerg_cost += c_em
            daily[d] = c_plan + c_em
        if d == q2.REPORT_START - 1:
            E_feb = E
    return {"plan": plan_cost, "emerg": emerg_cost, "total": plan_cost + emerg_cost,
            "x": x_all, "E_feb": E_feb, "daily": daily}


def theory_delta(data: dict) -> dict:
    """理论分位数偏移：e = 预报 − 实际（白天样本），δ_q = F_e⁻¹(0.8)。"""
    errs = []
    for d in range(q2.N_DAY):
        e = q2._hour_to_slots(data["fc0"][d]) - data["pv_act"][d]
        m = data["pv_act"][d] > 500
        if m.any():
            errs.append(e[m])
    e = np.concatenate(errs)
    return {
        "n": int(e.size),
        "mean_kW": float(e.mean()),
        "sigma_kW": float(e.std()),
        "delta_q80_kW": float(np.quantile(e, 0.80)),
        "delta_normal_kW": float(0.8416 * e.std()),
    }


def main():
    pm.init(seed=42, root=str(ROOT))
    data = q2.load_all()
    t0 = time.time()

    base = simulate_det(data, 0.0)
    print(f"[基准 δ=0] 计划 {base['plan'] / 1e4:.1f} + 紧急 {base['emerg'] / 1e4:.1f} "
          f"= {base['total'] / 1e4:.1f} 万  （自检目标 1231.1 + 155.2 = 1386.3）")
    cache = {0.0: base}

    deltas = [0, 50, 100, 150, 200, 300, 400, 500]
    rows = []
    for dl in deltas:
        r = cache.get(float(dl))
        if r is None:
            r = simulate_det(data, float(dl))
            cache[float(dl)] = r
        rows.append({"delta_kW": dl, "计划费/万元": round(r["plan"] / 1e4, 1),
                     "紧急费/万元": round(r["emerg"] / 1e4, 1),
                     "总费用/万元": round(r["total"] / 1e4, 1)})
        print(f"  δ={dl:4d} kW: 计划 {r['plan'] / 1e4:7.1f} + 紧急 {r['emerg'] / 1e4:6.1f} "
              f"= {r['total'] / 1e4:7.1f} 万   ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "code" / "outputs" / "q2_bias_delta_scan.csv", index=False, encoding="utf-8-sig")

    best = int(df["总费用/万元"].idxmin())
    d_star = float(df.loc[best, "delta_kW"])
    print(f"\n确定性最优偏差 δ* = {d_star:.0f} kW（费用 {df.loc[best, '总费用/万元']} 万）")

    th = theory_delta(data)
    print(f"理论分位数：e = 预报−实际（n={th['n']}），均值 {th['mean_kW']:+.1f} kW、"
          f"σ {th['sigma_kW']:.1f} kW → δ_q80 = {th['delta_q80_kW']:.0f} kW（正态近似 {th['delta_normal_kW']:.0f} kW）")

    # 把理论 δ 也加入缓存
    d_q = float(round(th["delta_q80_kW"] / 50) * 50)     # 归到 50 的倍数
    if d_q not in cache:
        cache[d_q] = simulate_det(data, d_q)
    print(f"理论 δ 取整为 {d_q:.0f} kW：费用 {cache[d_q]['total'] / 1e4:.1f} 万")

    # ---- 对冲策略 ----
    t1 = time.time()
    xh = q2.run_hedge(data, n_scen=20, seed=7)
    plan_h = float(sum(data["price"] @ xh[d] for d in range(q2.REPORT_START, q2.N_DAY)))
    print(f"对冲计划完成（{time.time() - t1:.0f}s），计划费 {plan_h / 1e4:.1f} 万")

    # ---- 蒙特卡洛（共同随机数） ----
    n_years = 50
    strategies = {
        "基准（无偏）": (base["x"], base["E_feb"], base["plan"]),
        f"费用偏差 δ*={d_star:.0f}kW": (cache[d_star]["x"], cache[d_star]["E_feb"], cache[d_star]["plan"]),
        f"理论分位数 δ={d_q:.0f}kW": (cache[d_q]["x"], cache[d_q]["E_feb"], cache[d_q]["plan"]),
        "场景对冲": (xh, base["E_feb"], plan_h),
    }
    rows_mc = []
    for name, (x, e_feb, plan) in strategies.items():
        t2 = time.time()
        mc, _ = q2.run_mc(data, x, e_feb, n_years=n_years, seed=42)
        total = plan + mc
        p95 = float(np.percentile(total, 95))
        rows_mc.append({
            "策略": name, "计划费/万元": round(plan / 1e4, 1),
            "紧急费均值/万元": round(float(mc.mean()) / 1e4, 1),
            "总费用均值/万元": round(float(total.mean()) / 1e4, 1),
            "总费用std/万元": round(float(total.std()) / 1e4, 2),
            "P95/万元": round(p95 / 1e4, 1),
            "CVaR95/万元": round(float(total[total >= p95].mean()) / 1e4, 1),
        })
        print(f"  {name:<18} 均值 {total.mean() / 1e4:7.1f}  std {total.std() / 1e4:5.2f}  "
              f"P95 {p95 / 1e4:7.1f}  ({time.time() - t2:.0f}s)")

    df_mc = pd.DataFrame(rows_mc)
    df_mc.to_csv(ROOT / "code" / "outputs" / "q2_bias_mc.csv", index=False, encoding="utf-8-sig")
    print("\n===== Q2 预测目标对照（确定性 + MC 50 年） =====")
    print(df.to_string(index=False))
    print()
    print(df_mc.to_string(index=False))
    print(f"\n总用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
