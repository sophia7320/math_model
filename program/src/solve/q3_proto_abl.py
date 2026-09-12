"""Q3 原型第三批：调整点消融、信息/纠偏价值分解、结算口径对照、蒙特卡洛尾部。

主口径：λ=1（纯官方预报），评估"调整机制本身"的价值，与题面口径一致。

实验：
  ① 调整点消融：8 种调整时刻组合（含无调整 = Q2），量化 6/12/18 各点的边际价值
  ② 价值分解：V_B 无调整、V_C 有调整但不用新预报、V_A 现实 → 信息价值 vs 纠偏价值
  ③ 结算口径：final（计划 vs 最终值一次结算）vs sequential（逐次结算）
  ④ 蒙特卡洛：残余误差整日块重采样，比较"有调整 vs 无调整"的费用分布与尾部风险

运行（program/ 下）：
    uv run python -m solve.q3_proto_abl              # 全部（约 6 分钟）
    uv run python -m solve.q3_proto_abl --skip-mc    # 只做 ①②③
    uv run python -m solve.q3_proto_abl --years 30
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from solve import q2
from solve import q3_proto as qp
from solve.common import E0, ROOT, T
from solve.core.residual import latest_forecast

DAY_START, N_DAYS = 31, 334
DAYS = list(range(DAY_START, DAY_START + N_DAYS))
LAM = 1.0
EMERG_MULT = qp.EMERG_MULT


def total_of(data, lam=LAM, **kw) -> float:
    return float(sum(qp.simulate_day(data, D, lam, **kw)["total"] for D in DAYS))


# ---------------------------------------------------------------------------
# ① 调整点消融
# ---------------------------------------------------------------------------
def run_ablation(data) -> dict:
    combos = [(), (6,), (12,), (18,), (6, 12), (6, 18), (12, 18), (6, 12, 18)]
    names = {(): "无调整（Q2）", (6,): "仅 6:00", (12,): "仅 12:00", (18,): "仅 18:00",
             (6, 12): "6+12", (6, 18): "6+18", (12, 18): "12+18", (6, 12, 18): "全部（Q3-A）"}
    res = {}
    t0 = time.time()
    for c in combos:
        res[c] = total_of(data, adj_hours=c)
        print(f"  {names[c]:<12} {res[c] / 1e4:9.1f} 万   ({time.time() - t0:.0f}s)")

    base = res[()]
    rows = [{"调整点": names[c], "总费用/万元": round(res[c] / 1e4, 1),
             "vs 无调整": f"{100 * (res[c] / base - 1):+.2f}%"} for c in combos]
    df = pd.DataFrame(rows)
    print("\n① 调整点消融（λ=1，334 天）：")
    print(df.to_string(index=False))
    print("\n逐点边际节省（万元）：")
    print(f"  单独开 6:00        : {(res[()] - res[(6,)]) / 1e4:6.1f}")
    print(f"  单独开 12:00       : {(res[()] - res[(12,)]) / 1e4:6.1f}")
    print(f"  单独开 18:00       : {(res[()] - res[(18,)]) / 1e4:6.1f}")
    print(f"  6:00 之后再开 12:00 : {(res[(6,)] - res[(6, 12)]) / 1e4:6.1f}")
    print(f"  6+12 之后再开 18:00 : {(res[(6, 12)] - res[(6, 12, 18)]) / 1e4:6.1f}")
    df.to_csv(ROOT / "code" / "outputs" / "q3_proto_ablation.csv", index=False, encoding="utf-8-sig")
    return res


# ---------------------------------------------------------------------------
# ② 信息价值 vs 纠偏价值
# ---------------------------------------------------------------------------
def run_decomposition(data, res_abl):
    V_B = res_abl[()]                                              # 无调整
    V_A = res_abl[(6, 12, 18)]                                     # 现实
    V_C = total_of(data, adj_hours=(6, 12, 18), use_new_fc=False)  # 有调整但无新预报
    print("\n② 调整机制价值分解（万元）：")
    print(f"  V_B 无调整                : {V_B / 1e4:8.1f}")
    print(f"  V_C 有调整、无新预报       : {V_C / 1e4:8.1f}")
    print(f"  V_A 有调整、新预报（现实）  : {V_A / 1e4:8.1f}")
    print(f"  总价值 V_B−V_A            : {(V_B - V_A) / 1e4:8.1f}（{100 * (V_B - V_A) / V_B:.2f}%）")
    print(f"    ├ 信息价值 V_C−V_A      : {(V_C - V_A) / 1e4:8.1f}")
    print(f"    └ 纠偏价值 V_B−V_C      : {(V_B - V_C) / 1e4:8.1f}")
    return {"V_B": V_B, "V_C": V_C, "V_A": V_A}


# ---------------------------------------------------------------------------
# ③ 结算口径
# ---------------------------------------------------------------------------
def run_settle(data):
    print("\n③ 结算口径对照：")
    rows = []
    for lam in (1.0, 0.5):
        for settle in ("final", "sequential"):
            tot = total_of(data, lam=lam, settle=settle)
            rows.append({"λ": lam, "口径": settle, "总费用/万元": round(tot / 1e4, 1)})
            print(f"  λ={lam}  {settle:<11} {tot / 1e4:9.1f} 万")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "code" / "outputs" / "q3_proto_settle.csv", index=False, encoding="utf-8-sig")
    return df


# ---------------------------------------------------------------------------
# ④ 蒙特卡洛：尾部风险
# ---------------------------------------------------------------------------
def _stats(x):
    p95 = float(np.percentile(x, 95))
    return {"mean": float(x.mean()), "std": float(x.std()),
            "P95": p95, "CVaR95": float(x[x >= p95].mean())}


def run_mc(data, years=50, seed=42):
    price = data["price"]
    months = np.asarray(data["months"])

    # 策略 A（λ=1）的固定决策
    x_plan_all, x_final_all, fix_adj, fix_no = {}, {}, {}, {}
    t0 = time.time()
    for D in DAYS:
        r = qp.simulate_day(data, D, LAM)
        x_plan_all[D], x_final_all[D] = r["x_plan"], r["x_final"]
        fix_adj[D] = float(price @ r["x_final"]) + float(
            (0.5 * price * np.abs(r["x_final"] - r["x_plan"])).sum())
        fix_no[D] = float(price @ r["x_plan"])
    print(f"  固定决策预计算完成（{time.time() - t0:.0f}s）")

    # 残余误差矩阵（kW）：最新预报 − 实际、0:00 预报 − 实际
    R_latest = np.zeros((365, T))
    R_q2 = np.zeros((365, T))
    for D in range(365):
        R_latest[D] = latest_forecast(data, D) - data["pv_act"][D]
        R_q2[D] = q2._hour_to_slots(data["fc0"][D]) - data["pv_act"][D]

    rng = np.random.default_rng(seed)
    cost_adj = np.zeros(years)
    cost_no = np.zeros(years)
    emerg_adj = np.zeros(years)
    emerg_no = np.zeros(years)
    t0 = time.time()
    for rep in range(years):
        for D in DAYS:
            pool = np.where(months == months[D])[0]
            idx = int(rng.choice(pool))
            load_kwh = data["load"][D] / 6.0

            pv_s = np.clip(latest_forecast(data, D) + R_latest[idx], 0.0, None) / 6.0
            ex = q2.exec_day(price, load_kwh, pv_s, x_final_all[D], E0)
            ec = EMERG_MULT * float(price @ ex["e"])
            emerg_adj[rep] += ec
            cost_adj[rep] += fix_adj[D] + ec

            pv_s0 = np.clip(q2._hour_to_slots(data["fc0"][D]) + R_q2[idx], 0.0, None) / 6.0
            ex0 = q2.exec_day(price, load_kwh, pv_s0, x_plan_all[D], E0)
            ec0 = EMERG_MULT * float(price @ ex0["e"])
            emerg_no[rep] += ec0
            cost_no[rep] += fix_no[D] + ec0
        if (rep + 1) % 10 == 0:
            print(f"  MC {rep + 1}/{years}，{time.time() - t0:.0f}s")

    sa, sn = _stats(cost_adj), _stats(cost_no)
    print("\n④ 蒙特卡洛（%d 年，共同随机数）：" % years)
    print(f"  {'指标':<22}{'无调整（Q2）':>14}{'有调整（Q3-A）':>16}")
    for k, lab in (("mean", "总费用均值/万元"), ("std", "总费用标准差/万元"),
                   ("P95", "总费用 P95/万元"), ("CVaR95", "总费用 CVaR95/万元")):
        print(f"  {lab:<22}{sn[k] / 1e4:>14.1f}{sa[k] / 1e4:>16.1f}")
    print(f"  {'紧急费均值/万元':<22}{emerg_no.mean() / 1e4:>14.1f}{emerg_adj.mean() / 1e4:>16.1f}")
    print(f"  {'紧急费 P95/万元':<22}{np.percentile(emerg_no, 95) / 1e4:>14.1f}"
          f"{np.percentile(emerg_adj, 95) / 1e4:>16.1f}")
    print(f"  总费用均值下降 {100 * (1 - sa['mean'] / sn['mean']):+.2f}%，"
          f"std 下降 {100 * (1 - sa['std'] / sn['std']):+.1f}%，"
          f"P95 下降 {100 * (1 - sa['P95'] / sn['P95']):+.1f}%")

    out = pd.DataFrame({
        "无调整_总费用_元": cost_no, "有调整_总费用_元": cost_adj,
        "无调整_紧急费_元": emerg_no, "有调整_紧急费_元": emerg_adj,
    })
    out.to_csv(ROOT / "code" / "outputs" / "q3_proto_mc.csv", index=False, encoding="utf-8-sig")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--skip-abl", action="store_true")
    ap.add_argument("--years", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qp.pm.init(seed=args.seed, root=str(ROOT))
    data = qp.load_extended()

    res_abl = None
    if not args.skip_abl:
        print("① 调整点消融（8 种组合）……")
        res_abl = run_ablation(data)
        print("\n② 价值分解……")
        run_decomposition(data, res_abl)
        print("\n③ 结算口径……")
        run_settle(data)

    if not args.skip_mc:
        print("\n④ 蒙特卡洛尾部……")
        run_mc(data, years=args.years, seed=args.seed)


if __name__ == "__main__":
    main()
