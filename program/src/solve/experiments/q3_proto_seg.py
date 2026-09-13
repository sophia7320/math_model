"""Q3 原型第二批：单 λ 滚动策略改进（W/EWMA/范围约束）+ 分时段 λ 网格。

第一部分（秒级，复用 q3_proto 的全网格费用表）：
    C 基线（W=7）之外，评估 W∈{3,14,30,60}、指数加权窗口、λ 范围约束。
第二部分（约 8~10 分钟）：分时段 λ（[0,12h) 与 [12h,24h) 各一个权重）7×7 组合网格，
    滚动窗口选择（无前视）与离线最优（参考）。

用法（program/ 下）：
    uv run python -m solve.experiments.q3_proto_seg              # 全部
    uv run python -m solve.experiments.q3_proto_seg --skip-seg   # 仅第一部分
    uv run python -m solve.experiments.q3_proto_seg --skip-single
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from solve import q2
from solve import q3_proto as qp
from solve.common import E0, ROOT, T

SEG_GRID = np.round(np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]), 2)
DAY_START, N_DAYS = 31, 334      # 与 q3_proto 全年表一致
I_START = 7                      # 评估起点（给滚动窗口预热；所有策略统一）
W_SEG = 7


# ---------------------------------------------------------------------------
# 单 λ 表：滚动规则对比
# ---------------------------------------------------------------------------
def load_single_table():
    path = ROOT / "code" / "outputs" / "q3_proto_lambda_table.csv"
    df = pd.read_csv(path, index_col=0)
    grid = np.array([float(c.split("=")[1]) for c in df.columns])
    return df.to_numpy(float), grid, df.index.to_numpy()


def _pick(rows: np.ndarray, grid: np.ndarray, alpha=None, lo=0.0, hi=1.0) -> int:
    if len(rows) == 0:
        return int(np.argmin(np.abs(grid - 0.5)))
    if alpha is None:
        score = rows.mean(axis=0)
    else:
        w = alpha ** np.arange(len(rows) - 1, -1, -1, dtype=float)
        score = (rows * w[:, None]).sum(axis=0) / w.sum()
    valid = (grid >= lo) & (grid <= hi)
    return int(np.argmin(np.where(valid, score, np.inf)))


def analyze_single(table, grid, dates):
    n = table.shape[0]
    idx_a = len(grid) - 1
    rules = {}

    for W in (3, 7, 14, 30, 60):
        cost = np.empty(n)
        for i in range(n):
            i0 = max(0, i - W)
            li = _pick(table[i0:i], grid)
            cost[i] = table[i, li]
        rules[f"动态 W={W}"] = cost

    for W, a in ((14, 0.6), (30, 0.8), (14, 0.3)):
        cost = np.empty(n)
        for i in range(n):
            i0 = max(0, i - W)
            li = _pick(table[i0:i], grid, alpha=a)
            cost[i] = table[i, li]
        rules[f"EWMA W={W} a={a}"] = cost

    for W, lo, hi in ((7, 0.3, 0.7), (30, 0.2, 0.8), (14, 0.3, 0.7)):
        cost = np.empty(n)
        for i in range(n):
            i0 = max(0, i - W)
            li = _pick(table[i0:i], grid, lo=lo, hi=hi)
            cost[i] = table[i, li]
        rules[f"限幅[{lo},{hi}] W={W}"] = cost

    # 固定与参考
    i_half = int(np.argmin(np.abs(grid - 0.5)))
    rules["固定 λ=0.5"] = table[:, i_half].copy()
    rules["A 纯官方"] = table[:, idx_a].copy()
    # 离线最优（仅参考，信息前视）
    rules["离线最优(参考)"] = table.min(axis=1)

    ev = slice(I_START, None)
    A = table[ev, idx_a].sum()
    rows = []
    for name, cost in rules.items():
        tot = cost[ev].sum()
        rows.append({
            "规则": name,
            "总费用/万元": round(tot / 1e4, 1),
            "vs A": f"{100 * (tot / A - 1):+.2f}%",
        })
    out = pd.DataFrame(rows)
    print("\n===== 单 λ 表：滚动规则对比（%d~%d 天，n=%d）=====" % (I_START + 1, n, n - I_START))
    print(out.to_string(index=False))
    return out, rules


# ---------------------------------------------------------------------------
# 分时段 λ 网格
# ---------------------------------------------------------------------------
def _day_cost(data, D, lam_vec):
    """分时段组合的单日 Q3 费用（标量四元组）。"""
    price = data["price"]
    load_kwh = data["load"][D] / 6.0
    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = qp.hist_forecast(data, D)
    pv_plan = (lam_vec * f0 + (1.0 - lam_vec) * ph) / 6.0

    x_plan, E_plan, _ = q2.plan_day(price, load_kwh, pv_plan, E0, eps=qp.EPS_TH)
    x_final = x_plan.copy()
    for t0, fc, pub in ((36, data["fc6"][D], 6), (72, data["fc12"][D], 12), (108, data["fc18"][D], 18)):
        pv_new = np.clip(qp.fc_slots(fc, pub), 0.0, None) / 6.0
        x_adj, _E_adj = qp.adjust_day(price, load_kwh, pv_new, x_plan, float(E_plan[t0 - 1]), t0)
        x_final[t0:] = x_adj

    ex = q2.exec_day(price, load_kwh, data["pv_act"][D] / 6.0, x_final, E0)
    base = float((price * np.minimum(x_plan, x_final)).sum())
    dev_lo = float((0.5 * price * np.maximum(x_plan - x_final, 0.0)).sum())
    dev_hi = float((1.5 * price * np.maximum(x_final - x_plan, 0.0)).sum())
    emerg = float((qp.EMERG_MULT * price * ex["e"]).sum())
    plan_cost = float(price @ x_plan)
    total = base + dev_lo + dev_hi + emerg
    return total, plan_cost, base + dev_lo + dev_hi - plan_cost, emerg


def run_seg_table(force=False):
    out = ROOT / "code" / "outputs" / "q3_proto_seg_table.npz"
    if out.exists() and not force:
        z = np.load(out)
        print(f"分时段表命中缓存：{out.name}")
        return z["table"], z["dates"]

    data = qp.load_extended()
    ng = len(SEG_GRID)
    table = np.zeros((N_DAYS, ng, ng))
    dates = [data["dates"][d] for d in range(DAY_START, DAY_START + N_DAYS)]
    ts = time.time()
    for i, D in enumerate(range(DAY_START, DAY_START + N_DAYS)):
        cost_grid = np.zeros((ng, ng))
        for a_i, lam_a in enumerate(SEG_GRID):
            for p_i, lam_p in enumerate(SEG_GRID):
                lam_vec = np.concatenate([np.full(72, lam_a), np.full(72, lam_p)])
                cost_grid[a_i, p_i] = _day_cost(data, D, lam_vec)[0]
        table[i] = cost_grid
        if (i + 1) % 20 == 0 or i == N_DAYS - 1:
            print(f"  分时段 {i + 1}/{N_DAYS} 天，用时 {time.time() - ts:.0f}s")
    np.savez_compressed(out, table=table, dates=np.array(dates))
    print(f"分时段表已保存：{out}")
    return table, np.array(dates)


def analyze_seg(table, dates, single_table, grid):
    n = table.shape[0]
    ng = len(SEG_GRID)
    cost_seg = np.empty(n)
    cost_seg_off = np.empty(n)
    picks = []
    for i in range(n):
        i0 = max(0, i - W_SEG)
        score = table[i0:i].mean(axis=0) if i > i0 else table[:1].mean(axis=0)
        a_i, p_i = np.unravel_index(np.argmin(score), (ng, ng))
        cost_seg[i] = table[i, a_i, p_i]
        picks.append((SEG_GRID[a_i], SEG_GRID[p_i]))
        a_i2, p_i2 = np.unravel_index(np.argmin(table[i]), (ng, ng))
        cost_seg_off[i] = table[i, a_i2, p_i2]
    picks = np.array(picks)

    i_half = int(np.argmin(np.abs(grid - 0.5)))
    idx_a = len(grid) - 1
    i_half_seg = int(np.argmin(np.abs(SEG_GRID - 0.5)))
    ev = slice(I_START, None)

    def report(name, cost):
        tot = cost[ev].sum()
        a_tot = single_table[ev, idx_a].sum()
        print(f"  {name:<26}{tot / 1e4:>10.1f}   {100 * (tot / a_tot - 1):>+7.2f}%")

    print("\n===== 分时段 λ 对比（n=%d）=====" % (n - I_START))
    print(f"  {'策略':<26}{'总费用/万元':>10}   {'vs A':>7}")
    report("A 纯官方", single_table[:, idx_a])
    report("固定 λ=0.5（单）", single_table[:, i_half])
    report("固定 0.5/0.5（分时段）", table[:, i_half_seg, i_half_seg])
    # 单 λ 动态 W=7（复算）
    c7 = np.empty(n)
    for i in range(n):
        i0 = max(0, i - 7)
        li = _pick(single_table[i0:i], grid)
        c7[i] = single_table[i, li]
    report("动态单 λ W=7", c7)
    report("动态分时段 W=7", cost_seg)
    report("分时段离线最优(参考)", cost_seg_off)

    print("\n  分时段滚动规则扫描（W × 限幅）：")
    for W in (14, 30, 60):
        for lo, hi in ((0.0, 1.0), (0.2, 0.8), (0.3, 0.7)):
            cost = np.empty(n)
            for i in range(n):
                i0 = max(0, i - W)
                score = table[i0:i].mean(axis=0) if i > i0 else table[:1].mean(axis=0)
                valid = (SEG_GRID >= lo - 1e-9) & (SEG_GRID <= hi + 1e-9)
                score = np.where(valid, score, np.inf)
                a_i, p_i = np.unravel_index(np.argmin(score), (ng, ng))
                cost[i] = table[i, a_i, p_i]
            report(f"分时段 W={W} 限[{lo},{hi}]", cost)

    # 单 λ 最优规则（同评估区间，用于公平对比）
    c30 = np.empty(n)
    for i in range(n):
        i0 = max(0, i - 30)
        li = _pick(single_table[i0:i], grid, lo=0.2, hi=0.8)
        c30[i] = single_table[i, li]
    report("单λ W=30 限[0.2,0.8]", c30)
    print(f"  分时段滚动选择：λ_am 均值 {picks[I_START:, 0].mean():.2f}，"
          f"λ_pm 均值 {picks[I_START:, 1].mean():.2f}")

    df = pd.DataFrame({
        "日期": dates,
        "动态分时段费用": cost_seg,
        "动态单λ费用": c7,
        "λ_am": picks[:, 0],
        "λ_pm": picks[:, 1],
    })
    df.to_csv(ROOT / "code" / "outputs" / "q3_proto_seg_picks.csv", index=False, encoding="utf-8-sig")
    return cost_seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-seg", action="store_true")
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--force-seg", action="store_true")
    args = ap.parse_args()

    single, grid, dates = load_single_table()
    if not args.skip_single:
        analyze_single(single, grid, dates)

    if not args.skip_seg:
        table, dates2 = run_seg_table(force=args.force_seg)
        analyze_seg(table, dates2, single, grid)


if __name__ == "__main__":
    main()
