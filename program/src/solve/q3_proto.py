"""C 题 问题三原型：官方预报 + 历史预测组合、滚动成本标定（兼容门面 + 原型入口）。

口径（原型，正式口径见 solve/q3.py）：
- 0:00 计划：光伏预测 = λ·官方 f0 + (1−λ)·历史预测（口径 E 加权；u 来自 Q2E 缓存）
- 6/12/18 调整：用附件 3 最新预报对未执行时段重优化（偏差按 50%/150% 结算）
- 执行：购电承诺 take-or-pay，储能再调度（日循环），缺口 5 倍价紧急购电
- 结算：p·min(plan,adj) + 0.5p·(plan−adj)⁺ + 1.5p·(adj−plan)⁺ + 5p·e

对比策略：
- A：λ=1（纯官方）；B：λ=0.5（固定组合）；C：滚动窗口 W 天费用最优 λ（动态组合）
- D：λ=0（纯历史）；P：完美下界（0:00 用实际光伏做计划，不调整）

实现分层（本文件只保留原型批处理 main 与兼容导出）：
- 调整/对冲 LP  solve.core.lp；槽位/残差池  solve.core.slots / solve.core.residual
- 单日模拟      solve.flows.q3_day；数据扩展  solve.data.attachments
- 历史预报/平滑 solve.models.adaptive / solve.models.weights

运行（program/ 下）：
    uv run python -m solve.q3_proto --start 52 --days 67
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import program as pm
from solve.common import EMERG_MULT, EPS_TH, ROOT, T
from solve.core.lp import adjust_day, adjust_day_hedge, exec_day, exec_segment_hindsight
from solve.core.residual import causal_residual_pool, latest_forecast
from solve.core.slots import fc_slots
from solve.data.attachments import load_extended
from solve.flows.q3_day import (
    forecast_residual,
    perfect_day,
    simulate_day,
    simulate_day_rt,
    simulate_day_rt_hedge,
)
from solve.models.adaptive import hist_forecast
from solve.models.weights import make_smooth_u
from solve.models.weights import softmax as _softmax

LAMBDA_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 2)   # 11 个组合权重
W_WINDOW = 7            # 滚动标定窗口（天）
DAY_TABLE_START = 52    # 建表起点（给滚动窗口预热）
DAY_EVAL_START = 59     # 评估起点 2025-03-01
DAY_END = 119           # 评估终点（不含）= 2025-04-30


# ---------------------------------------------------------------------------
# 主流程：λ 网格成本表 + 滚动窗口选择（无前视）
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

    # ---- 成本表：行=日历日，列=λ 网格；simulate_day 为事后执行口径（原型对照） ----
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

    # ---- 评估（滚动窗口在 table 行上，无前视）----
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
