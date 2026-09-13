"""Q4 电价预测参数结构原型（仿口径 E）：三源自适应加权的误差与费用验证。

结构：
    p̂(D,t) = v1·P(D−1,t) + v2·P(D−7,t) + v3·P̄(t)，v = softmax 凸权重（网格 0.1）
- 误差口径：全期 MAE 网格扫描（找最优权重结构）
- 费用口径：预测价 → 计划 LP（take-or-pay）→ 执行按真实价（缺口 5 倍真实价）
  · 事后最优权重（参考上界）
  · 滚动 W=7 标定（可部署，无前视）

运行（program/ 下）：uv run python -m solve.q4_price_fit

口径（2026-09-13 统一 v1.2）：计划输入 = κ·L̂ 与 max(P̂−m,0)（consistency.py），
执行 = 逐槽因果 free（无段末硬目标）；末端自由。表随执行/风险口径变化时须重跑。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import program as pm
from solve import consistency as cs
from solve import q2
from solve.common import DATA_C, E0, ROOT, T
from solve.data.attachments import load_extended
from solve.models.adaptive import hist_forecast_asof, hist_load_forecast_asof
from solve.models.price import (  # 唯一实现（models/price.py），此处再导出
    forecast_price,
    forecast_price_next_asof,
)
from solve.models.weights import simplex_grid  # 唯一实现（models/weights.py）

DAY_START = q2.REPORT_START      # 31
N_DAY = q2.N_DAY                 # 365


def mae_of(p4, p_typ, v, days):
    errs = [np.abs(forecast_price(D, v, p4, p_typ) - p4[D]).mean() for D in days]
    return float(np.mean(errs))


def main():
    pm.init(seed=42, root=str(ROOT))
    data = load_extended()
    p_typ = data["price"]  # 附件 1 = 逐槽均值
    df4 = pm.read_table(DATA_C / "附件4.xlsx")
    p4 = df4.iloc[:, 1:145].to_numpy(float)
    days = list(range(DAY_START, N_DAY))
    grid = simplex_grid(0.1)
    t0 = time.time()

    # ---- 1) 误差口径（逐日矩阵，供滚动误差标定复用）----
    rows = []
    daily_mae = np.zeros((N_DAY, len(grid)))
    for j, v in enumerate(grid):
        for D in range(7, N_DAY):
            daily_mae[D, j] = np.abs(forecast_price(D, v, p4, p_typ) - p4[D]).mean()
        rows.append({"v1_D-1": v[0], "v2_D-7": v[1], "v3_典型日": v[2],
                     "MAE/元": float(daily_mae[days, j].mean())})
    df = pd.DataFrame(rows).sort_values("MAE/元")
    best_err = df.iloc[0]
    print("=== 误差口径网格（前 8）===")
    print(df.head(8).to_string(index=False))
    for name, vec in (("D−1 单源", (1, 0, 0)), ("D−7 单源", (0, 1, 0)),
                      ("典型日单源", (0, 0, 1)), ("等权", (1 / 3, 1 / 3, 1 / 3))):
        print(f"  {name:<10} MAE = {mae_of(p4, p_typ, vec, days):.4f} 元")

    # ---- 2) 费用：一次遍历（事后最优 + 滚动标定共用）----
    print("\n费用网格（一次遍历，66 组）……")
    daily_costs = np.zeros((N_DAY, len(grid)))
    E_state = np.full(len(grid), float(E0))
    for D in range(N_DAY):
        for j, v in enumerate(grid):
            p_hat = forecast_price(D, v, p4, p_typ) if D >= 7 else p4[D]
            p_hat1 = forecast_price_next_asof(D, v, p4, p_typ)
            load0 = cs.kappa_load(hist_load_forecast_asof(data, D, D)) / 6.0
            load1 = cs.kappa_load(hist_load_forecast_asof(data, D + 1, D)) / 6.0
            pv0 = cs.margin_pv(q2._hour_to_slots(data["fc0"][D])) / 6.0
            pv1 = cs.margin_pv(hist_forecast_asof(data, D + 1, D)) / 6.0
            xh, _Eh, _ = q2.plan_horizon(
                np.concatenate([p_hat, p_hat1]), np.concatenate([load0, load1]),
                np.concatenate([pv0, pv1]), E_state[j], None, eps=1e-3,
            )
            x = xh[:T]
            ex = q2.exec_segment_causal(
                data["load"][D] / 6.0, data["pv_act"][D] / 6.0, x, E_state[j], None,
            )
            E_state[j] = float(ex["E"][-1])
            daily_costs[D, j] = float(p4[D] @ x + q2.EMERG_MULT * (p4[D] @ ex["e"]))
        if (D + 1) % 60 == 0:
            print(f"  {D + 1}/{N_DAY} 天 （{time.time() - t0:.0f}s）")

    idx = {(round(v[0], 4), round(v[1], 4), round(v[2], 4)): j for j, v in enumerate(grid)}
    after_tot = daily_costs[days].sum(axis=0)
    rows_c = [{"v1_D-1": v[0], "v2_D-7": v[1], "v3_典型日": v[2],
               "总费用/万元": round(after_tot[j] / 1e4, 1)} for j, v in enumerate(grid)]
    dfc = pd.DataFrame(rows_c).sort_values("总费用/万元")
    print("=== 费用口径·事后最优（前 8）===")
    print(dfc.head(8).to_string(index=False))

    # ---- 3) 滚动 W=7 标定（无前视，复用 daily_costs）----
    roll = 0.0
    picks = []
    for D in days:
        i0 = max(DAY_START, D - 7)
        j = (idx[(0.0, 0.0, 1.0)] if i0 == D
             else int(np.argmin(daily_costs[i0:D].mean(axis=0))))
        roll += daily_costs[D, j]
        picks.append(grid[j])
    print(f"滚动 W=7 总费用 {roll / 1e4:.1f} 万元，平均权重 "
          f"v=({np.mean([p[0] for p in picks]):.2f}, {np.mean([p[1] for p in picks]):.2f}, "
          f"{np.mean([p[2] for p in picks]):.2f})")

    # ---- 滚动（误差标定，对照）----
    roll_e = 0.0
    pe = []
    for D in days:
        i0 = max(DAY_START, D - 7)
        j = (idx[(0.0, 0.0, 1.0)] if i0 == D
             else int(np.argmin(daily_mae[i0:D].mean(axis=0))))
        roll_e += daily_costs[D, j]
        pe.append(grid[j])
    print(f"滚动 W=7（误差标定）总费用 {roll_e / 1e4:.1f} 万元，平均权重 "
          f"v=({np.mean([p[0] for p in pe]):.2f}, {np.mean([p[1] for p in pe]):.2f}, "
          f"{np.mean([p[2] for p in pe]):.2f})")

    # ---- 对照（全部从 daily_costs 取）----
    print("\n=== 对照（334 天费用）===")
    for name, v in (("D−1 单源", (1, 0, 0)), ("D−7 单源", (0, 1, 0)),
                    ("典型日单源", (0, 0, 1)), ("近似等权(0.3/0.3/0.4)", (0.3, 0.3, 0.4))):
        j = idx[(round(v[0], 4), round(v[1], 4), round(v[2], 4))]
        print(f"  {name:<18}: {daily_costs[days, j].sum() / 1e4:7.1f} 万元")
    ve = (round(best_err['v1_D-1'], 4), round(best_err['v2_D-7'], 4), round(best_err['v3_典型日'], 4))
    if ve in idx:
        print(f"  误差最优权重 {ve}: {daily_costs[days, idx[ve]].sum() / 1e4:7.1f} 万元")
    vc = (round(dfc.iloc[0]['v1_D-1'], 4), round(dfc.iloc[0]['v2_D-7'], 4), round(dfc.iloc[0]['v3_典型日'], 4))
    print(f"  费用最优权重 {vc}: {daily_costs[days, idx[vc]].sum() / 1e4:7.1f} 万元")
    print(f"  滚动 W=7（费用标定）  : {roll / 1e4:7.1f} 万元")
    print(f"  滚动 W=7（误差标定）  : {roll_e / 1e4:7.1f} 万元")

    df.to_csv(ROOT / "code" / "outputs" / "q4_price_fit_mae.csv", index=False, encoding="utf-8-sig")
    dfc.to_csv(ROOT / "code" / "outputs" / "q4_price_fit_cost.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(ROOT / "code" / "outputs" / "q4_price_fit_daily.npz",
                        daily_costs=daily_costs, daily_mae=daily_mae, grid=np.array(grid))
    print(f"\n已保存 code/outputs/q4_price_fit_*.csv（总用时 {time.time() - t0:.0f}s）")


if __name__ == "__main__":
    main()
