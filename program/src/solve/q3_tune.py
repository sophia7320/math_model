"""Q3 对冲扩展下的历史风险参数时间留出实验（λ / κ / m）。

在统一结构（2 日滚动 + 因果负荷 + EWMA h=5 预测 + κ/m + （Δ负荷, Δ光伏）联合残差
40 情景对冲）下，对已退出正式方案的扩展模型做两级扫描：

A. 粗筛：λ=0.7 固定，κ∈{1.0, 1.01, 1.015, 1.02} × m∈{0, 25, 50, 75}（16 组）；
B. 细选：最优 (κ, m) 上，λ∈{0.3, 0.5, 0.7, 0.9, 1.0}（5 组）。

协议：开发期 2025-02-01 ~ 06-30 只负责选参；冻结验证期 2025-07-01 ~ 12-31
不参与选择（同 q2_tune 的时间留出做法）。参数经 ``consistency.py`` 透传，
随机流由 (seed, D, pub) 固定，各配置共用随机数。

输出：code/outputs/q3e_tune_unified_{coarse,fine}.csv、
      figures/Q3E_参数搜索.pdf、RESULTS_REPORT 章节。

运行（program/ 下）：uv run python -m solve.q3_tune
"""
from __future__ import annotations

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import program as pm
from solve import consistency as cs
from solve import q3_proto as qp
from solve.common import ROOT
from solve.io.report import record

REP = list(range(cs.START_DAY, 365))
LAM_FIX = 0.7
KAPPA_GRID = (1.0, 1.01, 1.015, 1.02)
MARGIN_GRID = (0.0, 25.0, 50.0, 75.0)
LAM_GRID = (0.3, 0.5, 0.7, 0.9, 1.0)
SEED = 7
WORKERS = 6

_P = None  # 子进程全局负载


def _init_worker(payload) -> None:
    global _P
    _P = payload


def _task(cfg) -> dict:
    lam, kappa, margin = cfg
    p = _P
    data = p["data"]
    t0 = time.time()
    E = float(qp.E0)
    costs = np.zeros(len(REP))
    for i, D in enumerate(REP):
        r = qp.simulate_day_rt_hedge(
            data, D, lam, adj_lam=lam, n_scen=cs.N_SCEN, seed=SEED,
            kappa=kappa, margin=margin, e_start=E)
        costs[i] = r["total"]
        E = float(r["E_end"])
    split = p["split"]
    dev = float(costs[:split].sum()) / 1e4
    val = float(costs[split:].sum()) / 1e4
    print(f"  λ={lam:g} κ={kappa:g} m={margin:g}: 开发 {dev:.1f} 万 / "
          f"冻结 {val:.1f} 万（{time.time() - t0:.0f}s）")
    return {"配置": f"λ={lam:g} κ={kappa:g} m={margin:g}", "lam": lam,
            "kappa": kappa, "margin": margin,
            "开发期(2-6月)/万元": round(dev, 1),
            "冻结验证期(7-12月)/万元": round(val, 1),
            "全年描述值/万元": round(dev + val, 1)}


def _run(payload, cfgs, workers=WORKERS) -> pd.DataFrame:
    with Pool(workers, initializer=_init_worker, initargs=(payload,)) as pool:
        rows = pool.map(_task, cfgs)
    return pd.DataFrame(rows)


def main() -> None:
    pm.init(seed=42, root=str(ROOT))
    t0 = time.time()
    data = qp.load_extended()
    split = next(i for i, d in enumerate(REP)
                 if data["dates"][d] == cs.HOLDOUT_VAL_START)
    payload = {"data": data, "split": split}
    print(f"对冲扩展历史参数搜索：开发期到 2025-06-30（{split} 天），"
          f"冻结验证期起 {cs.HOLDOUT_VAL_START}；情景数 {cs.N_SCEN}，workers={WORKERS}")

    # ---------- A. 粗筛 κ × m（λ 固定） ----------
    cfgs = [(LAM_FIX, k, m) for k in KAPPA_GRID for m in MARGIN_GRID]
    coarse = _run(payload, cfgs)
    coarse = coarse.sort_values(
        ["开发期(2-6月)/万元", "冻结验证期(7-12月)/万元"]).reset_index(drop=True)
    coarse["开发期选中"] = coarse.index == 0
    coarse["验证期排名"] = coarse["冻结验证期(7-12月)/万元"].rank(method="min").astype(int)
    coarse.to_csv(pm.outputs_dir() / "q3e_tune_unified_coarse.csv",
                  index=False, encoding="utf-8-sig")
    best = coarse.iloc[0]
    print(f"粗筛选中：{best['配置']}（开发 {best['开发期(2-6月)/万元']} 万，"
          f"冻结 {best['冻结验证期(7-12月)/万元']} 万，验证排名 {best['验证期排名']}）")

    # ---------- B. 细选 λ ----------
    cfgs2 = [(lam, float(best["kappa"]), float(best["margin"])) for lam in LAM_GRID]
    fine = _run(payload, cfgs2, workers=min(WORKERS, len(cfgs2)))
    fine = fine.sort_values(
        ["开发期(2-6月)/万元", "冻结验证期(7-12月)/万元"]).reset_index(drop=True)
    fine["开发期选中"] = fine.index == 0
    fine["验证期排名"] = fine["冻结验证期(7-12月)/万元"].rank(method="min").astype(int)
    fine.to_csv(pm.outputs_dir() / "q3e_tune_unified_fine.csv",
                index=False, encoding="utf-8-sig")
    best2 = fine.iloc[0]
    print(f"细选选中：{best2['配置']}（开发 {best2['开发期(2-6月)/万元']} 万，"
          f"冻结 {best2['冻结验证期(7-12月)/万元']} 万，验证排名 {best2['验证期排名']}）")

    # ---------- 图 ----------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
    ax = axes[0]
    for k in KAPPA_GRID:
        g = coarse[np.isclose(coarse["kappa"], k)]
        ax.plot(g["margin"], g["开发期(2-6月)/万元"], marker="o", ms=3,
                label=f"κ={k:g}（开发期）")
    ax.set_xlabel("光伏折减 m / kW")
    ax.set_ylabel("开发期费用 / 万元")
    ax.set_title("(a) 粗筛 κ×m（开发期）", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    ax.plot(fine["lam"], fine["开发期(2-6月)/万元"], marker="o", label="开发期（选参）")
    ax.plot(fine["lam"], fine["冻结验证期(7-12月)/万元"], marker="s", label="冻结验证期")
    ax.set_xlabel("组合权重 λ")
    ax.set_ylabel("费用 / 万元")
    ax.set_title("(b) λ 细选", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    pm.save_fig(fig, "Q3E_参数搜索",
                data=pd.concat([coarse.assign(阶段="粗筛κ×m"),
                                fine.assign(阶段="细选λ")], ignore_index=True))

    # ---------- 报告 ----------
    record("问题三 对冲扩展历史参数搜索：粗筛（κ×m）", coarse,
           note=("已退出正式方案的对冲扩展（2 日滚动+因果负荷+EWMA+联合残差 40 情景）下，"
                 f"λ={LAM_FIX:g} 固定，κ∈{KAPPA_GRID}、m∈{MARGIN_GRID}；"
                 "开发期 2–6 月选参，7–12 月冻结验证；常用随机数（seed=7）。"))
    record("问题三 对冲扩展历史参数搜索：λ 细选", fine,
           note=(f"在粗筛最优 (κ,m)=({best['kappa']:g},{best['margin']:g}) 上扫描 "
                 f"λ∈{LAM_GRID}；开发期选择、冻结验证。"))
    record("问题三 对冲扩展参数搜索：汇总",
           pd.DataFrame([{"阶段": "粗筛", **best.to_dict()},
                         {"阶段": "细选", **best2.to_dict()}]),
           note=("这是对冲扩展的历史搜索，不再决定 v1.4 正式策略；验证期仅报告排名。"
                 "完整表见 code/outputs/q3e_tune_unified_*.csv；"
                 "图 figures/Q3E_参数搜索.pdf。"))
    print(f"完成，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
