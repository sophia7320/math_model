"""Q3 主方案灵敏度：情景数 / 场景池窗口 / 组合权重 λ（因果口径，不改变正式结果）。

对照 GPT 版"场景数 5/10/20 + λ 窗口"的灵敏度思路；本工作区 Q3 的 λ 为固定值
（无滚动窗口），因此等价地补测：
  - 对冲情景数 n_scen ∈ {5, 10, 20, 40}，并对 10/20/40 使用 3 个随机种子
  - 场景池回看窗口 lookback ∈ {30, 60, 90}（正式 90；同日样本 ≥14 时优先同月）
  - 组合权重 λ ∈ {0.5, 0.7, 0.9}（正式 0.7，0:00 与调整层同值）

运行（program/ 下）：uv run python -m solve.q3_sensitivity
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve import q3_proto as qp
from solve.common import ROOT

BETA = 0.1
DAYS = list(range(q2.REPORT_START, q2.N_DAY))

_P = None


def _init(payload):
    global _P
    _P = payload


def _task(t):
    vi, D, n_scen, pmm, plb, lam, seed = t
    r = qp.simulate_day_rt_hedge(
        _P, D, lam, adj_lam=lam, n_scen=n_scen, seed=seed,
        pool_min_month=pmm, pool_lookback=plb,
    )
    return (vi, D, float(r["plan_cost"]), float(r["adjust_net"]), float(r["emerg"]),
            float(r["e"].sum()), int(r.get("scenario_max_day", -1)))


VARIANTS = [
    ("正式（情景 40 / seed=7）", dict(n_scen=40, pmm=14, plb=90, lam=0.7, seed=7)),
    ("情景数 5 / seed=7", dict(n_scen=5, pmm=14, plb=90, lam=0.7, seed=7)),
    ("情景数 10 / seed=7", dict(n_scen=10, pmm=14, plb=90, lam=0.7, seed=7)),
    ("情景数 10 / seed=17", dict(n_scen=10, pmm=14, plb=90, lam=0.7, seed=17)),
    ("情景数 10 / seed=27", dict(n_scen=10, pmm=14, plb=90, lam=0.7, seed=27)),
    ("情景数 20 / seed=7", dict(n_scen=20, pmm=14, plb=90, lam=0.7, seed=7)),
    ("情景数 20 / seed=17", dict(n_scen=20, pmm=14, plb=90, lam=0.7, seed=17)),
    ("情景数 20 / seed=27", dict(n_scen=20, pmm=14, plb=90, lam=0.7, seed=27)),
    ("情景数 40 / seed=17", dict(n_scen=40, pmm=14, plb=90, lam=0.7, seed=17)),
    ("情景数 40 / seed=27", dict(n_scen=40, pmm=14, plb=90, lam=0.7, seed=27)),
    ("场景池回看 30 天", dict(n_scen=10, pmm=14, plb=30, lam=0.7, seed=7)),
    ("场景池回看 60 天", dict(n_scen=10, pmm=14, plb=60, lam=0.7, seed=7)),
    ("组合权重 λ=0.5", dict(n_scen=10, pmm=14, plb=90, lam=0.5, seed=7)),
    ("组合权重 λ=0.9", dict(n_scen=10, pmm=14, plb=90, lam=0.9, seed=7)),
]


def record(section: str, data, note: str = "") -> None:
    path = pm.reports_dir() / "RESULTS_REPORT.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        marker = f"### {section}"
        pos = text.find(marker)
        if pos != -1:
            end = text.find("\n### ", pos + len(marker))
            text = text[:pos] if end == -1 else text[:pos] + text[end + 1:]
            path.write_text(text, encoding="utf-8")
    pm.record_result(section, data, note=note)


def main():
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q3-sens")
    t0 = time.time()
    data = qp.load_extended()
    data["U_SMOOTH"] = qp.make_smooth_u(data, BETA)

    tasks = [(vi, D, v["n_scen"], v["pmm"], v["plb"], v["lam"], v["seed"])
             for vi, (_, v) in enumerate(VARIANTS) for D in DAYS]
    n = len(VARIANTS)
    plan = np.zeros(n)
    adj = np.zeros(n)
    em = np.zeros(n)
    em_kwh = np.zeros(n)
    viol = np.zeros(n, dtype=int)
    nw = min(8, os.cpu_count() or 1)
    from multiprocessing import Pool

    with Pool(nw, initializer=_init, initargs=(data,)) as pool:
        for k, (vi, D, c_plan, c_adj, c_em, e_kwh, smax) in enumerate(
                pool.imap_unordered(_task, tasks, chunksize=8), start=1):
            plan[vi] += c_plan
            adj[vi] += c_adj
            em[vi] += c_em
            em_kwh[vi] += e_kwh
            viol[vi] += int(smax >= D)
            if k % 500 == 0:
                print(f"  {k}/{len(tasks)}，用时 {time.time() - t0:.0f}s")

    out = pd.DataFrame({
        "配置": [name for name, _ in VARIANTS],
        "计划费/万元": np.round(plan / 1e4, 1),
        "调整净额/万元": np.round(adj / 1e4, 1),
        "紧急费/万元": np.round(em / 1e4, 1),
    })
    out["总费用/万元"] = (out["计划费/万元"] + out["调整净额/万元"] + out["紧急费/万元"]).round(1)
    base = float(out.loc[0, "总费用/万元"])
    out["相对正式/%"] = (100 * (out["总费用/万元"] / base - 1)).round(2)
    out["场景前视违规天数"] = viol
    pm.save_outputs(out, "q3_sensitivity")
    print(out.to_string(index=False))

    record(
        "问题三 主方案灵敏度：情景数 / 场景池窗口 / 组合权重（因果口径）",
        out,
        note=("对正式主方案（组合 λ=0.7、三点调整、情景对冲、逐槽因果）做单因素灵敏度："
              "对冲情景数使用 seed=7/17/27 检查随机稳定性，并单独改变场景池回看窗口、"
              "组合权重 λ；场景前视违规天数应全为 0。"
              "数据 code/outputs/q3_sensitivity.csv。"),
    )
    log.info("Q3 灵敏度完成（{:.0f}s）", time.time() - t0)
    return out


if __name__ == "__main__":
    main()
