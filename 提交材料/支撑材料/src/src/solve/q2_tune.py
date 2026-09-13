"""Q2 口径 E 参数搜索：标定窗口 / EWMA / 决策裕度 / 风险目标 / 权重收缩。

扫描对象（全部因果执行，2025-02-01 ~ 12-31，334 天）：
A. 标定窗口 W（网格表 argmin）与 EWMA 半衰期（对日费用行做指数衰减加权）；
B. 决策裕度：全局常数 m、逐时段/逐小时/逐月分位数（数据分布参考值）、
   仿射 m(t)=a·q80(t)+b、比例式 P̂(1−ρ)、负荷抬升 κ_L、动态 κ（滚动分位数）；
C. 风险目标：窗口选择目标改成 (1−λ)·均值 + λ·CVaR20（日费用最差 20% 均值）；
D. 权重收缩：w ← (1−α)·w* + α/3（抑制极端权重）。
E. 精扫与稳健性：κ 细网格、κ×m、κ×小时裕度、动态 κ、分半验证。

参考值来自 `solve/experiments/q2e_structure.py`（报童临界比 0.8 → 残差分位数）。
运行（在 program/ 下）：uv run python -m solve.q2_tune
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pandas as pd

import program as pm
from solve import consistency as cs
from solve import q2
from solve.common import E0, RESULTS_DIR, ROOT, progress, stage
from solve.q2_adaptive import AdaptiveWeightModel
from solve.io.report import record

REPORT_START = q2.REPORT_START
REP = list(range(REPORT_START, q2.N_DAY))
EPS_PLAN = AdaptiveWeightModel.EPS_PLAN
HOUR_OF_SLOT = np.arange(q2.T) // 6

_P = None  # 子进程全局负载


# ---------------------------------------------------------------------------
# 权重序列构造
# ---------------------------------------------------------------------------
def seqs_grid(model: AdaptiveWeightModel, W: int):
    """窗口 W 的网格 argmin 权重序列（官方口径）。"""
    return model.weights(W=W, use_gd=False)


def seqs_ewma(model: AdaptiveWeightModel, hl: float):
    """日费用行指数衰减加权（半衰期 hl 天）；hl≥1e8 退化为历史均值。

    统一实现见 ``consistency.ewma_weights_from_table``（唯一参数源）；
    本函数仅做 {day: (w, u)} → 报送期列表的接口适配。
    """
    table = cs.ewma_weights_from_table(model.C, model.grid, hl=hl)
    return [table[d] for d in REP]


def seqs_risk(model: AdaptiveWeightModel, W: int, lam: float, q: float = 0.2):
    """窗口目标 (1−λ)·均值 + λ·CVaR_q（最差 q 比例的日费用均值）。"""
    grid = model.grid
    out = []
    for d in REP:
        lo = max(model.START, d - W)
        block = model.C[lo:d]
        mean = block.mean(axis=0)
        k = max(1, int(np.ceil(q * block.shape[0])))
        cvar = np.sort(block, axis=0)[-k:].mean(axis=0)
        score = (1 - lam) * mean + lam * cvar
        i, j = np.unravel_index(np.argmin(score), mean.shape)
        out.append((grid[i].copy(), grid[j].copy()))
    return out


def seqs_shrink(model: AdaptiveWeightModel, W: int, alpha: float):
    """权重向等权收缩：(1−α)·w* + α/3（网格基座）。"""
    base = model.weights(W=W, use_gd=False)
    return [((1 - alpha) * w + alpha / 3.0, (1 - alpha) * u + alpha / 3.0)
            for (w, u) in base]


def shrink_seq(seq, alpha: float):
    """对任意权重序列做向等权收缩。"""
    return [((1 - alpha) * w + alpha / 3.0, (1 - alpha) * u + alpha / 3.0)
            for (w, u) in seq]


# ---------------------------------------------------------------------------
# 裕度参考值（分布结构）
# ---------------------------------------------------------------------------
def net_residuals(model: AdaptiveWeightModel, ws_list):
    """净需求残差 e 与负荷预测 L̂（kW），形状均为 (n_days, 144)。"""
    e = np.empty((len(REP), q2.T))
    lh = np.empty_like(e)
    for i, d in enumerate(REP):
        w, u = ws_list[i]
        l, p = model.forecast_kw(w, u, d)
        lh[i] = np.clip(l, 0.0, None)
        e[i] = (model.L[d] - model.P[d]) - (l - p)
    return e, lh


def margin_refs(model: AdaptiveWeightModel, seq, months) -> dict:
    """由残差分布得到参考裕度：全局/时段/小时/月/比例式/负荷系数。"""
    e, lh = net_residuals(model, seq)
    refs = {
        "global": float(np.quantile(e, 0.80)),
        "slot": np.quantile(e, 0.80, axis=0),
        "hour": np.array([np.quantile(e[:, 6 * h:6 * (h + 1)], 0.80)
                          for h in range(24)]),
        "month": np.array([
            np.quantile(e[months[REP] == m], 0.80) if (months[REP] == m).any() else 0.0
            for m in range(1, 13)]),
        "kappa": 1.0 + float(np.quantile((e / np.maximum(lh, 1.0)).ravel(), 0.80)),
    }
    ratios = []
    for i, d in enumerate(REP):
        w, u = seq[i]
        l, p = model.forecast_kw(w, u, d)
        mask = p > 500.0
        if mask.any():
            ratios.append(e[i][mask] / p[mask])
    refs["rho"] = float(np.quantile(np.concatenate(ratios), 0.80))
    refs["e"] = e
    return refs


def rolling_kappa(model: AdaptiveWeightModel, seq, window: int = 60, q: float = 0.8):
    """动态负荷系数：每日用过去 window 天残差比 e/L̂ 的 q 分位数（无前视）。"""
    e, lh = net_residuals(model, seq)
    ratio = e / np.maximum(lh, 1.0)
    kappa = np.ones(len(REP))
    for i in range(len(REP)):
        lo = max(0, i - window)
        if i - lo >= 14:
            kappa[i] = 1.0 + float(np.quantile(ratio[lo:i].ravel(), q))
    return kappa


# ---------------------------------------------------------------------------
# 并行评估
# ---------------------------------------------------------------------------
def _init_worker(payload):
    global _P
    _P = payload


def _task(t):
    ci, wsid, msid, d = t
    p = _P
    w, u = p["seqs"][wsid][d - REPORT_START]
    l = w[0] * p["L"][d - 7] + w[1] * p["L"][d - 14] + w[2] * p["L_typ"]
    pp = u[0] * p["P"][d - 1] + u[1] * p["P"][d - 2] + u[2] * p["P_typ"]
    spec = p["margins"][msid]
    if spec is not None:
        kind = spec["kind"]
        if kind == "pv_global":
            pp = pp - spec["m"]
        elif kind == "pv_slot":
            pp = pp - spec["m"]
        elif kind == "pv_hour":
            pp = pp - spec["m"][HOUR_OF_SLOT]
        elif kind == "pv_month":
            pp = pp - spec["m"][p["months"][d] - 1]
        elif kind == "pv_affine":
            pp = pp - (spec["a"] * spec["m"][HOUR_OF_SLOT] + spec["b"])
        elif kind == "pv_prop":
            pp = pp * (1.0 - spec["rho"])
        elif kind == "load_kappa":
            l = l * spec["k"]
        elif kind == "load_kappa_dyn":
            l = l * spec["kappa"][d - REPORT_START]
        elif kind == "combo":
            l = l * spec["k"]
            pp = pp - spec["m"]
        elif kind == "combo_hour":
            l = l * spec["k"]
            pp = pp - spec["m"][HOUR_OF_SLOT]
    l = np.clip(l, 0.0, None) / 6.0
    pp = np.clip(pp, 0.0, None) / 6.0
    x, _E, _ = q2.plan_day(p["price"], l, pp, E0, eps=EPS_PLAN)
    ex = q2.exec_day_causal(p["price"], p["L"][d] / 6.0, p["P"][d] / 6.0, x, E0)
    return ci, d - REPORT_START, float(p["price"] @ x), float(q2.EMERG_MULT * (p["price"] @ ex["e"]))


def evaluate(model: AdaptiveWeightModel, configs: list[dict],
             workers: int | None = None, desc: str = "批量评估配置"):
    """并行评估配置；返回 (汇总表, 日计划费矩阵, 日紧急费矩阵)。"""
    seqs, sid_by_obj = [], {}
    for cfg in configs:
        key = id(cfg["seq"])
        if key not in sid_by_obj:
            sid_by_obj[key] = len(seqs)
            seqs.append(cfg["seq"])
    margins, mid_by_key = [], {}
    for cfg in configs:
        key = id(cfg["margin"]) if cfg["margin"] is not None else -1
        if key not in mid_by_key:
            mid_by_key[key] = len(margins)
            margins.append(cfg["margin"])
    tasks = []
    for ci, cfg in enumerate(configs):
        sid = sid_by_obj[id(cfg["seq"])]
        mid = mid_by_key[id(cfg["margin"]) if cfg["margin"] is not None else -1]
        for d in REP:
            tasks.append((ci, sid, mid, d))

    payload = {
        "seqs": seqs, "margins": margins,
        "L": model.L, "P": model.P, "L_typ": model.L_typ, "P_typ": model.P_typ,
        "price": model.price, "months": model.data["months"],
    }
    t0 = time.time()
    n = len(configs)
    daily_plan = np.zeros((n, len(REP)))
    daily_emerg = np.zeros((n, len(REP)))
    nw = workers or min(8, os.cpu_count() or 1)
    stage(desc, f"{n} 个配置 × {len(REP)} 天单日规划+因果执行（{nw} 进程）")
    if nw > 1:
        from multiprocessing import Pool

        with Pool(nw, initializer=_init_worker, initargs=(payload,)) as pool:
            it = progress(pool.imap_unordered(_task, tasks, chunksize=64),
                          desc=desc, total=len(tasks), unit="任务")
            for ci, di, c_plan, c_em in it:
                daily_plan[ci, di] = c_plan
                daily_emerg[ci, di] = c_em
    else:
        _init_worker(payload)
        for t in progress(tasks, desc=desc, unit="任务"):
            ci, di, c_plan, c_em = _task(t)
            daily_plan[ci, di] = c_plan
            daily_emerg[ci, di] = c_em
    df = pd.DataFrame({
        "配置": [c["名称"] for c in configs],
        "计划购电费/万元": np.round(daily_plan.sum(axis=1) / 1e4, 1),
        "紧急购电费/万元": np.round(daily_emerg.sum(axis=1) / 1e4, 1),
    })
    df["总费用/万元"] = (df["计划购电费/万元"] + df["紧急购电费/万元"]).round(1)
    print(f"  批量评估 {len(configs)} 个配置完成（{time.time() - t0:.0f}s）")
    return df, daily_plan, daily_emerg


def simulate_two_day(model: AdaptiveWeightModel, seq, kappa: float, margin: float,
                     detail: bool = False, show_progress: bool = True):
    """2 日滚动正式口径：日末 SOC 自由、跨日连续，规划窗口末端自由（不锚定）。"""
    E = float(E0)
    rows = []
    x_plans = np.zeros((q2.N_DAY, q2.T)) if detail else None
    recs = [None] * q2.N_DAY if detail else None
    it = (progress(REP, desc="2 日滚动逐日模拟", unit="天") if show_progress else REP)
    for i, d in enumerate(it):
        w, u = seq[i]
        l0, p0 = model.forecast_kw(w, u, d)
        # 次日预测在 d 日 0:00 形成：负荷的 d+1−7/d+1−14 已知；
        # 光伏不能读取尚未实现的 P[d]，只能沿用 P[d−1]/P[d−2]/典型日组合。
        l1 = w[0] * model.L[d - 6] + w[1] * model.L[d - 13] + w[2] * model.L_typ
        p1 = u[0] * model.P[d - 1] + u[1] * model.P[d - 2] + u[2] * model.P_typ
        l0 = cs.kappa_load(l0, kappa)
        l1 = cs.kappa_load(l1, kappa)
        p0 = cs.margin_pv(p0, margin)
        p1 = cs.margin_pv(p1, margin)
        xh, Eh, _ = q2.plan_horizon(
            np.tile(model.price, 2), np.concatenate([l0, l1]) / 6.0,
            np.concatenate([p0, p1]) / 6.0, E, None, eps=EPS_PLAN,
        )
        x = xh[:q2.T]
        # 统一执行策略（consistency.EXEC_POLICY="free"）：无段末硬目标
        ex = q2.exec_segment_causal(
            model.L[d] / 6.0, model.P[d] / 6.0, x, E, None)
        if detail:
            x_plans[d] = x
            recs[d] = {
                "c": ex["c"], "d": ex["d"], "s": ex["s"], "e": ex["e"],
                "E": ex["E"], "E_start": E, "E_end": float(ex["E"][-1]),
            }
        rows.append({
            "日期": model.dates[d],
            "计划购电量/kWh": float(x.sum()),
            "计划购电费/元": float(model.price @ x),
            "紧急购电量/kWh": float(ex["e"].sum()),
            "紧急购电费/元": float(q2.EMERG_MULT * (model.price @ ex["e"])),
            "日初SOC/kWh": E,
            "日末SOC/kWh": float(ex["E"][-1]),
        })
        E = float(ex["E"][-1])
    df = pd.DataFrame(rows)
    return (df, x_plans, recs) if detail else df


def rolling_holdout_search() -> pd.DataFrame:
    """在 2 日滚动结构下做小范围开发/冻结验证，正式参数只由 2–6 月选择。"""
    pm.init(root=str(ROOT))
    model = AdaptiveWeightModel().load().build_table()
    split_at = next(i for i, d in enumerate(REP) if model.dates[d] == "2025-07-01")
    rows = []
    combos = [(h, kappa, margin) for h in (3.0, 5.0, 7.0)
              for kappa in (1.01, 1.015, 1.02) for margin in (25.0, 50.0, 75.0)]
    stage("2 日滚动时间留出", f"{len(combos)} 个候选 × {len(REP)} 天"
          "（288 槽两日 LP + 因果执行，逐候选串行；约 3–5 分钟）")
    for h, kappa, margin in progress(combos, desc="2 日滚动候选", unit="候选"):
        seq = seqs_ewma(model, h)
        df = simulate_two_day(model, seq, kappa, margin, show_progress=False)
        costs = df["计划购电费/元"].to_numpy() + df["紧急购电费/元"].to_numpy()
        rows.append({
            "配置": f"EWMA h={h:g} κ={kappa:g}+m={margin:g}",
            "h": h, "kappa": kappa, "margin": margin,
            "开发期(2-6月)/万元": round(float(costs[:split_at].sum()) / 1e4, 1),
            "冻结验证期(7-12月)/万元": round(float(costs[split_at:].sum()) / 1e4, 1),
            "全年描述值/万元": round(float(costs.sum()) / 1e4, 1),
        })
    out = pd.DataFrame(rows).sort_values(
        ["开发期(2-6月)/万元", "冻结验证期(7-12月)/万元"]
    ).reset_index(drop=True)
    out["开发期选中"] = out.index == 0
    out["验证期排名"] = out["冻结验证期(7-12月)/万元"].rank(method="min").astype(int)
    out.to_csv(pm.outputs_dir() / "q2e_tune_2day_holdout.csv", index=False, encoding="utf-8-sig")
    record(
        "问题二 2日滚动参数时间留出验证",
        out.head(10),
        note=("每个候选均按 48 小时预测窗口逐日滚动；当前日末 SOC 不固定并跨日传递，"
              "规划窗口末端完全自由（不锚定终值）。2–6 月选参，7–12 月冻结验证；完整表见 "
              "code/outputs/q2e_tune_2day_holdout.csv。"),
    )
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    pm.init(seed=42, root=str(ROOT))
    model = AdaptiveWeightModel().load().build_table()
    months = model.data["months"]
    rep_str = f"{model.dates[REPORT_START]} ~ {model.dates[-1]}"

    # ---------- 阶段 1：窗口与 EWMA ----------
    W_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 21, 25, 28, 30, 32, 35,
              45, 60, 90, 120, 180, 240, 300]
    HL_LIST = [0.5, 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 30, 60, 120, 1e9]
    cfgs = [{"名称": f"网格 W={W}", "seq": seqs_grid(model, W), "margin": None}
            for W in W_LIST]
    cfgs += [{"名称": ("历史均值" if hl > 1e8 else f"EWMA h={hl}"),
              "seq": seqs_ewma(model, hl), "margin": None} for hl in HL_LIST]
    df1, dp1, de1 = evaluate(model, cfgs, desc="阶段1 窗口与 EWMA")
    df1.to_csv(pm.outputs_dir() / "q2e_tune_window.csv", index=False, encoding="utf-8-sig")

    grid_rows = df1[df1["配置"].str.startswith("网格")]
    best_grid = grid_rows.loc[grid_rows["总费用/万元"].idxmin()]
    w_best = int(best_grid["配置"].split("=")[1])
    ew_rows = df1[df1["配置"].str.startswith("EWMA")]
    hl_best = float(ew_rows.loc[ew_rows["总费用/万元"].idxmin(), "配置"].split("=")[1])
    print(f"阶段1 网格最优：W*={w_best} = {best_grid['总费用/万元']} 万元；"
          f"EWMA 最优：h={hl_best:g} = {ew_rows['总费用/万元'].min()} 万元")

    seq_w7 = seqs_grid(model, 7)
    seq_best = seq_w7 if w_best == 7 else seqs_grid(model, w_best)
    seq_ew = seqs_ewma(model, hl_best)
    bases = {"W7": seq_w7}
    if w_best != 7:
        bases[f"W{w_best}"] = seq_best
    bases[f"EWMA h={hl_best:g}"] = seq_ew

    # ---------- 阶段 2：决策裕度（分布参考值） ----------
    refs = {name: margin_refs(model, seq, months) for name, seq in bases.items()}
    for name, r in refs.items():
        print(f"  {name} 参考裕度：全局 {r['global']:.0f} kW、"
              f"小时范围 {r['hour'].min():.0f}~{r['hour'].max():.0f}、"
              f"比例式 ρ={r['rho']:.3f}、κ_ref={r['kappa']:.3f}")

    cfgs2 = []
    for bname, seq in bases.items():
        r = refs[bname]
        for m in range(0, 501, 50):
            cfgs2.append({"名称": f"{bname} 全局裕度 m={m}", "seq": seq,
                          "margin": None if m == 0 else
                          {"kind": "pv_global", "m": float(m)}})
        for g in (0.5, 0.75, 1.0, 1.25, 1.5):
            cfgs2.append({"名称": f"{bname} 小时裕度 γ={g}", "seq": seq,
                          "margin": {"kind": "pv_hour", "m": r["hour"] * g}})
        for g in (0.75, 1.0, 1.25):
            cfgs2.append({"名称": f"{bname} 时段裕度 γ={g}", "seq": seq,
                          "margin": {"kind": "pv_slot", "m": r["slot"] * g}})
        cfgs2.append({"名称": f"{bname} 月裕度 γ=1", "seq": seq,
                      "margin": {"kind": "pv_month", "m": r["month"]}})
        for rho in (0.02, 0.04, 0.06, 0.08, 0.12):
            cfgs2.append({"名称": f"{bname} 比例裕度 ρ={rho}", "seq": seq,
                          "margin": {"kind": "pv_prop", "rho": rho}})
        for k in (1.005, 1.01, 1.015, 1.02):
            cfgs2.append({"名称": f"{bname} 负荷抬升 κ={k}", "seq": seq,
                          "margin": {"kind": "load_kappa", "k": k}})
    for a in (0.5, 0.75, 1.0, 1.25):
        for b in (50.0, 100.0, 150.0):
            cfgs2.append({"名称": f"W7 仿射裕度 a={a}, b={b}", "seq": seq_w7,
                          "margin": {"kind": "pv_affine", "a": a, "b": b,
                                     "m": refs["W7"]["hour"]}})
    df2, dp2, de2 = evaluate(model, cfgs2, desc="阶段2 决策裕度（分布参考值）")
    df2.to_csv(pm.outputs_dir() / "q2e_tune_margin.csv", index=False, encoding="utf-8-sig")
    top2 = df2.loc[df2["总费用/万元"].idxmin()]
    print(f"阶段2 最优：{top2['配置']} = {top2['总费用/万元']} 万元")

    # ---------- 阶段 3：风险目标与收缩 ----------
    cfgs3 = []
    for bname, seq in bases.items():
        for lam in (0.25, 0.5, 0.75, 1.0):
            for W in (7, 21, 30):
                cfgs3.append({"名称": f"{bname} 风险 λ={lam}, W={W}",
                              "seq": seqs_risk(model, W, lam), "margin": None})
        for alpha in (0.05, 0.1, 0.2, 0.35):
            cfgs3.append({"名称": f"{bname} 收缩 α={alpha}",
                          "seq": shrink_seq(seq, alpha), "margin": None})
    df3, dp3, de3 = evaluate(model, cfgs3, desc="阶段3 风险目标与收缩")
    df3.to_csv(pm.outputs_dir() / "q2e_tune_risk.csv", index=False, encoding="utf-8-sig")
    top3 = df3.loc[df3["总费用/万元"].idxmin()]
    print(f"阶段3 最优：{top3['配置']} = {top3['总费用/万元']} 万元")

    # ---------- 阶段 4：精扫 κ、组合、动态 κ ----------
    cfgs4 = []
    for bname, seq in bases.items():
        for k in (1.015, 1.02, 1.025, 1.03, 1.035, 1.04):
            cfgs4.append({"名称": f"{bname} κ={k}", "seq": seq,
                          "margin": {"kind": "load_kappa", "k": k}})
        for k in (1.015, 1.02, 1.025):
            for m in (50.0, 100.0, 150.0):
                cfgs4.append({"名称": f"{bname} κ={k}+m={m:.0f}", "seq": seq,
                              "margin": {"kind": "combo", "k": k, "m": m}})
        for g in (0.5, 0.75):
            cfgs4.append({"名称": f"{bname} κ=1.02+小时裕度γ={g}", "seq": seq,
                          "margin": {"kind": "combo_hour", "k": 1.02,
                                     "m": refs[bname]["hour"] * g}})
        cfgs4.append({"名称": f"{bname} 动态κ(60天q80)", "seq": seq,
                      "margin": {"kind": "load_kappa_dyn",
                                 "kappa": rolling_kappa(model, seq)}})
    # EWMA 基座的细网格（围绕 κ=1.02, m=50）
    seq_ew_name = f"EWMA h={hl_best:g}"
    if seq_ew_name in bases:
        seq = bases[seq_ew_name]
        for k in (1.02, 1.0225, 1.025):
            for m in (25.0, 50.0, 75.0):
                cfgs4.append({"名称": f"{seq_ew_name} κ={k}+m={m:.0f}", "seq": seq,
                              "margin": {"kind": "combo", "k": k, "m": m}})
    df4, dp4, de4 = evaluate(model, cfgs4, desc="阶段4 精扫 κ/组合/动态κ")
    df4.to_csv(pm.outputs_dir() / "q2e_tune_combo.csv", index=False, encoding="utf-8-sig")
    top4 = df4.loc[df4["总费用/万元"].idxmin()]
    print(f"阶段4 最优：{top4['配置']} = {top4['总费用/万元']} 万元")

    # ---------- 时间留出验证（开发期: 2–6 月；冻结验证期: 7–12 月） ----------
    # 不能按 334 天机械对半：真正的 7 月 1 日切点是第 150 个报告日。
    split_at = next(i for i, d in enumerate(REP) if model.dates[d] == "2025-07-01")
    tagged = []
    for family, df, dp, de in (
        ("窗口/EWMA", df1, dp1, de1),
        ("决策裕度", df2, dp2, de2),
        ("风险/收缩", df3, dp3, de3),
        ("组合精扫", df4, dp4, de4),
    ):
        for i in range(len(df)):
            tagged.append((family, df.iloc[i]["配置"], dp[i], de[i]))
    tagged = pd.DataFrame(tagged, columns=["族", "配置", "dp", "de"])

    split_rows = []
    for _, row in tagged.iterrows():
        dev = float(row["dp"][:split_at].sum() + row["de"][:split_at].sum()) / 1e4
        val = float(row["dp"][split_at:].sum() + row["de"][split_at:].sum()) / 1e4
        split_rows.append({
            "族": row["族"], "配置": row["配置"],
            "开发期(2-6月)/万元": round(dev, 1),
            "冻结验证期(7-12月)/万元": round(val, 1),
            "全年描述值/万元": round(dev + val, 1),
        })
    holdout = pd.DataFrame(split_rows).sort_values(
        ["开发期(2-6月)/万元", "冻结验证期(7-12月)/万元"]
    ).reset_index(drop=True)
    holdout["开发期选中"] = holdout.index == 0
    holdout["验证期排名"] = holdout["冻结验证期(7-12月)/万元"].rank(
        method="min"
    ).astype(int)
    selected = holdout.iloc[0]
    holdout.to_csv(pm.outputs_dir() / "q2e_tune_holdout.csv", index=False,
                   encoding="utf-8-sig")

    keep = {"网格 W=7", str(selected["配置"]), str(top4["配置"])}
    split = holdout[holdout["配置"].isin(keep)].copy()
    split.to_csv(pm.outputs_dir() / "q2e_tune_split.csv", index=False,
                 encoding="utf-8-sig")
    print("时间留出验证：开发期选中 {}（开发 {:.1f} 万，冻结验证 {:.1f} 万，验证排名 {}/{}）".format(
        selected["配置"], selected["开发期(2-6月)/万元"],
        selected["冻结验证期(7-12月)/万元"], selected["验证期排名"], len(holdout)))

    # ---------- 汇总 ----------
    official = df1[df1["配置"] == "网格 W=7"].iloc[0]
    all_best = pd.concat([df1, df2, df3, df4]).sort_values("总费用/万元")
    final = all_best.iloc[0]
    summary = pd.DataFrame({
        "方案": ["官方 W=7（网格）", f"最优配置：{final['配置']}"],
        "计划购电费/万元": [official["计划购电费/万元"], final["计划购电费/万元"]],
        "紧急购电费/万元": [official["紧急购电费/万元"], final["紧急购电费/万元"]],
        "总费用/万元": [official["总费用/万元"], final["总费用/万元"]],
        "相对官方/%": [0.0, round(100 * (final["总费用/万元"] / official["总费用/万元"] - 1), 2)],
    })
    print(summary.to_string(index=False))
    print(split.to_string(index=False))
    pm.save_outputs(summary, "q2e_tune_summary")
    pm.save_outputs(split, "q2e_tune_split")

    # 最优配置逐日表（供后续 result2 引用）
    best_row = tagged[tagged["配置"] == final["配置"]]
    if len(best_row):
        row = best_row.iloc[0]
        pd.DataFrame({
            "日期": [model.dates[d] for d in REP],
            "计划购电费/元": row["dp"],
            "紧急购电费/元": row["de"],
        }).to_csv(pm.outputs_dir() / "q2e_tune_daily_best.csv", index=False,
                  encoding="utf-8-sig")

    # ---------- 图：参数搜索 ----------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax = axes[0, 0]
    g = df1[df1["配置"].str.startswith("网格")]
    ax.plot([int(s.split("=")[-1]) for s in g["配置"]], g["总费用/万元"],
            marker="o", ms=3, color="#4C72B0", label="网格 W")
    ew = df1[df1["配置"].str.startswith("EWMA")]
    ax.plot([float(s.split("=")[-1]) for s in ew["配置"]], ew["总费用/万元"],
            marker="s", ms=3, color="#C44E52", label="EWMA h")
    ax.set_xscale("log")
    ax.set_xlabel("标定窗口 W / EWMA 半衰期 / 天")
    ax.set_ylabel("年度总费用 / 万元")
    ax.set_title("(a) 权重标定时间尺度", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for i, bname in enumerate(bases):
        color = ["#4C72B0", "#C44E52", "#55A868"][i % 3]
        gg = df2[df2["配置"].str.startswith(f"{bname} 全局裕度")]
        ax.plot([int(s.split("=")[-1]) for s in gg["配置"]], gg["总费用/万元"],
                marker="o", ms=3, color=color, label=f"{bname} 全局 m")
    ax.set_xlabel("全局裕度 m / kW")
    ax.set_ylabel("年度总费用 / 万元")
    ax.set_title("(b) 光伏裕度", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for i, bname in enumerate(bases):
        color = ["#4C72B0", "#C44E52", "#55A868"][i % 3]
        kk = df4[df4["配置"].str.startswith(f"{bname} κ=")
                 & ~df4["配置"].str.contains("\\+")]
        ax.plot([float(s.split("=")[-1]) for s in kk["配置"]], kk["总费用/万元"],
                marker="o", ms=3, color=color, label=f"{bname} 负荷 κ")
        ax.axvline(refs[bname]["kappa"], color=color, ls=":", lw=1.0, alpha=0.7)
    ax.set_xlabel("负荷抬升系数 κ（虚线 = 分布参考）")
    ax.set_ylabel("年度总费用 / 万元")
    ax.set_title("(c) 负荷裕度 κ（虚线 = 分布参考）", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    labels = [s.replace("：", "\n") for s in summary["方案"]]
    ax.bar(labels, summary["计划购电费/万元"], label="计划购电费", color="#4C72B0")
    ax.bar(labels, summary["紧急购电费/万元"], bottom=summary["计划购电费/万元"],
           label="紧急购电费", color="#C44E52")
    for i, v in enumerate(summary["总费用/万元"]):
        ax.text(i, v + 10, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("年度费用 / 万元")
    ax.set_title("(d) 官方 vs 最优配置", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_data = pd.concat([
        df1.assign(族="窗口/EWMA")[["配置", "总费用/万元", "族"]],
        df2.assign(族="裕度")[["配置", "总费用/万元", "族"]],
        df3.assign(族="风险/收缩")[["配置", "总费用/万元", "族"]],
        df4.assign(族="精扫")[["配置", "总费用/万元", "族"]],
    ], ignore_index=True)
    pm.save_fig(fig, "Q2E_参数搜索", data=fig_data)

    # ---------- 报告 ----------
    from solve.io.report import record

    record(
        "问题二 口径E 参数搜索：标定窗口与 EWMA",
        df1,
        note=(f"对照期 {rep_str}，因果执行；W=1 对应日度标定；"
              "EWMA 对日费用行做指数衰减加权（无前视）；历史均值 = 全历史等权。"),
    )
    record(
        "问题二 口径E 参数搜索：决策裕度",
        df2,
        note=("裕度为加在净需求（光伏）预测上的保守修正，参考值 = 残差 0.8 分位数"
              "（报童临界比 1−1/5）；小时/时段/月裕度使用各自通道的分布分位数，"
              "γ 为分位数缩放；比例式 = P̂(1−ρ)；负荷抬升 = κ·L̂。"),
    )
    record(
        "问题二 口径E 参数搜索：风险目标与权重收缩",
        df3,
        note="风险目标 = (1−λ)·窗口均值 + λ·CVaR20（最差 20% 日费用均值）；收缩 = (1−α)·w* + α/3。",
    )
    record(
        "问题二 口径E 参数搜索：κ 精扫与动态参考",
        df4,
        note=("κ 为负荷预测抬升系数（分布参考 κ_ref = 1 + q80(e/L̂) 见各基座行）；"
              "动态 κ = 过去 60 天 e/L̂ 的 0.8 分位数逐日重估（无前视）；"
              "组合 = κ 与光伏裕度 m/小时裕度叠加。"),
    )
    record(
        "问题二 口径E 参数搜索：时间留出验证",
        split,
        note=("开发期 2025-02-01~06-30 只负责选参，冻结验证期 2025-07-01~12-31 不参与选择；"
              "全部候选见 code/outputs/q2e_tune_holdout.csv。全年列仅作描述，不用于选参。"),
    )
    record(
        "问题二 口径E 调优配置汇总",
        summary,
        note=(f"最优配置 {final['配置']}：总费用 {final['总费用/万元']} 万元，"
              f"相对官方 W=7（{official['总费用/万元']} 万元）"
              f"{100 * (final['总费用/万元'] / official['总费用/万元'] - 1):+.2f}%。"
              "全部明细见 code/outputs/q2e_tune_*.csv；图 figures/Q2E_参数搜索.pdf。"),
    )
    return summary


def joint_residual_mc(model, seq, kappa, margin, x_plans, e_start,
                      n_years=100, seed=42):
    """统一口径年度分布：固定计划，重采样（Δ负荷, Δ光伏）同月整日联合残差块。

    名义 = κ·L̂ 与 max(P̂ − m, 0)（EWMA h=5 权重，构造同 ``simulate_two_day``）；
    残差 = 实际 − 名义；池同月优先、严格取目标日之前（``consistency``）。
    """
    data = model.data
    months = np.asarray(data["months"])
    wu = cs.ewma_weights_from_table(model.C, model.grid, hl=cs.EWMA_HL,
                                    first_target=15)
    L_nom = np.zeros((q2.N_DAY, q2.T))
    P_nom = np.zeros((q2.N_DAY, q2.T))
    for d in range(15, q2.N_DAY):
        w, u = wu[d]
        l = w[0] * model.L[d - 7] + w[1] * model.L[d - 14] + w[2] * model.L_typ
        p = u[0] * model.P[d - 1] + u[1] * model.P[d - 2] + u[2] * model.P_typ
        L_nom[d] = cs.kappa_load(np.clip(l, 0.0, None), kappa) / 6.0
        P_nom[d] = cs.margin_pv(np.clip(p, 0.0, None), margin) / 6.0
    RL = data["load"] / 6.0 - L_nom
    RP = data["pv_act"] / 6.0 - P_nom
    rng = np.random.default_rng(seed)
    emerg = np.zeros(n_years)
    kwh = np.zeros(n_years)
    stage("联合残差蒙特卡洛", f"{n_years} 个模拟年 × {len(REP)} 天"
          "（同月整日残差块重采样 + 逐日因果回放）")
    for y in progress(range(n_years), desc="联合残差 MC", unit="年"):
        E = float(e_start)
        for d in REP:
            pool = cs.scenario_indices(months, d)
            j = int(rng.choice(pool))
            load_s = np.clip(L_nom[d] + RL[j], 0.0, None)
            pv_s = np.clip(P_nom[d] + RP[j], 0.0, None)
            # 统一执行策略（free）：无段末硬目标
            ex = q2.exec_segment_causal(load_s, pv_s, x_plans[d], E, None)
            kwh[y] += float(ex["e"].sum())
            emerg[y] += float(q2.EMERG_MULT * (model.price @ ex["e"]))
            E = float(ex["E"][-1])
    return emerg, kwh


def write_result2_tuned(W: float = 5.0, kappa: float = 1.02,
                        margin: float = 25.0) -> dict:
    """把 2 日滚动时间留出选定配置写入 ``results/result2.xlsx``。

    流程：EWMA 权重序列 → 48 小时计划（当前日末 SOC 自由）→ 逐槽因果执行 →
    备份现官方文件 → 写出 + 回读校验 + 报告。默认参数只由 2–6 月开发期选择，
    7–12 月作为冻结验证期，不参与选择。
    """
    from solve.q2_adaptive import _verify_result2

    pm.init(root=str(ROOT))
    log = pm.get_logger("q2-tune")
    model = AdaptiveWeightModel().load().build_table()
    seq = seqs_ewma(model, W)

    stage("生成官方 result2", f"2 日滚动 {len(REP)} 天"
          f"（EWMA h={W:g} + κ={kappa:g} + m={margin:g}）→ 回读校验 → 图与年度 MC")
    df, x_plans, recs = simulate_two_day(model, seq, kappa, margin, detail=True)
    plan = float(df["计划购电费/元"].sum())
    emerg = float(df["紧急购电费/元"].sum())

    cur = RESULTS_DIR / "result2.xlsx"
    if cur.exists():
        backup = pm.outputs_dir() / "result2_pre_2day_backup.xlsx"
        if not backup.exists():
            shutil.copy2(cur, backup)
            log.info("2 日滚动修正前 result2 备份 -> {}", backup)
    out = q2.write_result2(model.dates, model.price, x_plans, recs, out_name="result2.xlsx")
    checks = _verify_result2(out, df)
    pm.save_outputs(df, "q2e_tuned_daily")

    # ---- 官方口径图与年度成本分布（论文引用；全部来自本官方计划）----
    import matplotlib.pyplot as plt

    idx = np.arange(len(df))
    month_starts = [i for i, s in enumerate(df["日期"]) if s.endswith("-01")]
    month_labels = [df["日期"].iloc[i][5:7] + "月" for i in month_starts]

    fig, ax = pm.line(idx, df["紧急购电量/kWh"].to_numpy(),
                      xlabel="日期", ylabel="紧急购电量 / kWh")
    ax.set_xticks(month_starts)
    ax.set_xticklabels(month_labels)
    pm.save_fig(fig, "Q2_逐日紧急购电", data=df[["日期", "紧急购电量/kWh"]])

    fig2, ax2 = pm.line(
        idx, [df["日初SOC/kWh"].to_numpy(), df["日末SOC/kWh"].to_numpy()],
        labels=["日初储电量", "日末储电量"], xlabel="日期", ylabel="储电量 / kWh")
    ax2.set_xticks(month_starts)
    ax2.set_xticklabels(month_labels)
    pm.save_fig(fig2, "Q2_储能轨迹",
                data=df[["日期", "日初SOC/kWh", "日末SOC/kWh"]])

    mc_costs, _mc_kwhs = joint_residual_mc(
        model, seq, kappa, margin, x_plans,
        float(df["日初SOC/kWh"].iloc[0]), n_years=100, seed=42)
    total = plan + mc_costs
    p95 = float(np.percentile(total, 95))
    fig3, ax3 = plt.subplots(figsize=(7, 4.3))
    ax3.hist(total, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
    ax3.axvline(total.mean(), color="#C44E52", ls="--",
                label=f"均值 {total.mean():,.0f} 元")
    ax3.axvline(p95, color="#55A868", ls=":", label=f"P95 {p95:,.0f} 元")
    ax3.set_xlabel("年度总购电费 / 元")
    ax3.set_ylabel("频数")
    ax3.legend()
    pm.save_fig(fig3, "Q2_总费用分布",
                data=pd.DataFrame({"年度总购电费_元": total}))
    record(
        "问题二 年度总成本分布（2 日滚动·蒙特卡洛）",
        {
            "模拟年数": 100,
            "年度总成本均值/元": float(total.mean()),
            "年度总成本标准差/元": float(total.std()),
            "P5/元": float(np.percentile(total, 5)),
            "P50/元": float(np.percentile(total, 50)),
            "P95/元": p95,
            "CVaR95（尾部均值）/元": float(total[total >= p95].mean()),
            "年度紧急费用均值/元": float(mc_costs.mean()),
            "年度紧急费用P95/元": float(np.percentile(mc_costs, 95)),
        },
        note=("固定 2 日滚动官方计划，重采样（Δ负荷, Δ光伏）同月整日联合残差块"
              "（相对统一名义预测 κ/m），保留日内与两通道相关；"
              "逐日执行沿用统一策略 free（无段末硬目标）、SOC 跨日连续；"
              "仅作事后风险评价，不参与在线决策；图 figures/Q2_总费用分布.pdf。"),
    )
    record(
        f"问题二 口径E（2日滚动时间留出选定：EWMA h={W:g} + κ={kappa:g} + m={margin:g}）官方 result2",
        {
            "计划购电费/万元": round(plan / 1e4, 1),
            "紧急购电费/万元": round(emerg / 1e4, 1),
            "总费用/万元": round((plan + emerg) / 1e4, 1),
            "天数": len(df),
            **checks,
        },
        note=(
            f"官方 results/result2.xlsx 由 2 日滚动时间留出配置生成：2–6 月开发期选择 EWMA "
            f"h={W:g}、κ={kappa:g}、m={margin:g} kW，7–12 月冻结验证不参与选参；"
            "每天以 48 小时为规划窗口、只执行次日，当前日末 SOC 不固定并传递到下一日，"
            "规划窗口末端完全自由（不锚定任何终值）。"
            "旧 W=7 日循环版仍保留在 "
            "code/outputs/result2_W7_backup.xlsx，留出修正前版本保留在 "
            "code/outputs/result2_pre_2day_backup.xlsx；复现："
            "uv run python -m solve.q2_tune --result2。"
            "图 figures/Q2_逐日紧急购电.pdf、Q2_储能轨迹.pdf、Q2_总费用分布.pdf。"
        ),
    )
    log.info("result2.xlsx（2 日滚动时间留出选定）写出完成：{}，总费用 {:.1f} 万元（计划 {:.1f} + 紧急 {:.1f}）",
             out, (plan + emerg) / 1e4, plan / 1e4, emerg / 1e4)
    return {"plan": plan, "emerg": emerg, "checks": checks, "out": out}


def make_weight_figure(W: float = 5.0, smooth: int = 15) -> None:
    """重生成权重演化图（EWMA h=5 标定口径，与官方模型一致）。

    模型权重取自离散网格（步长 0.2），逐日绘制锯齿明显；图中曲线为
    **显示用**居中滑动平均（窗口 ``smooth`` 天），不改变模型口径；
    原始逐日权重同时保存在作图数据 CSV 的原始列中。
    """
    import matplotlib.pyplot as plt

    pm.init(root=str(ROOT))
    model = AdaptiveWeightModel().load().build_table()
    seq = seqs_ewma(model, W)
    dates = [model.dates[d] for d in REP]
    Wm = np.array([w for w, _ in seq])
    Um = np.array([u for _, u in seq])

    def _roll(A):
        return pd.DataFrame(A).rolling(smooth, center=True, min_periods=1).mean().to_numpy()

    Ws, Us = _roll(Wm), _roll(Um)
    w_cols = ["w1_L(d-7)", "w2_L(d-14)", "w3_典型日"]
    u_cols = ["u1_P(d-1)", "u2_P(d-2)", "u3_典型日"]
    df = pd.DataFrame({"日期": dates,
                       **{c: Wm[:, i] for i, c in enumerate(w_cols)},
                       **{c: Um[:, i] for i, c in enumerate(u_cols)},
                       **{c + "_平滑": Ws[:, i] for i, c in enumerate(w_cols)},
                       **{c + "_平滑": Us[:, i] for i, c in enumerate(u_cols)}})
    x = np.arange(len(dates))
    month_starts = [i for i, s in enumerate(dates) if s.endswith("-01")]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    for ax, cols, ylabel, labels in (
        (axes[0], w_cols, "负荷权重", ["d-7", "d-14", "典型日"]),
        (axes[1], u_cols, "光伏权重", ["d-1", "d-2", "典型日"]),
    ):
        sm = [df[c + "_平滑"].to_numpy() for c in cols]
        ax.stackplot(x, sm, labels=labels, alpha=0.9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.9)
    axes[1].set_xticks(month_starts)
    axes[1].set_xticklabels([dates[i][5:7] + "月" for i in month_starts])
    axes[1].set_xlabel("日期")
    pm.save_fig(fig, "Q2E_权重演化", data=df)
    print(f"Q2E_权重演化.pdf 已按 EWMA h={W:g} 重新生成")


if __name__ == "__main__":
    if "--result2" in __import__("sys").argv:
        write_result2_tuned()
    elif "--rolling-holdout" in __import__("sys").argv:
        rolling_holdout_search()
    elif "--fig-weights" in __import__("sys").argv:
        make_weight_figure()
    else:
        main()
