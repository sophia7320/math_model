"""Q2 口径 E 参数搜索：标定窗口 / EWMA / 决策裕度 / 风险目标 / 权重收缩。

扫描对象（全部因果执行，2025-02-01 ~ 12-31，334 天）：
A. 标定窗口 W（网格表 argmin）与 EWMA 半衰期（对日费用行做指数衰减加权）；
B. 决策裕度：全局常数 m、逐时段/逐小时/逐月分位数（数据分布参考值）、
   仿射 m(t)=a·q80(t)+b、比例式 P̂(1−ρ)、负荷抬升 κ_L、动态 κ（滚动分位数）；
C. 风险目标：窗口选择目标改成 (1−λ)·均值 + λ·CVaR20（日费用最差 20% 均值）；
D. 权重收缩：w ← (1−α)·w* + α/3（抑制极端权重）。
E. 精扫与稳健性：κ 细网格、κ×m、κ×小时裕度、动态 κ、分半验证。

参考值来自 `q2e_structure.py`（报童临界比 0.8 → 残差分位数）。
运行（在 program/ 下）：uv run python -m solve.q2_tune
"""
from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve.common import E0, RESULTS_DIR, ROOT
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
    """日费用行指数衰减加权（半衰期 hl 天）；hl≥1e8 退化为历史均值。"""
    grid = model.grid
    decay = float(np.exp(-np.log(2.0) / hl)) if hl < 1e8 else 1.0
    M = None
    for k in range(model.START, REPORT_START - 1):  # 预热：仅纳入 d=31 之前的历史日
        M = model.C[k].copy() if M is None else decay * M + (1 - decay) * model.C[k]
    out = []
    for d in REP:
        M = model.C[d - 1].copy() if M is None else decay * M + (1 - decay) * model.C[d - 1]
        i, j = np.unravel_index(np.argmin(M), M.shape)
        out.append((grid[i].copy(), grid[j].copy()))
    return out


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
             workers: int | None = None):
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
    if nw > 1:
        from multiprocessing import Pool

        with Pool(nw, initializer=_init_worker, initargs=(payload,)) as pool:
            for ci, di, c_plan, c_em in pool.imap_unordered(_task, tasks, chunksize=64):
                daily_plan[ci, di] = c_plan
                daily_emerg[ci, di] = c_em
    else:
        _init_worker(payload)
        for t in tasks:
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
    df1, dp1, de1 = evaluate(model, cfgs)
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
    df2, dp2, de2 = evaluate(model, cfgs2)
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
    df3, dp3, de3 = evaluate(model, cfgs3)
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
    df4, dp4, de4 = evaluate(model, cfgs4)
    df4.to_csv(pm.outputs_dir() / "q2e_tune_combo.csv", index=False, encoding="utf-8-sig")
    top4 = df4.loc[df4["总费用/万元"].idxmin()]
    print(f"阶段4 最优：{top4['配置']} = {top4['总费用/万元']} 万元")

    # ---------- 分半验证（H1: 2–6 月；H2: 7–12 月） ----------
    half = len(REP) // 2
    tagged = []
    for df, dp, de in ((df1, dp1, de1), (df2, dp2, de2), (df4, dp4, de4)):
        for i in range(len(df)):
            tagged.append((df.iloc[i]["配置"], dp[i], de[i]))
    tagged = pd.DataFrame(tagged, columns=["配置", "dp", "de"])
    candidates = ["网格 W=7"]
    for tag in (top2["配置"], top4["配置"]):
        if tag not in candidates:
            candidates.append(tag)
    dyn_tags = [t for t in tagged["配置"] if "动态κ" in t]
    if dyn_tags:
        candidates.append(dyn_tags[0])
    split_rows = []
    for tag in candidates:
        row = tagged[tagged["配置"] == tag].iloc[0]
        p1 = float(row["dp"][:half].sum() + row["de"][:half].sum()) / 1e4
        p2 = float(row["dp"][half:].sum() + row["de"][half:].sum()) / 1e4
        split_rows.append({"配置": tag, "上半年(2-6月)/万元": round(p1, 1),
                           "下半年(7-12月)/万元": round(p2, 1),
                           "全年/万元": round(p1 + p2, 1)})
    split = pd.DataFrame(split_rows)
    split.to_csv(pm.outputs_dir() / "q2e_tune_split.csv", index=False, encoding="utf-8-sig")

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
    from solve.q2e_structure import record

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
        "问题二 口径E 参数搜索：分半验证",
        split,
        note="上半年 2025-02-01~06-30，下半年 2025-07-01~12-31；用于检查裕度参数的跨期稳健性。",
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


def write_result2_tuned(W: float = 5.0, kappa: float = 1.02, margin: float = 50.0) -> dict:
    """把调优配置（EWMA h=5 + κ=1.02 + m=50 kW）写入 results/result2.xlsx。

    流程：EWMA 权重序列 → 逐日（计划用 κ/m 修正后的预测）→ 逐槽因果执行 →
    备份现官方文件到 code/outputs/result2_W7_backup.xlsx → 写出 + 回读校验 + 报告。
    """
    from solve.q2_adaptive import _verify_result2

    pm.init(root=str(ROOT))
    log = pm.get_logger("q2-tune")
    model = AdaptiveWeightModel().load().build_table()
    seq = seqs_ewma(model, W)

    x_plans = np.zeros((q2.N_DAY, q2.T))
    recs = [None] * q2.N_DAY
    rows = []
    for i, d in enumerate(REP):
        w, u = seq[i]
        l_kw, p_kw = model.forecast_kw(w, u, d)
        l_kw = np.clip(l_kw * kappa, 0.0, None)
        p_kw = np.clip(p_kw - margin, 0.0, None)
        x, _E, _ = q2.plan_day(model.price, l_kw / 6.0, p_kw / 6.0, E0, eps=EPS_PLAN)
        ex = q2.exec_day_causal(model.price, model.L[d] / 6.0, model.P[d] / 6.0, x, E0)
        x_plans[d] = x
        recs[d] = {"c": ex["c"], "d": ex["d"], "s": ex["s"], "e": ex["e"],
                   "E": ex["E"], "E_start": E0, "E_end": float(ex["E"][-1])}
        rows.append({
            "日期": model.dates[d],
            "计划购电量/kWh": float(x.sum()), "计划购电费/元": float(model.price @ x),
            "紧急购电量/kWh": float(ex["e"].sum()),
            "紧急购电费/元": float(q2.EMERG_MULT * (model.price @ ex["e"])),
        })
    df = pd.DataFrame(rows)
    plan = float(df["计划购电费/元"].sum())
    emerg = float(df["紧急购电费/元"].sum())

    cur = RESULTS_DIR / "result2.xlsx"
    if cur.exists():
        shutil.copy2(cur, pm.outputs_dir() / "result2_W7_backup.xlsx")
        log.info("旧（W=7）result2 备份 -> {}", pm.outputs_dir() / "result2_W7_backup.xlsx")
    out = q2.write_result2(model.dates, model.price, x_plans, recs, out_name="result2.xlsx")
    checks = _verify_result2(out, df)
    pm.save_outputs(df, "q2e_tuned_daily")
    record(
        "问题二 口径E（调优：EWMA h=5 + κ=1.02 + m=50）官方 result2 结果与校验",
        {
            "计划购电费/万元": round(plan / 1e4, 1),
            "紧急购电费/万元": round(emerg / 1e4, 1),
            "总费用/万元": round((plan + emerg) / 1e4, 1),
            "天数": len(df),
            **checks,
        },
        note=(
            "官方 results/result2.xlsx 由调优配置生成：权重为 EWMA（半衰期 5 天）逐日费用标定，"
            "预测修正 κ=1.02（负荷抬升）、m=50 kW（光伏折扣）；计划 LP 与因果执行与口径 E 相同。"
            "旧 W=7 版备份 code/outputs/result2_W7_backup.xlsx；复现："
            "uv run python -m solve.q2_tune --result2。"
        ),
    )
    log.info("result2.xlsx（调优）写出完成：{}，总费用 {:.1f} 万元（计划 {:.1f} + 紧急 {:.1f}）",
             out, (plan + emerg) / 1e4, plan / 1e4, emerg / 1e4)
    return {"plan": plan, "emerg": emerg, "checks": checks, "out": out}


def make_weight_figure(W: float = 5.0) -> None:
    """重生成权重演化图（EWMA h=5 标定口径，与官方模型一致）。"""
    import matplotlib.pyplot as plt

    pm.init(root=str(ROOT))
    model = AdaptiveWeightModel().load().build_table()
    seq = seqs_ewma(model, W)
    dates = [model.dates[d] for d in REP]
    Wm = np.array([w for w, _ in seq])
    Um = np.array([u for _, u in seq])
    w_cols = ["w1_L(d-7)", "w2_L(d-14)", "w3_典型日"]
    u_cols = ["u1_P(d-1)", "u2_P(d-2)", "u3_典型日"]
    df = pd.DataFrame({"日期": dates,
                       **{c: Wm[:, i] for i, c in enumerate(w_cols)},
                       **{c: Um[:, i] for i, c in enumerate(u_cols)}})
    x = np.arange(len(dates))
    month_starts = [i for i, s in enumerate(dates) if s.endswith("-01")]
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    for ax, cols, ylabel, labels in (
        (axes[0], w_cols, "负荷权重", ["d-7", "d-14", "典型日"]),
        (axes[1], u_cols, "光伏权重", ["d-1", "d-2", "典型日"]),
    ):
        ax.stackplot(x, [df[c].to_numpy() for c in cols], labels=labels, alpha=0.9)
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
    elif "--fig-weights" in __import__("sys").argv:
        make_weight_figure()
    else:
        main()
