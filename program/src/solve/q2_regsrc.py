"""Q2 口径 E 候选改进：回归型参考量（用户思路）——探针 + 全流程测试。

原口径：L^ = w1·L(d−7) + w2·L(d−14) + w3·典型日（可信度权重由费用标定）。
新思路：把前两个"来源"换成两个回归预测参考量：
  参考量1（周期型）：由 y(d−28), y(d−21), y(d−14), y(d−7) 做线性回归，预测 d；
  参考量2（连续型）：由 y(d−4), y(d−3), y(d−2), y(d−1) 做线性回归，预测 d；
之后仍按三源（参考量1、参考量2、典型日）可信度权重组合，光伏同样处理。

两种"线性回归"实现：
  T1 趋势外推：4 点等间隔拟合直线并外推一步（闭式权重）；
  T2 滚动多元回归：4 个滞后为特征，最近 60 天同槽 OLS，逐日重估（无前视）。

模式：
  uv run python -m solve.q2_regsrc            # 探针：精度 + 固定权重费用
  uv run python -m solve.q2_regsrc --full     # 全流程：甲（严格）/乙（光伏适配）各建表 + 扫描
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

import program as pm
from solve import q2
from solve.common import E0, ROOT
from solve.q2_adaptive import AdaptiveWeightModel

REP = list(range(q2.REPORT_START, q2.N_DAY))
LAG_W = (28, 21, 14, 7)   # 周期型滞后（相隔 7 天）
LAG_C = (4, 3, 2, 1)      # 连续型滞后
LAG_P2 = (8, 6, 4, 2)     # 光伏备选：相隔 2 天


# ---------------------------------------------------------------------------
# 预测器
# ---------------------------------------------------------------------------
def extrap_coef(n: int) -> np.ndarray:
    """n 个等间隔点（坐标 −n..−1）拟合直线并外推至 0 的线性权重。"""
    xs = -np.arange(n, 0, -1, dtype=float)
    A = np.stack([np.ones(n), xs], axis=1)
    return A @ np.linalg.inv(A.T @ A)[:, 0]


def t1_predict(series: np.ndarray, lags, d: int):
    pts = [(x, series[d - lag]) for x, lag in zip(-np.arange(len(lags), 0, -1), lags)
           if d - lag >= 0]
    if len(pts) < 2:
        return None
    c = extrap_coef(len(pts))
    return np.tensordot(c, np.stack([y for _, y in pts]), axes=(0, 0))


def t2_predict(series: np.ndarray, lags, d: int, Wr: int = 60):
    """同槽滚动 OLS：特征 = 4 个滞后值，样本 = 最近 Wr 天（k ≤ d−1）。"""
    ks = np.arange(max(0, d - Wr), d)
    ks = ks[ks >= max(lags)]
    n = len(ks)
    if n < 8:
        return None
    Y = series[ks]                                            # (n,144)
    X = np.stack([series[ks - lag] for lag in lags], axis=1)  # (n,4,144)
    Xd = np.concatenate([np.ones((n, 1, q2.T)), X], axis=1)   # (n,5,144)
    XtX = np.einsum("nij,nkj->ikj", Xd, Xd) + 1e-8 * np.eye(5)[:, :, None]
    XtY = np.einsum("nij,nj->ij", Xd, Y)
    beta = np.linalg.solve(XtX.transpose(2, 0, 1), XtY.T[..., None])[..., 0]
    x0 = np.stack([series[d - lag] for lag in lags], axis=1)   # (144,4)
    return beta[:, 0] + np.einsum("sj,sj->s", beta[:, 1:], x0)


def build_sources(L: np.ndarray, P: np.ndarray, d_from: int = 14, t2_wr: int = 60) -> dict:
    """为标定表所需的所有天构建 10 个参考量矩阵（早期样本不足退化为趋势外推）。"""
    keys = ("L_T1w", "L_T1c", "L_T2w", "L_T2c",
            "P_T1w", "P_T1c", "P_T2w", "P_T2c", "P_T1p2", "P_T2p2")
    src = {k: np.full((q2.N_DAY, q2.T), np.nan) for k in keys}
    for d in range(d_from, q2.N_DAY):
        for key, series, lags in (("L_T1w", L, LAG_W), ("L_T1c", L, LAG_C),
                                  ("P_T1w", P, LAG_W), ("P_T1c", P, LAG_C),
                                  ("P_T1p2", P, LAG_P2)):
            v = t1_predict(series, lags, d)
            if v is not None:
                src[key][d] = np.clip(v, 0.0, None)
        for key, series, lags in (("L_T2w", L, LAG_W), ("L_T2c", L, LAG_C),
                                  ("P_T2w", P, LAG_W), ("P_T2c", P, LAG_C),
                                  ("P_T2p2", P, LAG_P2)):
            v = t2_predict(series, lags, d, Wr=t2_wr)
            fallback = src[key.replace("T2", "T1")][d]
            if v is not None:
                src[key][d] = np.clip(v, 0.0, None)
            elif not np.isnan(fallback).any():
                src[key][d] = fallback
    return src


def metrics(y, yhat) -> tuple[float, float]:
    e = yhat - y
    return float(np.abs(e).mean()), float(np.sqrt((e ** 2).mean()))


def shift(arr: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(arr)
    out[k:] = arr[:-k]
    return out


# ---------------------------------------------------------------------------
# 回归源模型（口径 E 变体：两个回归参考量 + 典型日）
# ---------------------------------------------------------------------------
class RegSourceModel(AdaptiveWeightModel):
    """把 forecast 换成"两个回归参考量 + 典型日"的口径 E 变体（标定框架不变）。"""

    CACHE_TAG = "q2e_regsrc"   # 与官方 q2e 缓存隔离（q3_proto 只认 tag=q2e 的表）

    def __init__(self, sources, name: str = "regsrc", **kwargs):
        super().__init__(**kwargs)
        self.S1L, self.S2L, self.S1P, self.S2P = sources
        self.NAME = name

    def forecast_kw(self, w, u, d: int):
        l = w[0] * self.S1L[d] + w[1] * self.S2L[d] + w[2] * self.L_typ
        p = u[0] * self.S1P[d] + u[1] * self.S2P[d] + u[2] * self.P_typ
        return l, p

    def forecast(self, w, u, d: int):
        l, p = self.forecast_kw(w, u, d)
        return np.clip(l, 0.0, None) / 6.0, np.clip(p, 0.0, None) / 6.0

    def _cache_key(self) -> str:
        h = hashlib.md5()
        for arr in (self.S1L, self.S2L, self.S1P, self.S2P, self.L_typ, self.P_typ):
            h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
        h.update(np.asarray([self.grid_step, self.steps, self.lr, self.eps_grad], float).tobytes())
        h.update(q2.CAUSAL_POLICY_VERSION.encode("utf-8"))
        h.update(self.NAME.encode("utf-8"))
        return h.hexdigest()[:12]

    def _one_day(self, d: int):
        """只算网格行（本变体不用 GD 精化，省 1/4 建表时间）。"""
        n = len(self.grid)
        row = np.empty((n, n))
        for i, w in enumerate(self.grid):
            for j, u in enumerate(self.grid):
                row[i, j] = self.day_cost(d, w, u)
        c = float(row.min())
        return d, row, np.zeros(6), c, c


# ---------------------------------------------------------------------------
# 并行评估（含裕度）
# ---------------------------------------------------------------------------
_W = None


def _w_init(payload):
    global _W
    _W = payload


def _w_task(t):
    ci, si, mi, d = t
    p = _W
    w, u = p["seqs"][si][d - q2.REPORT_START]
    l = w[0] * p["S1L"][d] + w[1] * p["S2L"][d] + w[2] * p["L_typ"]
    pp = u[0] * p["S1P"][d] + u[1] * p["S2P"][d] + u[2] * p["P_typ"]
    spec = p["margins"][mi]
    if spec is not None:
        kind = spec["kind"]
        if kind == "pv_global":
            pp = pp - spec["m"]
        elif kind == "load_kappa":
            l = l * spec["k"]
        elif kind == "combo":
            l = l * spec["k"]
            pp = pp - spec["m"]
    l = np.clip(l, 0.0, None) / 6.0
    pp = np.clip(pp, 0.0, None) / 6.0
    x, _E, _ = q2.plan_day(p["price"], l, pp, E0, eps=1e-3)
    ex = q2.exec_day_causal(p["price"], p["L"][d] / 6.0, p["P"][d] / 6.0, x, E0)
    return ci, d - q2.REPORT_START, float(p["price"] @ x), float(q2.EMERG_MULT * (p["price"] @ ex["e"]))


def eval_configs(model: RegSourceModel, configs, workers: int | None = None) -> pd.DataFrame:
    """configs: list of (名称, 权重序列, 裕度 spec)；返回年度费用表。"""
    seqs, sid, margins, mid = [], {}, [], {}
    for name, seq, spec in configs:
        if id(seq) not in sid:
            sid[id(seq)] = len(seqs)
            seqs.append(seq)
        key = -1 if spec is None else id(spec)
        if key not in mid:
            mid[key] = len(margins)
            margins.append(spec)
    tasks = [(ci, sid[id(seq)], mid[-1 if spec is None else id(spec)], d)
             for ci, (_, seq, spec) in enumerate(configs) for d in REP]
    payload = {"seqs": seqs, "margins": margins,
               "S1L": model.S1L, "S2L": model.S2L, "S1P": model.S1P, "S2P": model.S2P,
               "L_typ": model.L_typ, "P_typ": model.P_typ,
               "price": model.price, "L": model.L, "P": model.P}
    n = len(configs)
    plan = np.zeros(n)
    emerg = np.zeros(n)
    nw = workers or min(8, os.cpu_count() or 1)
    t0 = time.time()
    if nw > 1:
        from multiprocessing import Pool

        with Pool(nw, initializer=_w_init, initargs=(payload,)) as pool:
            for ci, di, c_plan, c_em in pool.imap_unordered(_w_task, tasks, chunksize=64):
                plan[ci] += c_plan
                emerg[ci] += c_em
    else:
        _w_init(payload)
        for t in tasks:
            ci, di, c_plan, c_em = _w_task(t)
            plan[ci] += c_plan
            emerg[ci] += c_em
    df = pd.DataFrame({
        "配置": [c[0] for c in configs],
        "计划购电费/万元": np.round(plan / 1e4, 1),
        "紧急购电费/万元": np.round(emerg / 1e4, 1),
    })
    df["总费用/万元"] = (df["计划购电费/万元"] + df["紧急购电费/万元"]).round(1)
    print(f"  批量评估 {n} 个配置完成（{time.time() - t0:.0f}s）")
    return df


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------
def run_probe(model: AdaptiveWeightModel):
    L, P = model.L, model.P
    t0 = time.time()
    src = build_sources(L, P, d_from=q2.REPORT_START)
    print(f"预测器构造完成（{time.time() - t0:.0f}s）")

    model.build_table()
    ws = model.weights(W=7, use_gd=False)
    E_L = np.full((q2.N_DAY, q2.T), np.nan)
    E_P = np.full((q2.N_DAY, q2.T), np.nan)
    for i, d in enumerate(REP):
        w, u = ws[i]
        l_kwh, p_kwh = model.forecast(w, u, d)
        E_L[d], E_P[d] = l_kwh * 6.0, p_kwh * 6.0

    rows = []
    for name, series, typ in (("负荷", L, model.L_typ), ("光伏", P, model.P_typ)):
        y = np.stack([series[d] for d in REP])
        base = {"原始 d−7": series[np.array(REP) - 7],
                "原始 d−14": series[np.array(REP) - 14],
                "原始 d−1": series[np.array(REP) - 1],
                "原始 d−2": series[np.array(REP) - 2],
                "典型日": np.stack([typ for _ in REP])}
        preds = dict(base)
        if name == "负荷":
            preds.update({k: np.stack([src[k][d] for d in REP]) for k in
                          ("L_T1w", "L_T1c", "L_T2w", "L_T2c")})
            preds["口径E(W=7)"] = np.stack([E_L[d] for d in REP])
        else:
            preds.update({k: np.stack([src[k][d] for d in REP]) for k in
                          ("P_T1w", "P_T1c", "P_T2w", "P_T2c", "P_T1p2", "P_T2p2")})
            preds["口径E(W=7)"] = np.stack([E_P[d] for d in REP])
        for k, v in preds.items():
            mae, rmse = metrics(y, v)
            rows.append({"变量": name, "预测器": k, "MAE/kW": round(mae, 1),
                         "RMSE/kW": round(rmse, 1)})
    acc = pd.DataFrame(rows)
    pm.save_outputs(acc, "q2_regsrc_probe")
    print(acc.to_string(index=False))

    def cost_of(load_src, pv_src, w, u):
        plan = emerg = 0.0
        for d in REP:
            l = np.clip(w[0] * load_src[0][d] + w[1] * load_src[1][d]
                        + w[2] * load_src[2][d], 0.0, None) / 6.0
            p = np.clip(u[0] * pv_src[0][d] + u[1] * pv_src[1][d]
                        + u[2] * pv_src[2][d], 0.0, None) / 6.0
            x, _E, _ = q2.plan_day(model.price, l, p, E0, eps=1e-3)
            ex = q2.exec_day_causal(model.price, L[d] / 6.0, P[d] / 6.0, x, E0)
            plan += float(model.price @ x)
            emerg += float(q2.EMERG_MULT * (model.price @ ex["e"]))
        return plan, emerg

    typL = np.stack([model.L_typ] * q2.N_DAY)
    typP = np.stack([model.P_typ] * q2.N_DAY)
    L7, L14 = shift(L, 7), shift(L, 14)
    P1, P2 = shift(P, 1), shift(P, 2)
    cases = [
        ("老口径 等权(L_{d−7},L_{d−14},典型)", (L7, L14, typL), (P1, P2, typP),
         (1 / 3, 1 / 3, 1 / 3)),
        ("新T1 等权(周期,连续,典型)", (src["L_T1w"], src["L_T1c"], typL),
         (src["P_T1w"], src["P_T1c"], typP), (1 / 3, 1 / 3, 1 / 3)),
        ("新T2 等权(周期,连续,典型)", (src["L_T2w"], src["L_T2c"], typL),
         (src["P_T2w"], src["P_T2c"], typP), (1 / 3, 1 / 3, 1 / 3)),
        ("新T2 (0.4,0.4,0.2)", (src["L_T2w"], src["L_T2c"], typL),
         (src["P_T2w"], src["P_T2c"], typP), (0.4, 0.4, 0.2)),
    ]
    rows = []
    for name, load_src, pv_src, w in cases:
        t1 = time.time()
        plan, emerg = cost_of(load_src, pv_src, w, w)
        tot = (plan + emerg) / 1e4
        rows.append({"配置": name, "计划费/万元": round(plan / 1e4, 1),
                     "紧急费/万元": round(emerg / 1e4, 1), "总费用/万元": round(tot, 1)})
        print(f"  {name}: {tot:.1f} 万（{time.time() - t1:.0f}s）")
    cost = pd.DataFrame(rows)
    pm.save_outputs(cost, "q2_regsrc_cost_probe")
    return acc, cost


# ---------------------------------------------------------------------------
# 全流程（甲/乙）
# ---------------------------------------------------------------------------
def run_variant(name: str, sources, workers: int | None = None):
    from solve.q2_tune import seqs_ewma

    print(f"\n===== 变体 {name}：建表 =====")
    model = RegSourceModel(sources, name=name).load()
    model.build_table()
    t0 = time.time()

    cfgs = []
    for W in (1, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120):
        cfgs.append((f"{name} W={W}", model.weights(W=W, use_gd=False), None))
    for hl in (1, 2, 3, 5, 7, 10, 14, 21, 30):
        cfgs.append((f"{name} EWMA h={hl}", seqs_ewma(model, hl), None))
    df1 = eval_configs(model, cfgs, workers)
    df1.to_csv(pm.outputs_dir() / f"q2_regsrc_{name}_weights.csv",
               index=False, encoding="utf-8-sig")
    print(df1.sort_values("总费用/万元").head(5).to_string(index=False))

    best = df1.sort_values("总费用/万元").iloc[0]
    best_name = best["配置"]
    best_seq = next(seq for nm, seq, _ in cfgs if nm == best_name)
    print(f"变体 {name} 权重最优：{best_name} = {best['总费用/万元']} 万元")

    cfgs2 = []
    for mv in (50, 100, 150, 200, 250, 300):
        cfgs2.append((f"{best_name} m={mv}", best_seq, {"kind": "pv_global", "m": float(mv)}))
    for k in (1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05):
        cfgs2.append((f"{best_name} κ={k}", best_seq, {"kind": "load_kappa", "k": k}))
    for k in (1.01, 1.015, 1.02, 1.025, 1.03):
        for mv in (25.0, 50.0, 75.0, 100.0):
            cfgs2.append((f"{best_name} κ={k}+m={mv:.0f}", best_seq,
                          {"kind": "combo", "k": k, "m": mv}))
    df2 = eval_configs(model, cfgs2, workers)
    df2.to_csv(pm.outputs_dir() / f"q2_regsrc_{name}_margin.csv",
               index=False, encoding="utf-8-sig")
    best2 = df2.sort_values("总费用/万元").iloc[0]
    print(f"变体 {name} 含裕度最优：{best2['配置']} = {best2['总费用/万元']} 万元"
          f"（{time.time() - t0:.0f}s）")
    return df1, df2, best2


def run_full(workers: int | None = None, only: list[str] | None = None,
             tag: str | None = None):
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q2-regsrc")
    t0 = time.time()
    base = AdaptiveWeightModel().load()
    src60 = build_sources(base.L, base.P, 14, t2_wr=60)
    src120 = build_sources(base.L, base.P, 14, t2_wr=120)
    for s in (src60, src120):
        s["L7"], s["L14"] = shift(base.L, 7), shift(base.L, 14)
    variants = {
        "甲": (src60, {"L1": "L_T2w", "L2": "L_T2c", "P1": "P_T2w", "P2": "P_T2c"}),
        "乙": (src60, {"L1": "L_T2w", "L2": "L_T2c", "P1": "P_T2c", "P2": "P_T2p2"}),
        "丙": (src60, {"L1": "L7", "L2": "L14", "P1": "P_T2c", "P2": "P_T2p2"}),
        "丁": (src60, {"L1": "L7", "L2": "L14", "P1": "P_T2c", "P2": "P_T2w"}),
        "丙Wr120": (src120, {"L1": "L7", "L2": "L14", "P1": "P_T2c", "P2": "P_T2p2"}),
    }
    if only:
        variants = {k: v for k, v in variants.items() if k in only}
    summary = []
    for name, (s, sel) in variants.items():
        sources = (s[sel["L1"]], s[sel["L2"]], s[sel["P1"]], s[sel["P2"]])
        _, _, best2 = run_variant(name, sources, workers)
        summary.append({"变体": name, "最优配置": best2["配置"],
                        "计划费/万元": best2["计划购电费/万元"],
                        "紧急费/万元": best2["紧急购电费/万元"],
                        "总费用/万元": best2["总费用/万元"],
                        "相对官方1467.4/%": round(100 * (best2["总费用/万元"] / 1467.4 - 1), 2),
                        "相对旧最优1406.6/%": round(100 * (best2["总费用/万元"] / 1406.6 - 1), 2)})
    out = pd.DataFrame(summary)
    suffix = f"_{tag}" if tag else ""
    pm.save_outputs(out, f"q2_regsrc_full_summary{suffix}")
    print("\n===== 汇总 =====")
    print(out.to_string(index=False))
    log.info("全流程完成（{:.0f}s）", time.time() - t0)
    return out


if __name__ == "__main__":
    if "--wr" in sys.argv:
        run_full(only=["丙", "丙Wr120"], tag="wr")
    elif "--full" in sys.argv:
        run_full()
    else:
        pm.init(seed=42, root=str(ROOT))
        _m = AdaptiveWeightModel().load()
        run_probe(_m)
