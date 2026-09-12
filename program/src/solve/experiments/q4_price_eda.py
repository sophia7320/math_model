"""Q4 附件 4 波动电价结构分析（EDA）：日内/周内/年内结构、自相关、预测源可解释性。

运行（program/ 下）：uv run python -m solve.experiments.q4_price_eda
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import program as pm
from solve.common import DATA_C, ROOT

T = 144


def _acf_curve(x: np.ndarray, max_lag: int, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    n = len(x)
    denom = float((x * x).sum())
    lags = np.arange(1, max_lag + 1, step)
    vals = np.array([float((x[: n - k] * x[k:]).sum() / denom) for k in lags])
    return lags, vals


def main():
    pm.init(seed=42, root=str(ROOT))
    df = pm.read_table(DATA_C / "附件4.xlsx")
    price = df.iloc[:, 1:145].to_numpy(float)
    dates = pd.to_datetime(df.iloc[:, 0])

    a1 = pm.read_table(DATA_C / "附件1.xlsx")
    p1 = a1["电价"].to_numpy(float)

    out = {}
    out["基本统计"] = {
        "均值": float(price.mean()), "标准差": float(price.std()),
        "最小": float(price.min()), "P25": float(np.percentile(price, 25)),
        "中位": float(np.median(price)), "P75": float(np.percentile(price, 75)),
        "P95": float(np.percentile(price, 95)), "P99": float(np.percentile(price, 99)),
        "最大": float(price.max()),
        "偏度": float(pd.Series(price.ravel()).skew()),
        "超额峰度": float(pd.Series(price.ravel()).kurt()),
    }
    print("=== 基本统计 ===")
    for k, v in out["基本统计"].items():
        print(f"  {k}: {v:.4f}")

    # 附件 1 = 逐槽全年均值？
    diff = np.abs(price.mean(axis=0) - p1)
    print(f"\n附件 1 电价 vs 附件 4 逐槽均值：最大差 {diff.max():.2e}（应为 0）")
    out["附件1_vs_均值_最大差"] = float(diff.max())

    # ---- 结构分解 ----
    m_slot = price.mean(axis=0)                     # 日内形状
    day_mean = price.mean(axis=1)                   # 每天均值
    resid_intra = price - m_slot                    # 扣除日内形状
    R2_intra = 1 - resid_intra.var() / price.var()
    weekday = np.array([d.weekday() for d in dates])
    month = np.array([d.month for d in dates])
    # 周内/月内对"天均值"的解释
    wd_eff = np.array([day_mean[weekday == w].mean() for w in weekday])
    mo_eff = np.array([day_mean[month == m].mean() for m in month])
    R2_wd = 1 - (day_mean - wd_eff).var() / day_mean.var()
    R2_mo = 1 - (day_mean - mo_eff).var() / day_mean.var()
    print(f"\n=== 方差分解 ===")
    print(f"  日内形状 R²（144 槽均值解释总方差）: {R2_intra:.3f}")
    print(f"  周内效应对天均值的 R²: {R2_wd:.3f}")
    print(f"  月内效应对天均值的 R²: {R2_mo:.3f}")
    out["方差分解"] = {"日内形状R2": float(R2_intra), "周内R2天均值": float(R2_wd),
                      "月内R2天均值": float(R2_mo)}

    print("\n  周内均值（周一~周日）:")
    print("   ", np.round([day_mean[weekday == w].mean() for w in range(7)], 4))
    print("  月均值:")
    print("   ", np.round([day_mean[month == m].mean() for m in range(1, 13)], 4))

    # ---- 自相关 ----
    lags, acf = _acf_curve(price.ravel(), 1008, 1)   # 到 7 天（逐槽）
    print(f"\n=== 自相关（槽级）===")
    for k in (1, 6, 36, 144, 288, 432, 720, 1008):
        idx = np.where(lags == k)[0]
        if len(idx):
            print(f"  lag {k:4d} 槽（{k * 10 / 60:5.1f} h）: {acf[idx[0]]:+.3f}")
    dm_lags, dm_acf = _acf_curve(day_mean, 14)
    print("  天均值 ACF（lag 1..7 天）:", np.round(dm_acf[:7], 3))
    out["ACF"] = {"slot_lags": lags.tolist(), "slot_acf": acf.tolist(),
                  "daymean_lags": dm_lags.tolist(), "daymean_acf": dm_acf.tolist()}

    # ---- 预测源可解释性（D 与源的正确对齐）----
    P = price
    print(f"\n=== 预测源相关性（D 与源，逐槽）===")
    r_all = {}
    for lag, name in ((1, "D−1 同时刻"), (2, "D−2 同时刻"), (7, "D−7 同时刻")):
        a, b = P[lag:], P[:-lag]
        rr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        r_all[name] = rr
        print(f"  {name:<12}: r = {rr:.3f}")
    rr = float(np.corrcoef(P.ravel(), np.tile(p1, (365, 1)).ravel())[0, 1])
    r_all["典型日(附件1)"] = rr
    print(f"  {'典型日(附件1)':<12}: r = {rr:.3f}")
    out["源相关"] = r_all

    # 三源联合回归（逐槽最小二乘，全样本）
    m = 365 - 7
    X = np.stack([P[6:365 - 1], P[5:365 - 2], P[:365 - 7]], axis=2)  # (n,144,3)
    Y = P[7:]
    Xf = X.reshape(-1, 3)
    Yf = Y.ravel()
    Xd = np.column_stack([Xf, np.ones(len(Xf))])
    coef, *_ = np.linalg.lstsq(Xd, Yf, rcond=None)
    pred = Xd @ coef
    R2 = 1 - ((Yf - pred) ** 2).sum() / ((Yf - Yf.mean()) ** 2).sum()
    print(f"\n三源（D−1/D−2/D−7）联合回归 R² = {R2:.3f}，系数 {np.round(coef[:3], 3)}")
    out["三源回归"] = {"R2": float(R2), "coef": coef[:3].tolist()}

    # ---- 极端价格 ----
    thr = np.percentile(price, 99)
    mask = price >= thr
    slot_share = mask.sum(axis=0) / mask.sum()
    top_slots = np.argsort(slot_share)[::-1][:10]
    hot_hours = sorted(set(int(s * 10 // 60) for s in top_slots))
    print(f"\n=== 极端价格（≥P99={thr:.3f}）===")
    print(f"  样本数 {int(mask.sum())}（占 {100 * mask.mean():.2f}%），最集中时段小时：{hot_hours}")
    out["极端价格"] = {"P99": float(thr), "n": int(mask.sum()), "热点小时": hot_hours}

    # ---- 与负荷/光伏的相关性 ----
    sheets = pm.read_sheets(DATA_C / "附件2.xlsx")
    load = sheets["小区负载"].iloc[:, 1:145].to_numpy(float)
    pv = sheets["光伏发电实际功率"].iloc[:, 1:145].to_numpy(float)
    print(f"\n=== 与负荷/光伏的同时刻相关 ===")
    for name, arr in (("负荷", load), ("光伏", pv)):
        rr = float(np.corrcoef(price.ravel(), arr.ravel())[0, 1])
        print(f"  {name}: r = {rr:.3f}")
        out.setdefault("相关_负荷光伏", {})[name] = rr

    # ---- 电价峰谷的日内时段特征（用于口径说明）----
    lo = np.argsort(m_slot)[:12]
    hi = np.argsort(m_slot)[::-1][:12]
    print("\n日内均值最低时段（槽索引）:", lo.tolist())
    print("日内均值最高时段（槽索引）:", sorted(hi.tolist()))
    out["日内极值槽"] = {"最低": sorted(lo.tolist()), "最高": sorted(hi.tolist())}

    with open(ROOT / "code" / "outputs" / "q4_price_eda.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # ---- 图：ACF + 结构分解 ----
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags / 6.0, acf, lw=0.9)
    for k, lab in ((144 / 6, "1 天"), (720 / 6, "5 天"), (1008 / 6, "7 天")):
        ax.axvline(k, color="gray", ls="--", lw=0.7)
        ax.text(k, ax.get_ylim()[1] * 0.9, lab, fontsize=8, ha="center")
    ax.set_xlabel("滞后 / 小时"); ax.set_ylabel("ACF")
    pm.save_fig(fig, "Q4_电价自相关", data=pd.DataFrame({"滞后_小时": lags / 6.0, "ACF": acf}))

    fig2, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    t_h = (np.arange(T) + 0.5) / 6.0
    axes[0].plot(t_h, m_slot, lw=1.2); axes[0].set_xlabel("时刻 / h"); axes[0].set_ylabel("电价 / (元·kWh$^{-1}$)")
    axes[0].set_title("日内均值曲线", fontsize=9)
    axes[1].bar(np.arange(7), [day_mean[weekday == w].mean() for w in range(7)],
                color="#4C72B0"); axes[1].set_xticks(np.arange(7))
    axes[1].set_xticklabels(["一", "二", "三", "四", "五", "六", "日"])
    axes[1].set_title("周内均值（天均值）", fontsize=9)
    axes[2].bar(np.arange(1, 13), [day_mean[month == m].mean() for m in range(1, 13)],
                color="#55A868")
    axes[2].set_title("月均值（天均值）", fontsize=9)
    fig2.tight_layout()
    pm.save_fig(fig2, "Q4_电价结构分解",
                data=pd.DataFrame({"时段": np.arange(T), "日内均值": m_slot}))
    print("\n已保存 code/outputs/q4_price_eda.json 与两张图")


if __name__ == "__main__":
    main()
