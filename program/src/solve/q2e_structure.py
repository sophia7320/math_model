"""Q2 口径 E：误差分布结构分析与参数参考值推导（先验版）。

对官方口径（历史自适应加权 W=7 网格选择）在报送期 2025-02-01 ~ 12-31 的
预测误差做分布结构分析，把模型参数与数据分布挂钩：

1. 净需求偏差裕度 m_ref（对应"决策最优预测"的报童临界比 0.8）：
   m_ref = F_{e}^{-1}(0.8)，e = 实际净需求 − 预测净需求；
   给出全局 / 逐时段 / 逐小时 / 逐月 / 比例式（e 相对预报光伏）五种口径；
2. 标定窗口 W 与 EWMA 半衰期 h_ref：由日度偏差序列的 AR(1) 记忆推得；
3. 风险参考：官方计划的日费用尾部分布（P95/CVaR95），为风险厌恶系数 λ 扫描定标。

输出：code/outputs/q2e_structure_{slots,hours,months}.csv、q2e_structure_summary.json
图：figures/Q2E_误差结构与参考值.pdf
运行（在 program/ 下）：uv run python -m solve.q2e_structure
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy import stats

import program as pm
from solve import q2
from solve.common import ROOT
from solve.q2_adaptive import AdaptiveWeightModel


def record(section: str, data, note: str = "") -> None:
    """写结果报告：先删除同名旧章节再追加。"""
    path = pm.reports_dir() / "RESULTS_REPORT.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        marker = f"### {section}"
        pos = text.find(marker)
        if pos != -1:
            end = text.find("\n### ", pos + len(marker))
            text = text[:pos] if end == -1 else text[:pos] + text[end + 1 :]
            path.write_text(text, encoding="utf-8")
    pm.record_result(section, data, note=note)


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """样本自相关函数（0..max_lag）。"""
    x = np.asarray(x, float)
    x = x - x.mean()
    out = [1.0]
    for k in range(1, max_lag + 1):
        out.append(float(np.corrcoef(x[:-k], x[k:])[0, 1]))
    return np.array(out)


def ar1_half_life(x: np.ndarray) -> tuple[float, float]:
    """AR(1) 系数与半衰期（天）；系数 ≤0 时半衰期记 NaN。"""
    x = np.asarray(x, float)
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    hl = float(np.log(0.5) / np.log(rho)) if 0.0 < rho < 1.0 else float("nan")
    return rho, hl


def forecast_kw(model: AdaptiveWeightModel, w, u, d: int):
    """第 d 天预测功率（kW，未裁剪）。"""
    l = w[0] * model.L[d - 7] + w[1] * model.L[d - 14] + w[2] * model.L_typ
    p = u[0] * model.P[d - 1] + u[1] * model.P[d - 2] + u[2] * model.P_typ
    return l, p


def main() -> dict:
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q2e-structure")
    t0 = time.time()
    model = AdaptiveWeightModel().load().build_table()
    months = model.data["months"]

    rep = list(range(q2.REPORT_START, q2.N_DAY))
    ws = model.weights(W=7, use_gd=False)  # 官方 W=7 网格序列

    n = len(rep)
    e_net = np.zeros((n, q2.T))   # 实际净需求 − 预测净需求（kW）
    e_pv = np.zeros((n, q2.T))    # 预报光伏 − 实际光伏（kW）
    e_load = np.zeros((n, q2.T))  # 预报负荷 − 实际负荷（kW）
    p_hat = np.zeros((n, q2.T))
    for i, d in enumerate(rep):
        w, u = ws[i]
        l, p = forecast_kw(model, w, u, d)
        p_hat[i] = np.clip(p, 0.0, None)
        e_load[i] = l - model.L[d]
        e_pv[i] = p - model.P[d]
        e_net[i] = (model.L[d] - model.P[d]) - (l - p)

    # ---- 分布形状（全样本 / 白天样本）----
    day_mask = p_hat > 500.0  # 光照样本（按预报出力定义）
    x_all = e_net.ravel()
    x_day = e_net[day_mask]
    shape = {
        "样本数（全时段）": int(x_all.size),
        "样本数（白天）": int(x_day.size),
        "均值/kW（全时段）": float(x_all.mean()),
        "标准差/kW（全时段）": float(x_all.std()),
        "偏度（全时段）": float(stats.skew(x_all)),
        "峰度（全时段）": float(stats.kurtosis(x_all)),
        "均值/kW（白天）": float(x_day.mean()),
        "标准差/kW（白天）": float(x_day.std()),
        "偏度（白天）": float(stats.skew(x_day)),
    }
    qq = {f"q{int(q * 100)}": float(np.quantile(x_day, q))
          for q in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)}

    # ---- 裕度参考值 ----
    m_global_all = float(np.quantile(x_all, 0.80))
    m_global_day = float(np.quantile(x_day, 0.80))
    m_slot = np.quantile(e_net, 0.80, axis=0)            # 144
    m_hour = np.array([np.quantile(e_net[:, 6 * h:6 * (h + 1)], 0.80)
                       for h in range(24)])
    m_month = np.array([
        np.quantile(e_net[months[rep] == mm], 0.80) if (months[rep] == mm).any() else np.nan
        for mm in range(1, 13)
    ])
    # 比例式：白天 e_net / 预报光伏 的 0.8 分位
    ratio = (e_net[day_mask] / p_hat[day_mask])
    rho_ref = float(np.quantile(ratio, 0.80))
    # 条件结构：按预报出力分桶的 q80（检查裕度是否随出力增大）
    buckets = [(500, 2000), (2000, 3500), (3500, 5000), (5000, 8000)]
    bucket_rows = []
    for lo, hi in buckets:
        mask = (p_hat > lo) & (p_hat <= hi)
        if mask.any():
            bucket_rows.append({
                "预报光伏区间/kW": f"({lo},{hi}]",
                "样本数": int(mask.sum()),
                "q80/kW": float(np.quantile(e_net[mask], 0.80)),
                "均值/kW": float(e_net[mask].mean()),
            })

    # ---- 时间记忆：日度偏差的 AR(1) 与 ACF ----
    b_net = e_net.mean(axis=1)                       # 日度净需求偏差
    with np.errstate(invalid="ignore"):
        day_bias = np.array([
            e_net[i][day_mask[i]].mean() if day_mask[i].any() else np.nan
            for i in range(n)
        ])
    good = ~np.isnan(day_bias)
    rho_net, hl_net = ar1_half_life(b_net)
    rho_day, hl_day = ar1_half_life(day_bias[good])
    u1 = np.array([w[0] for w, _ in ws])             # 光伏 d−1 权重序列
    rho_u1, hl_u1 = ar1_half_life(u1)
    acf_net = acf(b_net, 30)
    acf_u1 = acf(u1, 30)

    # ---- 风险：官方计划日费用尾部分布 ----
    df_official = model.simulate(ws)
    daily_total = (df_official["计划购电费/元"] + df_official["紧急购电费/元"]).to_numpy()
    p95 = float(np.percentile(daily_total, 95))
    risk = {
        "日费用均值/元": float(daily_total.mean()),
        "日费用标准差/元": float(daily_total.std()),
        "日费用P95/元": p95,
        "日费用CVaR95/元": float(daily_total[daily_total >= p95].mean()),
        "尾部溢价(P95−均值)/均值": float((p95 - daily_total.mean()) / daily_total.mean()),
        "日费用偏度": float(stats.skew(daily_total)),
        "有紧急购电天数占比": float((df_official["紧急购电量/kWh"] > 1e-3).mean()),
    }

    # ---- 导出 ----
    pd.DataFrame({
        "槽序号": np.arange(1, q2.T + 1),
        "e_net_均值/kW": e_net.mean(axis=0),
        "e_net_标准差/kW": e_net.std(axis=0),
        "e_net_q80/kW": m_slot,
    }).to_csv(pm.outputs_dir() / "q2e_structure_slots.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "小时": np.arange(1, 25),
        "e_net_均值/kW": [e_net[:, 6 * h:6 * (h + 1)].mean() for h in range(24)],
        "e_net_标准差/kW": [e_net[:, 6 * h:6 * (h + 1)].std() for h in range(24)],
        "e_net_q80/kW": m_hour,
    }).to_csv(pm.outputs_dir() / "q2e_structure_hours.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "月": np.arange(1, 13),
        "e_net_均值/kW": [
            e_net[months[rep] == mm].mean() if (months[rep] == mm).any() else np.nan
            for mm in range(1, 13)],
        "e_net_q80/kW": m_month,
    }).to_csv(pm.outputs_dir() / "q2e_structure_months.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "滞后/天": np.arange(31),
        "日度净需求偏差ACF": acf_net,
        "光伏d−1权重ACF": acf_u1,
    }).to_csv(pm.outputs_dir() / "q2e_structure_acf.csv", index=False, encoding="utf-8-sig")

    summary = {
        **shape,
        **qq,
        "m80_全时段/kW": m_global_all,
        "m80_白天/kW": m_global_day,
        "m80_逐时段范围/kW": [float(m_slot.min()), float(m_slot.max())],
        "m80_逐小时范围/kW": [float(m_hour.min()), float(m_hour.max())],
        "m80_逐月范围/kW": [float(np.nanmin(m_month)), float(np.nanmax(m_month))],
        "rho_ref_比例式": rho_ref,
        "日度偏差AR1系数": rho_net,
        "日度偏差半衰期/天": hl_net,
        "白天偏差AR1系数": rho_day,
        "白天偏差半衰期/天": hl_day,
        "光伏d−1权重AR1系数": rho_u1,
        "光伏权重半衰期/天": hl_u1,
        **risk,
    }
    (pm.outputs_dir() / "q2e_structure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图：四联 ----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax = axes[0, 0]
    ax.hist(x_day, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(m_global_day, color="#C44E52", ls="--",
               label=f"白天 q80 = {m_global_day:.0f} kW")
    ax.set_xlabel("净需求偏差 e / kW")
    ax.set_ylabel("频数")
    ax.set_title("(a) 偏差分布（白天）", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    tt = np.arange(q2.T) / 6.0
    ax.plot(tt, m_slot, color="#4C72B0", lw=1.0, label="逐时段 q80")
    ax.step(np.arange(1, 25), m_hour, where="mid", color="#C44E52",
            lw=1.4, label="逐小时 q80")
    ax.axhline(m_global_all, color="#55A868", ls=":", lw=1.2,
               label=f"全局 q80 = {m_global_all:.0f} kW")
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("裕度参考 / kW")
    ax.set_title("(b) 分位数裕度曲线", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    lags = np.arange(31)
    ax.bar(lags, acf_net, color="#4C72B0", alpha=0.85, label="日度净需求偏差")
    ax.plot(lags, acf_u1, color="#C44E52", marker="o", ms=2.5, lw=1.0,
            label="光伏 d−1 权重")
    if np.isfinite(hl_net):
        ax.axvline(hl_net, color="#55A868", ls="--", lw=1.2,
                   label=f"半衰期 ≈ {hl_net:.0f} 天")
    ax.set_xlabel("滞后 / 天")
    ax.set_ylabel("自相关")
    ax.set_title("(c) 时间记忆", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    hours = np.arange(24)
    hmean = np.array([e_net[:, 6 * h:6 * (h + 1)].mean() for h in range(24)])
    hstd = np.array([e_net[:, 6 * h:6 * (h + 1)].std() for h in range(24)])
    ax.plot(hours + 0.5, hmean, color="#4C72B0", marker="o", ms=2.5, label="均值")
    ax.fill_between(hours + 0.5, hmean - hstd, hmean + hstd,
                    color="#4C72B0", alpha=0.2, label="±1σ")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("时刻 / h")
    ax.set_ylabel("净需求偏差 / kW")
    ax.set_title("(d) 日内偏差结构", fontsize=9)
    ax.legend(fontsize=8)

    fig.tight_layout()
    pm.save_fig(fig, "Q2E_误差结构与参考值", data=pd.DataFrame({
        "槽序号": np.arange(1, q2.T + 1),
        "m80_逐时段_kW": m_slot,
        "m80_逐小时_kW": np.repeat(m_hour, 6),
    }))

    # ---- 报告 ----
    record(
        "问题二 口径E 误差分布结构与参数参考值",
        {
            "样本数（全时段/白天）": f"{shape['样本数（全时段）']}/{shape['样本数（白天）']}",
            "偏差均值/kW（白天）": round(shape["均值/kW（白天）"], 1),
            "偏差标准差/kW（白天）": round(shape["标准差/kW（白天）"], 1),
            "偏差偏度（白天）": round(shape["偏度（白天）"], 3),
            "q0.8 裕度/kW（全时段）": round(m_global_all, 0),
            "q0.8 裕度/kW（白天）": round(m_global_day, 0),
            "逐时段 q80 范围/kW": (
                f"{summary['m80_逐时段范围/kW'][0]:.0f} ~ "
                f"{summary['m80_逐时段范围/kW'][1]:.0f}"),
            "逐小时 q80 范围/kW": (
                f"{summary['m80_逐小时范围/kW'][0]:.0f} ~ "
                f"{summary['m80_逐小时范围/kW'][1]:.0f}"),
            "逐月 q80 范围/kW": (
                f"{summary['m80_逐月范围/kW'][0]:.0f} ~ "
                f"{summary['m80_逐月范围/kW'][1]:.0f}"),
            "比例式参考 rho_ref": round(rho_ref, 4),
            "日度偏差 AR(1)": round(rho_net, 3),
            "日度偏差半衰期/天": round(hl_net, 1),
            "白天偏差半衰期/天": round(hl_day, 1),
            "光伏 d−1 权重半衰期/天": round(hl_u1, 1),
            "日费用均值/元": round(risk["日费用均值/元"], 1),
            "日费用 P95/元": round(p95, 1),
            "日费用 CVaR95/元": round(risk["日费用CVaR95/元"], 1),
            "尾部溢价（P95−均值）/均值": round(risk["尾部溢价(P95−均值)/均值"], 3),
            "有紧急购电天数占比": round(risk["有紧急购电天数占比"], 3),
        },
        note=(
            "对官方口径 E（W=7 网格权重）在 2025-02-01~12-31 的预测误差做分布结构分析："
            "偏差 e = 实际净需求 − 预测净需求。报童临界比 1−1/5=0.8 对应的最优裕度为 e 的 "
            "0.8 分位数；分时段/分小时/分月/比例式参考见 code/outputs/q2e_structure_*.csv；"
            "时间记忆用于 W 与 EWMA 半衰期参考；日费用尾部用于风险厌恶系数 λ 扫描。"
            "图 figures/Q2E_误差结构与参考值.pdf。"
        ),
    )
    log.info("结构分析完成（{:.0f}s）：m80_全={:.0f} m80_白天={:.0f} ρ_net={:.3f} "
             "半衰期={:.1f}天", time.time() - t0, m_global_all, m_global_day,
             rho_net, hl_net)
    return summary


if __name__ == "__main__":
    main()
