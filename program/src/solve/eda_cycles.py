"""C 题附件数据周期结构 EDA：日周期与周周期可视化。

数据：附件2（小区负载、光伏发电实际功率，365×144）、附件4（实时电价，365×144）、
附件1（典型日 144 槽；实测等于附件2/4 的全年逐槽均值，作为日周期基准线）。

输出：
- figures/数据_日周期.pdf、figures/数据_周周期.pdf
  作图数据：code/outputs/figure_data/数据_日周期.csv、数据_周周期.csv
- code/outputs/eda_cycles_stats.json（日内/周内解释方差、ACF、星期效应检验）
- reports/RESULTS_REPORT.md 章节「附件数据周期结构（EDA）」

运行（program/ 下）：uv run python -m solve.eda_cycles
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

import program as pm
from solve.common import DATA_C, ROOT, T

HOUR = T // 24          # 每小时 6 槽
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
BLUE_TEAL_YELLOW_RED = ["#2166ac", "#1fa8c9", "#77d7c8", "#fff3a5", "#fdae61", "#d73027"]
SERIES = {              # 显示名 -> (附件数组键, 单位, 颜色)
    "小区负载": ("load", "kW", "#2166ac"),
    "光伏出力": ("pv", "kW", "#1fa8c9"),
    "实时电价": ("price", "元/kWh", "#d73027"),
}


def _acf(x: np.ndarray, lag: int) -> float:
    """去均值序列滞后 lag 槽的自相关。"""
    y = x - x.mean()
    return float((y[:-lag] * y[lag:]).sum() / (y * y).sum())


def _acf_curve(x: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    lags = np.arange(1, max_lag + 1)
    return lags, np.array([_acf(x, int(k)) for k in lags])


def _weekday_r2(day_mean: np.ndarray, weekday: np.ndarray) -> float:
    """星期效应（各星期几均值）对天均值的解释方差比例。"""
    eff = np.array([day_mean[weekday == w].mean() for w in weekday])
    return float(1 - np.var(day_mean - eff) / np.var(day_mean))


def record(section: str, data, note: str = "") -> None:
    """写结果报告：先删同名旧章节再追加（pm.record_result 为追加模式）。"""
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


def main() -> None:
    t0 = time.perf_counter()
    pm.init(seed=42, root=str(ROOT))

    load = pm.read_table(DATA_C / "附件2.xlsx", sheet="小区负载")
    pv = pm.read_table(DATA_C / "附件2.xlsx", sheet="光伏发电实际功率")
    price = pm.read_table(DATA_C / "附件4.xlsx")
    a1 = pm.read_table(DATA_C / "附件1.xlsx")

    dates = pd.to_datetime(load.iloc[:, 0])
    weekday = dates.dt.weekday.to_numpy()
    data = {
        "load": load.iloc[:, 1:145].to_numpy(float),
        "pv": pv.iloc[:, 1:145].to_numpy(float),
        "price": price.iloc[:, 1:145].to_numpy(float),
    }
    a1_ref = {
        "load": a1["小区负载"].to_numpy(float),
        "pv": a1["光伏发电预测功率"].to_numpy(float),
        "price": a1["电价"].to_numpy(float),
    }

    t_h = (np.arange(T) + 0.5) * 10 / 60          # 槽中心时刻 / h
    hour_mean: dict[str, np.ndarray] = {}          # 日内均值曲线（144 槽）
    wd_slot: dict[str, np.ndarray] = {}            # 星期几×槽均值（7×144）
    slot_stats: dict[str, dict] = {}

    for name, (key, unit, _c) in SERIES.items():
        arr = data[key]
        m = arr.mean(axis=0)
        hour_mean[key] = m
        wd_slot[key] = np.stack([arr[weekday == w].mean(axis=0) for w in range(7)])
        day_mean = arr.mean(axis=1)
        r2_intra = 1 - np.var(arr - m) / np.var(arr)
        r2_wd = _weekday_r2(day_mean, weekday)
        kr = stats.kruskal(*[day_mean[weekday == w] for w in range(7)])
        wknd = np.isin(weekday, (4, 5))            # 数据中周五+周六为低谷
        diff_a1 = float(np.abs(a1_ref[key] - m).max())
        slot_stats[name] = {
            "单位": unit,
            "日内形状R2": round(float(r2_intra), 3),
            "周内R2_天均值": round(r2_wd, 3),
            "ACF_lag1天": round(_acf(arr.ravel(), 144), 3),
            "ACF_lag7天": round(_acf(arr.ravel(), 1008), 3),
            "周五六天均值": round(float(day_mean[wknd].mean()), 2),
            "其余天均值": round(float(day_mean[~wknd].mean()), 2),
            "周五六相对偏离_pct": round(float(100 * (day_mean[wknd].mean() / day_mean.mean() - 1)), 1),
            "Kruskal_p": float(kr.pvalue),
            "附件1与年均最大差": diff_a1,
            "峰值时刻_h": float(t_h[int(np.argmax(m))]),
        }
        print(f"{name}: 日内R2={slot_stats[name]['日内形状R2']:.3f} "
              f"周内R2={slot_stats[name]['周内R2_天均值']:.3f} "
              f"ACF(1d/7d)={slot_stats[name]['ACF_lag1天']:.3f}/{slot_stats[name]['ACF_lag7天']:.3f} "
              f"周五六偏离={slot_stats[name]['周五六相对偏离_pct']:+.1f}% "
              f"Kruskal p={kr.pvalue:.2e}")

    # ---- 图 1：日周期（均值 + P25–P75 带 + 附件1 典型日）----
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (name, (key, unit, color)) in zip(axes, SERIES.items()):
        arr = data[key]
        p25, p75 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
        ax.fill_between(t_h, p25, p75, color=color, alpha=0.22, lw=0,
                        label="P25–P75 带")
        ax.plot(t_h, hour_mean[key], color=color, lw=1.5, label="全年均值")
        ax.plot(t_h, a1_ref[key], color="k", lw=0.8, ls="--", label="附件1 典型日")
        ax.axvline(slot_stats[name]["峰值时刻_h"], color="gray", ls=":", lw=0.7)
        ax.set_xlabel("时刻 / h")
        ax.set_ylabel(f"{name} / {unit}")
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 6))
        ax.set_title(name, fontsize=9)
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    day_df = pd.DataFrame({
        "时刻_h": np.round(t_h, 3),
        **{f"{name}_{stat}": vals for name, (key, _u, _c) in SERIES.items()
           for stat, vals in (("均值", hour_mean[key]),
                              ("P25", np.percentile(data[key], 25, axis=0)),
                              ("P75", np.percentile(data[key], 75, axis=0)),
                              ("附件1", a1_ref[key]))},
    })
    pm.save_fig(fig, "数据_日周期", data=day_df)

    # ---- 图 2：周周期（星期几×时刻热力图，槽级数据 + 双线性平滑；色带=蓝青→黄红）----
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue_teal_yellow_red",
                                             BLUE_TEAL_YELLOW_RED)
    fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.0))
    for ax, (name, (key, unit, _c)) in zip(axes2, SERIES.items()):
        grid = wd_slot[key]
        im = ax.imshow(grid, aspect="auto", origin="upper", cmap=cmap,
                       interpolation="bilinear")
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(WEEKDAYS, fontsize=7)
        for lbl in ax.get_yticklabels():
            if lbl.get_text() in ("周五", "周六"):
                lbl.set_color("#C44E52")
        ax.set_xticks(np.arange(0, T + 1, 6 * HOUR))
        ax.set_xticklabels(np.arange(0, 25, 6), fontsize=7)
        ax.set_xlabel("时刻 / h")
        ax.set_title(f"{name} / {unit}", fontsize=9)
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig2.tight_layout()
    week_df = pd.DataFrame({
        "星期": np.repeat(WEEKDAYS, T),
        "时刻_h": np.round(np.tile(t_h, 7), 3),
    })
    for name, (key, _u, _c) in SERIES.items():
        week_df[name] = wd_slot[key].ravel().round(4)
    pm.save_fig(fig2, "数据_周周期", data=week_df)

    # ---- 统计落盘 + 报告章节 ----
    stats_out = {name: slot_stats[name] for name in SERIES}
    pm.save_outputs(stats_out, "eda_cycles_stats")

    flat = {
        f"{name}·{k}": v for name, s in stats_out.items() for k, v in s.items()
    }
    record(
        "附件数据周期结构（EDA）",
        flat,
        note=("数据：附件2（负荷/光伏实际，365×144）、附件4（电价，365×144）；"
              "统计含义：日内形状R2=144 槽均值对总方差的解释比例，周内R2=星期几均值对天均值的解释比例，"
              "ACF 为槽级去均值自相关。结论：负荷与电价具『日内+周内』双重周期，"
              "周内低谷为周五+周六（非通常意义的周六周日），光伏仅日内周期、无周内效应；"
              "附件1 三序列=附件2/4 的全年逐槽均值（最大差见上表），可作典型日基准。"
              "图 figures/数据_日周期.pdf、figures/数据_周周期.pdf；"
              "报告章节由 solve.eda_cycles 重跑时自动覆盖。"),
    )

    dt = time.perf_counter() - t0
    print(f"\n完成：figures/数据_日周期.pdf、figures/数据_周周期.pdf、"
          f"code/outputs/eda_cycles_stats.json（{dt:.1f} s）")


if __name__ == "__main__":
    main()
