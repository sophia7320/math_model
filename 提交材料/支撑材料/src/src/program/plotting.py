"""论文级绘图工具：中文字体、PDF 矢量输出、常用图型一套搞定。

约定
----
- 所有函数返回 ``(fig, ax)``（或自定义说明），并可选 ``save="图名"`` 参数：
  传入即按 ``figures/图名.pdf`` 保存，同时把作图数据存到
  ``code/outputs/figure_data/图名.csv``（满足 RESULTS_REPORT 数据可追溯要求）。
- 不在图内写大标题：标题交给论文 caption。
- 默认输出矢量 PDF（300 dpi），也支持 ``save_fig(..., formats=("pdf","png"))``。

用法::

    import program as pm

    pm.init()  # 必须：设置中文字体与论文风格
    pm.line(x, [y1, y2], labels=["预测值", "真实值"], xlabel="时间/s", ylabel="温度/℃", save="q1_temp")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib.pyplot as plt

from .config import figure_data_dir, figures_dir

_FONT_READY = False


def _bootstrap_font() -> None:
    """未调用 ``config.init()`` 时的兜底中文字体设置（只做一次）。"""
    global _FONT_READY
    if _FONT_READY:
        return
    _FONT_READY = True
    from .config import find_cjk_font

    name = find_cjk_font()
    if name:
        current = list(plt.rcParams.get("font.sans-serif", []))
        if name not in current:
            plt.rcParams["font.sans-serif"] = [name] + [c for c in current if c != "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def save_fig(
    fig,
    name: str | Path,
    *,
    outdir: str | Path | None = None,
    formats: tuple[str, ...] = ("pdf",),
    dpi: int | None = None,
    close: bool = False,
    data: Any = None,
    transparent: bool = False,
    tight: bool = True,
) -> list[Path]:
    """保存图形为论文可用的矢量文件，并可选保存作图数据。

    Parameters
    ----------
    fig : Figure
        matplotlib 图对象。
    name : str | Path
        文件名（可带子目录或扩展名）。带扩展名时 ``formats`` 被忽略。
    outdir : 路径 | None
        输出目录，默认 ``figures/``。
    formats : tuple[str, ...]
        输出格式，默认只输出 ``pdf``；预览可加 ``"png"``。
    data : DataFrame | dict | ndarray | None
        作图数据，保存到 ``code/outputs/figure_data/<name>.csv``（或 json）。

    Returns
    -------
    list[Path]
        实际写出的文件路径列表。
    """
    _bootstrap_font()
    out = Path(outdir) if outdir is not None else figures_dir()
    name_p = Path(str(name))
    stem = name_p.stem if name_p.suffix else name_p.name
    base = out / name_p.parent / stem
    fmts = [name_p.suffix.lstrip(".").lower()] if name_p.suffix else list(formats)

    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass

    paths: list[Path] = []
    for f in fmts:
        p = base.with_suffix("." + f)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, transparent=transparent)
        paths.append(p)

    if data is not None:
        import pandas as pd

        ddir = figure_data_dir()
        rel = base.relative_to(out)
        ddir = ddir / rel.parent
        ddir.mkdir(parents=True, exist_ok=True)
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame()
            if isinstance(data, pd.DataFrame):
                data.to_csv(ddir / f"{stem}.csv", index=False, encoding="utf-8-sig")
            elif isinstance(data, dict):
                try:
                    pd.DataFrame(data).to_csv(
                        ddir / f"{stem}.csv", index=False, encoding="utf-8-sig"
                    )
                except ValueError:  # 不等长序列
                    from .dataio import save_json

                    save_json(data, ddir / f"{stem}.json")
            else:
                arr = np.asarray(data)
                if arr.ndim == 2:
                    pd.DataFrame(arr).to_csv(
                        ddir / f"{stem}.csv", index=False, encoding="utf-8-sig"
                    )
                else:
                    pd.DataFrame({stem: arr}).to_csv(
                        ddir / f"{stem}.csv", index=False, encoding="utf-8-sig"
                    )
        except Exception as err:  # 数据记录失败不阻断绘图
            print(f"[save_fig] 作图数据保存失败：{err}")

    if close:
        plt.close(fig)
    return paths


def _get_ax(ax, figsize=None):
    _bootstrap_font()
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _maybe_save(save, fig, data=None):
    if save is None or save is False:
        return
    save_fig(fig, save, data=data, close=False)


def _as_series_list(ys: Any) -> list[np.ndarray]:
    """把多种输入的 y 统一成「一组曲线」：list[ndarray]。"""
    if isinstance(ys, np.ndarray) and ys.ndim == 2:
        return [ys[:, j] for j in range(ys.shape[1])]
    if isinstance(ys, (list, tuple)) and len(ys) > 0 and hasattr(ys[0], "__len__"):
        return [np.asarray(y) for y in ys]
    return [np.asarray(ys)]


# ---------------------------------------------------------------------------
# 常用图型
# ---------------------------------------------------------------------------
def line(
    x: Sequence,
    ys: Any,
    *,
    labels: Sequence[str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    marker: str | None = None,
    linestyle: str = "-",
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    legend: bool = True,
    **plot_kw: Any,
):
    """折线图（支持多条曲线）。

    ``ys`` 可以是 1D 序列、2D 数组（每列一条曲线）或一组序列；
    ``labels`` 给每条曲线命名。返回 ``(fig, ax)``。
    """
    fig, ax = _get_ax(ax, figsize)
    series = _as_series_list(ys)
    for i, y in enumerate(series):
        lab = labels[i] if labels and i < len(labels) else None
        ax.plot(x, y, marker=marker, linestyle=linestyle, label=lab, **plot_kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    has_label = labels is not None or any(
        not ln.get_label().startswith("_") for ln in ax.get_lines()
    )
    if legend and has_label:
        ax.legend()
    data: dict[str, Any] = {"x": np.asarray(x).ravel()}
    multi = len(series) > 1
    for i, y in enumerate(series):
        data[f"y{i + 1}" if multi else "y"] = np.asarray(y).ravel()
    _maybe_save(save, fig, data=data)
    return fig, ax


def scatter(
    x: Sequence,
    y: Sequence,
    *,
    xlabel: str = "",
    ylabel: str = "",
    label: str | None = None,
    s: float | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """散点图。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    ax.scatter(x, y, label=label, s=s, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    _maybe_save(save, fig, data={"x": x, "y": y})
    return fig, ax


def scatter_fit(
    x: Sequence,
    y: Sequence,
    *,
    degree: int = 1,
    xlabel: str = "",
    ylabel: str = "",
    show_eq: bool = True,
    color: str | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """散点 + 多项式拟合线（含 R² / 表达式标注）。

    返回 ``(fig, ax, info)``，``info = {"coef", "r2", "rmse"}``。
    """
    from .fitting import poly_str

    fig, ax = _get_ax(ax, figsize)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    ax.scatter(xa, ya, **kw)
    coef = np.polyfit(xa, ya, degree)
    xx = np.linspace(xa.min(), xa.max(), 200)
    yy = np.polyval(coef, xx)
    yhat = np.polyval(coef, xa)
    ss_res = float(np.sum((ya - yhat) ** 2))
    ss_tot = float(np.sum((ya - ya.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((ya - yhat) ** 2)))
    line_label = f"拟合曲线 R² = {r2:.4f}"
    ax.plot(xx, yy, color=color, label=line_label)
    if show_eq and degree <= 3:
        ax.text(
            0.03, 0.97, f"$y = {poly_str(coef)}$\n$R^2 = {r2:.4f}$",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _maybe_save(save, fig, data={"x": xa, "y": ya})
    return fig, ax, {"coef": coef, "r2": r2, "rmse": rmse}


def bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    errors: Sequence[float] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    orient: str = "v",
    color: str | None = None,
    value_labels: bool = False,
    rot: float = 0,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """柱状图（``orient="h"`` 为条形图）。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    pos = np.arange(len(labels))
    if orient == "h":
        ax.barh(pos, values, xerr=errors, color=color, capsize=3, **kw)
        ax.set_yticks(pos, labels)
        ax.set_xlabel(xlabel or ylabel)
        if value_labels:
            for i, v in enumerate(values):
                ax.text(v, i, f" {v:.3g}", va="center")
    else:
        ax.bar(pos, values, yerr=errors, color=color, capsize=3, **kw)
        ax.set_xticks(pos, labels)
        if rot:
            plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot > 0 else "center")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if value_labels:
            for i, v in enumerate(values):
                ax.text(i, v, f"{v:.3g}", ha="center", va="bottom")
    _maybe_save(save, fig, data={"类别": list(labels), "数值": list(values)})
    return fig, ax


def bar_group(
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    xlabel: str = "",
    ylabel: str = "",
    rot: float = 0,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """分组柱状图。``series = {"方法A": [...], "方法B": [...]}``。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    x = np.arange(len(labels))
    n = len(series)
    width = 0.8 / max(n, 1)
    for i, (name, vals) in enumerate(series.items()):
        offset = (i - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.92, label=name, **kw)
    ax.set_xticks(x, labels)
    if rot:
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right" if rot > 0 else "center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    _maybe_save(save, fig, data={"类别": list(labels), **{k: list(v) for k, v in series.items()}})
    return fig, ax


def hist(
    x: Sequence[float],
    *,
    bins: int = 30,
    density: bool = True,
    kde: bool = True,
    xlabel: str = "",
    ylabel: str = "",
    label: str | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """直方图 + 可选 KDE 密度曲线（默认密度归一）。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    xa = np.asarray(x, dtype=float)
    ylab = ylabel or ("概率密度" if density else "频数")
    ax.hist(xa, bins=bins, density=density, alpha=0.75, label=label, **kw)
    if kde and len(xa) >= 5:
        from scipy.stats import gaussian_kde

        kde_obj = gaussian_kde(xa)
        grid = np.linspace(xa.min(), xa.max(), 200)
        ax.plot(grid, kde_obj(grid), color="crimson", lw=1.8, label="KDE 密度")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylab)
    if label or (kde and len(xa) >= 5):
        ax.legend()
    _maybe_save(save, fig, data={"值": xa})
    return fig, ax


def box(
    data: Mapping[str, Sequence[float]] | np.ndarray,
    *,
    xlabel: str = "",
    ylabel: str = "",
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """箱线图。``data`` 为 ``{组名: 数据}`` 或 2D 数组（每列一组）。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    if isinstance(data, Mapping):
        groups = {str(k): np.asarray(v, dtype=float) for k, v in data.items()}
    else:
        arr = np.asarray(data, dtype=float)
        arr = arr[:, None] if arr.ndim == 1 else arr
        groups = {f"组{j + 1}": arr[:, j] for j in range(arr.shape[1])}
    ax.boxplot(
        list(groups.values()),
        tick_labels=list(groups.keys()),
        patch_artist=True,
        **kw,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _maybe_save(save, fig, data=groups)
    return fig, ax


def heatmap(
    matrix: Any,
    *,
    xticklabels: Sequence[str] | None = None,
    yticklabels: Sequence[str] | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rot: float = 45,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """热力图。``matrix`` 支持 DataFrame（自动取行列名）或 ndarray。返回 ``(fig, ax)``。"""
    import pandas as pd

    fig, ax = _get_ax(ax, figsize)
    if isinstance(matrix, pd.DataFrame):
        xticklabels = xticklabels or [str(c) for c in matrix.columns]
        yticklabels = yticklabels or [str(i) for i in matrix.index]
        arr = matrix.to_numpy(dtype=float)
    else:
        arr = np.asarray(matrix, dtype=float)
    im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    n_r, n_c = arr.shape
    ax.set_xticks(range(n_c), xticklabels if xticklabels is not None else range(n_c))
    ax.set_yticks(range(n_r), yticklabels if yticklabels is not None else range(n_r))
    if rot:
        plt.setp(ax.get_xticklabels(), rotation=rot, ha="right")
    if annot and n_r * n_c <= 200:
        finite = arr[np.isfinite(arr)]
        vmax_abs = max(float(np.abs(finite).max()), 1e-12) if finite.size else 1.0
        for i in range(n_r):
            for j in range(n_c):
                val = arr[i, j]
                txt = format(val, fmt) if np.isfinite(val) else "—"
                shade = "white" if np.isfinite(val) and abs(val) > 0.6 * vmax_abs else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=shade)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    if cbar_label:
        cb.set_label(cbar_label)
    ax.grid(False)
    _maybe_save(
        save, fig,
        data=pd.DataFrame(arr, index=yticklabels, columns=xticklabels),
    )
    return fig, ax


def corr_heatmap(
    df,
    *,
    method: str = "pearson",
    annot: bool = True,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """相关系数热力图（-1~1 固定色阶，适合论文）。返回 ``(fig, ax)``。"""
    from .stats import corr_matrix

    corr = corr_matrix(df, method=method)
    fig, ax = heatmap(
        corr, annot=annot, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, cbar_label="相关系数",
        save=save, figsize=figsize, ax=ax,
    )
    return fig, ax


def radar(
    labels: Sequence[str],
    series: Mapping[str, Sequence[float]],
    *,
    normalize: bool = False,
    fill: bool = True,
    ylim: tuple[float, float] | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
):
    """雷达图（综合评价/多方案对比）。``series = {"方案A": [...], ...}``。

    ``normalize=True`` 时按指标（列）做 min-max 归一化到 [0, 1]。
    返回 ``(fig, ax)``。
    """
    _bootstrap_font()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="polar")
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_c = angles + angles[:1]

    mat = np.array([list(v) for v in series.values()], dtype=float)
    if normalize:
        lo = mat.min(axis=0)
        hi = mat.max(axis=0)
        rng = np.where(hi - lo == 0, 1, hi - lo)
        mat = (mat - lo) / rng

    for (name, _), vals in zip(series.items(), mat):
        vals_c = np.concatenate([vals, vals[:1]])
        ax.plot(angles_c, vals_c, label=str(name))
        if fill:
            ax.fill(angles_c, vals_c, alpha=0.10)
    ax.set_xticks(angles, [str(l) for l in labels])
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    _maybe_save(save, fig, data={"指标": list(labels), **{k: list(v) for k, v in series.items()}})
    return fig, ax


def dual_axis(
    x: Sequence,
    y1: Sequence,
    y2: Sequence,
    *,
    labels: tuple[str, str] = ("左轴", "右轴"),
    xlabel: str = "",
    ylabel1: str = "",
    ylabel2: str = "",
    colors: tuple[str, str] = ("#4C72B0", "#C44E52"),
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """双纵轴折线图（量纲不同的两条曲线）。返回 ``(fig, ax, ax2)``。"""
    fig, ax = _get_ax(ax, figsize)
    ax2 = ax.twinx()
    ax.plot(x, y1, color=colors[0], label=labels[0])
    ax2.plot(x, y2, color=colors[1], label=labels[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel1 or labels[0], color=colors[0])
    ax2.set_ylabel(ylabel2 or labels[1], color=colors[1])
    ax.tick_params(axis="y", colors=colors[0])
    ax2.tick_params(axis="y", colors=colors[1])
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="best")
    ax.grid(True, alpha=0.3)
    _maybe_save(save, fig, data={"x": list(x), labels[0]: list(y1), labels[1]: list(y2)})
    return fig, ax, ax2


def pie(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    startangle: float = 90,
    autopct: str | None = "%1.1f%%",
    explode: Sequence[float] | None = None,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """饼图（占比结构）。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    ax.pie(values, labels=labels, startangle=startangle, autopct=autopct, explode=explode)
    ax.axis("equal")
    _maybe_save(save, fig, data={"类别": list(labels), "数值": list(values)})
    return fig, ax


def errorbar(
    x: Sequence,
    y: Sequence,
    yerr: Sequence | None = None,
    *,
    xlabel: str = "",
    ylabel: str = "",
    label: str | None = None,
    fmt: str = "o-",
    capsize: float = 3,
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
    **kw: Any,
):
    """带误差棒的折线/散点图。返回 ``(fig, ax)``。"""
    fig, ax = _get_ax(ax, figsize)
    ax.errorbar(x, y, yerr=yerr, fmt=fmt, capsize=capsize, label=label, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend()
    data = {"x": list(x), "y": list(y)}
    if yerr is not None:
        data["yerr"] = list(np.asarray(yerr).ravel())
    _maybe_save(save, fig, data=data)
    return fig, ax


def surface3d(
    X: Any,
    Y: Any,
    Z: Any,
    *,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    cmap: str = "viridis",
    elev: float = 30,
    azim: float = -60,
    cbar: bool = True,
    save: str | Path | None = None,
    figsize: tuple[float, float] = (7.5, 5.5),
):
    """三维曲面图（参数敏感性、二维优化景观、调参曲面）。返回 ``(fig, ax)``。"""
    _bootstrap_font()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=elev, azim=azim)
    if cbar:
        fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=zlabel)
    _maybe_save(
        save, fig,
        data={"X": np.asarray(X).ravel(), "Y": np.asarray(Y).ravel(), "Z": np.asarray(Z).ravel()},
    )
    return fig, ax


def roc(
    y_true: Sequence,
    y_score: Sequence[float],
    *,
    xlabel: str = "假阳性率 FPR",
    ylabel: str = "真阳性率 TPR",
    save: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ax=None,
):
    """二分类 ROC 曲线（含 AUC 标注）。返回 ``(fig, ax, auc_value)``。"""
    from sklearn.metrics import auc as sk_auc
    from sklearn.metrics import roc_curve

    fig, ax = _get_ax(ax, figsize)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = float(sk_auc(fpr, tpr))
    ax.plot(fpr, tpr, lw=1.8, label=f"ROC（AUC = {auc_val:.4f}）")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right")
    _maybe_save(save, fig, data={"FPR": fpr, "TPR": tpr})
    return fig, ax, auc_val
