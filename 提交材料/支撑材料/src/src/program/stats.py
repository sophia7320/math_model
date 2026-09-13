"""统计检验与回归分析（统一的 TestResult 输出，含中文结论）。

覆盖数学建模高频统计需求：正态性、t 检验、方差分析、卡方、相关性、
一元/多元线性回归、置信区间与 bootstrap。

用法::

    import program as pm

    r = pm.stats.compare_groups(group_a, group_b, group_c)   # 自动选 ANOVA / Kruskal
    print(r)                                                  # 直接可读的中文结论
    pm.stats.ols(df, y="销量", xs=["价格", "广告费"]).summary()  # 多元回归摘要
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from .utils import fmt


# ---------------------------------------------------------------------------
# 统一检验结果
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    """假设检验结果（``str(r)`` 直接给出中文结论）。"""

    name: str
    statistic: float
    pvalue: float
    h0: str = ""
    alpha: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        """p < alpha 时为 True（拒绝 H0）。"""
        return bool(self.pvalue < self.alpha)

    def __str__(self) -> str:
        sign = "<" if self.significant else "≥"
        verdict = "拒绝 H0" if self.significant else "不拒绝 H0"
        msg = (
            f"【{self.name}】统计量 = {fmt(self.statistic)}，"
            f"p = {fmt(self.pvalue)} {sign} α = {self.alpha} → {verdict}"
        )
        if self.h0:
            msg += f"（H0：{self.h0}）"
        if self.extra:
            extras = "，".join(f"{k} = {fmt(v)}" for k, v in self.extra.items())
            msg += f"\n附：{extras}"
        return msg

    def to_dict(self) -> dict[str, Any]:
        d = {"检验": self.name, "统计量": self.statistic, "p值": self.pvalue,
             "结论": "拒绝 H0" if self.significant else "不拒绝 H0"}
        d.update(self.extra)
        return d


def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    s2 = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / max(n1 + n2 - 2, 1)
    return float((a.mean() - b.mean()) / np.sqrt(s2)) if s2 > 0 else float("nan")


# ---------------------------------------------------------------------------
# 分布与均值检验
# ---------------------------------------------------------------------------
def normality(x: Sequence[float], alpha: float = 0.05) -> TestResult:
    """正态性检验：n ≤ 5000 用 Shapiro-Wilk，否则用 D'Agostino K²。"""
    a = _clean(x)
    n = len(a)
    if n < 8:
        raise ValueError(f"正态性检验至少需要 8 个有效样本，当前 {n}")
    if n <= 5000:
        stat, p = sps.shapiro(a)
        method = "Shapiro-Wilk"
    else:
        stat, p = sps.normaltest(a)
        method = "D'Agostino K²"
    return TestResult(
        f"正态性检验（{method}）", float(stat), float(p),
        h0="数据服从正态分布", alpha=alpha, extra={"n": n},
    )


def t_test_one(x: Sequence[float], mu: float = 0, alpha: float = 0.05) -> TestResult:
    """单样本 t 检验（样本均值是否等于 ``mu``）。"""
    a = _clean(x)
    stat, p = sps.ttest_1samp(a, mu)
    return TestResult(
        "单样本 t 检验", float(stat), float(p),
        h0=f"总体均值等于 {mu}", alpha=alpha,
        extra={"样本均值": a.mean(), "样本量": len(a)},
    )


def t_test_ind(
    a: Sequence[float],
    b: Sequence[float],
    *,
    equal_var: bool = False,
    alpha: float = 0.05,
) -> TestResult:
    """独立双样本 t 检验；默认 Welch（不假设方差齐性）。"""
    x, y = _clean(a), _clean(b)
    stat, p = sps.ttest_ind(x, y, equal_var=equal_var)
    return TestResult(
        "独立双样本 t 检验（Welch）" if not equal_var else "独立双样本 t 检验",
        float(stat), float(p),
        h0="两组均值相等", alpha=alpha,
        extra={"均值A": x.mean(), "均值B": y.mean(),
               "nA": len(x), "nB": len(y), "Cohen's d": _cohens_d(x, y)},
    )


def t_test_paired(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> TestResult:
    """配对样本 t 检验。"""
    x, y = _clean(a), _clean(b)
    if len(x) != len(y):
        raise ValueError("配对检验要求两组长度相同")
    stat, p = sps.ttest_rel(x, y)
    return TestResult(
        "配对样本 t 检验", float(stat), float(p),
        h0="差值的均值为 0", alpha=alpha,
        extra={"均值差": (x - y).mean(), "n": len(x)},
    )


def f_test_var(a: Sequence[float], b: Sequence[float], alpha: float = 0.05) -> TestResult:
    """方差齐性 F 检验（两样本）。"""
    x, y = _clean(a), _clean(b)
    stat = x.var(ddof=1) / y.var(ddof=1)
    df1, df2 = len(x) - 1, len(y) - 1
    p = 2 * min(sps.f.cdf(stat, df1, df2), 1 - sps.f.cdf(stat, df1, df2))
    return TestResult(
        "方差齐性 F 检验", float(stat), float(p),
        h0="两组方差相等", alpha=alpha,
        extra={"方差A": x.var(ddof=1), "方差B": y.var(ddof=1)},
    )


# ---------------------------------------------------------------------------
# 多样本比较
# ---------------------------------------------------------------------------
def anova(*groups: Sequence[float], alpha: float = 0.05) -> TestResult:
    """单因素方差分析（ANOVA，H0：各组均值相等）。"""
    gs = [_clean(g) for g in groups]
    if len(gs) < 2:
        raise ValueError("ANOVA 至少需要两组")
    stat, p = sps.f_oneway(*gs)
    return TestResult(
        "单因素方差分析 ANOVA", float(stat), float(p),
        h0="各组均值相等", alpha=alpha,
        extra={f"组{i + 1}均值": g.mean() for i, g in enumerate(gs)},
    )


def kruskal(*groups: Sequence[float], alpha: float = 0.05) -> TestResult:
    """Kruskal-Wallis 非参数检验（不满足正态/方差齐性时使用）。"""
    gs = [_clean(g) for g in groups]
    stat, p = sps.kruskal(*gs)
    return TestResult(
        "Kruskal-Wallis 非参数检验", float(stat), float(p),
        h0="各组分布相同", alpha=alpha,
    )


def compare_groups(*groups: Sequence[float], alpha: float = 0.05) -> TestResult:
    """多组均值差异比较（自动选择方法）。

    先检验每组正态性（Shapiro）与方差齐性（Levene）：
    全部通过 → ANOVA；否则 → Kruskal-Wallis 非参数检验。
    """
    gs = [_clean(g) for g in groups]
    if len(gs) < 2:
        raise ValueError("至少需要两组数据")
    normal_ok = all(len(g) >= 8 and sps.shapiro(g).pvalue > alpha for g in gs)
    var_ok = sps.levene(*gs).pvalue > alpha
    if normal_ok and var_ok:
        return anova(*gs, alpha=alpha)
    r = kruskal(*gs, alpha=alpha)
    r.name += "（自动选择：正态性/方差齐性未通过）"
    return r


def chi2_test(table: Any, *, correction: bool = False, alpha: float = 0.05) -> TestResult:
    """列联表卡方独立性检验。

    Parameters
    ----------
    table : array-like | DataFrame
        列联表（行 × 列的观测频数）。
    """
    arr = np.asarray(table, dtype=float)
    chi2, p, dof, expected = sps.chi2_contingency(arr, correction=correction)
    n = arr.sum()
    r, c = arr.shape
    cramers_v = float(np.sqrt(chi2 / (n * min(r - 1, c - 1)))) if min(r - 1, c - 1) > 0 else float("nan")
    return TestResult(
        "卡方独立性检验", float(chi2), float(p),
        h0="行变量与列变量独立", alpha=alpha,
        extra={"自由度": dof, "Cramér's V": cramers_v,
               "最小期望频数": float(expected.min())},
    )


# ---------------------------------------------------------------------------
# 相关性与回归
# ---------------------------------------------------------------------------
def corr_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    method: str = "pearson",
    alpha: float = 0.05,
) -> TestResult:
    """两变量相关性检验（pearson / spearman / kendall）。"""
    xa, ya = _clean(x), _clean(y)
    n = min(len(xa), len(ya))
    if method == "pearson":
        stat, p = sps.pearsonr(xa[:n], ya[:n])
    elif method == "spearman":
        stat, p = sps.spearmanr(xa[:n], ya[:n])
    elif method == "kendall":
        stat, p = sps.kendalltau(xa[:n], ya[:n])
    else:
        raise ValueError(f"未知相关方法：{method!r}")
    return TestResult(
        f"{method} 相关检验", float(stat), float(p),
        h0="两变量不相关", alpha=alpha, extra={"n": n},
    )


def corr_matrix(df: pd.DataFrame, *, method: str = "pearson") -> pd.DataFrame:
    """数值列相关系数矩阵（pearson / spearman / kendall）。"""
    num = df.select_dtypes(include="number")
    return num.corr(method=method)


@dataclass
class LinReg:
    """一元线性回归结果：``y = slope * x + intercept``。"""

    slope: float
    intercept: float
    r: float
    r2: float
    pvalue: float
    stderr: float
    n: int

    def predict(self, x: Sequence[float]) -> np.ndarray:
        return self.slope * np.asarray(x, dtype=float) + self.intercept

    def __str__(self) -> str:
        return (
            f"y = {fmt(self.slope, 5)} x + {fmt(self.intercept, 5)}"
            f"（R² = {fmt(self.r2, 5)}, p = {fmt(self.pvalue)}, n = {self.n}）"
        )


def linregress_summary(x: Sequence[float], y: Sequence[float]) -> LinReg:
    """一元线性回归摘要（含 R²、斜率标准误、p 值）。"""
    xa, ya = _clean(x), _clean(y)
    n = min(len(xa), len(ya))
    res = sps.linregress(xa[:n], ya[:n])
    return LinReg(
        slope=float(res.slope), intercept=float(res.intercept),
        r=float(res.rvalue), r2=float(res.rvalue) ** 2,
        pvalue=float(res.pvalue), stderr=float(res.stderr), n=n,
    )


def ols(df: pd.DataFrame, y: str, xs: Sequence[str], *, add_constant: bool = True):
    """多元线性回归（statsmodels OLS），返回结果对象。

    打印 ``.summary()`` 查看完整统计信息；用 :func:`ols_text` 直接拿字符串。
    """
    import statsmodels.api as sm

    X = df[list(xs)].astype(float)
    if add_constant:
        X = sm.add_constant(X)
    model = sm.OLS(df[y].astype(float), X).fit()
    return model


def ols_text(df: pd.DataFrame, y: str, xs: Sequence[str], *, add_constant: bool = True) -> str:
    """多元线性回归摘要（纯文本，可直接贴进报告/论文附录）。"""
    return str(ols(df, y, xs, add_constant=add_constant).summary())


# ---------------------------------------------------------------------------
# 区间估计
# ---------------------------------------------------------------------------
def ci_mean(x: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    """总体均值的 t 置信区间，返回 ``(下界, 上界)``。"""
    a = _clean(x)
    n = len(a)
    se = a.std(ddof=1) / np.sqrt(n)
    h = se * sps.t.ppf(1 - alpha / 2, n - 1)
    return float(a.mean() - h), float(a.mean() + h)


@dataclass
class BootstrapCI:
    """Bootstrap 置信区间结果。"""

    stat: float
    lo: float
    hi: float
    se: float
    alpha: float
    n_boot: int

    def __str__(self) -> str:
        lv = f"{1 - self.alpha:.0%}"
        return (
            f"Bootstrap {lv}% CI = [{fmt(self.lo)}, {fmt(self.hi)}]，"
            f"点估计 = {fmt(self.stat)}，标准误 = {fmt(self.se)}（B = {self.n_boot}）"
        )


def bootstrap_ci(
    x: Sequence[float],
    stat_fn=np.mean,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap 置信区间（对任意统计量 ``stat_fn``）。

    Parameters
    ----------
    x : 序列
        样本数据。
    stat_fn : callable
        统计量函数，默认均值。
    """
    a = _clean(x)
    rng = np.random.default_rng(seed)
    stats_boot = np.array(
        [stat_fn(rng.choice(a, size=len(a), replace=True)) for _ in range(n_boot)]
    )
    lo, hi = np.percentile(stats_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapCI(
        stat=float(stat_fn(a)), lo=float(lo), hi=float(hi),
        se=float(stats_boot.std(ddof=1)), alpha=alpha, n_boot=n_boot,
    )
