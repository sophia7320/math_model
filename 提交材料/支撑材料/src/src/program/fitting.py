"""曲线拟合、插值与平滑。

- :func:`fit_curve`：任意函数最小二乘拟合（curve_fit 包装，带 R²/标准误）
- :func:`polyfit` / :func:`poly_scan`：多项式拟合与自动定阶
- :func:`interp` / :func:`spline_smooth`：插值与平滑样条
- :func:`smooth`：Savitzky-Golay / 滑动平均 / 指数平滑

用法::

    import numpy as np
    from program import fitting

    def growth(x, a, b, c):
        return a / (1 + np.exp(-b * (x - c)))     # Logistic

    r = fitting.fit_curve(growth, t, y, p0=[10, 0.5, 5], names=["K", "r", "t0"])
    print(r.summary())
    y_hat = r.predict(t_new)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .utils import fmt


# ---------------------------------------------------------------------------
# 拟合结果容器
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    """拟合结果：``predict`` 求预测值，``summary`` 打印摘要。"""

    func: Callable[..., np.ndarray]
    params: np.ndarray
    stderr: np.ndarray | None
    r2: float
    rmse: float
    n: int
    names: list[str]
    kind: str = ""
    x: Any = None
    y: Any = None

    def predict(self, x: Sequence[float]) -> np.ndarray:
        """用拟合参数计算预测值。"""
        return self.func(np.asarray(x, dtype=float), *self.params)

    @property
    def residuals(self) -> np.ndarray:
        if self.x is None or self.y is None:
            raise ValueError("未保存原始数据，无法计算残差")
        return np.asarray(self.y, dtype=float) - self.predict(self.x)

    def param_dict(self) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, self.params)}

    def summary(self) -> str:
        lines = [
            f"拟合结果{'（' + self.kind + '）' if self.kind else ''}："
            f"R² = {fmt(self.r2, 5)}，RMSE = {fmt(self.rmse, 5)}，n = {self.n}"
        ]
        for i, (name, v) in enumerate(zip(self.names, self.params)):
            if self.stderr is not None and i < len(self.stderr) and np.isfinite(self.stderr[i]):
                lines.append(f"  {name} = {fmt(v, 6)} ± {fmt(self.stderr[i], 3)}")
            else:
                lines.append(f"  {name} = {fmt(v, 6)}")
        return "\n".join(lines)

    __str__ = summary


def _metrics(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return r2, float(np.sqrt(np.mean((y - y_hat) ** 2)))


# ---------------------------------------------------------------------------
# 通用曲线拟合
# ---------------------------------------------------------------------------
def fit_curve(
    func: Callable[..., Any],
    x: Sequence[float],
    y: Sequence[float],
    p0: Sequence[float] | None = None,
    *,
    bounds: tuple = (-np.inf, np.inf),
    names: Sequence[str] | None = None,
    sigma: Sequence[float] | None = None,
    maxfev: int = 20000,
    kind: str = "",
) -> FitResult:
    """非线性最小二乘拟合 ``y ≈ func(x, *params)``。

    Parameters
    ----------
    func : callable
        ``f(x, a, b, ...) -> y``，第一个参数是自变量。
    x, y : 序列
        观测数据。
    p0 : 序列 | None
        参数初值（建议给出，避免陷入局部最优）。
    bounds : tuple
        ``(下界, 上界)``，元素为标量或与参数等长的序列。
    names : 序列 | None
        参数名（用于摘要），默认 ``p0, p1, ...``。

    Returns
    -------
    FitResult
        ``r.predict(x_new)`` / ``r.summary()`` / ``r.param_dict()``。
    """
    from scipy.optimize import curve_fit

    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    popt, pcov = curve_fit(
        func, xa, ya, p0=None if p0 is None else np.asarray(p0, dtype=float),
        sigma=sigma, bounds=bounds, maxfev=maxfev,
    )
    diag = np.clip(np.diag(pcov), 0, None) if pcov is not None else np.full_like(popt, np.nan)
    perr = np.sqrt(diag)
    if names is None:
        names = [f"p{i}" for i in range(len(popt))]
    r2, rmse = _metrics(ya, func(xa, *popt))
    return FitResult(
        func=func, params=np.asarray(popt), stderr=np.asarray(perr),
        r2=r2, rmse=rmse, n=len(xa), names=list(names),
        kind=kind or getattr(func, "__name__", ""), x=xa, y=ya,
    )


# ---------------------------------------------------------------------------
# 多项式
# ---------------------------------------------------------------------------
def _poly_func(x: np.ndarray, *coef: float) -> np.ndarray:
    return np.polyval(coef, x)


def polyfit(
    x: Sequence[float],
    y: Sequence[float],
    degree: int = 1,
) -> FitResult:
    """多项式拟合（含系数标准误，``params`` 按降幂排列）。"""
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    coef, cov = np.polyfit(xa, ya, degree, cov=True)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    r2, rmse = _metrics(ya, np.polyval(coef, xa))
    names = [f"a{i}" for i in range(degree + 1)]
    return FitResult(
        func=_poly_func, params=np.asarray(coef), stderr=np.asarray(perr),
        r2=r2, rmse=rmse, n=len(xa), names=names, kind=f"多项式(deg={degree})",
        x=xa, y=ya,
    )


def poly_str(coef: Sequence[float], var: str = "x", nd: int = 4) -> str:
    """多项式系数（降幂）→ 可读表达式，如 ``0.5x^{2} - 1.2x + 3``。"""
    coef = np.asarray(coef, dtype=float)
    deg = len(coef) - 1
    terms: list[tuple[str, str]] = []
    for i, c in enumerate(coef):
        power = deg - i
        if abs(c) < 1e-12:
            continue
        sign = "-" if c < 0 else "+"
        mag = fmt(abs(c), nd)
        if power == 0:
            body = mag
        elif power == 1:
            body = f"{mag}{var}"
        else:
            body = f"{mag}{var}^{{{power}}}"
        terms.append((sign, body))
    if not terms:
        return "0"
    s = terms[0][1] if terms[0][0] == "+" else "-" + terms[0][1]
    for sign, body in terms[1:]:
        s += f" {sign} {body}"
    return s


def poly_scan(
    x: Sequence[float],
    y: Sequence[float],
    max_degree: int = 5,
) -> "pd.DataFrame":
    """扫描 1~max_degree 阶多项式，返回对比表（用于自动定阶）。

    列：``degree, r2, adj_r2, rmse, aic`` —— R² 提升有限时选更简单的阶数。
    """
    import pandas as pd

    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    n = len(xa)
    rows = []
    for deg in range(1, max_degree + 1):
        coef = np.polyfit(xa, ya, deg)
        yhat = np.polyval(coef, xa)
        r2, rmse = _metrics(ya, yhat)
        sse = float(np.sum((ya - yhat) ** 2))
        k = deg + 1
        adj = 1 - (1 - r2) * (n - 1) / max(n - k - 1, 1)
        aic = n * np.log(max(sse / n, 1e-300)) + 2 * k
        rows.append({"degree": deg, "r2": r2, "adj_r2": adj, "rmse": rmse, "aic": aic})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 插值与平滑
# ---------------------------------------------------------------------------
def interp(
    x: Sequence[float],
    y: Sequence[float],
    x_new: Sequence[float],
    kind: str = "cubic",
) -> np.ndarray:
    """一维插值：``kind ∈ {"linear", "cubic", "pchip", "nearest"}``。

    曲线光滑推荐 ``"cubic"``；防震荡（数据陡变）推荐 ``"pchip"``。
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    xb = np.asarray(x_new, dtype=float).ravel()
    if kind == "linear":
        return np.interp(xb, xa, ya)
    if kind == "cubic":
        from scipy.interpolate import CubicSpline

        return CubicSpline(xa, ya)(xb)
    if kind == "pchip":
        from scipy.interpolate import PchipInterpolator

        return PchipInterpolator(xa, ya)(xb)
    if kind == "nearest":
        idx = np.abs(xb[:, None] - xa[None, :]).argmin(axis=1)
        return ya[idx]
    raise ValueError(f"不支持的插值类型：{kind!r}")


def spline_smooth(
    x: Sequence[float],
    y: Sequence[float],
    s: float | None = None,
    k: int = 3,
) -> Callable[[np.ndarray], np.ndarray]:
    """平滑样条（UnivariateSpline），返回可调用对象。

    ``s`` 越大越光滑（``None`` 自动按残差平方和目标选择）。
    """
    from scipy.interpolate import UnivariateSpline

    return UnivariateSpline(np.asarray(x, float).ravel(), np.asarray(y, float).ravel(), s=s, k=k)


def smooth(
    y: Sequence[float],
    window: int = 7,
    *,
    method: str = "savgol",
    polyorder: int = 2,
) -> np.ndarray:
    """一维数据平滑。

    - ``"savgol"``：Savitzky-Golay 滤波（默认，保峰形，实验数据首选）
    - ``"moving"``：居中滑动平均
    - ``"ewma"``：指数加权平均
    """
    import pandas as pd

    ya = np.asarray(y, dtype=float).ravel()
    window = int(window)
    if method == "savgol":
        from scipy.signal import savgol_filter

        w = window if window % 2 == 1 else window + 1
        w = min(w, len(ya) if len(ya) % 2 == 1 else len(ya) - 1)
        return savgol_filter(ya, w, min(polyorder, max(w - 2, 1) if w > 2 else 1))
    if method == "moving":
        return pd.Series(ya).rolling(window, center=True, min_periods=1).mean().to_numpy()
    if method == "ewma":
        return pd.Series(ya).ewm(span=window, adjust=False).mean().to_numpy()
    raise ValueError(f"未知平滑方法：{method!r}")
