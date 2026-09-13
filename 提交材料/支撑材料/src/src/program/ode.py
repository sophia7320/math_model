"""常微分方程：初值问题求解与参数反演（传染病、动力学、种群模型）。

``func`` 统一使用 ``solve_ivp`` 的接口 ``f(t, y, *params)``（注意不是
``odeint`` 的 ``func(y, t)``），如 SIR 模型::

    def sir(t, y, beta, gamma):
        S, I, R = y
        return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

    sol = pm.ode.solve_ivp_model(sir, [0.99, 0.01, 0], (0, 60), params=(0.3, 0.1), t_eval=t)

用法（参数反演）::

    fit = pm.ode.fit_ode_params(sir, [0.99, 0.01, 0], t_data, y_data, p0=[0.2, 0.1],
                                bounds=([0.001, 0.001], [2, 2]), param_names=["β", "γ"])
    print(fit.summary())
    y_sim = fit.simulate(t_grid)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import fmt


def solve_ivp_model(
    func: Callable[..., Any],
    y0: Sequence[float],
    t_span: tuple[float, float],
    *,
    params: Sequence[float] = (),
    t_eval: Sequence[float] | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    **kwargs: Any,
):
    """求解常微分方程组初值问题，返回 ``scipy.integrate.OdeResult``。

    ``func(t, y, *params)``；默认提高精度（rtol=1e-6, atol=1e-9）。
    结果用 ``sol.t``（时间）与 ``sol.y``（形状 ``(变量数, 时间点数)``）访问。
    """
    from scipy.integrate import solve_ivp

    y0a = np.atleast_1d(np.asarray(y0, dtype=float))
    f = lambda t, y: func(t, y, *params)
    return solve_ivp(
        f, t_span, y0a, t_eval=t_eval, method=method, rtol=rtol, atol=atol, **kwargs
    )


@dataclass
class ODEFitResult:
    """ODE 参数反演结果。"""

    params: np.ndarray
    param_names: list[str]
    cost: float
    rmse: float
    r2: float
    success: bool
    func: Callable[..., Any]
    y0: np.ndarray
    t_span: tuple[float, float]
    t_data: np.ndarray
    y_data: np.ndarray
    method: str = "RK45"

    def simulate(self, t_eval: Sequence[float] | None = None) -> np.ndarray:
        """用拟合参数模拟，返回 ``(时间点数, 变量数)`` 数组。"""
        t = (
            np.asarray(t_eval, dtype=float).ravel()
            if t_eval is not None
            else self.t_data
        )
        sol = solve_ivp_model(
            self.func,
            self.y0,
            self.t_span,
            params=self.params,
            t_eval=t,
            method=self.method,
        )
        return sol.y.T

    predict = simulate

    def summary(self) -> str:
        lines = [
            f"ODE 参数反演{'成功' if self.success else '失败'}："
            f"RMSE = {fmt(self.rmse, 5)}，R² = {fmt(self.r2, 5)}"
        ]
        for name, v in zip(self.param_names, self.params):
            lines.append(f"  {name} = {fmt(v, 6)}")
        return "\n".join(lines)

    __str__ = summary


def fit_ode_params(
    func: Callable[..., Any],
    y0: Sequence[float],
    t_data: Sequence[float],
    y_data: Sequence[float],
    p0: Sequence[float],
    *,
    bounds: tuple = (-np.inf, np.inf),
    param_names: Sequence[str] | None = None,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    verbose: bool = False,
) -> ODEFitResult:
    """用最小二乘反演 ODE 参数（``scipy.optimize.least_squares``）。

    Parameters
    ----------
    func : callable
        ``f(t, y, *params)``。
    y0 : 序列
        初值（已知且固定；如需一起拟合请自行扩展参数向量）。
    t_data, y_data : 序列
        观测时间点（需单调递增）与观测值；``y_data`` 形状 ``(n_t,)``
        或 ``(n_t, n_vars)``。
    p0 : 序列
        参数初值。
    bounds : tuple
        ``(下界, 上界)``；强烈建议给出，避免参数跑到无物理意义区域。
    """
    from scipy.optimize import least_squares

    t = np.asarray(t_data, dtype=float).ravel()
    Yobs = np.asarray(y_data, dtype=float)
    if Yobs.ndim == 1:
        Yobs = Yobs[:, None]
    y0a = np.atleast_1d(np.asarray(y0, dtype=float))
    t_span = (float(t.min()), float(t.max()))

    def residual(p: np.ndarray) -> np.ndarray:
        try:
            sol = solve_ivp_model(
                func,
                y0a,
                t_span,
                params=p,
                t_eval=t,
                method=method,
                rtol=rtol,
                atol=atol,
            )
            if not sol.success or sol.y.shape[1] != len(t):
                return np.full(Yobs.size, 1e6)
            return (sol.y.T - Yobs).ravel()
        except Exception:
            return np.full(Yobs.size, 1e6)

    res = least_squares(
        residual,
        np.asarray(p0, dtype=float),
        bounds=bounds,
        verbose=2 if verbose else 0,
    )
    resid = res.fun
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((Yobs - Yobs.mean()) ** 2))
    r2 = 1 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    if param_names is None:
        param_names = [f"p{i}" for i in range(len(res.x))]
    return ODEFitResult(
        params=np.asarray(res.x),
        param_names=list(param_names),
        cost=float(res.cost),
        rmse=rmse,
        r2=r2,
        success=bool(res.success),
        func=func,
        y0=y0a,
        t_span=t_span,
        t_data=t,
        y_data=Yobs,
        method=method,
    )
