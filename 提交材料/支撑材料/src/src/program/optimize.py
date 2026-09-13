"""优化求解：线性/整数规划、非线性规划、全局优化与经典离散问题。

- :func:`solve_lp`：线性规划 / 混合整数线性规划（HiGHS，支持最大化）
- :func:`solve_nlp`：非线性规划（scipy.minimize 包装）
- :func:`global_minimize`：全局优化（差分进化）
- :func:`knapsack` / :func:`assignment` / :func:`tsp_nearest_neighbor` + :func:`tsp_2opt`

优化类赛题注意：先保证「可行解」，再优化目标值；结果用
:func:`program.report.record_result` 记录目标值与约束校验。

用法::

    import program as pm

    r = pm.optimize.solve_lp(
        c=[3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18],
        maximize=True,
    )
    print(r.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .utils import fmt


# ---------------------------------------------------------------------------
# 结果容器
# ---------------------------------------------------------------------------
@dataclass
class OptResult:
    """优化结果（``str(r)`` 直接看结论）。"""

    x: np.ndarray
    fun: float
    success: bool
    message: str
    solver: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        xs = np.asarray(self.x).ravel()
        x_str = ", ".join(f"{v:.6g}" for v in xs[:10]) + (" …" if len(xs) > 10 else "")
        status = "成功" if self.success else "失败"
        msg = (
            f"【{self.solver or '优化'}】{status}：目标值 = {fmt(self.fun, 8)}"
            f"\n  最优解 x = [{x_str}]"
        )
        if not self.success and self.message:
            msg += f"\n  信息：{self.message}"
        return msg

    __str__ = summary


# ---------------------------------------------------------------------------
# 线性 / 整数规划
# ---------------------------------------------------------------------------
def solve_lp(
    c: Sequence[float],
    A_ub: Any = None,
    b_ub: Any = None,
    A_eq: Any = None,
    b_eq: Any = None,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    *,
    maximize: bool = False,
    integer: bool | Sequence[int] | None = None,
    options: dict | None = None,
) -> OptResult:
    """线性规划（``integer=None``）或混合整数线性规划（HiGHS）。

    Parameters
    ----------
    c : 序列
        目标函数系数。
    A_ub, b_ub, A_eq, b_eq : 数组 | None
        不等式约束 ``A_ub x ≤ b_ub`` 与等式约束 ``A_eq x = b_eq``。
    bounds : 序列[(下界, 上界)] | None
        变量边界，``None`` 表示 ±∞；整体默认为 ``(0, +∞)``（非负）。
    maximize : bool
        是否最大化（默认最小化）。
    integer : bool | 序列 | None
        ``None``/``False`` 连续；``True`` 全部整数；序列则逐变量指定
        （1 = 整数，0 = 连续，可用于 0-1 变量配合 bounds=(0,1)）。

    Returns
    -------
    OptResult
    """
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp

    c_arr = np.asarray(c, dtype=float).ravel()
    n = len(c_arr)
    sign = -1.0 if maximize else 1.0

    if integer is None or integer is False:
        res = linprog(
            sign * c_arr, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs", options=options or {},
        )
        x = res.x if res.x is not None else np.full(n, np.nan)
        return OptResult(
            x=np.asarray(x), fun=float(sign * res.fun) if res.fun is not None else float("nan"),
            success=bool(res.success), message=str(res.message),
            solver="HiGHS 线性规划", extra={"maximize": maximize},
        )

    # 整数规划
    integrality = (
        np.ones(n, dtype=int) if integer is True
        else np.asarray(integer, dtype=int)
    )
    if bounds is None:
        bl = Bounds(np.zeros(n), np.full(n, np.inf))
    else:
        lb = np.array([b[0] if b[0] is not None else -np.inf for b in bounds], dtype=float)
        ub = np.array([b[1] if b[1] is not None else np.inf for b in bounds], dtype=float)
        bl = Bounds(lb, ub)
    cons = []
    if A_ub is not None:
        cons.append(
            LinearConstraint(np.asarray(A_ub, dtype=float), -np.inf, np.asarray(b_ub, dtype=float))
        )
    if A_eq is not None:
        b_eq_arr = np.asarray(b_eq, dtype=float)
        cons.append(LinearConstraint(np.asarray(A_eq, dtype=float), b_eq_arr, b_eq_arr))
    res = milp(
        sign * c_arr, constraints=cons if cons else (),
        integrality=integrality, bounds=bl, options=options or {},
    )
    x = res.x if res.x is not None else np.full(n, np.nan)
    return OptResult(
        x=np.asarray(x), fun=float(sign * res.fun) if res.fun is not None else float("nan"),
        success=bool(res.success), message=str(res.message),
        solver="HiGHS 整数规划", extra={"maximize": maximize, "n_integer": int(integrality.sum())},
    )


# ---------------------------------------------------------------------------
# 非线性与全局优化
# ---------------------------------------------------------------------------
def solve_nlp(
    func: Callable[[np.ndarray], float],
    x0: Sequence[float],
    *,
    bounds: Sequence[tuple[float | None, float | None]] | None = None,
    constraints: Any = None,
    maximize: bool = False,
    method: str | None = None,
    options: dict | None = None,
) -> OptResult:
    """非线性规划（``scipy.optimize.minimize`` 包装，支持最大化与约束）。

    Parameters
    ----------
    func : callable
        ``f(x) -> float``。
    x0 : 序列
        初值（建议多试几个初值以判断局部/全局）。
    bounds : 序列[(下界, 上界)] | None
        变量边界。
    constraints : dict | 序列 | None
        scipy 约束格式，如
        ``{"type": "ineq", "fun": lambda x: x[0] + x[1] - 1}``。
    maximize : bool
        是否最大化。
    """
    from scipy.optimize import minimize

    sign = -1.0 if maximize else 1.0

    def f(x: np.ndarray) -> float:
        return sign * float(func(x))

    res = minimize(
        f, np.asarray(x0, dtype=float), method=method,
        bounds=bounds, constraints=constraints, options=options or {},
    )
    return OptResult(
        x=np.asarray(res.x), fun=float(sign * res.fun), success=bool(res.success),
        message=str(res.message), solver=f"minimize/{method or '默认方法'}",
        extra={"maximize": maximize, "iterations": getattr(res, "nit", None)},
    )


def global_minimize(
    func: Callable[[np.ndarray], float],
    bounds: Sequence[tuple[float, float]],
    *,
    maximize: bool = False,
    seed: int = 42,
    **de_kwargs: Any,
) -> OptResult:
    """全局优化（差分进化，适合多峰/非凸目标，结果可复现）。

    ``de_kwargs`` 透传 ``scipy.optimize.differential_evolution``，
    如 ``maxiter=200, popsize=30, tol=1e-8``。
    """
    from scipy.optimize import differential_evolution

    sign = -1.0 if maximize else 1.0
    res = differential_evolution(
        lambda x: sign * float(func(x)), bounds, seed=seed, polish=True, **de_kwargs
    )
    return OptResult(
        x=np.asarray(res.x), fun=float(sign * res.fun), success=bool(res.success),
        message=str(res.message), solver="differential_evolution",
        extra={"maximize": maximize, "nfev": int(res.nfev)},
    )


# ---------------------------------------------------------------------------
# 经典离散问题
# ---------------------------------------------------------------------------
@dataclass
class KnapsackResult:
    """0-1 背包结果。"""

    best_value: float
    items: list[int]
    total_weight: float

    def summary(self) -> str:
        return (
            f"【0-1 背包】最优价值 = {fmt(self.best_value)}，"
            f"总重量 = {fmt(self.total_weight)}，选中物品（0 起） = {self.items}"
        )

    __str__ = summary


def knapsack(
    values: Sequence[float],
    weights: Sequence[float],
    capacity: float,
) -> KnapsackResult:
    """0-1 背包（动态规划，O(n·C)）。

    要求 ``weights`` 与 ``capacity`` 为整数（非整数请先统一缩放为整数，如乘 1000）。
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights).ravel()
    if not np.allclose(w, np.round(w)) or not float(capacity).is_integer():
        raise ValueError("weights 与 capacity 必须为整数（可先统一放大为整数）")
    w = w.astype(int)
    W = int(capacity)
    n = len(v)
    dp = np.zeros(W + 1)
    keep = np.zeros((n, W + 1), dtype=bool)
    for i in range(n):
        for c in range(W, w[i] - 1, -1):
            cand = dp[c - w[i]] + v[i]
            if cand > dp[c]:
                dp[c] = cand
                keep[i, c] = True
    c = W
    items: list[int] = []
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            items.append(i)
            c -= w[i]
    items.reverse()
    return KnapsackResult(
        best_value=float(dp[W]), items=items,
        total_weight=float(w[items].sum()) if items else 0.0,
    )


@dataclass
class Assignment:
    """指派问题结果。"""

    rows: np.ndarray
    cols: np.ndarray
    total: float

    def pairs(self) -> list[tuple[int, int]]:
        return [(int(r), int(c)) for r, c in zip(self.rows, self.cols)]

    def summary(self) -> str:
        return f"【指派问题】最小总成本 = {fmt(self.total)}，匹配 = {self.pairs()}"

    __str__ = summary


def assignment(cost: Any, *, maximize: bool = False) -> Assignment:
    """指派问题（匈牙利算法）。

    ``cost[i, j]`` 表示第 i 个工人做第 j 项任务的成本（最大化时传收益矩阵）。
    """
    from scipy.optimize import linear_sum_assignment

    c = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(c, maximize=maximize)
    return Assignment(rows=rows, cols=cols, total=float(c[rows, cols].sum()))


def distance_matrix(points: Any, metric: str = "euclidean") -> np.ndarray:
    """点坐标 → 距离矩阵（``scipy.spatial.distance.cdist``）。"""
    from scipy.spatial.distance import cdist

    return cdist(np.asarray(points, dtype=float), np.asarray(points, dtype=float), metric=metric)


def _tour_length(dist: np.ndarray, tour: Sequence[int]) -> float:
    t = np.asarray(tour, dtype=int)
    return float(dist[t, np.roll(t, -1)].sum())


def tsp_nearest_neighbor(dist: np.ndarray, start: int = 0) -> list[int]:
    """TSP 最近邻构造启发式，返回访问顺序（不含回到起点）。"""
    n = dist.shape[0]
    unvisited = set(range(n))
    unvisited.discard(start)
    tour = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur, j])
        tour.append(nxt)
        unvisited.discard(nxt)
        cur = nxt
    return tour


def tsp_2opt(
    dist: np.ndarray,
    tour: Sequence[int] | None = None,
    *,
    max_iter: int = 5000,
) -> tuple[list[int], float]:
    """TSP 2-opt 局部搜索改进，返回 ``(访问顺序, 总路程)``。

    ``tour=None`` 时先用最近邻构造再改进。确定性算法，可复现。

    用法::

        d = pm.optimize.distance_matrix(points)
        tour, length = pm.optimize.tsp_2opt(d)
    """
    dist = np.asarray(dist, dtype=float)
    t = list(tour) if tour is not None else tsp_nearest_neighbor(dist)
    n = len(t)
    improved = True
    it = 0
    while improved and it < max_iter:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = t[i], t[i + 1]
                c, d = t[j], t[(j + 1) % n]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                    improved = True
                    it += 1
                    if it >= max_iter:
                        break
            if it >= max_iter:
                break
    return t, _tour_length(dist, t)
