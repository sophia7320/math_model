# 优化求解 optimize

优化类赛题的常用武器库：从线性规划到 TSP 路径规划，不用再翻 scipy 文档。

## 线性 / 整数规划

**生产计划**经典写法（`maximize=True` 时直接最大化利润，不用手写负号）：

```python
import program as pm

r = pm.optimize.solve_lp(
    c=[3, 5],                                  # 目标系数（两种产品利润）
    A_ub=[[1, 0], [0, 2], [3, 2]],             # 不等式约束 A x ≤ b
    b_ub=[4, 12, 18],
    bounds=[(0, None), (0, None)],             # 变量上下界
    maximize=True,
)
print(r.summary())
# 【HiGHS 线性规划】成功：目标值 = 36
#   最优解 x = [2, 6]

# 0-1 选址 / 整数产量：integer 参数控制
r2 = pm.optimize.solve_lp(
    c=[5, 4, 3],
    A_ub=[[2, 3, 1]], b_ub=[10],
    bounds=[(0, 1)] * 3,
    maximize=True,
    integer=[1, 1, 1],                         # 全整数；也可 [1,0,1] 混合
)
```

## 非线性 / 全局优化

```python
# 非线性规划（可带约束、可最大化）
r = pm.optimize.solve_nlp(
    lambda v: (v[0] - 1) ** 2 + (v[1] - 2) ** 2,
    x0=[0, 0],
    bounds=[(-5, 5), (-5, 5)],
)

# 多峰目标：差分进化全局优化（固定种子，结果可复现）
r = pm.optimize.global_minimize(lambda v: np.sin(3 * v[0]) + (v[0] - 0.5) ** 2,
                                bounds=[(0, 3)], seed=42)
```

## 经典离散问题

```python
# 0-1 背包（整数权重，非整数先统一放大）
kp = pm.optimize.knapsack([60, 100, 120], [10, 20, 30], capacity=50)
print(kp)   # 最优价值 = 220，选中物品 = [1, 2]

# 指派问题（成本矩阵，自动求最小成本匹配）
asg = pm.optimize.assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
print(asg.pairs())    # [(0, 1), (1, 0), (2, 2)]

# TSP：坐标 → 距离矩阵 → 最近邻构造 → 2-opt 改进
d = pm.optimize.distance_matrix(city_points)
tour, length = pm.optimize.tsp_2opt(d)        # 确定性算法，可复现
```

## 函数速查

| 函数 | 说明 |
| --- | --- |
| `solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, maximize, integer)` | LP / MILP（HiGHS）；`integer=True/[1,0,...]` |
| `solve_nlp(func, x0, bounds, constraints, maximize)` | 非线性规划 |
| `global_minimize(func, bounds, seed=42)` | 差分进化全局优化 |
| `knapsack(values, weights, capacity)` | 0-1 背包（DP） |
| `assignment(cost)` | 指派问题（匈牙利算法） |
| `distance_matrix(points)` | 坐标 → 距离矩阵 |
| `tsp_nearest_neighbor(dist)` / `tsp_2opt(dist)` | TSP 构造 + 改进，返回 `(tour, length)` |

## 小贴士

!!! warning "默认是最小化、默认非负"
    - `solve_lp` / `solve_nlp` / `global_minimize` 都默认**最小化**，最大化要传 `maximize=True`。
    - 变量边界默认 `(0, +∞)`；需要负值必须显式写 `bounds=[(None, None), ...]`。

!!! tip "优化类论文的证据链"
    1. 先给**可行解**（哪怕不是最优），再给优化结果；
    2. 用 `r.success` / `r.message` 核验求解状态；
    3. 对关键参数做灵敏度分析（改动 ±10% 看目标值变化），
       图用 `pm.line`，数值用 `pm.record_result` 记录。

!!! tip "约束校验模板"
    ```python
    import numpy as np
    lhs = np.asarray(A_ub) @ r.x
    assert np.all(lhs <= np.asarray(b_ub) + 1e-6), "约束被违反！"
    pm.record_result("约束校验", {"最大违反量": float(np.max(lhs - b_ub))})
    ```
