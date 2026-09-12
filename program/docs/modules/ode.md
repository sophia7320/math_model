# 微分方程 ode

传染病（SIR/SEIR）、种群增长、人口预测、物理动力系统……这类题目的两步走：
**正问题**（已知参数求演化）和**反问题**（已知数据反演参数）。

约定：模型函数写作 `f(t, y, *params)`（`solve_ivp` 风格，**不是** `odeint` 的 `f(y, t)`）。

## 正问题：求解 + 出图

```python
import numpy as np
import program as pm

def sir(t, y, beta, gamma):
    S, I, R = y
    return [-beta * S * I, beta * S * I - gamma * I, gamma * I]

t = np.linspace(0, 60, 121)
sol = pm.ode.solve_ivp_model(
    sir, y0=[0.99, 0.01, 0.0], t_span=(0, 60),
    params=(0.35, 0.12), t_eval=t,
)
# sol.y 形状为 (变量数, 时间点数)
pm.line(t, [sol.y[0], sol.y[1], sol.y[2]], labels=["S", "I", "R"],
        xlabel="时间/d", ylabel="人群占比", save="q1_sir")
```

## 反问题：用观测数据反演参数

```python
fit = pm.ode.fit_ode_params(
    sir, y0=[0.99, 0.01, 0.0],
    t_data=t_obs, y_data=I_obs,          # 观测数据（时间需单调递增）
    p0=[0.3, 0.1],                       # 参数初值
    bounds=([0.01, 0.01], [2.0, 1.0]),   # 强烈建议给物理边界
    param_names=["β", "γ"],
)
print(fit.summary())
# ODE 参数反演成功：RMSE = 0.0031，R² = 0.9978
#   β = 0.351 ± ...
#   γ = 0.119 ± ...

I_sim = fit.simulate(t)                  # 用最优参数重放，画拟合对比
pm.line(t, [I_obs, I_sim[:, 1]], labels=["观测", "模型拟合"],
        xlabel="时间/d", ylabel="感染比例", save="q1_fit")
```

## 函数速查

| 函数 | 说明 |
| --- | --- |
| `solve_ivp_model(func, y0, t_span, params=(), t_eval=..., method="RK45")` | 求解初值问题，返回 `OdeResult`（`sol.t` / `sol.y`） |
| `fit_ode_params(func, y0, t_data, y_data, p0, bounds=..., param_names=...)` | 最小二乘参数反演，返回 `ODEFitResult` |
| `ODEFitResult.summary()` / `.simulate(t)` | 打印摘要 / 用最优参数模拟 |

## 小贴士

!!! tip "精度设置"
    `solve_ivp_model` 默认 `rtol=1e-6, atol=1e-9`，比 scipy 原版更精细，
    适合参数反演（反演对数值误差敏感）。

!!! warning "三类常见错误"
    1. **函数签名写反**：必须是 `f(t, y, ...)`；报 `too many positional arguments`
       时先检查这一点。
    2. **t_eval 超出 t_span**：会报错，`t_span` 要完整包住观测时间。
    3. **参数跑到物理无意义区域**：给 `bounds`，例如 `β, γ > 0`。

!!! tip "刚度问题"
    如果求解很慢或结果爆炸，换 `method="LSODA"` 或 `"BDF"`（自动处理刚性方程）。
