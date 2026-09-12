# 快速开始

## 1. 前置条件

工具包已经安装在 `program/` 项目的 uv 环境中，**不需要任何额外安装**。
Runner 固定用 `uv run`（不要用裸 `python` / `pip`）。

```powershell
# 在 program/ 目录下确认一切正常
uv run python tests/smoke_test.py     # 75 项冒烟检查，全绿即就绪
```

## 2. 运行你的脚本

按工作区约定，赛题代码放在 `program/src/program/` 下（如 `problem1.py`）。
两种运行方式：

=== "方式 A（推荐）"

    在**赛题工作区根目录**执行：

    ```powershell
    uv run --project program python program/src/program/problem1.py
    ```

    此时当前目录 = 工作区根，输出自动落在：

    ```
    工作区根/
    ├── figures/            # 图（PDF）
    ├── reports/            # RESULTS_REPORT.md
    └── code/outputs/       # 中间数据、图表数据、日志
    ```

=== "方式 B"

    在 `program/` 目录执行：

    ```powershell
    uv run python src/program/problem1.py
    ```

    输出落在 `program/` 下。若希望仍输出到工作区根，在脚本里写：

    ```python
    pm.init(root="..")     # 或设置环境变量 MATHMODEL_ROOT
    ```

!!! note "为什么强调运行目录？"
    所有输出路径 = **项目根**（默认取当前工作目录）+ 固定子目录。
    换目录运行时用 `pm.init(root="...")` 显式指定即可，不要手写 `os.path`。

## 3. 第一个完整脚本

```python
# program/src/program/problem1.py
"""问题一：某工厂生产计划优化。"""

import numpy as np
import program as pm

# ---------- 初始化 ----------
pm.init(seed=42)          # 中文字体 + 论文风格 + 固定随机种子

# ---------- 数据 ----------
df = pm.read_table("data/附件1.xlsx", sheet="Sheet1")
pm.record_result("数据概览", pm.summarize(df), level=2)

# ---------- 求解：最大化利润 ----------
profit = [3, 5]                                   # 两种产品单位利润
r = pm.optimize.solve_lp(
    c=profit,
    A_ub=[[1, 0], [0, 2], [3, 2]],                # 资源约束
    b_ub=[4, 12, 18],
    maximize=True,
)
print(r.summary())
pm.record_result(
    "问题一结果",
    {"最大利润": r.fun, "产品1产量": r.x[0], "产品2产量": r.x[1]},
    note=f"求解器状态：{r.message}",
)

# ---------- 出图 ----------
x = np.linspace(0, 4, 100)
pm.line(x, [3 * x, 5 * (18 - 3 * x) / 2], labels=["产品 1 利润", "产品 2 利润"],
        xlabel="产品 1 产量", ylabel="利润/万元", save="q1_profit")
```

运行后你会得到：

- 终端打印最优解摘要；
- `figures/q1_profit.pdf` —— 矢量论文图；
- `code/outputs/figure_data/q1_profit.csv` —— 作图数据（可追溯）；
- `reports/RESULTS_REPORT.md` —— 追加了「数据概览」「问题一结果」两节。

## 4. 常用命令速查

| 目的 | 命令 |
| --- | --- |
| 运行脚本（推荐方式） | `uv run --project program python program/src/program/problem1.py` |
| 跑测试 | `uv run --project program python program/tests/smoke_test.py` |
| 看模块清单 | `uv run --project program program` |
| 浏览本手册 | `uv run mkdocs serve`（在 program/ 下） |
| 加新依赖 | `uv add <包名>`，然后提交 `pyproject.toml` 与 `uv.lock` |

## 5. 接下来读什么

- 想画图 → [绘图手册](modules/plotting.md)（附 8 张示例图）
- 在读数据上卡住 → [数据读写](modules/dataio.md)
- 不知道用哪个检验/模型 → [统计 · 拟合 · 评估](modules/stats-fitting.md)
- 优化题不会下手 → [优化求解](modules/optimize.md)
- 想照着套路抄 → [实战食谱](cookbook.md)
