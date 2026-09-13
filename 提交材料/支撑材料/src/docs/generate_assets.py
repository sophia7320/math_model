"""生成文档示例图（docs/assets/*.png）。

运行::

    uv run python docs/generate_assets.py

生成脚本本身也是 program 库的用法示例：中文字体、PDF/PNG 输出、作图数据记录。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import program as pm

DOCS = Path(__file__).resolve().parent
ASSETS = DOCS / "assets"
ASSETS.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
# root 指向 docs/，避免示例图生成时在工作区根产生额外目录
pm.init(root=DOCS, seed=42, verbose=False)


def save(fig, name: str) -> None:
    pm.save_fig(fig, name, outdir=ASSETS, formats=("png",), dpi=150, close=True, tight=False)
    print(f"  已生成 assets/{name}.png")


# 1. 双轴折线 ---------------------------------------------------------------
fig, ax, ax2 = pm.dual_axis(
    np.arange(1, 13),
    [12, 15, 18, 24, 28, 31, 33, 32, 28, 22, 17, 13],
    [30, 45, 80, 120, 160, 210, 230, 200, 140, 90, 55, 35],
    labels=("平均气温/℃", "降水量/mm"),
    xlabel="月份",
    ylabel1="气温/℃",
    ylabel2="降水/mm",
)
save(fig, "dual_axis")

# 2. 散点 + 多项式拟合 -------------------------------------------------------
x = rng.uniform(0, 10, 60)
y = 0.8 * x**2 - 3 * x + 5 + rng.normal(0, 4, 60)
fig, ax, info = pm.scatter_fit(x, y, degree=2, xlabel="自变量 x", ylabel="响应 y", s=22)
save(fig, "scatter_fit")
print(f"    （拟合 R² = {info['r2']:.4f}）")

# 3. 相关矩阵热力图 ----------------------------------------------------------
df = pd.DataFrame(
    rng.normal(size=(200, 4)),
    columns=["价格", "销量", "广告费", "评分"],
)
df["销量"] = -0.6 * df["价格"] + 0.5 * df["广告费"] + rng.normal(0, 0.5, 200)
fig, ax = pm.corr_heatmap(df, save=None)
save(fig, "corr_heatmap")

# 4. 雷达图（多方案综合评价） -------------------------------------------------
fig, ax = pm.radar(
    ["成本", "效率", "稳定性", "环保性", "可扩展性"],
    {
        "方案 A": [8, 6, 7, 9, 5],
        "方案 B": [6, 9, 8, 6, 7],
        "方案 C": [7, 7, 9, 7, 8],
    },
    normalize=True,
)
save(fig, "radar")

# 5. 三维曲面（参数敏感性） ---------------------------------------------------
XX, YY = np.meshgrid(np.linspace(0.1, 2.0, 60), np.linspace(0.1, 2.0, 60))
ZZ = 1.5 * XX * np.exp(-XX) + 0.8 * YY * np.exp(-0.6 * YY)
fig, ax = pm.surface3d(XX, YY, ZZ, xlabel="参数 α", ylabel="参数 β", zlabel="目标函数值")
save(fig, "surface3d")

# 6. ROC 曲线 ----------------------------------------------------------------
n = 300
y_true = (rng.uniform(size=n) < 0.5).astype(int)
score = np.clip(0.6 * y_true + rng.normal(0.25, 0.25, n), 0, 1)
fig, ax, auc = pm.roc(y_true, score)
save(fig, "roc")
print(f"    （AUC = {auc:.4f}）")

# 7. 分组柱状图（模型对比） ---------------------------------------------------
fig, ax = pm.bar_group(
    ["线性回归", "随机森林", "XGBoost"],
    {"R²": [0.82, 0.93, 0.95], "RMSE": [0.31, 0.19, 0.17]},
    xlabel="模型",
    ylabel="指标值",
)
save(fig, "bar_group")

# 8. 箱线图 + 直方图 ----------------------------------------------------------
fig, ax = pm.box(
    {
        "对照组": rng.normal(10, 2, 80),
        "实验组 A": rng.normal(12.5, 2.2, 80),
        "实验组 B": rng.normal(9.2, 1.8, 80),
    },
    ylabel="响应值",
)
save(fig, "box_group")

print(f"\n全部示例图已输出到 {ASSETS}")
