# program —— 数学建模便利工具包

面向 CUMCM / 华为杯 / MCM 的 Python 工具包：把「读数据 → 建模 → 出图 → 记录结果」
的重复工作封装成一套函数，并与本工作区的六阶段流水线（`figures/`、`reports/RESULTS_REPORT.md`、
`code/outputs/`）严格对齐。

## 快速开始

```python
import program as pm

pm.init()                                  # 中文字体 + 论文风格 + 随机种子
df = pm.read_table("data/附件1.xlsx")       # 数据读取（csv/xlsx/json/pkl，GBK 自动回退）
r = pm.ml.fit_regressor("rf", X, y)        # 训练 + 交叉验证 + 评估
pm.record_result("问题一结果", r.test_metrics)
pm.line(x, y, xlabel="t", ylabel="y", save="q1_curve")   # → figures/q1_curve.pdf
```

## 模块一览

| 模块 | 用途 |
| --- | --- |
| `config` / `utils` | 初始化、路径、随机种子、计时、日志、缓存 |
| `dataio` | 数据读写、缺失处理、数据概览、训练/测试划分 |
| `plotting` | 论文级绘图（PDF + 数据记录）：折线/散点/热图/雷达/3D/ROC 等 15 种 |
| `stats` | 统计检验、相关、回归、置信区间（中文结论） |
| `fitting` | 曲线拟合、多项式、插值、平滑 |
| `metrics` | 回归 / 分类 / 聚类评估指标 |
| `optimize` | LP / MILP / NLP / 全局优化 / 背包 / 指派 / TSP |
| `ml` | 机器学习训练 + 交叉验证 + 模型对比 + 聚类 + PCA |
| `ode` | 微分方程求解与参数反演 |
| `report` | RESULTS_REPORT 写入、LaTeX 表格、结果文件输出 |

## 运行与测试

```powershell
# 在工作区根目录运行赛题脚本（推荐）
uv run --project program python program/src/program/problem1.py

# 冒烟测试（75 项检查）
uv run --project program python program/tests/smoke_test.py

# 查看模块清单
uv run --project program program
```

## 文档

| 文档 | 读者 | 内容 |
| --- | --- | --- |
| **`docs/`（人类可读手册）** | 人 | 快速开始、模块教程、8 张示例图画廊、实战食谱、FAQ。`uv run mkdocs serve` 在线浏览，或 `uv run mkdocs build` 后直接打开 `site/index.html`（支持离线双击） |
| **`AGENTS.md`** | AI 助手 | 紧凑的 API 速查、输出约定、常见坑与端到端示例 |
| 各模块 docstring | 开发者 | 函数级用法示例 |
