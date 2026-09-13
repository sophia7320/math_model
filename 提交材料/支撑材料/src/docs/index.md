# program · 数学建模工具包

面向 **CUMCM / 华为杯 / MCM** 的 Python 便利库，把「读数据 → 建模 → 出图 → 记录结果」
这条流水线上的重复劳动全部封装好，并且输出的文件结构与本工作区的竞赛工作流
（`figures/`、`reports/RESULTS_REPORT.md`、`code/outputs/`）严格对齐。

> 这份手册是**给人看**的：讲清楚每个模块解决什么问题、什么时候用、怎么用。
> 给 AI 的 API 速查表在项目根目录的 `AGENTS.md`。

## 亮点

<div class="grid cards" markdown>

- :material-rocket-launch: **一行初始化**

    中文字体、论文风格、随机种子，一个 `pm.init()` 全部就位。

- :material-file-table: **什么格式都能读**

    Excel 多工作表、GBK 编码的 CSV、JSON……自动识别，缺失值一键处理。

- :material-chart-line: **论文级绘图**

    15 种常用图型，直接输出矢量 PDF；作图数据自动留档，随时可追溯。

- :material-check-decagram: **结果可复现**

    随机过程全部固定种子；关键数值统一写入 `RESULTS_REPORT.md`，论文引用不迷路。

</div>

## 30 秒示例

```python
import numpy as np
import program as pm

pm.init()                                    # ① 初始化（必做）

df = pm.read_table("data/附件1.xlsx")         # ② 读数据
pm.record_result("数据概览", pm.summarize(df))

x = np.linspace(0, 24, 200)
pm.line(x, np.exp(-0.13 * x), xlabel="时间 t/h", ylabel="浓度 c/(mg/L)",
        save="q1_curve")                      # ③ 出图 → figures/q1_curve.pdf
```

## 模块总览

| 模块 | 一句话说明 |
| --- | --- |
| [`config` + `utils`](modules/report-utils.md) | 初始化、路径、随机种子、计时、日志、缓存 |
| [`dataio`](modules/dataio.md) | 读数据、洗数据、看数据 |
| [`plotting`](modules/plotting.md) | 论文级绘图（含 8 张示例图画廊） |
| [`stats` / `fitting` / `metrics`](modules/stats-fitting.md) | 统计检验、曲线拟合、模型评估 |
| [`optimize`](modules/optimize.md) | 线性 / 整数 / 非线性规划，背包、指派、TSP |
| [`ml`](modules/ml.md) | 机器学习训练、交叉验证、聚类、PCA |
| [`ode`](modules/ode.md) | 微分方程求解与参数反演 |
| [`report`](modules/report-utils.md) | 结果写报告、导出 LaTeX 表格 |

## 从这里开始

1. [:octicons-arrow-right-24: 快速开始](getting-started.md) —— 环境、运行方式、第一个完整脚本
2. [:octicons-arrow-right-24: 实战食谱](cookbook.md) —— 预测 / 优化 / 评价 / 分类四类赛题套路
3. [:octicons-arrow-right-24: 常见问题](faq.md) —— 中文乱码、目录约定、报错排查

!!! tip "本地浏览本手册"
    ```powershell
    uv run mkdocs serve      # 打开 http://127.0.0.1:8000
    uv run mkdocs build      # 静态站点输出到 site/，可离线双击打开
    ```
