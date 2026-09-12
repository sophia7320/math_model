# CUMCM 2026 C 题：微网与外部电网电力调控

数学建模竞赛工作区。交付物 = **竞赛论文 + 可复现代码 + 图表**；当前主线为 CUMCM 2026 C 题
「微网（小区负载 + 光伏 + 储能）与外网之间的购电–储能联合优化调度」。

## 当前进度

> **2026-09-12 口径修订**：执行层由"事后 LP"统一改为**逐槽因果执行**（不读取未来实际值），
> 下表数字均为修订后口径；旧的段级/事后执行数字已废弃。

| 子问题 | 状态 | 关键结果 | 产物 |
| --- | --- | --- | --- |
| 问题一 典型日调度 | 完成 | 购电 59 482.7 kWh / 35 126.95 元，储能节省 26.9% | `results/result1.xlsx`、`figures/Q1_*.pdf` |
| 问题二 全年滚动（口径 D） | 完成 | 计划 1231.1 万 + 紧急 255.7 万 = **1486.8 万元**；朴素 MC 均值 1419 万、对冲 1399 万（−1.4%） | `results/result2.xlsx`（E 版）、`figures/Q2_*.pdf` |
| 问题二 自适应加权（口径 E，**官方 result2**） | 完成 | **调优版：EWMA h=5 + κ=1.02 + m=50 kW = 1406.6 万元**（计划 1329.1 + 紧急 77.5；原 W=7 版 1467.4 万，−4.14%）；纯历史、不用预报 | `results/result2.xlsx`、`figures/Q2E_*.pdf`、`code/outputs/q2e_tune_*.csv` |
| 问题二 ARIMA 预测对照 | 完成 | ARIMA(2,0,1) **2464.6 万元**（负荷 MAE 638.9、光伏 152.6 kW；未建模周周期） | `figures/Q2ARIMA_*.pdf`、`code/outputs/q2_arima_*.csv` |
| 问题三 日内滚动调整 | 完成 | 主方案（组合 λ=0.7 + 平滑 + 三点调整 + 无前视对冲）**1327.8 万元**；相对无调整官方 −12.7%、相对官方三点 −6.7% | `results/result3.xlsx`、`figures/Q3_*.pdf`、`code/outputs/q3_daily.csv` |
| 问题四 波动电价 | 完成 | **官方结果齐全**：Q2 层 result4-2（H 1558.6 / G 1547.8 万）、Q3 层 result4-3（H 1397.1 / G 1383.7 万）；2 日滚动跨日结构，电价信息价值 0.69% / 0.96% | `results/result4-2.xlsx`、`result4-3.xlsx`、`figures/Q4_*.pdf`、`code/outputs/q4_*.csv` |
| 论文 | 完成 | LaTeX（cumcmthesis 模板，21 页）：摘要 + 11 章 + 文献 + 附录；验收 PASS | `paper/main.pdf`、`论文_CUMCM2026_C题.pdf`、`reports/VERIFY_REPORT.md` |

口径说明：`result2.xlsx`（官方结果文件）由**口径 E 调优版（EWMA h=5 + κ=1.02 + m=50 kW）**生成
（1406.6 万元；旧 W=7 版备份在 `code/outputs/result2_W7_backup.xlsx`，旧口径 D 版备份在
`code/outputs/result2_D_backup.xlsx`）。参数搜索细节见 `reports/Q2_参数与分布结构专题.md`；
另有预测器对照实验（`uv run python -m solve.q2_arima`）。
详见 `reports/RESULTS_REPORT.md`；执行层信息结构见 `program/tests/solve_causal_test.py`。

## 快速开始

代码是 uv 项目（Python 3.14），**所有命令在 `program/` 目录下执行**：

```powershell
cd program

uv run python -m solve              # 问题一：典型日 LP（秒级）
uv run python -m solve.q2           # 问题二口径 D：全年滚动 + 100 年蒙特卡洛 + 对冲（约 2 分钟）
uv run python -m solve.q2_adaptive  # 问题二口径 E：自适应加权（首次建表约 5 分钟，之后走缓存）
uv run python -m solve.q2_tune --result2  # 用调优口径（EWMA+κ+m）生成官方 results/result2.xlsx
uv run python -m solve.q3           # 问题三主方案：result3 + 3 张图 + 报告章节（约 1 分钟）
uv run python tests/solve_causal_test.py      # 因果执行 + 无前视残差池回归测试
```

产物统一落在工作区根（由 `solve/common.py` 解析，不要另建输出目录）：

- `figures/*.pdf`：论文图（矢量 PDF，**中文命名**：`Q1_/Q2_/Q2E_/Q2ARIMA_/Q3_/Q4_` 前缀；逐图说明见 `figures/图表说明.md`）
- `reports/RESULTS_REPORT.md`：结果报告，**论文一切数值的唯一来源**
- `code/outputs/`：中间结果与图表数据；`code/outputs/cache/` 是可重建的耗时缓存
- `results/result*.xlsx`：按附件 5 模板填写的官方结果文件

## 目录结构

```text
math_model/
├─ CUMCM2026Problems/          赛题原文与附件（A~E 题，当前用 C 题）
├─ program/                    代码工程（uv 项目，独立 git 仓库）
│  ├─ src/program/             工具箱（import program as pm；11 模块、100+ 函数）
│  ├─ src/solve/               C 题求解（q1 / q2 / q2_adaptive / q2_arima / q3 / q4_price_* + 算法与设计文档）
│  ├─ data/C/                  附件 1~5（数据与官方结果模板）
│  ├─ tests/                   工具箱冒烟测试（smoke_test.py）+ 因果执行回归测试（solve_causal_test.py）
│  └─ AGENTS.md                工具箱 API 速查与常见坑（写代码前必读）
├─ figures/  reports/  code/  results/     产出目录（见上）
├─ C题解读.md                  题意、四问框架、交付格式与格式陷阱
├─ C题_经验总结.md             口径/单位/储能/踩坑清单/当前进度（写代码前必读）
├─ C题_预报误差分析.md         附件 3 预报误差规律（精度衰减、分布、相关结构）
├─ C题_电网基础背景知识.md     电力系统背景概念
├─ 数学建模函数速查表.md       scipy/sympy/torch 实测签名与坑
├─ 时间序列模型详解.md         ARIMA/SARIMA/ETS/GARCH 定阶、诊断与回测
├─ 建模参考资料/               7 类题型速查（含「选题速查表」）
├─ 数学建模模板/               CUMCM LaTeX 论文模板与提交检查清单
├─ 往年参考/MathModel-master/  往届资料库（只读）
└─ AGENTS.md                   AI 协作约定、uv 硬性约定、六阶段 skill 流水线
```

## 文档导航

| 想了解 | 看这里 |
| --- | --- |
| 题意与四问建模骨架 | `C题解读.md` |
| 踩坑经验、当前进度 | `C题_经验总结.md` |
| 口径 E 完整算法（公式/伪代码/结果） | `program/src/solve/口径E_自适应加权算法.md` |
| 工具箱 API 与输出约定 | `program/AGENTS.md`、`program/README.md` |
| 函数签名速查 | `数学建模函数速查表.md` |
| 选模型/方法 | `建模参考资料/README.md` |
| 写论文/提交 | `数学建模模板/提交前必读.md` |
| AI 协作、命令与工作流 | `AGENTS.md` |

## 工作约定（摘要）

- **Python 一律用 uv**：不用裸 `python` / `pip install`，不手动建 venv；加依赖 `uv add <pkg>`。
- 运行 `solve` 脚本时 cwd 必须是 `program/`（附件按 `./data/C/...` 相对路径读取）。
- 论文/图表里的一切数字必须能在 `reports/RESULTS_REPORT.md`、`results/` 或 `code/outputs/` 中找到来源。
- `pm.record_result` 为追加模式，重跑脚本先清理旧章节；`q2_adaptive.py` 已内置同名章节替换。
- `solve/` 里用 `multiprocessing.Pool`（Windows spawn）时，入口必须由 `if __name__ == "__main__":` 保护。
- 完整约定与六阶段 skill 流水线见 `AGENTS.md`。
