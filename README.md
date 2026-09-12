# CUMCM 2026 C 题：微网与外部电网电力调控

数学建模竞赛工作区。交付物 = **竞赛论文 + 可复现代码 + 图表**；当前主线为 CUMCM 2026 C 题
「微网（小区负载 + 光伏 + 储能）与外网之间的购电–储能联合优化调度」。

## 当前进度

> **2026-09-13 统一口径 v1.2 定稿**：唯一参数源 `program/src/solve/consistency.py`
> （EWMA h=5 + κ=1.02 + m=50 + （Δ负荷, Δ光伏）联合残差 40 情景 + 2 日滚动/末端自由 +
> 执行器 free）；Q2/Q3 已按该口径重算，Q4 两层为过渡版本（待重算，见 `todo.md` P1-1）。

| 子问题 | 状态 | 关键结果 | 产物 |
| --- | --- | --- | --- |
| 问题一 典型日调度 | 完成 | 购电 59 482.7 kWh / 35 126.95 元，储能节省 26.9% | `results/result1.xlsx`、`figures/Q1_*.pdf` |
| 问题二 全年滚动（口径 D） | 完成 | 口径 D 对照（附件 3/2 信息结构）；`--write-result2-d` 写备份 | `figures/Q2_*.pdf`、`code/outputs/q2_*.csv` |
| 问题二 自适应加权（口径 E，**官方 result2**） | 完成 | **1394.7 万元**（计划 1322.5 + 紧急 72.2；EWMA h=5 + κ=1.02 + m=50，2 日滚动 + free 执行） | `results/result2.xlsx`、`figures/Q2E_*.pdf`、`code/outputs/q2e_tune_*.csv` |
| 问题二 ARIMA 预测对照 | 完成 | ARIMA(2,0,1) 2464.6 万元（负荷 MAE 638.9、光伏 152.6 kW；未建模周周期） | `figures/Q2ARIMA_*.pdf`、`code/outputs/q2_arima_*.csv` |
| 问题三 日内滚动调整 | 完成 | 主方案（组合 λ=0.7 + 2 日滚动 + 三点调整 + 40 情景联合对冲 + free）**1350.8 万元**；对照 无调整 1455.7 → 三点官方 1358.9 | `results/result3.xlsx`、`figures/Q3_*.pdf`、`code/outputs/q3_daily.csv` |
| 问题四 波动电价 | 进行中 | G（题面已知电价）主口径过渡版本：Q2 层 1460.6 万、Q3 层 1413.0 万（H 扩展 1467.1 / 1421.5）；待按 v1.2 重算 | `results/result4-2.xlsx`、`result4-3.xlsx`、`figures/Q4_*.pdf` |
| 论文 | 待同步 | LaTeX（cumcmthesis）；正文数值与口径待随 v1.2 重算更新（todo P1-3） | `paper/main.pdf`、`论文_CUMCM2026_C题.pdf` |

口径说明：`result2.xlsx`（官方结果文件）由 `q2_tune.py --result2` 按**时间留出冠军
（EWMA h=5 + κ=1.02 + m=50）**生成（1394.7 万元；2 日滚动前的备份在
`code/outputs/result2_pre_2day_backup.xlsx`，旧 W=7/D 版备份在 `result2_W7_backup.xlsx`
与 `result2_D_backup.xlsx`）。口径条款见 `reports/模型一致性规范.md`，参数搜索细节见
`reports/Q2_参数与分布结构专题.md`；预测器对照实验
（`uv run python -m solve.experiments.q2_arima`）。
详见 `reports/RESULTS_REPORT.md`；唯一参数源守卫见 `program/tests/model_consistency_test.py`。

## 快速开始

代码是 uv 项目（Python 3.14），**所有命令在 `program/` 目录下执行**：

```powershell
cd program

uv run python -m solve              # 问题一：典型日 LP（秒级）
uv run python -m solve.q2_tune --result2  # 官方 result2（2 日滚动 + EWMA h=5 + κ=1.02 + m=50，约 10 分钟）
uv run python -m solve.q3           # 问题三主方案：result3 + 3 张图 + 报告章节（约 5 分钟）
uv run python tests/model_consistency_test.py  # 唯一参数源与统一构造守卫
uv run python tests/solve_causal_test.py       # 因果执行/无前视/发布时刻残差回归测试
uv run python tests/c_results_audit.py         # 五份结果文件结构审计
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
├─ program/                    代码工程（uv 项目，随根仓库统一管理）
│  ├─ src/program/             工具箱（import program as pm；11 模块、100+ 函数）
│  ├─ src/solve/               C 题求解（core/models/data/io/flows 分层；官方入口 q1~q4、q2_tune、q4_price_fit；experiments/ 专题脚本；legacy/ 早期脚本；docs/ 算法文档）
│  ├─ data/C/                  附件 1~5（数据与官方结果模板）
│  ├─ tests/                   工具箱冒烟测试 + 唯一参数源/因果执行/结果审计回归测试
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
| 踩坑经验、当前进度 | `C题_经验总结.md`、`todo.md`（交接清单与正式数字速查） |
| 统一口径条款与唯一参数源 | `reports/模型一致性规范.md`、`program/src/solve/consistency.py` |
| 执行器段末目标实验证据 | `reports/执行器段末目标实验.md` |
| 口径 E 完整算法（公式/伪代码/结果） | `program/src/solve/docs/口径E_自适应加权算法.md` |
| 工具箱 API 与输出约定 | `program/AGENTS.md`、`program/README.md` |
| 函数签名速查 | `数学建模函数速查表.md` |
| 选模型/方法 | `建模参考资料/README.md` |
| 写论文/提交 | `数学建模模板/提交前必读.md` |
| AI 协作、命令与工作流 | `AGENTS.md` |

## 工作约定（摘要）

- **Python 一律用 uv**：不用裸 `python` / `pip install`，不手动建 venv；加依赖 `uv add <pkg>`。
- 运行 `solve` 脚本时 cwd 必须是 `program/`（附件按 `./data/C/...` 相对路径读取）。
- 论文/图表里的一切数字必须能在 `reports/RESULTS_REPORT.md`、`results/` 或 `code/outputs/` 中找到来源。
- `pm.record_result` 为追加模式，重跑脚本先清理旧章节（`solve.io.report.record` 自带同名章节替换）。
- `solve/` 里用 `multiprocessing.Pool`（Windows spawn）时，入口必须由 `if __name__ == "__main__":` 保护。
- 完整约定与六阶段 skill 流水线见 `AGENTS.md`。
