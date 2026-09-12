# CUMCM 2026 C 题交付计划

## 目标

完成可提交的 C 题论文、5 个官方 Excel 结果文件（result1/2/3/4-2/4-3）、
可复现求解代码与全部论文图表。所有正式数字只从 `reports/RESULTS_REPORT.md`、
`results/` 和 `code/outputs/` 引用。

## 当前口径（2026-09-12 修订）

- 竞赛：CUMCM 2026 C 题；语言：中文；排版引擎进入论文阶段前由用户选定（Typst / LaTeX）。
- 调度粒度 10 分钟，功率统一为 kWh/槽（kW ÷ 6）。
- 储能：充放双向效率 0.9，SOC 1200–10800 kWh，不售电、富余弃电，日循环 E(0)=E(24)=6000 kWh。
- Q2/Q3 主结果：无前视预测 + **逐槽因果执行**（`q2.exec_segment_causal`）；事后 LP 只作前视下界。
- Q3：最终调整量相对原计划一次结算；逐次结算作敏感性。
- Q4：实时波动电价；口径 G（完全信息）与 H（历史预测）对照；主口径 H 用三源凸权重 + 动态修正。

## 阶段和交付物

1. 建模设计：`reports/ANALYSIS_MODELING_REPORT.md`（与 plan/todo 同批补建）。
2. Q1/Q2 结果：`result1.xlsx`、`result2.xlsx`、报告章节与图。✅
3. Q3：`solve/q3.py`、`result3.xlsx`、Q3 图与结果章节。✅
4. Q4 波动电价：`result4-2.xlsx`、`result4-3.xlsx`。✅
5. 技术路线图与模型流程图（drawio → PDF）。
6. 论文撰写、编译与最终验收。

## 验收标准

- 任一目标日的预报、参数与场景不得使用该日及未来实际数据。
- 功率平衡、SOC 动态、容量/功率/终端约束全部回代通过（`tests/solve_causal_test.py`）。
- Excel 的 sheet 名、行列、汇总值与紧急购电事件通过回读勾稽。
- 每张 PDF 图有可追溯的 CSV 源数据（`code/outputs/figure_data/`）。
- 论文数字与 `RESULTS_REPORT.md` 完全一致。
