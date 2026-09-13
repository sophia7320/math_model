# C题论文写作记录（v1.4）

日期：2026-09-13。分支：`gpt`。排版：LaTeX / `cumcmthesis`。

## 本轮决策

Q3 正式方案由“组合预测 + 三点调整 + 场景对冲”改为“组合预测 + 三点调整、无对冲”。对冲结构仍保留公式、代码和消融结果，但明确作为未获留出数据支持的负结果扩展。决策依据见 `reports/Q3_对冲取舍实验.md`：开发期对冲低 0.20 万元，冻结验证期无对冲低 2.30 万元，全年无对冲低 2.10 万元且紧急购电更少。

本轮使用 `3coding-visual` 重算正式结果与图，使用 `5writing` 同步正文、摘要和指定日期表，再用 `6verity` 完成文本门禁、编译、数值和视觉验收。

## 正式数值映射

| 内容 | 来源 | v1.4 正式值 |
|---|---|---:|
| Q1 | `results/result1.xlsx` | 35126.95 元/日 |
| Q2 | `q2e_tuned_daily.csv`、`result2.xlsx` | 1395.7 万元 |
| Q3 | `q3_daily.csv`、`result3.xlsx` | 1347.1 万元 |
| Q3 对冲扩展 | `q3_daily_hedge_v1.3.csv`、`q3_ablation.csv` | 1349.2 万元 |
| Q4-2 G | `q4_result42_daily.csv`、`result4-2.xlsx` | 1459.8 万元 |
| Q4-3 G | `q4_result43_daily.csv`、`result4-3.xlsx` | 1410.3 万元 |
| H 扩展 | `RESULTS_REPORT.md` 当前 Q4 章节 | 1465.9 / 1418.4 万元 |

Q3 正式分项为计划 1290.5、调整净额 25.2、紧急 31.4 万元；Q4-3 G 分项为买入 1333.5、偏差 36.3、紧急 40.5 万元。分项独立舍入可能与总额相差 0.1 万元。

## 论文同步

- 摘要、问题三、问题四、灵敏度、模型评价与复现说明均已切换到 v1.4。
- `paper/tables/q3.tex` 从新 `result3.xlsx` 与 `q3_daily.csv` 机械重生成，四个指定日期为 3月20日、6月21日、9月23日、12月21日。
- 论文明确区分：两日初始规划中每天的日末电量不是常数；实际日末 SOC 自由并跨日传递；日内调整使用动态日末参考，但实际执行不追赶固定目标。
- `paper/main.pdf` 两遍 XeLaTeX 编译为 19 页；根目录副本 `论文_CUMCM2026_C题.pdf` 与其 SHA256 相同。

## 可复现运行

- `uv run python -m solve.q3`：约 23 秒，正式 Q3 无对冲。
- `uv run python -m solve.q4 --result4-3`：约 16 秒，正式 Q4-3 无对冲。
- `uv run python -m solve.q3_ablation`：无对冲单配置约 7–8 秒；40 情景对冲单配置约 164–165 秒。
- 测试：`model_consistency_test.py`、`solve_causal_test.py`、`c_results_audit.py` 均 PASS。

## 文件 SHA256

| 文件 | SHA256 |
|---|---|
| `paper/main.pdf` | `f6200c34a7e558a735217e161079750d9a647f67b7bd9c8a3ae8312f22716621` |
| `result1.xlsx` | `5351de53ac07eb497cbb630a9ba12c45b220c5ec950aae125f5f4bddce57a762` |
| `result2.xlsx` | `c88e35025e382efd76786938ddbd4a4e6c45077a21f709eda9463cbb37ff0791` |
| `result3.xlsx` | `c55ee0a09d144187ba5e367e144108f0e35875c2478eb35061436e857323e58f` |
| `result4-2.xlsx` | `1ab9a14b2b08e4f319b9d4b3ccbb71377ee08a3a92cab6ac19aabe47c6c9d150` |
| `result4-3.xlsx` | `c3f9b9b77ee8d2e10dc6cc5150da77a4f7e30bb7a749437c287e5754a3be90da` |

## 披露边界

本实验否定的是当前 40 情景、期望费用目标下的对冲实现，不是否定所有随机或鲁棒优化。既有 `q3e_tune_unified_*` 与 `q3_sensitivity.csv` 属于对冲扩展历史实验，不再决定 v1.4 正式策略。
