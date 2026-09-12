# CUMCM 2026 C 题进度（2026-09-13）

## 已完成

- [x] 题面、附件、时间相位与单位核对
- [x] Q1 连续 LP、`result1.xlsx`、图表与约束校验
- [x] Q2 口径 D 确定性 + 蒙特卡洛 + 对冲（因果口径重算）
- [x] Q2 口径 E 自适应加权（W=7，1467.4 万，因果口径）
- [x] Q3 主方案 `result3.xlsx`（1327.8 万，场景前视违规 0 天）
- [x] 逐槽因果执行器 + 无前视残差池 + 回归测试 `tests/solve_causal_test.py`
- [x] Q2E 平滑/窗口/ARIMA 对照按因果口径刷新
- [x] Q4 电价 EDA + 三源参数结构 + 动态修正（设计文档 `Q4_电价预测结构设计.md`）
- [x] `plan.md` / `todo.md` / `reports/ANALYSIS_MODELING_REPORT.md` 补建
- [x] Q4 官方结果：`result4-2`（H 1558.6 / G 1547.8 万）+ `result4-3`（H 1397.1 / G 1383.7 万），
  2 日滚动跨日结构 + H 主口径；校验全过（前视违规 0）
- [x] 偏差专题与 Q3 关键消融按因果口径复算 → 更新两份报告
- [x] Q4 收尾：电价权重演化图
- [x] 技术路线图（五带模板 → `figures/技术路线图.pdf`；`reports/DRAWIO_REPORT.md`）
- [x] 论文与验收（`reports/VERIFY_REPORT.md` PASS）
- [x] Q2 参数搜索与分布结构专题（`reports/Q2_参数与分布结构专题.md`；公式流程图）
- [x] 回归参考量探索（甲/乙/丙/丁 + 动态窗口；丙 1403.2 万）——**用户决定搁置**（产物留存）
- [x] **执行 B（2026-09-13）：Q2 调优配置（EWMA h=5 + κ=1.02 + m=50 = 1406.6 万）写入
  `results/result2.xlsx`**，旧 W=7 版备份 `code/outputs/result2_W7_backup.xlsx`；
  论文同步为 21 页并重编译（摘要/对照表/参数灵敏度/结论/附录均已更新），
  writing_check 与 `reports/VERIFY_REPORT.md` PASS
- [x] Q3 灵敏度补测（情景数 5/10/20、场景池窗口 30/60/90、λ=0.5/0.9；前视违规全 0）
- [x] 五份结果文件结构审计 `tests/c_results_audit.py`（PASS；参考 GPT math_git 经验收编）
- [x] q2e 缓存隔离修复（`q2e_regsrc` + tag 选表；Q3 灵敏度复现 1327.7 万）

## 待完成

- [ ] （提交前）填写 `数学建模模板/support_materials/AI工具使用详情` 并生成 PDF、按需打包支撑材料
- [ ] 最终提交检查：`uv run python tests/c_results_audit.py` + `tests/solve_causal_test.py` +
  论文与根目录副本一致性（`paper/main.pdf` ↔ `论文_CUMCM2026_C题.pdf`）

## 已搁置（产物留存，随时可拾起）

- 回归参考量探索（用户 2026-09-13 决定先不考虑）：`program/src/solve/experiments/q2_regsrc.py`；
  结论——纯替换变差（甲 1518 / 乙 1522 万）；光伏回归混合版（丙）1403.2 万（−4.38% vs 官方，
  比旧最优仅省 3.4 万）；动态窗口 Wr=120 → 1405.8 万。数据 `code/outputs/q2_regsrc_*.csv`。

## 口径备忘

- 正式执行口径 = 逐槽因果（`q2.exec_segment_causal`）；`q2.exec_day` / `exec_segment_hindsight`
  是前视下界，只能作对照。
- 缓存键含 `q2.CAUSAL_POLICY_VERSION`；改执行策略必须升版本号，否则旧缓存污染新结果。
- 官方 result2 = 调优口径（EWMA h=5 + κ=1.02 + m=50）；`cache/q2e` 只放官方表（tag=q2e），
  变体表在 `cache/q2e_regsrc`（`load_extended` 按 tag 选表）。
- 历史报告（Q3 探索、偏差专题）部分数字为修订前口径；正式引用以 `RESULTS_REPORT.md` 为准。
- **预测层缩放/校准已否决**（全局 scale、差幅 γ、高峰 δ、滚动仿射、isotonic、分位映射）：
  计划 LP 对价格一次齐次，保序变换被决策吸收，费用不变；凸组合范围限制不修复。
  探针记录见 `program/src/solve/Q4_电价预测结构设计.md` §6。
