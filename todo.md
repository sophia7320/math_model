# CUMCM 2026 C 题进度（2026-09-13 口径统一改造，交接清单）

> **本文件是当前唯一进度清单，交接给新会话请先读这里**，再读：
> `reports/模型一致性规范.md`（v1.1 口径条款）、`reports/执行器段末目标实验.md`（执行器修复证据）、
> `reports/RESULTS_REPORT.md`（数值唯一来源）、`AGENTS.md`（工程约定）。
> 状态：**统一口径已落地，Q3 已按最终口径重算；Q2/Q4 官方结果与论文待重算同步。**

---

## 一、当前一句话状态

统一口径（EWMA h=5 + （Δ负荷, Δ光伏）联合残差 40 情景 + 因果负荷 +
2 日滚动/末端自由 + **执行器无段末硬目标**）已全部落地；
**Q2 已重算 = 1394.7 万**（自由执行器下留出冠军改选为 **κ=1.02 / m=50**，旧冠军 1.015/75 = 1406.4 万）；
**Q3 已重算 = 1360.3 万**（旧"Q3 > Q2"异常已消除，但用的是旧参数 1.015/75）；
⚠ **暂态不一致**：`consistency.py` 仍为 1.015/75，而 result2 已用 1.02/50——待 P0-2 重选参后统一定稿
（若 Q3 也支持 1.02/50 则更新 consistency.py + 规范 + 守卫测试，并重算 Q3）；Q4 两层为过渡版本；论文数值待统一同步。

---

## 二、2026-09-13 已完成（本节全部为当日新增）

- [x] **双轨合流**：gpt 分支修正并入（1/S 归一、发布时刻残差、因果负荷预测、时间留出选参、
  Q4 题面 G 主口径、40 情景收敛）；本机并行实现 `q2_roll/q3_roll` 归档删除（`archive/parallel_roll_2026-09-13/`）
- [x] **终端口径统一为"完全自由"**：去除 48 小时视野远端 6000 锚定（探针证明与锚定同值）
- [x] **统一口径规范 v1.1**：`reports/模型一致性规范.md`；**唯一参数源** `program/src/solve/consistency.py`
  （`EWMA_HL=5`、`KAPPA=1.015`、`MARGIN=75`、`N_SCEN=40`、`SCEN_MIN_SAME_MONTH=14`、
  `SCEN_LOOKBACK=90`、`START_DAY=31`、`EXEC_POLICY="free"`）；守卫测试 `tests/model_consistency_test.py`（PASS）
- [x] **预测统一**：Q3/Q4 历史预测换用 EWMA h=5 权重（`q3_proto.load_extended()` 生成 `EWMA_WU`；
  U_SMOOTH/TH 仅作回退）；κ/m 透传 Q3 计划与调整层；Q4 两层修复（**移除"当日实际负荷混入计划"bug**、
  因果负荷、κ/m、联合场景）
- [x] **场景统一**：（Δ负荷, Δ光伏）同月整日联合残差块、40 情景
  （`q3_proto.joint_residual_blocks`；Q2 官方 MC 用 `q2_tune.joint_residual_mc`）
- [x] **Q3 参数搜索（注：旧执行器下）**：`q3_tune.py` 粗筛选中 κ=1.015/m=75；细选 λ=0.7
  （开发期 2–6 月选参、7–12 月冻结）；表 `code/outputs/q3e_tune_unified_*.csv`
- [x] **执行器段末硬目标修复（关键）**：`q2.exec_segment_causal` 支持 `e_end=None`（自由）；
  `q3_proto.run_exec` 策略开关；**全部官方路径默认 free**；小样本证据
  `reports/执行器段末目标实验.md`（夏季 60 天 329.4→289.1 万，−12.2%；秋季 47 天 197.0→188.6 万，−4.3%；
  `free` 与 `dayend` 同值；SOC 仍全幅 [1200,10800]）
- [x] **Q3 重算（最终口径）**：`result3.xlsx` = **1360.3 万**（计划 1298.2 + 调整净额 +20.6 + 紧急 41.6）；
  对照：无调整 1460.8、三点官方 1357.1；审计 PASS、前视违规 0、跨日衔接 0；段末紧急尖峰消失
- [x] **清尾**：`q3_sensitivity/q3_ablation` 的失效 `U_SMOOTH` 行清除、`q3_ablation` 对齐 `cs.N_SCEN`、
  `q2_bias.py` 标注历史口径；RESULTS_REPORT 头部改为 v1.1 状态表、删除归档脚本的过时 Q3 章节
- [x] **codegraph 接入**：`.codegraph/` 已初始化并入 `.gitignore`（84 文件/1581 节点）；
  改码后 `codegraph sync`；注意属性调用（`cs.xxx`/`qp.xxx`）漏解析，**必须 grep 兜底复核**

---

## 三、待办（按优先级，新会话从这里继续）

### P0（先做这三件，冻结 Q2/Q3 正式数字）

- [x] **P0-1 Q2 官方按自由执行器重算（2026-09-13 完成）**
  - 留出重选：冠军由 `1.015/75` 改选为 **`h=5 κ=1.02 m=50`**（开发 574.7 / 验证 820.0 / 全年 1394.7 万，验证期排名 3）
  - `write_result2_tuned` 默认参数已更新；`result2.xlsx` 重写 = **1394.7 万**（计划 1322.5 + 紧急 72.2；旧版 1406.4）
  - 顺手清理：`joint_residual_mc` 移除未用的 `targets` 参数；MC 报告 note 修正为 free 口径；
    旧官方章节（1.015/75）已在 RESULTS_REPORT 标注为历史存档
  - 验收：`tests/c_results_audit.py` PASS、`tests/model_consistency_test.py` PASS
- [ ] **P0-2 Q3 参数搜索在自由执行器下重跑**
  ```powershell
  uv run python -m solve.q3_tune     # ~20 分钟（16+5 组，6 进程）
  ```
  当前主方案（λ=0.7+对冲）在最终口径下为 1360.3，**略差于"三点官方"基线 1357.1（+0.23%）**；
  重跑后若最优参数/策略不同（例如 λ 更低、或"λ=1+关对冲"），更新 `q3.py` 的 `LAM/ADJ_LAM/hedge`
  并 `uv run python -m solve.q3` 重算 result3（~5 分钟）。
- [ ] **P0-3 Q2/Q3 同口径配对对照**（两问都重算后）：出一张"同预测 / 同 κ/m / 同联合场景，
  仅机制开关不同"的对照表（无调整 → 三点 → +对冲 → +组合），供论文讲"机制价值"；
  建议做成一次性探针脚本（预期 15–30 分钟）。

### P1（正式数字冻结后）

- [ ] **P1-1 Q4 两层按统一口径重算**：`uv run python -m solve.q4 --result4-2` + `--result4-3`
  （执行器已 free、因果负荷、κ/m、联合场景；G 主口径；预计 10–20 分钟）；
  若 H 扩展要保持一致，先 `uv run python -m solve.q4_price_fit`（~6 分钟）再跑 result4
- [ ] **P1-2 图表与报告同步**：Q2 三图随 P0-1 自动重生成；RESULTS_REPORT 最终整理
  （多代章节归并/标注；头部已有 v1.1 索引）
- [ ] **P1-3 论文同步**：`paper/main.tex` 数值与结构描述（"末端自由 / 无段末目标 / 联合场景 /
  统一参数 / Q3 机制分解"）；`xelatex` 两遍；根副本 `论文_CUMCM2026_C题.pdf` 更新；
  `reports/VERIFY_REPORT.md` 重跑

### P2（提交前）

- [ ] **P2-1 文档同步**：`AGENTS.md`（脚本清单加 `consistency/q3_tune/exec_policy_probe`；执行策略）、
  `README.md`、`C题_经验总结.md`
- [ ] **P2-2《AI 工具使用详情》**审校定稿（日期/措辞待参赛队确认）+ **支撑材料打包**
  （src/ data/ results/ figures/ reports/ README.txt + AI 详情 PDF；与论文附录 B 逐项一致）
- [ ] **P2-3** 可选：`q2_bias.py` / `q3_ablation.py` 复跑适配（当前仅历史标注）
- [ ] **P2-4** 提交：`git add -A && git commit`（当前分支 `temp`，工作区有大量未提交改动）

### 已知口径 caveat（论文里如实说明即可，不必阻塞）

- `q2e` 权重标定成本表仍是"日循环执行"代理口径（`day_cost` 未随 free 执行器重标定）；
  如需严格一致需改造标定评估（成本高，暂不做）。
- Q3 主方案 vs 三点官方在最终口径下差 +0.23%（P0-2 重选参后再定论文表述）。

---

## 四、口径与命令备忘（v1.1）

- **唯一参数源**：`program/src/solve/consistency.py`；改口径先改它 + 规范文档 + 守卫测试。
- **官方执行** = 逐槽因果 + **无段末硬目标（free）**；`q2.exec_day` / `exec_segment_hindsight`
  仍是前视下界，只能作对照。
- 信息集：Q1 附件 1；Q2 附件 1/2（历史自适应，不用附件 3）；Q3/Q4 可用附件 3（λ 组合）；
  Q4 价格 G = 附件 4 已知（主口径）、H 为历史预测扩展。
- 起点：**2-1 0:00 以 E0=6000 起步**（1 月仅预测预热）；规划窗口 2 日、末端自由；Q2 执行 1 天、
  Q3/Q4 执行窗 6 h（决策 0/6/12/18）。
- 固定检查：`uv run python tests/model_consistency_test.py`（守卫）、
  `tests/c_results_audit.py`（五份结果结构/勾稽/跨日连续）、`tests/solve_causal_test.py`（因果/无前视）。
- 运行环境：`workdir=program`；脚本含中文先设 `$env:PYTHONIOENCODING='utf-8'`。
- codegraph：改码后 `codegraph sync`；查询用 `callers/callees/impact`，**属性调用漏报须 grep 复核**。

## 五、当前正式数字速查（2026-09-13 06:15）

| 文件 | 数值 | 状态 |
| --- | --- | --- |
| `results/result1.xlsx` | 35126.95 元/日（59482.7 kWh，弃光 0） | 正式（题面日循环） |
| `results/result2.xlsx` | **1394.7 万**（计划 1322.5 + 紧急 72.2；冠军 EWMA h=5 + κ=1.02 + m=50） | 已按 v1.1 重算 ✓ |
| `results/result3.xlsx` | **1360.3 万**（计划 1298.2 + 调整 +20.6 + 紧急 41.6；参数仍为 κ=1.015/m=75） | 已重算，但参数待 P0-2 统一定稿 |
| `results/result4-2.xlsx` | G 口径过渡版本 | 待 P1-1 重算 |
| `results/result4-3.xlsx` | G 口径过渡版本 | 待 P1-1 重算 |

对照（最终口径）：Q3 无调整 1460.8 → 三点官方 1357.1 → 主方案 1360.3；
结构调整机制价值 −7.1%（相对无调整），组合+对冲增量 ≈ +0.23%（待 P0-2 重选参后定稿）。

## 六、历史已完成（更早，保留备查）

- [x] Q1 连续 LP、result1 与校验；题面/附件/相位/单位核对（`C题_解读`/`经验总结`）
- [x] Q2 口径 D/E 全套探索、ARIMA 对照、参数与分布结构专题
- [x] Q3 消融/灵敏度（旧口径，数字为历史）、无前视场景池与因果执行器
- [x] Q4 电价 EDA/三源结构/动态修正设计；双轨合流前的 H 主口径版本（历史）
- [x] 技术路线图/流程图、论文 21 页版与验收（数字将随本轮重算更新）
- [x] 回归参考量探索（用户决定搁置；`q2_regsrc`，产物留存）
