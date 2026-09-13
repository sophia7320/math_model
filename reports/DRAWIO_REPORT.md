# DrawIO 图示生成报告

## 图示清单

| 文件 | 类型 | 来源依据 | 用途 | 状态 |
| --- | --- | --- | --- | --- |
| `figures/技术路线图.drawio` / `.pdf` / `.png` / `.content.json` | 五带技术路线图（scibox-diagram 模板 `roadmap-5band`） | `reports/ANALYSIS_MODELING_REPORT.md`、`reports/RESULTS_REPORT.md` | 论文「问题分析/总体框架」节；一页概览四问递进与统一建模引擎 | ✅ 已导出 |
| `figures/Q2改进模型_流程图.drawio` / `.pdf` / `.png` | 手写 XML 纵向流程图（scibox-diagram 路径 B） | `reports/Q2_参数与分布结构专题.md`、`solve/experiments/q2e_structure.py`/`q2_tune.py` | 第二问改进模型（EWMA 标定 + κ/m 裕度 + 因果执行）的公式级流程说明；可作论文 Q2 模型图 | ✅ 已导出 |
| `figures/模型一图流.drawio` / `.pdf` / `.png` | 手写 XML 全模型总览（2400×1460 横版，四问泳道 + 共用机制带 + 验证交付带） | `reports/模型信息总表_论文手版.md`、`.opencode/skills/scibox-diagram/references/authoring.md` | 一图涵盖:数据口径 → 统一预测与风险机制 → Q1–Q4 泳道(信息/预测/计划/调整对冲/执行/结算/输出) → 验证与交付;含全部核心公式 | ✅ 已导出 |
| `figures/核心算法流程图.drawio` / `.pdf` / `.png` | 手写 XML 纵向流程图（scibox-diagram 路径 B；700×990，主列 7 盒 + 跨日滚动回路） | `paper/sections/5–8`、`reports/模型一致性规范.md`、`program/src/solve/consistency.py` | 论文正文「统一核心算法」图：多源预测→κ/m 裕度→两日计划 LP→日内调整 LP→逐槽因果执行→结算，逐盒核心公式 + 四问口径差异；v1.4 正式口径（无场景对冲） | ✅ 已导出 |

## 未生成图示及原因

- **子问题求解流程图（fig_flow_q1…）**：四问共享同一"统一建模引擎"（预测→计划→调整→执行，
  见路线图第③带），正文以公式、数据图与消融表承载细节，单独流程图会与路线图重复；
- **数据处理流程图**：单位换算、时间相位、信息时序审计三个口径要点已并入路线图第②带；
- **模型结构图**：路线图第③带即模型框架总览；如论文修改阶段仍需细化，可按
  `scibox-diagram` 的 `framework-3col` 模板补画。

## 导出与自检记录

- 渲染：`roadmap_5band.py`（内容源 `figures/技术路线图.content.json`）→ 容量检查通过（75 图元），
  两处超框文案已按预算缩短；
- 版式体检：`check_layout.py` → **0 FAIL / 0 WARN**；
- 导出：draw.io 桌面版 CLI（`%LOCALAPPDATA%\Programs\draw.io\draw.io.exe`）：
  `-x -f pdf --crop` → `技术路线图.pdf`（矢量，论文用）；
  `-x -f png -s 1 -b 0 --width 954` → `技术路线图.png`（预览）；
- 目视自检（PNG）：文字无溢出、箭头方向正确、五带配色统一、节点无重叠。
- `Q2改进模型_流程图`：手写 XML（1220×1500，主列 8 盒 + 右列 2 注释 + 费用回填回路），
  体检 **0 FAIL / 0 WARN**，`-x -f pdf --crop` / `-x -f png -s 2 -b 10` 导出；
  目视两轮：修正变音符歧义（改用 `^`=预测、`′`=修正、`_typ`=典型日），现无溢出、箭头语义正确。
- `模型一图流`：生成器脚本产出手写 XML（2400×1460；标题带 + 数据带 + 共用机制带 + 决策时序轴
  + 四问泳道 + 验证交付带，共 102 图元；母线式 B→C→四问分流，泳道内 7 盒串联 + 左侧“递进”箭头）；
  体检 **0 FAIL / 0 WARN**（一次行超宽与字号层级已修）；导出 `-x -f pdf --crop` /
  `-x -f png -s 1 --crop`；目视整图 + 四处 2× 放大（Q1 约束盒、Q3 对冲公式盒、题面参数盒、
  κ/m 与联合残差盒）均无溢出、无压线、字号层级 3 档（13/16/20）。
- `核心算法流程图`：生成器 `figures/核心算法流程图.gen.py`（含与 `check_layout.py` 同模型的中文
  字宽预检）→ 700×990，8 顶点 + 7 连接器；体检 **0 FAIL / 0 WARN**；`-x -f pdf --crop` /
  `-x -f png -s 2 -b 10 --width 1400` 导出；目视两轮：① 无溢出/压线、箭头方向与跨日回路正确；
  ② 跨日滚动标签移至 ④/⑤ 间隙、改「上一日实际末值」措辞后重渲复核。公式与参数逐条对照
  `program/src/solve/consistency.py`（κ=1.02、m=25 kW、λ=0.7、EWMA h=5、288 槽末端自由、
  free 执行、无场景对冲）。**注意**：`figures/模型一图流.drawio` 仍为旧口径（κ=1.015/m=75、
  Q3 开对冲），与本图不一致，引用旧图前需按 v1.4 修订重导。

## 给论文阶段的嵌入建议

- 位置：第 1 章「问题重述与研究思路」末尾，整页横排（`sidewaysfigure`）或 `0.95\textwidth` 缩放；
- 建议 caption：**图 1 微网购电—储能联合调度四问递进建模的技术路线**。
- `核心算法流程图`：建议置于第 2 章「模型建立」开头的统一求解框架小节，`width=0.9\textwidth`
  （裁剪后 474×693 pt，缩放后高约 21.9 cm，整页浮图）；
  建议 caption：**图 X 统一核心算法流程（Q1–Q4 共用引擎）**。
