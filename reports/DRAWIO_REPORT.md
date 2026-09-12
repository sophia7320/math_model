# DrawIO 图示生成报告

## 图示清单

| 文件 | 类型 | 来源依据 | 用途 | 状态 |
| --- | --- | --- | --- | --- |
| `figures/技术路线图.drawio` / `.pdf` / `.png` / `.content.json` | 五带技术路线图（scibox-diagram 模板 `roadmap-5band`） | `reports/ANALYSIS_MODELING_REPORT.md`、`reports/RESULTS_REPORT.md` | 论文「问题分析/总体框架」节；一页概览四问递进与统一建模引擎 | ✅ 已导出 |
| `figures/Q2改进模型_流程图.drawio` / `.pdf` / `.png` | 手写 XML 纵向流程图（scibox-diagram 路径 B） | `reports/Q2_参数与分布结构专题.md`、`solve/experiments/q2e_structure.py`/`q2_tune.py` | 第二问改进模型（EWMA 标定 + κ/m 裕度 + 因果执行）的公式级流程说明；可作论文 Q2 模型图 | ✅ 已导出 |

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

## 给论文阶段的嵌入建议

- 位置：第 1 章「问题重述与研究思路」末尾，整页横排（`sidewaysfigure`）或 `0.95\textwidth` 缩放；
- 建议 caption：**图 1 微网购电—储能联合调度四问递进建模的技术路线**。
