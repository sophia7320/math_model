# AGENTS.md

数学建模竞赛工作区（CUMCM/华为杯/MCM 等）：最终交付物是竞赛论文 + 可复现代码 + 图表。代码统一放 `program/`（uv 项目，Python 3.14）；**整个工作区由根目录 `.git` 单仓库统一管理**（论文/结果/图表/报告/技能配置/代码；忽略缓存、`.venv` 与 LaTeX 中间产物，规则见根 `.gitignore`）。提交：`git add -A && git commit`（在根目录执行一次即含 `program/`）。

**当前主线：CUMCM 2026 C 题**（微网与外部电网电力调控，题面在 `CUMCM2026Problems/C题/`）。根目录工作笔记：`C题解读.md`、`C题_电网基础背景知识.md`、`C题_预报误差分析.md`、`C题_经验总结.md`——**写 C 题代码前先读「经验总结」**（口径、单位换算、储能终端条件、踩坑清单见 §1~§3）。另有早期 `A题_*.md` 笔记。

**C 题进度（2026-09）**：Q1–Q4 官方结果齐全——`result1~3.xlsx`、**`result4-2.xlsx`（Q2 层：H 1558.6 / G 1547.8 万）与 `result4-3.xlsx`（Q3 层：H 1397.1 / G 1383.7 万）**，Q4 用 2 日滚动跨日结构 + H 主口径；**论文与验收已完成**（LaTeX/cumcmthesis，`paper/main.pdf` 21 页；`reports/VERIFY_REPORT.md` PASS；论文副本 `论文_CUMCM2026_C题.pdf`）。**2026-09-12 起全部正式数字按"逐槽因果执行"口径重算**；**2026-09-13 Q2 官方 result2 升级为调优版：EWMA h=5 + κ=1.02 + m=50 kW = 1406.6 万（原 W=7 版 1467.4 万，−4.14%）**，Q3 主方案 1327.8 万。论文素材（关键数字、负结果、口径讨论）：`reports/Q3_方案探索报告.md`、`reports/预测目标专题_精度最优vs决策最优.md`、`reports/Q2_参数与分布结构专题.md`（前两者部分数字为修订前口径，正式引用以 `RESULTS_REPORT.md` 为准）。

## Python 一律用 uv（硬性约定，覆盖 skill 文档里的写法）

各阶段 skill 文档中写的 `python3` / `pip install` 一律改用 uv 等价命令执行。

- **代码全部放 `program/`**（uv 项目，Python 3.14）：工具箱包在 `program/src/program/`（`import program as pm`，11 模块）；**赛题求解代码放 `program/src/solve/`**，运行必须 cwd=`program/`（如 `uv run python -m solve.q3`）：
  - 正式求解：`q1.py`（问题一）；`q2.py`（问题二口径 D）；`q2_adaptive.py`（口径 E 自适应加权历史模型；**官方 result2 已改由 `q2_tune.py --result2` 生成**，此脚本仅用于复现旧 W=7 版）；`q2_tune.py`（**Q2 参数搜索**：`--result2` 写官方调优版（EWMA h=5+κ=1.02+m=50）、`--fig-weights` 重生成 EWMA 权重图，整跑约 4 分钟）；`q3.py`（**问题三主方案，一键生成 result3.xlsx + 中文图 + 报告章节，约 1 分钟**）；`q3_sensitivity.py`（Q3 灵敏度：情景数/场景池窗口/λ，约 1 分钟）；`q4.py`（**问题四主程序：结构探针 `--probe` / `--probe-q3`，官方结果 `--result4-2` / `--result4-3`**）；`q2_arima.py`（ARIMA 对照，首次约 5 分钟）。
  - 专题/探索（保留勿删）：`q2_bias.py` + `q2_bias_roll.py`（有偏预测 vs 场景对冲）；`q2e_smooth.py`（Q2E 平滑回测，负结果）；`q2_regsrc.py`（**回归参考量探索，已搁置**：探针 / `--full` / `--wr`，产物 `code/outputs/q2_regsrc_*.csv`）；`q3_ablation.py`（Q3 关键消融：对冲/组合/信息退化，写入报告章节）；`q3_proto.py`（**Q3 核心库**：三层结算 LP / 实时执行 / 场景对冲 / `fc_slots` 相位映射）及 `q3_proto_{seg,abl,rt,adjmix,final,lh,smooth}.py`；`q4_price_{eda,fit,dyn}.py`（Q4 电价结构/参数/动态修正）。
  - 测试：`tests/solve_causal_test.py`（因果执行/残差池回归）、`tests/c_results_audit.py`（**五份官方结果文件结构审计**，提交前固定检查）。
  - 早期脚本 `2.py`/`arima.py`/`data_reader.py`/`lp_model.py`/`solve1.py`/`solve2.py` 勿动。
- **Q3 口径坑（改 Q3 相关代码前必读）**：
  - **执行口径（2026-09-12 修订）**：正式口径为**逐槽因果执行**（`q2.exec_segment_causal`/`q2.exec_day_causal`，每槽只读当前已实现值）；`q2.exec_day`（全天事后 LP）与 `q3_proto.exec_segment_hindsight`（段内事后 LP）**只能当前视下界**，不得进入正式结果。前视会系统性低估紧急购电，别把"事后最优"当可实现策略。
  - **无前视场景池**：对冲/重采样只能用目标日之前的残差块（`q3_proto.causal_residual_pool`；池参数 `min_same_month`/`lookback` 经 `simulate_day_rt_hedge` 透传，默认 14/90，Q3 灵敏度用）；缓存的哈希必须含 `q2.CAUSAL_POLICY_VERSION`，改执行策略要升版本号（否则旧缓存会污染新结果）。
  - **相位**：`q2._hour_to_slots` 只适用于 0:00 发布的预报；6:00/12:00/18:00 发布的预报必须用 `q3_proto.fc_slots(fc24, 发布时刻)` 映射到槽，否则整体错位 6/12/18 小时（费用翻倍级错误）。
  - **信息退化**：调整层预报必须与 0:00 计划层同口径；调整层单用官方会抹掉组合预测的收益（λ=0.5 时"仅开 6:00"曾因此由改善转为恶化）。
  - **参数平滑适用性**：新旧混合 `β·new+(1−β)·old` 只对日频抖动大的权重有效（Q3 的 W=1 光伏权重，β≈0.1）；对已平滑的 W=7 负荷权重一律有害（Q2E 回测）。
  - **回归测试**：改执行/场景相关代码后跑 `uv run python tests/solve_causal_test.py`（因果执行约束回代 + 残差池无前视断言）。
- **Q4 口径（已定稿）**：主口径 H（历史电价：滚动费用标定 v + β=0.1 动态修正），G（完全信息）作对照；正式结果为 **2 日滚动跨日结构**（`q4.py` 的 `plan_horizon`）。复现：`uv run python -m solve.q4 --result4-2` / `--result4-3`（各约 1–2 分钟）。**注意**：v 权重由 `code/outputs/q4_price_fit_daily.npz` 逐日费用表滚动选择，该表依赖执行口径——改执行策略后需 `uv run python -m solve.q4_price_fit` 重算（约 3 分钟）再跑 `q4.py`。
- **产物统一落在工作区根**：
  - `figures/*.pdf`——**文件名中文**（如 `Q3_策略费用对比.pdf`），命名规范与逐图说明见 `figures/图表说明.md`；作图数据同步在 `code/outputs/figure_data/<同名>.csv`。非数据图（`技术路线图.pdf`、`Q2改进模型_流程图.pdf`，源 `.drawio`，生成记录 `reports/DRAWIO_REPORT.md`）也在本目录。新增/改名图后同步更新说明文档与代码里的 save 名。
  - `reports/RESULTS_REPORT.md`（论文唯一数值来源）；`results/result*.xlsx`（共 5 个：`result1/2/3.xlsx` + `result4-2/4-3.xlsx`。`result2.xlsx` = 调优口径 1406.6 万，旧 W=7 版备份 `code/outputs/result2_W7_backup.xlsx`、旧 D 版 `result2_D_backup.xlsx`；`result3.xlsx` 由 `solve.q3` 生成 1327.8 万；`result4-2/4-3.xlsx` 由 `solve.q4` 生成）。
  - `code/outputs/`：`cache/q2e/*.npz`（**只放官方表，`tag=q2e`**）、`cache/q2e_regsrc/*.npz`（回归源变体表，`RegSourceModel.CACHE_TAG`）——**q3_proto.load_extended 按 tag 选表，变体表不要写进 `cache/q2e`**；`cache/q2arima/*.npz` 删除后需重算约 13 分钟；`q4_price_fit_daily.npz` 为 Q4 电价逐日费用/误差表（动态修正复用它）。
  - 输出根由 `solve/common.py::workspace_root()` 解析（`MATHMODEL_ROOT` > `program/` 上一级 > cwd），不要另建输出目录。
- 附件在 `program/data/C/`；`solve/data_reader.py` 用相对路径 `./data/C/...`——**运行这类脚本时 cwd 必须是 `program/`**。
- torch/torchvision 走 pyproject 的 CUDA 13.0 轮子源（`[[tool.uv.index]] pytorch-cu130`），不要改用默认 PyPI 源。**本机 GPU 可用**（RTX 5060 / sm_120 / torch 2.14.0+cu130）；深度学习试验在 `program/experiments/q3_dl/`（结论：单年数据下 DL 季节性风险大、作为组合第三源仅小幅增量，已搁置——未明确要求勿重跑）。
- **一次性探索脚本**（依赖未进 pyproject 时）用 `--no-project` 临时注入，依赖速记：数值计算 numpy,scipy,matplotlib,pandas / 符号计算 sympy / 深度学习 torch / 读表 openpyxl。
- 不要用裸 `python` / `pip install`，也不要手动建/激活 venv（`.venv` 由 uv 维护）。
- 脚本含中文输出时先设 `$env:PYTHONIOENCODING='utf-8'`；控制台是 PowerShell 5.1 且默认 GBK，列/搜中文文件名先设 `[Console]::OutputEncoding=[Text.Encoding]::UTF8`，读文件内容优先用 Read/Glob/Grep 工具而非 Get-Content。
- **报告耗时（用户要求，2026-09-13）**：每次运行程序后，在回复中说明该次运行的耗时（脚本自带计时输出的一并引用）。
- 一次性临时脚本写到 `C:\Users\sophia\AppData\Local\Temp\opencode`，不要散落在仓库根目录。
- **重跑防重复**：`q3.py`、`q4.py`、`q3_ablation.py`、`q2_adaptive.py`、`q2_arima.py`、`q2_tune.py`、`q3_sensitivity.py` 自带 `record()`（先删同名旧章节再追加）；其余脚本用追加模式的 `pm.record_result`，重跑前手动清理。
- **多进程**：`solve/` 里用 `multiprocessing.Pool`（Windows 为 spawn）时，入口必须由 `if __name__ == "__main__":` 保护，否则子进程会递归导入主模块。
- `pm.optimize.solve_lp` 只返回解与目标值，**不含对偶价格**；需要 LP 对偶/灵敏度时直接调 `scipy.optimize.linprog(method="highs")` 取 `eqlin.marginals`。

## 工作流（skill 流水线，按序调用）

总入口是 `1start-mathmodel` skill，它会先询问偏好（排版引擎 Typst/LaTeX、竞赛类型、论文语言、子问题数），生成 `plan.md` + `todo.md`，再按序串联各阶段 skill：

| 阶段 | skill | 关键产物 |
| --- | --- | --- |
| 1 | `1start-mathmodel` | `plan.md`、`todo.md` |
| 2 | `2analysis-modeling` | 赛题拆解、建模报告 |
| 3 | `3coding-visual` | 代码、运行结果、图表 PDF、`RESULTS_REPORT.md` |
| 4 | `4drawio` | 技术路线图/流程图 PDF |
| 5 | `5writing` | 论文（Typst 或 LaTeX 编译） |
| 6 | `6verity` | 章节完整/数值一致/编译/提交检查 |

- `1start-mathmodel` 的偏好问题必须先问用户，不要替用户假设竞赛类型和语言（MCM/ICM 强制英文，其余默认中文）。
- 共享规范库 `.agents/skills/_references/math_modeling_norms.md`（写作规范、题型防错、图表规范）在需要领域判断时按需读取。
- 各阶段 skill 声明的中间产物（`figures/`、`code/outputs/`、`RESULTS_REPORT.md` 等）是下一阶段输入，不要跳过；论文/图表里的一切数字必须能在 `reports/RESULTS_REPORT.md`、`results/` 或 `code/outputs/` 中找到来源，禁止编造或"估算"。
- `doctor` skill 检查环境（typst / xelatex / python3 / drawio / pdftoppm / mutool / magick + numpy/pandas/matplotlib 等），仅手动触发。
- 六阶段之外的辅助 skill：`mathmodel-figure-templates`（SHAP/ROC/云雨图等内置科研图表模板）、`scibox-diagram`（draw.io 技术路线图/流程图模板），需要时单独调用。
- **本机工具链 quirks（实测）**：
  - draw.io CLI 不在 PATH：在 `%LOCALAPPDATA%\Programs\draw.io\draw.io.exe`；导出用 `-x -f pdf --crop -o 输出.pdf 输入.drawio`。`scibox-diagram` 的 `export_figure.py` 依赖 PATH 上的 `drawio`，找不到时直接调 exe；五带模板渲染用 `roadmap_5band.py <content.json> -o <出图.drawio>`，体检用 `check_layout.py`。
  - `6verity` 的 `writing_check.sh` 需 **Git Bash**（`C:\Program Files\Git\bin\bash.exe`；系统 `bash` 是 WSL 空壳会失败），且脚本调用 `python3`——本机 Windows Store 别名会拦截，需做指向 `program/.venv/Scripts/python.exe` 的 `python3` shim 并加进 PATH。
  - `writing_check.sh` 只认相对 `.tex` 的显式图片路径（不解析 `\graphicspath`），论文 `\includegraphics` 一律写 `../figures/xxx.pdf`。

## 论文编译

- 模板都在 `.agents/skills/5writing/templates/{zh,en}/`，每竞赛一套 Typst 目录 + 一套 `-latex` 目录（14 中 + 3 英，`default` 为通用模板）。
- **本工作区 C 题论文**在 `paper/`（单文件 `main.tex` + `cumcmthesis.cls`/`cumcm_official_final.sty`，类文件复制自 `数学建模模板/`）。编译：`cd paper; xelatex -interaction=nonstopmode main.tex`（两遍）；成品 `paper/main.pdf`（当前 21 页），根目录另有副本 `论文_CUMCM2026_C题.pdf`；验收报告 `reports/VERIFY_REPORT.md`。**摘要用模板的 `abstract` 环境（minipage 不跨页）——写长了会整块跳到第 2 页，须控制在一页内。** 注意：若 PDF 被阅读器占用会编译失败（`xdvipdfmx: Unable to open...`）——先关闭阅读器；或改 `-jobname=tmp` 编译后复制替换。
- Typst 引擎：`typst compile main.typ`（排版语法问题用 `typst-author` skill）。
- LaTeX 引擎：`xelatex -interaction=nonstopmode main.tex`，**必须跑两遍**解决交叉引用（中文英文皆然）。
- 引擎选定后不要中途混用。

## 项目级 skill 与 agent

- skill 实体只存 `.agents/skills/`（skills CLI 的 canonical 位置，OpenCode 原生识别该路径）；`.opencode/skills` 是指向它的 Junction，仅为与 `.opencode/agents` 配套，编辑哪边都是同一份文件。
- 冗余镜像 `.claude/skills`、`agent/skills` 已删除；skills CLI 安装/更新时若重新生成它们，可直接删掉。
- 安装/更新走 skills CLI（维护 `skills-lock.json`）：`npx skills add <owner/repo@skill>`、`npx skills check`、`npx skills update`。
- 当前锁定来源：`jihe520/MathModelAgent`（9 个 skill）、`jihe520/sci-box`（图表/流程图模板 ×2）、`k-dense-ai/scientific-agent-skills`（sympy）。
- 项目级 agent：`.opencode/agents/mathmodel.md`（primary，数学建模专用，驱动六阶段流水线）；新增/修改 agent 后需重启 OpenCode 才生效。

## 本仓库自有资料

- `program/`：数学建模 Python 便利工具包（`config/dataio/plotting/stats/fitting/metrics/optimize/ml/ode/report/utils` 共 11 模块、100+ 公开函数，覆盖读写数据→建模→出图→记录结果全流程）。**写赛题代码前先读 `program/AGENTS.md`**（API 速查、输出约定、与 `3coding-visual` 的衔接、常见坑）；冒烟测试 `uv run --project program python program/tests/smoke_test.py`。
- `数学建模函数速查表.md`：写 scipy/sympy/pytorch 代码前先查。函数签名按 scipy 1.18 / sympy 1.14 / torch 2.14 实测提取，含常见坑（`dblquad` 参数顺序、`odeint` 的 `func(y,t)`、`CrossEntropyLoss` 输入是 logits、`solve_ivp` 的 `y` 形状为 `(n, m)` 等）。
- `时间序列模型详解.md`：ARIMA/SARIMA/ETS/VAR/状态空间/GARCH 的公式、定阶、残差诊断、滚动回测与论文写法（statsmodels 0.15 实测代码 + 版本坑清单）。写时序预测代码前先查。
- `建模参考资料/`：7 类题型速查（评价决策 / 优化规划 / 预测预报 / 分类聚类与统计 / 微分方程 / 图论网络 / 通用算法与仿真），README 含「选题速查表」；选型拿不准时先查。
- `数学建模模板/`：CUMCM 格式 LaTeX 论文模板（`cumcmthesis.cls` + 一键编译脚本；`main.tex` 为完整 A 题示例论文，可直接对照写法），含 `提交前必读.md`、`官方合规核对表.md`、`AI工具使用详情_填写说明.md`，论文/提交阶段可用。
