# program —— 数学建模便利工具包（AI 使用文档）

> **在本工作区编写赛题代码前，先读本文档。** 工具包已安装在本项目的 uv 环境中，
> 所有函数都保证：中文注释、固定随机种子、流程可复现、输出与竞赛目录结构对齐。
>
> 运行测试确认可用：`uv run python tests/smoke_test.py`（75 项检查）。

---

## 0. 快速开始

### 运行方式

```powershell
# 方式 A（推荐）：在赛题工作区根目录执行 —— 输出自动落在 工作区根/figures、reports、code/outputs
uv run --project program python program/src/program/problem1.py

# 方式 B：在 program/ 目录执行（根 AGENTS.md 的默认约定）—— 输出落在 program/ 下
uv run python src/program/problem1.py

# 方式 C（当前 C 题的包式入口）：代码在 program/src/solve/，在 program/ 下运行
uv run python -m solve        # 问题一
uv run python -m solve.q2     # 问题二
```

> 工具箱模块在 `program/src/program/`（本文档 API 所属）；赛题求解代码按当前约定放
> `program/src/solve/`（包），其输出根由 `solve/common.py` 固定解析为工作区根。
> 输出目录由「项目根」决定，项目根默认 = **运行脚本时的当前工作目录**，
> 可用 `pm.init(root=...)` 或环境变量 `MATHMODEL_ROOT` 覆盖（方式 B 想输出到
> 工作区根时写 `pm.init(root="..")`）。

### 标准脚本骨架

```python
"""问题一：XXX。运行：uv run --project program python program/src/program/problem1.py"""
import numpy as np
import pandas as pd
import program as pm

pm.init()                                     # ① 必做：中文字体 + 论文风格 + seed=42（可改）

df = pm.read_table("data/附件1.xlsx")          # ② 读数据（csv/xlsx/json/pkl，自动处理 GBK）
pm.record_result("数据概览", pm.summarize(df), level=2)   #    概览写进 RESULTS_REPORT

# ③ 建模与求解（示例：线性规划 + 图）
r = pm.optimize.solve_lp([3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True)
print(r.summary())
pm.record_result("问题一结果", {"最优目标值": r.fun, "决策变量": list(r.x)}, note=str(r))

# ④ 出图（保存 PDF 到 figures/，作图数据自动存到 code/outputs/figure_data/）
x = np.linspace(0, 10, 200)
pm.line(x, [np.exp(-0.5 * x), np.exp(-0.2 * x)], labels=["方案A", "方案B"],
        xlabel="时间 t/s", ylabel="响应 y", save="q1_response")
```

---

## 1. 路径与输出约定（与 1start-mathmodel 目录结构对齐）

| 函数 / 参数 | 默认输出 | 用途 |
| --- | --- | --- |
| `pm.save_fig(fig, "name")` / 各绘图函数 `save="name"` | `figures/name.pdf` | 论文图（矢量 PDF） |
| 同上，附加 `data=` 或 `save=` 自动记录 | `code/outputs/figure_data/name.csv` | 图表数据来源（可追溯） |
| `pm.record_result(title, data)` | `reports/RESULTS_REPORT.md`（追加） | 结果报告（论文唯一数值来源） |
| `pm.save_outputs(data, "name")` | `code/outputs/name.csv/.json/.pkl` | 中间结果、结果表 |
| `pm.DataCache("q1")` | `code/outputs/cache/q1/*.pkl` | 耗时计算的缓存（避免重复跑） |
| 日志 `pm.get_logger("q1", log_file=...)` | `code/outputs/...` | 运行日志 |

- 目录不存在会自动创建，无需手写 `os.makedirs`。
- **不要**把图存到别处再手动搬运；**不要**在论文里引用未记录来源的数字。

---

## 2. 模块速查

### 2.1 config（配置）与 utils（工具）

| API | 说明 |
| --- | --- |
| `pm.init(seed=42, root=None, chinese=True, backend="Agg", figsize=(7,4.3))` | 一键初始化；返回生效配置 dict |
| `pm.set_project_root(path)` / `pm.project_root()` | 手动设置/读取项目根 |
| `pm.figures_dir()` / `pm.reports_dir()` / `pm.outputs_dir()` | 输出目录 Path |
| `pm.set_seed(seed)` | 固定 random/numpy/torch 种子（`init` 已调用） |
| `pm.Timer("label")` | 计时上下文管理器/装饰器，`t.elapsed` 取秒数 |
| `pm.get_logger("q1", log_file=None)` | loguru 日志器（`log.info(...)`） |
| `pm.DataCache("q1")` | `cache.get_or_compute(key, func, *args)` 缓存耗时结果 |
| `pm.fmt(x, nd=4)` | 数值 → 论文友好字符串（自动科学计数法） |
| `pm.markdown_table(data, headers=None)` | dict/DataFrame/list → Markdown 表 |
| `pm.utils.env_markdown()` | 运行环境（Python 与库版本）Markdown 表 |

### 2.2 dataio（数据读写与清洗）

| API | 说明 |
| --- | --- |
| `pm.read_table(path, sheet=0, sep=None, encoding=None)` | 自动识别 csv/tsv/txt/xlsx/json/pkl；中文 GBK 自动回退 |
| `pm.read_sheets(path)` | 读 Excel 全部工作表 → `{表名: DataFrame}` |
| `pm.save_table(df, path, index=False)` | 写 csv（utf-8-sig）/xlsx/json/pkl |
| `pm.dataio.save_json(obj, path)` / `load_json(path)` | 处理 numpy 类型的 JSON |
| `pm.summarize(df)` | 数据概览 Markdown（形状/列信息/前 5 行）→ 给 `pm.record` |
| `pm.dataio.missing_report(df)` / `column_report(df)` | 缺失 / 列级统计表 |
| `pm.dataio.fill_missing(df, strategy="auto")` | 缺失填充：auto=数值均值+类别众数；支持 mean/median/mode/zero/ffill/bfill/drop |
| `pm.dataio.clean_columns(df, lower=False)` | 列名去空格、空列名替换 |
| `pm.dataio.split_xy(df, target, drop=None)` | → `(X, y)` |
| `pm.dataio.train_test_split_df(df, target=None, test_size=0.2, seed=42, stratify=False)` | target=None 返回 `(train, test)`；否则返回 `X_train, X_test, y_train, y_test` |
| `pm.dataio.to_numpy(df, numeric_only=True)` | DataFrame → float ndarray |

### 2.3 plotting（绘图，论文级）

> 所有函数返回 `(fig, ax)`（`scatter_fit` 多一个 info，`dual_axis` 多 ax2，`roc` 多 auc）。
> 传入 `save="图名"` 即保存 `figures/图名.pdf` 并记录数据；**不在图内写标题**（交给论文 caption）。

| API | 说明 |
| --- | --- |
| `pm.save_fig(fig, name, formats=("pdf",), dpi=300, data=None)` | 底层保存；`formats=("pdf","png")` 出预览图 |
| `pm.line(x, ys, labels=[...], xlabel, ylabel, marker=None, save=None, ax=None)` | 折线，ys 可为 2D 数组/列表多条 |
| `pm.scatter(x, y, ...)` / `pm.scatter_fit(x, y, degree=1, ...)` | 散点 / 散点+拟合（返回 `info={"coef","r2","rmse"}`） |
| `pm.bar(labels, values, errors=None, orient="v", value_labels=False)` | 柱状/条形（`orient="h"`） |
| `pm.bar_group(labels, {"方法A": [...], "方法B": [...]})` | 分组柱状（模型对比） |
| `pm.hist(x, bins=30, kde=True)` / `pm.box({"组名": 数据})` | 分布图 |
| `pm.heatmap(matrix, annot=False, cmap="RdBu_r")` | 热力图（DataFrame 自动取行列名） |
| `pm.corr_heatmap(df, method="pearson")` | 相关矩阵热力图（固定 -1~1 色阶） |
| `pm.radar(labels, {"方案A": [...]}, normalize=False)` | 雷达图（综合评价） |
| `pm.dual_axis(x, y1, y2, labels=("左","右"))` | 双纵轴（量纲不同） |
| `pm.pie(labels, values)` / `pm.errorbar(x, y, yerr)` | 占比 / 误差棒 |
| `pm.surface3d(X, Y, Z, xlabel, ylabel, zlabel)` | 三维曲面（敏感性/调参景观） |
| `pm.roc(y_true, y_score)` | 二分类 ROC（返回 `(fig, ax, auc)`） |

### 2.4 stats（统计检验，中文结论）

> 检验结果统一为 `TestResult`：`print(r)` 直接给结论；`r.pvalue`、`r.significant`、`r.extra` 可用。

| API | 说明 |
| --- | --- |
| `pm.stats.normality(x)` | 正态性（n≤5000 Shapiro-Wilk，否则 D'Agostino） |
| `pm.stats.t_test_ind(a, b)` / `t_test_paired(a, b)` / `t_test_one(x, mu=0)` | t 检验（默认 Welch，extra 含 Cohen's d） |
| `pm.stats.anova(*groups)` / `kruskal(*groups)` | 方差分析 / 非参数替代 |
| `pm.stats.compare_groups(*groups)` | 自动选择 ANOVA 或 Kruskal（推荐） |
| `pm.stats.chi2_test(table)` | 列联表卡方（extra 含 Cramér's V） |
| `pm.stats.corr_test(x, y, method="pearson")` / `corr_matrix(df)` | 相关检验 / 相关矩阵 |
| `pm.stats.linregress_summary(x, y)` | 一元线性回归（斜率/截距/R²/p，可 `.predict`） |
| `pm.stats.ols(df, y="销量", xs=[...])` / `ols_text(...)` | 多元回归（statsmodels，`.summary()` 完整摘要） |
| `pm.stats.ci_mean(x)` / `bootstrap_ci(x, stat_fn=np.mean, seed=42)` | 置信区间 |

### 2.5 fitting（拟合/插值/平滑）

| API | 说明 |
| --- | --- |
| `fitting.fit_curve(func, x, y, p0, bounds=..., names=[...])` | 任意函数拟合 → `FitResult`（`.predict(x)` / `.summary()` / `.param_dict()`） |
| `fitting.polyfit(x, y, degree)` | 多项式拟合（`FitResult`，params 降幂） |
| `fitting.poly_str(coef)` | 系数 → 表达式字符串（LaTeX 公式可用） |
| `fitting.poly_scan(x, y, max_degree=5)` | 自动定阶对比表（r2/adj_r2/rmse/aic） |
| `fitting.interp(x, y, x_new, kind="cubic")` | 插值：linear/cubic/pchip/nearest |
| `fitting.spline_smooth(x, y, s=None)` | 平滑样条（返回可调用对象） |
| `fitting.smooth(y, window, method="savgol")` | 平滑：savgol/moving/ewma |

### 2.6 metrics（评估指标）

| API | 说明 |
| --- | --- |
| `pm.metrics.regression_metrics(y, yhat, n_features=None)` | MAE/MSE/RMSE/MAPE/SMAPE/R²/NSE/AdjR² → dict |
| `pm.metrics.classification_metrics(y, yhat, y_proba=None)` | accuracy/precision/recall/F1(macro,weighted)/AUC |
| `pm.metrics.confusion_table(y, yhat)` | 混淆矩阵 DataFrame |
| `pm.metrics.cluster_metrics(X, labels)` | 轮廓系数/CH/DB |
| `pm.metrics.str_metrics(d)` | 指标 dict → 多行文本（贴报告） |

### 2.7 optimize（优化）

| API | 说明 |
| --- | --- |
| `pm.optimize.solve_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, maximize=False, integer=None)` | LP/MILP（HiGHS）；`integer=True/[0,1,...]` 指定整数变量 |
| `pm.optimize.solve_nlp(func, x0, bounds, constraints, maximize=False)` | 非线性规划（minimize 包装） |
| `pm.optimize.global_minimize(func, bounds, seed=42)` | 差分进化全局优化（可复现） |
| `pm.optimize.knapsack(values, weights, capacity)` | 0-1 背包（整数权重） |
| `pm.optimize.assignment(cost)` | 指派问题（匈牙利） |
| `pm.optimize.distance_matrix(points)` | 坐标 → 距离矩阵 |
| `pm.optimize.tsp_nearest_neighbor(dist)` / `tsp_2opt(dist)` | TSP 构造 + 2-opt 改进（返回 `(tour, length)`） |

结果统一为 `OptResult`：`r.x`、`r.fun`、`r.success`、`r.summary()`。

### 2.8 ml（机器学习快捷流程）

| API | 说明 |
| --- | --- |
| `pm.ml.fit_regressor(model, X, y, test_size=0.2, cv=5, seed=42, scale="auto")` | 训练+CV+测试评估 → `TrainResult`（`.model/.test_metrics/.y_test/.y_pred/.summary()`） |
| `pm.ml.fit_classifier(model, X, y, ...)` | 同上（默认分层划分，含 `y_proba`） |
| `pm.ml.compare_models(["linear","ridge","rf"], X, y, task="regression")` | 多模型对比表 |
| `pm.ml.cross_validate(model, X, y, cv=5)` | 交叉验证（dict：mean/std/scores） |
| `pm.ml.kmeans(X, k, seed=42)` / `elbow(X, k_range)` | 聚类 / 肘部扫描 |
| `pm.ml.pca(X, n_components=2)` | 主成分（scores/loadings/解释率） |
| `pm.ml.standardize(X)` | Z-score → `(标准化数据, scaler)` |

可用模型字符串 —— 回归：`linear/ridge/lasso/rf/gbr/svr/knn/mlp`；分类：`logreg/rf/dt/svc/knn/mlp`。

### 2.9 ode（微分方程）

| API | 说明 |
| --- | --- |
| `pm.ode.solve_ivp_model(func, y0, t_span, params=(...), t_eval=..., method="RK45")` | 求解，`func(t, y, *params)`；返回 OdeResult（`sol.y` 形状 `(变量, 时间)`） |
| `pm.ode.fit_ode_params(func, y0, t_data, y_data, p0, bounds=..., param_names=[...])` | 参数反演 → `ODEFitResult`（`.summary()` / `.simulate(t)`） |

### 2.10 report（结果记录与论文素材）

| API | 说明 |
| --- | --- |
| `pm.record(text)` | 向 `reports/RESULTS_REPORT.md` 追加一段 Markdown |
| `pm.record_result(title, data, note=None, level=3)` | 标题 + 表格（dict 标量 → 指标表；DataFrame → 数据表） |
| `pm.record_environment()` | 写入运行环境（Python/库版本） |
| `pm.save_outputs(data, name, fmt=None)` | 结果落盘 `code/outputs/`（DataFrame→csv，dict→json） |
| `pm.to_latex(df, caption=..., label=...)` | DataFrame → LaTeX 表格（论文直接可用） |

---

## 3. 与 `3coding-visual` skill 的衔接（每个子问题必须完成）

1. **读数据** → `pm.read_table` + `pm.summarize`。
2. **建模求解** → 对应模块（optimize / ml / stats / ode / fitting），随机过程必有 `seed`。
3. **验证约束** → 手动检查并 `pm.record_result("约束校验", {...})`。
4. **出图** → ≥2 张图（如结果对比 + 灵敏度/收敛），`save=` 输出 PDF 到 `figures/`。
5. **记录结果** → `pm.record_result(...)` 写入 `reports/RESULTS_REPORT.md`：
   方法、关键数值（含单位）、图表文件名、校验结论。
6. **中间数据** → `pm.save_outputs(...)` 或 `save=` 自带的数据记录。

> 论文中的一切数字必须能在 `RESULTS_REPORT.md` / 结果表 / 图表数据中找到来源，禁止编造。

---

## 4. 常见坑（务必避免）

1. **先 `pm.init()`**：不初始化则中文乱码（方框）、负号显示异常。
2. **不要用 `df.to_markdown()`**：未安装 tabulate；用 `pm.markdown_table(df)`。
3. **ODE 函数签名是 `f(t, y, *params)`**（`solve_ivp` 版），不是 `odeint` 的 `f(y, t)`。
4. **优化默认最小化**：求最大传 `maximize=True`；变量默认非负，负下界要显式传
   `bounds=[(None, None), ...]`。
5. **`solve_lp` 整数变量**用 `integer=True`（全整数）或 `integer=[1,0,...]`（逐变量，
   0-1 变量配合 `bounds=[(0,1),...]`）。
6. **随机过程必须固定种子**：`pm.init(seed=...)` 或函数 `seed=` 参数；不要在多次运行间
   改种子还混用结果。
7. **pandas 3.x**：不要用 `df.append` / `inplace=True` 旧写法（CoW 模式）。
8. **图表不要写 `plt.title`**：标题由论文 caption 承担（skill 规定）。
9. **结果先记录再引用**：论文写作阶段只认 `RESULTS_REPORT.md` 里的数值。
10. **编号命名规范化**：图名用 `q1_xxx`、`q2_xxx`，结果文件用 `q1_xxx.csv`，
    便于论文引用与验收。

---

## 5. 完整示例（回归预测 + 统计 + 出图 + 报告）

```python
"""问题二：销量预测。运行：uv run --project program python program/src/program/problem2.py"""
import numpy as np
import pandas as pd
import program as pm
from program import fitting

pm.init(seed=0)
log = pm.get_logger("q2")

# 数据
df = pm.read_table("data/sales.xlsx")
df = pm.dataio.clean_columns(df)
df = pm.dataio.fill_missing(df, strategy="median", cols=["广告费"])

# 建模：多元回归 + 随机森林对比
ols = pm.stats.ols(df, y="销量", xs=["价格", "广告费", "季节指数"])
log.info(f"OLS R² = {ols.rsquared:.4f}")
res = pm.ml.fit_regressor("rf", df[["价格", "广告费", "季节指数"]], df["销量"], cv=5, seed=0)
pm.record_result("模型对比", pd.DataFrame({
    "模型": ["OLS", "随机森林"],
    "R2": [ols.rsquared, res.test_metrics["R2"]],
    "RMSE": [np.sqrt(ols.mse_resid), res.test_metrics["RMSE"]],
}))

# 图：预测 vs 真实 + 残差分布
pm.scatter(res.y_test, res.y_pred, xlabel="真实销量", ylabel="预测销量", save="q2_pred_vs_true")
pm.hist(res.y_test - res.y_pred, xlabel="残差", save="q2_residual")

# 灵敏度：广告费 ±20% 对预测的影响（重新预测）
for k, tag in [(0.8, "低"), (1.2, "高")]:
    X = res.X_test.copy()
    X["广告费"] = X["广告费"] * k
    pm.save_outputs(pd.DataFrame({"场景": [tag], "预测均值": [np.mean(res.model.predict(X))]}),
                    f"q2_sens_{tag}")

pm.record_environment()
```

---

## 6. 维护与测试

- 冒烟测试：`uv run python tests/smoke_test.py`（覆盖全部 11 模块、75 项断言）。
- 人类可读手册：`docs/`（MkDocs + Material）—— `uv run mkdocs build` 产出 `site/`（已 gitignore），
  或 `uv run mkdocs serve` 预览；示例图由 `uv run python docs/generate_assets.py` 再生。
  改动模块 API 后同步更新 `docs/` 与本文档。
- 新增依赖：`uv add <pkg>` 并提交 `pyproject.toml` / `uv.lock`。
- 模块清单：`uv run program`。
