# 实战食谱

四类常见赛题的「抄作业」骨架。把骨架里的列名、参数换成题目实际内容即可，
其余部分（记录、出图、种子）已经按竞赛要求配好。

---

## 🍚 套路一：预测类

> 特征 → 目标值；模型对比 → 评估 → 残差诊断。

```python
"""预测类赛题通用骨架。"""
import numpy as np
import program as pm

pm.init(seed=42)

# 1. 数据理解
df = pm.read_table("data/附件1.xlsx")
df = pm.dataio.clean_columns(df)
df = pm.dataio.fill_missing(df)
pm.record("## 数据理解\n\n" + pm.summarize(df))

X, y = pm.dataio.split_xy(df, target="目标列")

# 2. 多模型对比（5 折交叉验证 + 测试集）
table = pm.ml.compare_models(
    ["linear", "ridge", "rf", "gbr"], X, y, task="regression", cv=5, seed=42,
)
pm.record_result("模型对比", table)

# 3. 用最优模型出最终结果
best = str(table.iloc[0]["模型"])
res = pm.ml.fit_regressor(best, X, y, test_size=0.2, cv=5, seed=42)
pm.record_result("最终模型指标", res.test_metrics, note=f"模型：{best}")

# 4. 诊断图（论文标配：预测对比 + 残差 + 时序）
pm.scatter(res.y_test, res.y_pred, xlabel="真实值", ylabel="预测值", save="q2_pred_vs_true")
resid = np.asarray(res.y_test) - res.y_pred
pm.hist(resid, xlabel="残差", save="q2_residual")
pm.line(np.arange(len(resid)), [np.asarray(res.y_test), res.y_pred],
        labels=["真实值", "预测值"], xlabel="样本序号", ylabel="目标值", save="q2_series")
```

---

## 🍜 套路二：优化类

> 建目标函数与约束 → 求解 → 可行性校验 → 灵敏度分析。

```python
"""优化类赛题通用骨架。"""
import numpy as np
import pandas as pd
import program as pm

pm.init(seed=42)

c = [3, 5]                                  # 目标系数（换成题目实际数据）
A_ub = np.array([[1, 0], [0, 2], [3, 2]])   # 约束矩阵
b_ub = np.array([4, 12, 18])

# 1. 求解
r = pm.optimize.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, maximize=True)
assert r.success, r.message

# 2. 可行性校验（先可行、再最优）
violation = float(np.max(A_ub @ r.x - b_ub))
pm.record_result(
    "问题一结果",
    {f"x{i + 1}": v for i, v in enumerate(r.x)} | {"目标值": r.fun},
    note=f"最大约束违反量 = {violation:.2e}（≤ 0 即全部满足）",
)

# 3. 灵敏度：逐个关键参数 ±10%
rows = []
for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
    r2 = pm.optimize.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub * (1 + delta), maximize=True)
    rows.append({"扰动": f"{delta:+.0%}", "最优目标值": r2.fun})
sens = pd.DataFrame(rows)
pm.record_result("灵敏度分析", sens)
pm.line(sens["扰动"], sens["最优目标值"], marker="o",
        xlabel="约束参数扰动", ylabel="最优目标值", save="q1_sensitivity")

# 4. 方案对比
pm.bar(["原方案", "优化方案"], [28.0, r.fun], value_labels=True,
       ylabel="总收益/万元", save="q1_compare")
```

离散问题（背包 / 指派 / TSP）见 [优化手册](modules/optimize.md)。

---

## 🍣 套路三：评价类

> 指标归一化 → 定权重（熵权法）→ 综合得分 → 排序与可视化。

```python
"""评价类赛题通用骨架：熵权法 + 综合得分。"""
import numpy as np
import pandas as pd
import program as pm

pm.init()

names = ["方案A", "方案B", "方案C", "方案D"]        # 评价对象
indicators = ["成本", "效率", "稳定性", "环保性"]     # 指标名
raw = np.array([...])                               # 行=方案，列=指标
directions = np.array([-1, 1, 1, 1])                # 1 效益型，-1 成本型

# 1. min-max 归一化（成本型指标反向）
Z = (raw - raw.min(0)) / (raw.max(0) - raw.min(0))
Z[:, directions < 0] = 1 - Z[:, directions < 0]

# 2. 熵权法
P = (Z + 1e-12) / (Z + 1e-12).sum(0)
E = -(P * np.log(P)).sum(0) / np.log(len(Z))
W = (1 - E) / (1 - E).sum()

# 3. 得分与排名
score = Z @ W
rank = pd.Series(score, index=names).rank(ascending=False).astype(int)
table = pd.DataFrame({"方案": names, "综合得分": score, "排名": rank})
pm.record_result("综合评价结果", table)
pm.record_result("权重", pd.DataFrame({"指标": indicators, "权重": W}))

# 4. 图：得分排序 + 雷达
pm.bar(names, score, value_labels=True, ylabel="综合得分", save="q3_score")
pm.radar(indicators, {n: Z[i].tolist() for i, n in enumerate(names)}, save="q3_radar")
```

> 权重也可以换层次分析法（AHP）或主客观组合权重，得分计算框架不变。

---

## 🍱 套路四：分类 / 识别类

```python
"""分类赛题通用骨架。"""
import numpy as np
import program as pm

pm.init(seed=42)

df = pm.read_table("data/train.xlsx")
X, y = pm.dataio.split_xy(df, target="标签")

# 1. 多模型对比（默认分层划分）
table = pm.ml.compare_models(["logreg", "rf", "svc", "knn"], X, y,
                             task="classification", cv=5, seed=42)
pm.record_result("模型对比", table)

# 2. 最终模型
res = pm.ml.fit_classifier("rf", X, y, cv=5, seed=42)
pm.record_result("测试集指标", res.test_metrics)
pm.record_result("混淆矩阵", pm.metrics.confusion_table(res.y_test, res.y_pred))

# 3. ROC（二分类；多分类可对每个类做 one-vs-rest）
if res.y_proba is not None and len(np.unique(y)) == 2:
    pm.roc(res.y_test, res.y_proba[:, 1], save="q2_roc")
```

!!! tip "四个套路共通的三件事"
    1. **种子**：所有随机过程都有 `seed=42`，同一脚本任何时候跑结果一致；
    2. **记录**：每个关键数值都经过 `record_result` 进入 `RESULTS_REPORT.md`；
    3. **出图**：`save=` 的图自动进 `figures/`，作图数据自动进 `code/outputs/figure_data/`。
