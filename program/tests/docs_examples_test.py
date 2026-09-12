"""验证 docs/cookbook.md 中的核心示例代码可运行（预测 / 优化 / 评价三套骨架）。

运行::

    uv run python tests/docs_examples_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import program as pm

tmp = Path(tempfile.mkdtemp(prefix="docs_examples_"))
pm.init(root=tmp, seed=42, verbose=False)
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ 预测类骨架
n = 150
df = pd.DataFrame(
    {
        "价格": rng.uniform(5, 20, n),
        "广告费": rng.uniform(0, 10, n),
        "促销力度": rng.uniform(0, 1, n),
    }
)
df["销量"] = 80 - 2 * df["价格"] + 3 * df["广告费"] + 10 * df["促销力度"] + rng.normal(0, 3, n)
pm.record("## 数据理解\n\n" + pm.summarize(df))

X, y = pm.dataio.split_xy(df, target="销量")
table = pm.ml.compare_models(["linear", "ridge", "rf", "gbr"], X, y, task="regression", cv=5, seed=42)
pm.record_result("模型对比", table)
best = str(table.iloc[0]["模型"])
res = pm.ml.fit_regressor(best, X, y, test_size=0.2, cv=5, seed=42)
pm.record_result("最终模型指标", res.test_metrics, note=f"模型：{best}")
pm.scatter(res.y_test, res.y_pred, xlabel="真实值", ylabel="预测值", save="q2_pred_vs_true")
resid = np.asarray(res.y_test) - res.y_pred
pm.hist(resid, xlabel="残差", save="q2_residual")
pm.line(np.arange(len(resid)), [np.asarray(res.y_test), res.y_pred],
        labels=["真实值", "预测值"], xlabel="样本序号", ylabel="销量", save="q2_series")
assert res.test_metrics["R2"] > 0.8, res.test_metrics
print("OK  预测类骨架")

# ------------------------------------------------------------------ 优化类骨架
c = [3, 5]
A_ub = np.array([[1, 0], [0, 2], [3, 2]])
b_ub = np.array([4, 12, 18])
r = pm.optimize.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub, maximize=True)
assert r.success, r.message
violation = float(np.max(A_ub @ r.x - b_ub))
pm.record_result(
    "问题一结果",
    {f"x{i + 1}": v for i, v in enumerate(r.x)} | {"目标值": r.fun},
    note=f"最大约束违反量 = {violation:.2e}（≤ 0 即全部满足）",
)
rows = []
for delta in (-0.10, -0.05, 0.0, 0.05, 0.10):
    r2 = pm.optimize.solve_lp(c=c, A_ub=A_ub, b_ub=b_ub * (1 + delta), maximize=True)
    rows.append({"扰动": f"{delta:+.0%}", "最优目标值": r2.fun})
sens = pd.DataFrame(rows)
pm.record_result("灵敏度分析", sens)
pm.line(sens["扰动"], sens["最优目标值"], marker="o",
        xlabel="约束参数扰动", ylabel="最优目标值", save="q1_sensitivity")
pm.bar(["原方案", "优化方案"], [28.0, r.fun], value_labels=True,
       ylabel="总收益/万元", save="q1_compare")
assert abs(r.fun - 36) < 1e-8
print("OK  优化类骨架")

# ------------------------------------------------------------------ 评价类骨架
names = ["方案A", "方案B", "方案C", "方案D"]
indicators = ["成本", "效率", "稳定性", "环保性"]
raw = rng.uniform(1, 10, (4, 4))
directions = np.array([-1, 1, 1, 1])

Z = (raw - raw.min(0)) / (raw.max(0) - raw.min(0))
Z[:, directions < 0] = 1 - Z[:, directions < 0]
P = (Z + 1e-12) / (Z + 1e-12).sum(0)
E = -(P * np.log(P)).sum(0) / np.log(len(Z))
W = (1 - E) / (1 - E).sum()
score = Z @ W
rank = pd.Series(score, index=names).rank(ascending=False).astype(int)
table = pd.DataFrame({"方案": names, "综合得分": score, "排名": rank})
pm.record_result("综合评价结果", table)
pm.record_result("权重", pd.DataFrame({"指标": indicators, "权重": W}))
pm.bar(names, score, value_labels=True, ylabel="综合得分", save="q3_score")
pm.radar(indicators, {nm: Z[i].tolist() for i, nm in enumerate(names)}, save="q3_radar")
assert abs(W.sum() - 1) < 1e-9
print("OK  评价类骨架")

# ------------------------------------------------------------------ 分类类骨架
dfc = pd.DataFrame(rng.normal(size=(160, 3)), columns=["a", "b", "c"])
dfc["标签"] = (dfc["a"] + dfc["b"] > 0).astype(int)
Xc, yc = pm.dataio.split_xy(dfc, target="标签")
table_c = pm.ml.compare_models(["logreg", "rf", "knn"], Xc, yc, task="classification", cv=5, seed=42)
res_c = pm.ml.fit_classifier("rf", Xc, yc, cv=5, seed=42)
pm.record_result("测试集指标", res_c.test_metrics)
pm.record_result("混淆矩阵", pm.metrics.confusion_table(res_c.y_test, res_c.y_pred))
pm.roc(res_c.y_test, res_c.y_proba[:, 1], save="q2_roc")
assert res_c.test_metrics["accuracy"] > 0.9
print("OK  分类类骨架")

# ------------------------------------------------------------------ 产物检查
report = (tmp / "reports" / "RESULTS_REPORT.md").read_text(encoding="utf-8")
figs = sorted(p.name for p in (tmp / "figures").glob("*.pdf"))
fdata = sorted(p.name for p in (tmp / "code" / "outputs" / "figure_data").glob("*.csv"))
assert "模型对比" in report and "综合评价结果" in report
assert len(figs) == 8, figs
assert len(fdata) == 8, fdata
print(f"OK  报告与图表产物（{len(figs)} 图 / {len(fdata)} 数据）")
print("\n✅ 文档示例代码全部可运行")
