# 统计 · 拟合 · 评估

这一页覆盖三个紧密相关的模块：`stats`（假设检验与回归）、`fitting`（曲线拟合与插值）、
`metrics`（模型评估指标）。数据分析类、预测类赛题的主战场。

## 统计检验：先问「该用哪个」

| 你的问题 | 用哪个函数 | 说明 |
| --- | --- | --- |
| 数据是不是正态分布？ | `pm.stats.normality(x)` | n≤5000 用 Shapiro-Wilk |
| 两组均值有没有差异？ | `pm.stats.t_test_ind(a, b)` | 默认 Welch（不假设方差齐性） |
| 同一批对象前后对比？ | `pm.stats.t_test_paired(a, b)` | 配对 t 检验 |
| 三组以上均值差异？ | `pm.stats.compare_groups(a, b, c)` | **自动**在 ANOVA / Kruskal 间选择 |
| 两个分类变量有关联？ | `pm.stats.chi2_test(table)` | 列联表卡方 |
| 两变量相关程度？ | `pm.stats.corr_test(x, y)` | pearson / spearman / kendall |
| 某个指标随另一个变量怎么变？ | `pm.stats.linregress_summary(x, y)` | 一元线性回归（含 R²、p 值） |
| 多个因子的影响大小？ | `pm.stats.ols(df, y, xs)` | 多元回归，`.summary()` 完整统计表 |

所有检验返回统一的 `TestResult`，`print` 一下就是中文结论：

```python
import program as pm

r = pm.stats.t_test_ind(group_a, group_b)
print(r)
# 【独立双样本 t 检验（Welch）】统计量 = 3.92，p = 1.3e-4 < α = 0.05 → 拒绝 H0（H0：两组均值相等）
# 附：均值A = 12.3，均值B = 13.1，nA = 60，nB = 60，Cohen's d = 0.72
```

```python
# 多元线性回归：完整统计摘要可直接贴报告
model = pm.stats.ols(df, y="销量", xs=["价格", "广告费", "促销力度"])
print(model.summary())          # 或 pm.stats.ols_text(df, "销量", [...])
```

## 曲线拟合：`fit_curve` 是万能入口

```python
import numpy as np
from program import fitting

def logistic(x, K, r, x0):                    # 自定义模型函数
    return K / (1 + np.exp(-r * (x - x0)))

result = fitting.fit_curve(
    logistic, x, y,
    p0=[100, 0.5, 20],                        # 初值（重要）
    bounds=([0, 0, 0], [1e4, 10, 100]),       # 物理约束（推荐）
    names=["K", "r", "x0"],
)
print(result.summary())
# 拟合结果（logistic）：R² = 0.9987，RMSE = 0.42，n = 50
#   K = 98.32 ± 0.51
#   r = 0.512 ± 0.02
#   x0 = 19.8 ± 0.3

y_hat = result.predict(x_new)                 # 用拟合结果预测
```

**多项式场景有专门工具**：

```python
r = fitting.polyfit(x, y, degree=2)           # 多项式拟合
print(fitting.poly_str(r.params))             # 'y = 0.81x^2 - 3.2x + 4.6' 形式
table = fitting.poly_scan(x, y, max_degree=5) # 自动定阶对比表（R²/adj-R²/AIC）
```

**插值与平滑**：

```python
y_smooth = fitting.interp(x, y, x_new, kind="cubic")   # cubic / pchip / linear
spl = fitting.spline_smooth(x, y, s=0.01)               # 平滑样条（返回可调用）
y_sg = fitting.smooth(y_noisy, window=7, method="savgol")  # 保峰形滤波
```

## 模型评估：数字要写全

```python
m = pm.metrics.regression_metrics(y_true, y_pred, n_features=5)
print(pm.metrics.str_metrics(m))
#   MAE = 0.1523
#   RMSE = 0.1912
#   MAPE(%) = 4.12
#   R2 = 0.9614
#   AdjR2 = 0.9588
#   ...

# 分类任务
mc = pm.metrics.classification_metrics(y_true, y_pred, y_proba)
pm.metrics.confusion_table(y_true, y_pred)    # 混淆矩阵 DataFrame
```

!!! tip "指标怎么选"
    - 预测类：MAE / RMSE + R²（论文里至少给两个，量纲解释用 RMSE）；
    - 有零点或相对误差更重要：MAPE / SMAPE；
    - 分类不均衡数据：别只看 accuracy，用 F1-macro / AUC。

## 函数速查

| 模块 | 函数 | 说明 |
| --- | --- | --- |
| stats | `normality / t_test_one / t_test_ind / t_test_paired` | 正态性、t 检验族 |
| stats | `anova / kruskal / compare_groups` | 多组比较 |
| stats | `chi2_test / corr_test / corr_matrix` | 卡方、相关 |
| stats | `linregress_summary / ols / ols_text` | 回归 |
| stats | `ci_mean / bootstrap_ci` | 置信区间（bootstrap 可对任意统计量） |
| fitting | `fit_curve / polyfit / poly_str / poly_scan` | 拟合 |
| fitting | `interp / spline_smooth / smooth` | 插值、平滑 |
| metrics | `regression_metrics / classification_metrics` | 指标 dict |
| metrics | `confusion_table / cluster_metrics / str_metrics` | 混淆矩阵、聚类、格式化 |
