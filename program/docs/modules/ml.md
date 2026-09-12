# 机器学习 ml

预测 / 分类 / 聚类类赛题的快速通道：模型用字符串点名，训练、交叉验证、
标准化、评估一条龙，所有随机性都固定种子。

## 回归：一行完成训练 + 评估

```python
import program as pm
from program.dataio import split_xy

X, y = split_xy(df, target="销量")

res = pm.ml.fit_regressor("rf", X, y, test_size=0.2, cv=5, seed=42)
print(res.summary())
# 【回归】交叉验证（5 折，r2）：0.9321 ± 0.018
# 测试集指标：
#   MAE = 0.1523
#   RMSE = 0.1912
#   R2 = 0.9614
# ...

# 画预测效果（数据直接取训练结果，不用重跑）
pm.scatter(res.y_test, res.y_pred, xlabel="真实值", ylabel="预测值", save="q2_pred")
pm.hist(res.y_test - res.y_pred, xlabel="残差", save="q2_residual")
```

可用的模型字符串：

| 任务 | 可选模型 |
| --- | --- |
| 回归 | `linear` `ridge` `lasso` `rf`（随机森林） `gbr`（梯度提升） `svr` `knn` `mlp` |
| 分类 | `logreg` `rf` `dt`（决策树） `svc` `knn` `mlp` |

也可以传入任意自定义 sklearn 估计器（`fit_regressor(MyModel(), X, y)`）。

## 模型对比：给论文一张表

```python
table = pm.ml.compare_models(
    ["linear", "ridge", "rf", "gbr"], X, y,
    task="regression", cv=5, seed=42,
)
pm.record_result("模型对比", table)              # 直接进报告
pm.bar_group(table["模型"], {"R²": table["测试集_R2"]}, save="q2_compare")
```

## 分类

```python
res = pm.ml.fit_classifier("logreg", X, y, seed=42)
print(res.test_metrics)                          # accuracy / F1 / AUC...
pm.plotting.roc(res.y_test, res.y_proba[:, 1], save="q2_roc")
pm.record_result("分类结果", pm.metrics.confusion_table(res.y_test, res.y_pred))
```

## 聚类与降维

```python
# K-means：返回标签、原始尺度质心、轮廓系数等指标
cl = pm.ml.kmeans(X, k=3, seed=42)
print(cl.summary())

# 定 k：肘部扫描（inertia + 轮廓系数随 k 变化）
table = pm.ml.elbow(X, k_range=range(2, 11))
pm.line(table["k"], [table["inertia"]], xlabel="簇数 k", ylabel="inertia", save="q3_elbow")

# PCA：得分矩阵、载荷矩阵、累计方差解释率
pc = pm.ml.pca(df_numeric, n_components=3)
print(pc.summary())
pm.scatter(pc.scores[:, 0], pc.scores[:, 1], xlabel="PC1", ylabel="PC2", save="q3_pca")
```

## 小贴士

!!! tip "标准化不用自己记"
    `fit_regressor/fit_classifier` 的 `scale="auto"` 会对 SVR、KNN、MLP、逻辑回归等
    自动加 `StandardScaler` 管道，其余模型不动。想强制开关就传 `scale=True/False`。

!!! warning "交叉验证 < 报告的纪律"
    论文里报告模型性能时，**必须**同时给出交叉验证均值±标准差或测试集指标，
    不能只报训练集拟合效果。`res.cv_scores` 与 `res.test_metrics` 两个都有。

!!! note "数据泄漏自查"
    标准化、特征选择必须放在训练集上做（本库的管道已处理）；
    如果自己手写流程，切记先划分再 `fit` 预处理。
