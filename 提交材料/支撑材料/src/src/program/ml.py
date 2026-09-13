"""机器学习快捷流程：训练 / 交叉验证 / 评估一体化。

常用模型可用字符串直接指定，内部自动划分数据、标准化（按需）、
交叉验证并给出测试集指标，每个环节都固定随机种子。

- 回归：``"linear" "ridge" "lasso" "rf" "gbr" "svr" "knn" "mlp"``
- 分类：``"logreg" "rf" "dt" "svc" "knn" "mlp"``
- 无监督：:func:`kmeans` / :func:`elbow` / :func:`pca`

用法::

    import program as pm

    r = pm.ml.fit_regressor("rf", X, y, test_size=0.2, cv=5, seed=42)
    print(r.summary())
    print(pm.metrics.str_metrics(r.test_metrics))

    c = pm.ml.fit_classifier("logreg", X, y)
    pm.plotting.roc(y_test, proba, save="q2_roc")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .metrics import classification_metrics, cluster_metrics, regression_metrics, str_metrics
from .utils import fmt, set_seed


# ---------------------------------------------------------------------------
# 模型工厂
# ---------------------------------------------------------------------------
def _regressor_table(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.01, max_iter=10000),
        "rf": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "gbr": GradientBoostingRegressor(random_state=seed),
        "svr": SVR(),
        "knn": KNeighborsRegressor(),
        "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=seed),
    }


def _classifier_table(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "logreg": LogisticRegression(max_iter=2000, random_state=seed),
        "rf": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "dt": DecisionTreeClassifier(random_state=seed),
        "svc": SVC(probability=True),
        "knn": KNeighborsClassifier(),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=seed),
    }


_SCALE_NEEDED = {
    "SVR", "KNeighborsRegressor", "MLPRegressor",
    "LogisticRegression", "SVC", "KNeighborsClassifier", "MLPClassifier",
}


def make_model(model: Any, task: str = "regression", seed: int = 42):
    """字符串 → sklearn 估计器；传入估计器实例则原样返回。"""
    if not isinstance(model, str):
        return model
    key = model.lower().replace("-", "_").replace(" ", "_")
    table = _regressor_table(seed) if task == "regression" else _classifier_table(seed)
    if key not in table:
        raise ValueError(f"未知模型 {model!r}；可选：{sorted(table)}")
    return table[key]


def _needs_scaling(estimator: Any) -> bool:
    return type(estimator).__name__ in _SCALE_NEEDED


def _prepared(model: Any, task: str, seed: int, scale: str | bool):
    est = make_model(model, task, seed)
    if scale == "auto":
        scale = _needs_scaling(est)
    if scale:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        est = make_pipeline(StandardScaler(), est)
    return est


def _feature_names(X: Any) -> list[str] | None:
    if isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    return None


# ---------------------------------------------------------------------------
# 训练结果容器
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """监督学习训练结果。"""

    model: Any
    task: str
    cv_scores: np.ndarray
    cv_scoring: str
    test_metrics: dict[str, float]
    y_test: Any
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    X_test: Any = None
    feature_names: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"【{self.task}】交叉验证（{len(self.cv_scores)} 折，{self.cv_scoring}）："
                 f"{fmt(self.cv_scores.mean(), 5)} ± {fmt(self.cv_scores.std(), 3)}"]
        lines.append("测试集指标：")
        lines.append(str_metrics(self.test_metrics))
        return "\n".join(lines)

    __str__ = summary


# ---------------------------------------------------------------------------
# 监督学习快捷流程
# ---------------------------------------------------------------------------
def fit_regressor(
    model: Any = "rf",
    X: Any = None,
    y: Any = None,
    *,
    test_size: float = 0.2,
    cv: int = 5,
    seed: int = 42,
    scale: str | bool = "auto",
    scoring: str | None = None,
) -> TrainResult:
    """训练回归模型：划分 → 拟合 → 交叉验证 → 测试集评估。

    Parameters
    ----------
    model : str | 估计器
        模型名（见模块文档）或自定义 sklearn 估计器。
    X, y : 数据
        特征与目标（支持 DataFrame / ndarray）。
    cv : int
        交叉验证折数（在训练集上做）。
    scale : ``"auto"`` | bool
        是否标准化（``"auto"`` 对 SVR/KNN/MLP 等自动开启）。

    Returns
    -------
    TrainResult
        ``r.model`` 训练好的模型；``r.y_test`` / ``r.y_pred`` 供画图；
        ``r.test_metrics`` 指标 dict。
    """
    from sklearn.model_selection import cross_val_score, train_test_split

    set_seed(seed)
    est = _prepared(model, "regression", seed, scale)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)
    est.fit(Xtr, ytr)
    cv_scores = cross_val_score(est, Xtr, ytr, cv=cv, scoring=scoring or "r2")
    y_pred = est.predict(Xte)
    n_features = np.asarray(X).shape[1] if np.ndim(X) == 2 else None
    metrics = regression_metrics(yte, y_pred, n_features=n_features)
    return TrainResult(
        model=est, task="回归", cv_scores=np.asarray(cv_scores),
        cv_scoring=scoring or "r2", test_metrics=metrics,
        y_test=yte, y_pred=np.asarray(y_pred),
        X_test=Xte, feature_names=_feature_names(X),
    )


def fit_classifier(
    model: Any = "rf",
    X: Any = None,
    y: Any = None,
    *,
    test_size: float = 0.2,
    cv: int = 5,
    seed: int = 42,
    scale: str | bool = "auto",
    scoring: str | None = None,
    stratify: bool = True,
) -> TrainResult:
    """训练分类模型：划分（默认分层）→ 拟合 → 交叉验证 → 测试集评估。"""
    from sklearn.model_selection import cross_val_score, train_test_split

    set_seed(seed)
    est = _prepared(model, "classification", seed, scale)
    ya = np.asarray(y).ravel()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=ya if stratify else None,
    )
    est.fit(Xtr, ytr)
    cv_scores = cross_val_score(est, Xtr, ytr, cv=cv, scoring=scoring or "accuracy")
    y_pred = est.predict(Xte)
    y_proba = est.predict_proba(Xte) if hasattr(est, "predict_proba") else None
    metrics = classification_metrics(yte, y_pred, y_proba)
    return TrainResult(
        model=est, task="分类", cv_scores=np.asarray(cv_scores),
        cv_scoring=scoring or "accuracy", test_metrics=metrics,
        y_test=yte, y_pred=np.asarray(y_pred), y_proba=y_proba,
        X_test=Xte, feature_names=_feature_names(X),
    )


def cross_validate(
    model: Any,
    X: Any,
    y: Any,
    *,
    cv: int = 5,
    task: str | None = None,
    seed: int = 42,
    scale: str | bool = "auto",
    scoring: str | None = None,
) -> dict[str, Any]:
    """K 折交叉验证（返回均值/标准差/每折得分）。"""
    from sklearn.base import is_classifier
    from sklearn.model_selection import cross_val_score

    set_seed(seed)
    if task is None:
        probe = make_model(model, "regression", seed) if isinstance(model, str) else model
        task = "classification" if is_classifier(probe) else "regression"
    est = _prepared(model, task, seed, scale)
    scores = np.asarray(cross_val_score(est, X, y, cv=cv, scoring=scoring))
    return {
        "task": task,
        "scoring": scoring or ("accuracy" if task == "classification" else "r2"),
        "scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def compare_models(
    models: Sequence[Any],
    X: Any,
    y: Any,
    *,
    task: str = "regression",
    test_size: float = 0.2,
    cv: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """批量训练并对比多个模型，返回对比表（模型 / CV 均值 / CV 标准差 / 测试指标）。

    用法::

        table = pm.ml.compare_models(
            ["linear", "ridge", "rf", "gbr"], X, y, task="regression"
        )
    """
    rows = []
    key = "R2" if task == "regression" else "accuracy"
    for m in models:
        name = m if isinstance(m, str) else type(m).__name__
        res = (
            fit_regressor(m, X, y, test_size=test_size, cv=cv, seed=seed)
            if task == "regression"
            else fit_classifier(m, X, y, test_size=test_size, cv=cv, seed=seed)
        )
        row = {
            "模型": name,
            f"CV_{res.cv_scoring}_均值": res.cv_scores.mean(),
            f"CV_{res.cv_scoring}_标准差": res.cv_scores.std(),
            f"测试集_{key}": res.test_metrics.get(key, float("nan")),
            "测试集_RMSE": res.test_metrics.get("RMSE", float("nan")),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(f"测试集_{key}", ascending=False)


# ---------------------------------------------------------------------------
# 无监督
# ---------------------------------------------------------------------------
def standardize(X: Any) -> tuple[Any, Any]:
    """标准化（Z-score），返回 ``(标准化数据, scaler)``。

    DataFrame 输入返回 DataFrame（保持列名与索引），否则返回 ndarray。
    """
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    if isinstance(X, pd.DataFrame):
        arr = sc.fit_transform(X)
        return pd.DataFrame(arr, columns=X.columns, index=X.index), sc
    return sc.fit_transform(np.asarray(X, dtype=float)), sc


@dataclass
class ClusterResult:
    """聚类结果。"""

    labels: np.ndarray
    centers: np.ndarray
    metrics: dict[str, float]
    model: Any
    X_scaled: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        import collections

        counts = collections.Counter(self.labels.tolist())
        lines = [f"聚类完成：k = {len(counts)}，各簇样本数 = {dict(sorted(counts.items()))}"]
        lines.append(str_metrics(self.metrics))
        return "\n".join(lines)

    __str__ = summary


def kmeans(
    X: Any,
    k: int = 3,
    *,
    seed: int = 42,
    scale: bool = True,
    n_init: int = 10,
) -> ClusterResult:
    """K-means 聚类（返回标签、原始尺度质心与内部指标）。"""
    from sklearn.cluster import KMeans

    set_seed(seed)
    Xa = X
    sc = None
    if scale:
        Xa, sc = standardize(X)
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = model.fit_predict(Xa)
    centers = (
        sc.inverse_transform(model.cluster_centers_) if sc is not None
        else model.cluster_centers_
    )
    metrics = cluster_metrics(np.asarray(Xa, dtype=float), labels)
    return ClusterResult(
        labels=np.asarray(labels), centers=np.asarray(centers),
        metrics=metrics, model=model, X_scaled=Xa,
        extra={"k": k, "inertia": float(model.inertia_)},
    )


def elbow(
    X: Any,
    k_range: Sequence[int] = range(2, 11),
    *,
    seed: int = 42,
    scale: bool = True,
) -> pd.DataFrame:
    """肘部法则扫描：每个 k 的 inertia 与轮廓系数（用于确定最优簇数）。"""
    from sklearn.cluster import KMeans

    set_seed(seed)
    Xa = X
    if scale:
        Xa, _ = standardize(X)
    Xa = np.asarray(Xa, dtype=float)
    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = model.fit_predict(Xa)
        row = {"k": k, "inertia": float(model.inertia_)}
        row["silhouette"] = cluster_metrics(Xa, labels).get("silhouette", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class PCAResult:
    """主成分分析结果。"""

    scores: np.ndarray
    loadings: pd.DataFrame | None
    explained_ratio: np.ndarray
    model: Any
    feature_names: list[str] | None = None

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.explained_ratio)

    def summary(self) -> str:
        parts = [
            f"主成分方差解释率：{'  '.join(f'PC{i + 1}={r:.4f}' for i, r in enumerate(self.explained_ratio))}",
            f"累计解释率：{self.cumulative[-1]:.4f}",
        ]
        return "\n".join(parts)

    __str__ = summary


def pca(X: Any, n_components: int = 2, *, scale: bool = True) -> PCAResult:
    """主成分分析（返回得分矩阵、载荷矩阵、解释方差比）。"""
    from sklearn.decomposition import PCA

    Xa = X
    if scale:
        Xa, _ = standardize(X)
    Xa = np.asarray(Xa, dtype=float)
    model = PCA(n_components=n_components)
    scores = model.fit_transform(Xa)
    names = _feature_names(X)
    loadings = None
    if names is not None:
        loadings = pd.DataFrame(
            model.components_.T,
            index=names,
            columns=[f"PC{i + 1}" for i in range(model.n_components_)],
        )
    return PCAResult(
        scores=scores, loadings=loadings,
        explained_ratio=model.explained_variance_ratio_,
        model=model, feature_names=names,
    )
