"""模型评估指标（回归 / 分类 / 聚类）。

用法::

    import program as pm

    m = pm.metrics.regression_metrics(y_true, y_pred, n_features=5)
    print(pm.metrics.str_metrics(m))          # 多行文本，可直接贴报告
    pm.metrics.confusion_table(y_true, y_pred)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import fmt


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------
def regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    n_features: int | None = None,
) -> dict[str, float]:
    """回归指标：MAE / MSE / RMSE / MAPE / SMAPE / R² / NSE（+ 调整 R²）。

    MAPE/SMAPE 自动忽略 ``y_true = 0`` 且分母同时为 0 的点。
    """
    y = np.asarray(y_true, dtype=float).ravel()
    yhat = np.asarray(y_pred, dtype=float).ravel()
    if len(y) != len(yhat):
        raise ValueError(f"长度不一致：y_true {len(y)}，y_pred {len(yhat)}")
    n = len(y)
    err = y - yhat
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    denom = np.where(np.abs(y) > 1e-12, np.abs(y), np.nan)
    mape = float(np.nanmean(np.abs(err) / denom) * 100)
    smape_den = np.abs(y) + np.abs(yhat)
    smape = float(np.nanmean(np.where(smape_den > 1e-12, 2 * np.abs(err) / smape_den, np.nan)) * 100)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "SMAPE(%)": smape,
        "R2": r2,
        "NSE": nse,
        "MaxErr": float(np.max(np.abs(err))),
        "n": float(n),
    }
    if n_features is not None and n > n_features + 1:
        out["AdjR2"] = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
    return out


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------
def classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    y_proba: Sequence[float] | None = None,
    labels: Sequence | None = None,
) -> dict[str, float]:
    """分类指标：accuracy / precision / recall / F1（macro、weighted）与 AUC。

    ``y_proba`` 为每个类别的预测概率（形状 ``(n, n_classes)``），二分类时也可
    直接传正类概率（1D）。
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y, yp)),
        "precision_macro": float(precision_score(y, yp, average="macro", zero_division=0, labels=labels)),
        "recall_macro": float(recall_score(y, yp, average="macro", zero_division=0, labels=labels)),
        "f1_macro": float(f1_score(y, yp, average="macro", zero_division=0, labels=labels)),
        "f1_weighted": float(f1_score(y, yp, average="weighted", zero_division=0, labels=labels)),
        "n": float(len(y)),
    }
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        n_classes = len(np.unique(y))
        try:
            if n_classes > 2:
                out["auc_ovr"] = float(
                    roc_auc_score(y, proba, multi_class="ovr", labels=labels)
                )
            else:
                p1 = proba[:, 1] if proba.ndim == 2 else proba.ravel()
                out["auc"] = float(roc_auc_score(y, p1))
        except Exception:
            pass
    return out


def confusion_table(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Sequence | None = None,
) -> pd.DataFrame:
    """混淆矩阵（行 = 真实类别，列 = 预测类别）。"""
    return pd.crosstab(
        pd.Series(np.asarray(y_true).ravel(), name="真实"),
        pd.Series(np.asarray(y_pred).ravel(), name="预测"),
        dropna=False,
    )


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def cluster_metrics(X, labels: Sequence[int]) -> dict[str, float]:
    """聚类内部指标：轮廓系数、Calinski-Harabasz、Davies-Bouldin。

    轮廓系数越大越好；CH 越大越好；DB 越小越好。
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    Xa = np.asarray(X, dtype=float)
    lab = np.asarray(labels).ravel()
    out: dict[str, float] = {}
    if len(set(lab.tolist())) > 1:
        out["silhouette"] = float(silhouette_score(Xa, lab))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xa, lab))
        out["davies_bouldin"] = float(davies_bouldin_score(Xa, lab))
    return out


def str_metrics(metrics: dict[str, Any], nd: int = 4, indent: str = "  ") -> str:
    """把指标 dict 转成多行文本（``MAE = 1.234``），便于写入报告。"""
    return "\n".join(f"{indent}{k} = {fmt(v, nd)}" for k, v in metrics.items())
