"""C 题 问题二 对照：逐时段 ARIMA 预测 vs 自适应加权预测。

对比两个"预测器"在完全相同的信息结构下（每天 0:00 只看历史）的表现：

- 自适应加权（口径 E，`q2_adaptive.py`）：负荷 d-7/d-14/典型日，光伏 d-1/d-2/典型日，
  权重按历史实际费用标定；
- ARIMA(2,0,1)：每个 10 分钟槽（144 槽）× 负荷/光伏 各建一条日序列，滚动一步预测；
- 参考基线：典型日、持续预测（D-1）、口径 D。

滚动方式：每 REFIT_EVERY 天用扩展窗口重估参数，其余日期用 Kalman 滤波
append 新观测（refit=False）后再做 1 步预测。预测结果缓存到
``code/outputs/cache/q2arima/``。

评价两条线：
1. 预测精度：MAE / RMSE（kW，2025-02-01 ~ 12-31）；
2. 全年费用：把预测代入同一套计划/执行/紧急购电模型（与问题二口径一致）。

运行：uv run python -m solve.q2_arima   （在 program/ 目录；首次约 5 分钟，之后走缓存）
"""
from __future__ import annotations

import hashlib
import os
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import program as pm
from solve import q2
from solve.common import DATA_C, E0, ROOT
from solve.q2_adaptive import AdaptiveWeightModel, record

N_DAY = q2.N_DAY                  # 365
REPORT_START = q2.REPORT_START    # 31（2025-02-01）
EPS_PLAN = 1e-3
ORDER = (2, 0, 1)                 # 抽样槽位 AIC 对比后固定（负荷 8/12 最优，光伏并列最优）
REFIT_EVERY = 7                   # 参数重估周期（天）
WORKERS = min(8, os.cpu_count() or 1)

_DATA = None                      # 子进程数据（initializer 注入）


# ---------------------------------------------------------------------------
# ARIMA 预测器
# ---------------------------------------------------------------------------
def _init_worker(data) -> None:
    global _DATA
    _DATA = data


def _fit_slot(task):
    """单条序列（变量, 槽位）的滚动一步预测，返回 (which, slot, pred[365])。"""
    which, slot, order, refit_every = task
    warnings.filterwarnings("ignore")
    y = (_DATA["load"] if which == 0 else _DATA["pv_act"])[:, slot]
    pred = np.full(N_DAY, np.nan)
    res = None
    for d in range(REPORT_START, N_DAY):
        train = y[:d]
        if np.ptp(train) < 1e-9:                  # 常数序列（夜间光伏槽等）
            pred[d] = float(train[-1])
            continue
        try:
            if res is None or (d - REPORT_START) % refit_every == 0:
                res = ARIMA(train, order=order).fit(
                    method_kwargs={"warn_convergence": False})
            else:
                res = res.append([y[d - 1]], refit=False)
            pred[d] = float(res.forecast(1)[0])
        except Exception:                         # 拟合失败退化为持续预测
            pred[d] = float(y[d - 1])
    return which, slot, pred


class ArimaForecaster:
    """逐时段 ARIMA 滚动一步预测器（144 槽 × 负荷/光伏 2 变量）。

    Parameters
    ----------
    order : tuple
        ARIMA(p, d, q) 阶数，默认 (2, 0, 1)。
    refit_every : int
        参数重估周期（天）；其间用 append(refit=False) 更新状态。
    workers : int
        并行进程数（按序列并行），默认 min(8, CPU)。
    """

    def __init__(self, order=ORDER, refit_every=REFIT_EVERY, workers=WORKERS):
        self.order = tuple(order)
        self.refit_every = refit_every
        self.workers = workers
        self.data = None
        self.load_fc = self.pv_fc = None
        self.seconds = 0.0

    def load(self, data: dict | None = None) -> "ArimaForecaster":
        self.data = q2.load_all() if data is None else data
        return self

    def _cache_path(self):
        h = hashlib.md5()
        for arr in (self.data["load"], self.data["pv_act"]):
            h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
        h.update(np.asarray([*self.order, self.refit_every], float).tobytes())
        cache = pm.outputs_dir() / "cache" / "q2arima"
        cache.mkdir(parents=True, exist_ok=True)
        return cache / f"fc_{h.hexdigest()[:12]}.npz"

    def forecast_all(self) -> "ArimaForecaster":
        """并行预测，结果（kW，365×144）存 load_fc / pv_fc（31 日之前为 0）。"""
        pm.init(root=str(ROOT))
        path = self._cache_path()
        if path.exists():
            z = np.load(path)
            self.load_fc, self.pv_fc = z["load_fc"], z["pv_fc"]
            return self
        tasks = [(which, slot, self.order, self.refit_every)
                 for which in (0, 1) for slot in range(144)]
        t0 = time.time()
        if self.workers > 1:
            from multiprocessing import Pool

            with Pool(self.workers, initializer=_init_worker,
                      initargs=(self.data,)) as pool:
                results = pool.map(_fit_slot, tasks)
        else:
            _init_worker(self.data)
            results = [_fit_slot(tk) for tk in tasks]

        fl = np.zeros((N_DAY, 144))
        fp = np.zeros((N_DAY, 144))
        for which, slot, pred in results:
            (fl if which == 0 else fp)[:, slot] = np.nan_to_num(pred, nan=0.0)
        self.load_fc, self.pv_fc = np.clip(fl, 0.0, None), np.clip(fp, 0.0, None)
        self.seconds = time.time() - t0
        np.savez_compressed(path, load_fc=self.load_fc, pv_fc=self.pv_fc)
        return self


# ---------------------------------------------------------------------------
# 指标与费用
# ---------------------------------------------------------------------------
def mae_rmse(pred_kw: np.ndarray, true_kw: np.ndarray):
    """报告期内的 MAE / RMSE（kW，全部槽位与日期）。"""
    err = pred_kw[REPORT_START:] - true_kw[REPORT_START:]
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def annual_cost(load_fc_kwh: np.ndarray, pv_fc_kwh: np.ndarray, data: dict):
    """用给定预测跑全年计划 + 执行 + 紧急购电，返回 (计划费, 紧急费, 总费用)。"""
    price, load, pv = data["price"], data["load"], data["pv_act"]
    plan = emerg = 0.0
    for d in range(REPORT_START, N_DAY):
        x, _E, _ = q2.plan_day(price, load_fc_kwh[d], pv_fc_kwh[d], E0, eps=EPS_PLAN)
        ex = q2.exec_day_causal(price, load[d] / 6.0, pv[d] / 6.0, x, E0)
        plan += float(price @ x)
        emerg += float(q2.EMERG_MULT * (price @ ex["e"]))
    return plan, emerg, plan + emerg


def typical_forecast(data: dict):
    """附件 1 典型日（负荷 kW, 光伏 kW）平铺到全年。"""
    a1 = pm.read_table(DATA_C / "附件1.xlsx")
    lt = a1["小区负载"].to_numpy(float)
    pt = a1["光伏发电预测功率"].to_numpy(float)
    return np.tile(lt, (N_DAY, 1)), np.tile(pt, (N_DAY, 1))


def adaptive_forecast(data: dict, model: AdaptiveWeightModel):
    """口径 E 的预测（kW，365×144）：目标日 d 用 TH[d-1] 的权重。"""
    fl = np.zeros((N_DAY, 144))
    fp = np.zeros((N_DAY, 144))
    weights = model.weights(W=1, use_gd=True)
    for i, d in enumerate(range(REPORT_START, N_DAY)):
        w, u = weights[i]
        l_kwh, p_kwh = model.forecast(w, u, d)
        fl[d], fp[d] = l_kwh * 6.0, p_kwh * 6.0
    return fl, fp


def persistence_forecast(data: dict):
    """持续预测基线：用 D-1 的实际值。"""
    fl = np.zeros((N_DAY, 144))
    fp = np.zeros((N_DAY, 144))
    fl[REPORT_START:] = data["load"][REPORT_START - 1:-1]
    fp[REPORT_START:] = data["pv_act"][REPORT_START - 1:-1]
    return fl, fp


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_comparison(workers=WORKERS) -> dict:
    """跑 ARIMA，与自适应加权/典型日/持续预测比精度与费用，并写报告与图。"""
    import matplotlib.pyplot as plt

    pm.init(root=str(ROOT))
    log = pm.get_logger("q2arima")
    data = q2.load_all()

    log.info("ARIMA(2,0,1) 滚动预测中（{} 进程）...", workers)
    ar = ArimaForecaster(workers=workers).load(data).forecast_all()
    log.info("ARIMA 完成，用时 {:.0f}s", ar.seconds)

    model = AdaptiveWeightModel().load().build_table()   # load() 会补上典型日列
    fl_ad, fp_ad = adaptive_forecast(data, model)
    fl_typ, fp_typ = typical_forecast(data)
    fl_pst, fp_pst = persistence_forecast(data)

    rows = []
    for name, fl, fp in (
        ("自适应加权（E）", fl_ad, fp_ad),
        ("ARIMA(2,0,1)", ar.load_fc, ar.pv_fc),
        ("典型日（B）", fl_typ, fp_typ),
        ("持续预测（D-1）", fl_pst, fp_pst),
    ):
        mae_l, rmse_l = mae_rmse(fl, data["load"])
        mae_p, rmse_p = mae_rmse(fp, data["pv_act"])
        rows.append({
            "预测模型": name,
            "负荷MAE/kW": round(mae_l, 1), "负荷RMSE/kW": round(rmse_l, 1),
            "光伏MAE/kW": round(mae_p, 1), "光伏RMSE/kW": round(rmse_p, 1),
        })
    metrics = pd.DataFrame(rows)
    record(
        "问题二 对照：ARIMA 与自适应加权的预测精度（2025-02-01 ~ 12-31）",
        metrics,
        note=(
            "逐时段滚动一步预测（144 槽 × 334 天），全部只用 D-1 及以前的数据；"
            f"ARIMA(2,0,1) 每 {REFIT_EVERY} 天扩展窗口重估参数、其余日期 Kalman append。"
            "指标单位 kW；持续预测 = 用前一天同时刻实际值。"
        ),
    )
    pm.save_outputs(metrics, "q2_arima_metrics")

    cost_rows = []
    for name, fl, fp in (
        ("自适应加权（E）", fl_ad, fp_ad),
        ("ARIMA(2,0,1)", ar.load_fc, ar.pv_fc),
        ("典型日（B）", fl_typ, fp_typ),
    ):
        plan, emerg, tot = annual_cost(fl / 6.0, fp / 6.0, data)
        cost_rows.append({
            "预测模型": name,
            "计划购电费/万元": round(plan / 1e4, 1),
            "紧急购电费/万元": round(emerg / 1e4, 1),
            "总费用/万元": round(tot / 1e4, 1),
        })
    _x_d, recs_d = q2.run_deterministic(data)      # 口径 D 参考
    plan_d = sum(recs_d[d]["plan_cost"] for d in range(REPORT_START, N_DAY))
    emerg_d = sum(recs_d[d]["emerg_cost"] for d in range(REPORT_START, N_DAY))
    cost_rows.append({
        "预测模型": "0:00 预报（D，参考）",
        "计划购电费/万元": round(plan_d / 1e4, 1),
        "紧急购电费/万元": round(emerg_d / 1e4, 1),
        "总费用/万元": round((plan_d + emerg_d) / 1e4, 1),
    })
    costs = pd.DataFrame(cost_rows)
    record(
        "问题二 对照：ARIMA 与自适应加权的全年费用",
        costs,
        note=(
            "同一套计划/执行/紧急购电模型（take-or-pay、储能日循环、5 倍价紧急购电）；"
            "ARIMA 预测为统计拟合（未按费用标定），自适应加权权重按费用标定。"
        ),
    )
    pm.save_outputs(costs, "q2_arima_compare")

    # ---- 图：精度对比 + 费用对比 ----
    labels = metrics["预测模型"].tolist()
    pm.bar_group(
        labels,
        {"负荷MAE": metrics["负荷MAE/kW"].tolist(),
         "光伏MAE": metrics["光伏MAE/kW"].tolist()},
        ylabel="MAE / kW", rot=12, save="Q2ARIMA_误差对比",
    )
    pm.bar_group(
        costs["预测模型"].tolist(),
        {"计划购电费": costs["计划购电费/万元"].tolist(),
         "紧急购电费": costs["紧急购电费/万元"].tolist()},
        ylabel="年度费用 / 万元", rot=12, save="Q2ARIMA_费用对比",
    )

    # ---- 图：示例日曲线（2025-06-21）----
    d = 171
    t_h = (np.arange(144) + 0.5) / 6.0
    fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    for ax, actual, ad, arm, ylabel in (
        (axes[0], data["load"][d], fl_ad[d], ar.load_fc[d], "负荷 / kW"),
        (axes[1], data["pv_act"][d], fp_ad[d], ar.pv_fc[d], "光伏 / kW"),
    ):
        ax.plot(t_h, actual, color="#333333", lw=1.6, label="实际")
        ax.plot(t_h, ad, color="#4C72B0", lw=1.2, label="自适应加权")
        ax.plot(t_h, arm, color="#C44E52", lw=1.2, ls="--", label="ARIMA")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=3, fontsize=8)
    axes[1].set_xlabel("时刻 / h")
    pm.save_fig(fig, "Q2ARIMA_预测示例",
                data=pd.DataFrame({"时刻/h": t_h,
                                   "负荷实际": data["load"][d], "负荷_E": fl_ad[d],
                                   "负荷_ARIMA": ar.load_fc[d],
                                   "光伏实际": data["pv_act"][d], "光伏_E": fp_ad[d],
                                   "光伏_ARIMA": ar.pv_fc[d]}))

    log.info("对照完成：自适应 {:.1f} 万元 | ARIMA {:.1f} | D {:.1f}",
             costs.loc[0, "总费用/万元"], costs.loc[1, "总费用/万元"],
             costs.loc[3, "总费用/万元"])
    return {"metrics": metrics, "costs": costs,
            "load_fc": ar.load_fc, "pv_fc": ar.pv_fc}


if __name__ == "__main__":
    run_comparison()
