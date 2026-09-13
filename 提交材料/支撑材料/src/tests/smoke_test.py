"""program 工具包冒烟测试：覆盖全部 11 个模块，验证核心功能可运行且结果正确。

运行::

    uv run python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import program as pm

PASSED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"❌ {name} {detail}")
    PASSED.append(name)
    print(f"  ok - {name}")


def section(title: str) -> None:
    print(f"\n== {title} ==")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="program_smoke_"))
    pm.config.set_project_root(tmp)
    pm.init(seed=42, verbose=False)
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ utils
    section("utils / config")
    from program.utils import DataCache, Timer, env_markdown, fmt, markdown_table

    check("fmt", fmt(1234.5678) == "1235" and "e" in fmt(1.23e-6))
    check("markdown_table", "| a |" in markdown_table({"a": [1, 2]}, nd=3))
    with Timer("noop"):
        pass
    cache = DataCache("smoke", root=tmp / "cache")
    cache.save("k", {"v": 1})
    check("DataCache", cache.get_or_compute("k", lambda: 2)["v"] == 1)
    check("env_markdown", "numpy" in env_markdown())

    # ----------------------------------------------------------------- dataio
    section("dataio")
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2.0, np.nan, 6.0, 8.0, 10.0],
            "g": ["a", "b", "a", "b", "a"],
        }
    )
    csv_p = pm.save_table(df, tmp / "data" / "t.csv")
    check("csv roundtrip", len(pm.read_table(csv_p)) == 5)
    xlsx_p = pm.save_table(df, tmp / "data" / "t.xlsx")
    check("xlsx roundtrip", pm.read_table(xlsx_p)["y"].isna().sum() == 1)
    check("read_sheets", "Sheet1" in pm.read_sheets(xlsx_p))
    filled = pm.dataio.fill_missing(df)
    check("fill_missing", filled["y"].isna().sum() == 0)
    check("clean_columns", list(pm.dataio.clean_columns(df).columns) == ["x", "y", "g"])
    check("missing_report", pm.dataio.missing_report(df)["缺失数"].sum() == 1)
    s = pm.summarize(df)
    check("summarize", "数据概览" in s and "前 5 行" in s)
    X, y = pm.dataio.split_xy(filled, target="y")
    check("split_xy", X.shape == (5, 2) and len(y) == 5)
    Xtr, Xte, ytr, yte = pm.dataio.train_test_split_df(filled, target="y", seed=1)
    check("train_test_split_df(target)", len(Xtr) + len(Xte) == 5 and len(ytr) + len(yte) == 5)
    tr, te = pm.dataio.train_test_split_df(filled, seed=1)
    check("train_test_split_df(默认)", len(tr) + len(te) == 5)
    js_p = pm.dataio.save_json({"a": np.arange(3)}, tmp / "data" / "t.json")
    check("json roundtrip", pm.dataio.load_json(js_p)["a"] == [0, 1, 2])

    # --------------------------------------------------------------- plotting
    section("plotting")
    xs = np.linspace(0, 2 * np.pi, 100)
    pm.line(xs, [np.sin(xs), np.cos(xs)], labels=["sin", "cos"],
            xlabel="x", ylabel="y", save="p_line")
    check("line", (tmp / "figures" / "p_line.pdf").exists())
    pm.scatter(xs, np.sin(xs), xlabel="x", ylabel="sin", save="p_scatter")
    _, _, info = pm.scatter_fit(xs, 2 * xs + 1 + rng.normal(0, 0.3, 100),
                                xlabel="x", ylabel="y", save="p_fit")
    check("scatter_fit", abs(info["coef"][0] - 2) < 0.3)
    pm.bar(["A", "B", "C"], [1, 2, 3], ylabel="值", save="p_bar")
    pm.bar_group(["A", "B"], {"m1": [1, 2], "m2": [2, 1]}, save="p_bar_group")
    pm.hist(rng.normal(size=200), xlabel="x", save="p_hist")
    pm.box({"A": rng.normal(size=50), "B": rng.normal(1, 1, 50)}, save="p_box")
    pm.heatmap(df[["x", "y"]].fillna(0).to_numpy(), annot=True, save="p_heat")
    pm.corr_heatmap(filled[["x", "y"]], save="p_corr")
    pm.radar(["m1", "m2", "m3"], {"A": [1, 2, 3], "B": [3, 1, 2]}, save="p_radar")
    pm.dual_axis(xs, np.sin(xs), 100 * np.cos(xs), save="p_dual")
    pm.pie(["A", "B"], [3, 1], save="p_pie")
    pm.errorbar(xs[:10], np.sin(xs[:10]), yerr=0.1, save="p_err")
    XX, YY = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
    pm.surface3d(XX, YY, XX**2 + YY**2, xlabel="x", ylabel="y", zlabel="z", save="p_3d")
    yb = (rng.normal(size=200) > 0).astype(int)
    prob = np.clip(yb * 0.7 + rng.normal(0, 0.2, 200) + 0.2, 0, 1)
    _, _, auc = pm.roc(yb, prob, save="p_roc")
    check("roc", auc > 0.9)
    figs = sorted(p.name for p in (tmp / "figures").glob("*.pdf"))
    check("全部图已输出 PDF", len(figs) == 15, f"实际 {len(figs)}：{figs}")
    check(
        "图表数据已记录",
        all(
            (tmp / "code" / "outputs" / "figure_data" / f).exists()
            for f in ("p_line.csv", "p_fit.csv")
        ),
    )

    # ------------------------------------------------------------------ stats
    section("stats")
    a = rng.normal(0, 1, 100)
    b = rng.normal(0.6, 1, 100)
    r_t = pm.stats.t_test_ind(a, b)
    check("t_test_ind", r_t.significant and abs(r_t.statistic) > 2)
    check("normality", pm.stats.normality(a).pvalue > 0.01)
    check("t_test_one", not pm.stats.t_test_one(a, mu=0).significant)
    check("t_test_paired", pm.stats.t_test_paired(a, a + rng.normal(0, 0.1, 100)).pvalue is not None)
    check("anova", pm.stats.anova(*[rng.normal(m, 1, 60) for m in (0, 0, 1)]).pvalue < 0.05)
    check("kruskal", pm.stats.kruskal(a, b).pvalue < 0.05)
    check("compare_groups", pm.stats.compare_groups(a, b).pvalue < 0.05)
    check("chi2", pm.stats.chi2_test([[10, 20], [30, 15]]).pvalue < 0.05)
    check("corr_test", abs(pm.stats.corr_test(np.arange(50), np.arange(50)).statistic - 1) < 1e-9)
    df_c = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    check("corr_matrix", pm.stats.corr_matrix(df_c).shape == (3, 3))
    lr = pm.stats.linregress_summary(np.arange(20), 2 * np.arange(20) + 1)
    check("linregress", abs(lr.slope - 2) < 1e-9 and lr.r2 > 0.999)
    df_r = pd.DataFrame({"y": 3 * df_c["a"] - 2 * df_c["b"] + rng.normal(0, 0.1, 100), "x1": df_c["a"], "x2": df_c["b"]})
    ols_res = pm.stats.ols(df_r, y="y", xs=["x1", "x2"])
    check("ols", abs(ols_res.params["x1"] - 3) < 0.1)
    check("ols_text", "R-squared" in pm.stats.ols_text(df_r, y="y", xs=["x1", "x2"]))
    lo, hi = pm.stats.ci_mean(a)
    check("ci_mean", lo < a.mean() < hi)
    bci = pm.stats.bootstrap_ci(a, n_boot=2000, seed=1)
    check("bootstrap_ci", bci.lo < bci.stat < bci.hi)

    # ---------------------------------------------------------------- fitting
    section("fitting")
    from program import fitting

    def expo(x, aa, bb):
        return aa * np.exp(-bb * x)

    xf = np.linspace(0, 2, 60)
    yf = 3 * np.exp(-1.5 * xf) + rng.normal(0, 0.01, 60)
    fr = fitting.fit_curve(expo, xf, yf, p0=[1, 1], names=["a", "b"])
    check("fit_curve", abs(fr.params[0] - 3) < 0.1 and abs(fr.params[1] - 1.5) < 0.05)
    check("fit_curve.summary", "R²" in fr.summary() and "±" in fr.summary())
    pf = fitting.polyfit(xf, 2 * xf**2 - 3 * xf + 1, degree=2)
    check("polyfit", np.allclose(pf.params, [2, -3, 1], atol=1e-6))
    check("poly_str", "x^{2}" in fitting.poly_str(pf.params))
    check("poly_scan", len(fitting.poly_scan(xf, yf, max_degree=3)) == 3)
    xi = np.linspace(0.1, 1.9, 20)
    for kind in ("linear", "cubic", "pchip"):
        check(f"interp/{kind}", fitting.interp(xf, yf, xi, kind=kind).shape == (20,))
    spl = fitting.spline_smooth(xf, yf, s=len(xf) * 0.001)
    check("spline_smooth", spl(xi).shape == (20,))
    for m in ("savgol", "moving", "ewma"):
        check(f"smooth/{m}", fitting.smooth(yf, 7, method=m).shape == yf.shape)

    # ---------------------------------------------------------------- metrics
    section("metrics")
    yt = np.array([1.0, 2, 3, 4, 5])
    m = pm.metrics.regression_metrics(yt, yt + 0.1, n_features=2)
    check("regression_metrics", abs(m["MAE"] - 0.1) < 1e-9 and "AdjR2" in m)
    y_cls = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p_cls = np.array([0, 1, 0, 1, 0, 0, 0, 1])
    cm = pm.metrics.classification_metrics(y_cls, p_cls, y_proba=np.clip(p_cls + 0.1, 0, 1))
    check("classification_metrics", cm["accuracy"] > 0.8 and "auc" in cm)
    ct = pm.metrics.confusion_table(y_cls, p_cls)
    check("confusion_table", ct.shape == (2, 2))
    blobs = np.vstack([rng.normal(m, 0.3, (30, 2)) for m in (0, 3)]).astype(float)
    lab = np.r_[np.zeros(30, int), np.ones(30, int)]
    check("cluster_metrics", pm.metrics.cluster_metrics(blobs, lab)["silhouette"] > 0.8)
    check("str_metrics", "MAE" in pm.metrics.str_metrics(m))

    # --------------------------------------------------------------- optimize
    section("optimize")
    r_lp = pm.optimize.solve_lp(
        [3, 5], A_ub=[[1, 0], [0, 2], [3, 2]], b_ub=[4, 12, 18], maximize=True
    )
    check("solve_lp", r_lp.success and abs(r_lp.fun - 36) < 1e-8, str(r_lp))
    r_ip = pm.optimize.solve_lp([1, 1], A_ub=[[2, 2]], b_ub=[7], maximize=True, integer=True)
    check("solve_lp(integer)", r_ip.success and abs(r_ip.fun - 3) < 1e-8)
    r_nlp = pm.optimize.solve_nlp(lambda v: (v[0] - 1) ** 2 + (v[1] - 2) ** 2, x0=[0, 0])
    check("solve_nlp", abs(r_nlp.fun) < 1e-6 and np.allclose(r_nlp.x, [1, 2], atol=1e-3))
    r_g = pm.optimize.global_minimize(lambda v: (v[0] - 1.5) ** 2, bounds=[(0, 3)])
    check("global_minimize", abs(r_g.fun) < 1e-6)
    r_kp = pm.optimize.knapsack([60, 100, 120], [10, 20, 30], 50)
    check("knapsack", r_kp.best_value == 220 and r_kp.items == [1, 2])
    r_as = pm.optimize.assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    check("assignment", r_as.total == 5, str(r_as))
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    d = pm.optimize.distance_matrix(pts)
    tour_nn = pm.optimize.tsp_nearest_neighbor(d)
    tour_2o, length = pm.optimize.tsp_2opt(d, tour_nn)
    check("tsp", abs(length - 4.0) < 1e-9 and sorted(tour_2o) == [0, 1, 2, 3])

    # --------------------------------------------------------------------- ml
    section("ml")
    Xm = rng.normal(size=(120, 3))
    ym = Xm @ np.array([1.0, -2.0, 3.0]) + rng.normal(0, 0.1, 120)
    r_reg = pm.ml.fit_regressor("ridge", Xm, ym, cv=3, seed=0)
    check("fit_regressor", r_reg.test_metrics["R2"] > 0.99, r_reg.summary())
    Xc = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(2.5, 1, (60, 2))])
    yc = np.r_[np.zeros(60, int), np.ones(60, int)]
    r_cls = pm.ml.fit_classifier("logreg", Xc, yc, cv=3, seed=0)
    check("fit_classifier", r_cls.test_metrics["accuracy"] > 0.9 and r_cls.y_proba is not None)
    cv_res = pm.ml.cross_validate("rf", Xm, ym, cv=3, seed=0)
    check("cross_validate", 0 <= cv_res["mean"] <= 1)
    table = pm.ml.compare_models(["linear", "ridge"], Xm, ym, cv=3, seed=0)
    check("compare_models", len(table) == 2 and "测试集_R2" in table.columns)
    Xs, sc = pm.ml.standardize(pd.DataFrame(Xm, columns=["a", "b", "c"]))
    check("standardize", abs(float(np.asarray(Xs).mean())) < 1e-9)
    cl = pm.ml.kmeans(blobs, k=2, seed=0)
    check("kmeans", cl.metrics["silhouette"] > 0.8 and cl.centers.shape == (2, 2))
    elbow_df = pm.ml.elbow(blobs, k_range=range(2, 5), seed=0)
    check("elbow", len(elbow_df) == 3 and elbow_df["inertia"].is_monotonic_decreasing)
    pc = pm.ml.pca(pd.DataFrame(Xm, columns=["a", "b", "c"]), n_components=2)
    check("pca", pc.scores.shape == (120, 2) and pc.loadings is not None and pc.cumulative[-1] <= 1)

    # -------------------------------------------------------------------- ode
    section("ode")

    def sir(t, y, beta, gamma):
        s, i, r = y
        return [-beta * s * i, beta * s * i - gamma * i, gamma * i]

    t_grid = np.linspace(0, 40, 41)
    sol = pm.ode.solve_ivp_model(sir, [0.99, 0.01, 0], (0, 40), params=(0.4, 0.2), t_eval=t_grid)
    check("solve_ivp_model", sol.success and sol.y.shape == (3, 41))
    y_obs = sol.y.T + rng.normal(0, 0.002, (41, 3))
    fit_ode = pm.ode.fit_ode_params(
        sir, [0.99, 0.01, 0], t_grid, y_obs, p0=[0.3, 0.15],
        bounds=([0.05, 0.05], [1.0, 1.0]), param_names=["beta", "gamma"],
    )
    check("fit_ode_params", abs(fit_ode.params[0] - 0.4) < 0.05 and abs(fit_ode.params[1] - 0.2) < 0.03,
          fit_ode.summary())
    check("ODEFitResult.simulate", fit_ode.simulate(t_grid).shape == (41, 3))

    # ----------------------------------------------------------------- report
    section("report")
    pm.record_result("问题一结果", {"目标值": 36.0, "状态": "最优"}, note="smoke test 写入")
    pm.record_result("模型对比", table)
    pm.record("正文段落：验证 record 直接追加。")
    text = (tmp / "reports" / "RESULTS_REPORT.md").read_text(encoding="utf-8")
    check("record_result", "问题一结果" in text and "| 指标 | 数值 |" in text)
    check("record(DataFrame)", "模型对比" in text)
    tex = pm.to_latex(pd.DataFrame({"指标": ["R2"], "值": [0.9961]}), caption="测试表", label="tab:smoke")
    check("to_latex", "tabular" in tex)
    out_p = pm.save_outputs({"最优值": 36.0}, "q1_result")
    check("save_outputs(json)", out_p.exists() and out_p.suffix == ".json")
    out_csv = pm.save_outputs(table, "q1_models")
    check("save_outputs(csv)", out_csv.suffix == ".csv" and out_csv.exists())

    print(f"\n✅ SMOKE TEST OK：{len(PASSED)} 项检查全部通过")
    print(f"   临时输出目录：{tmp}")


if __name__ == "__main__":
    main()
