"""C 题重构金标测试：冻结公共函数在重构前（2026-09-13 tidy 分支起点）的数值行为。

重构期间使用：入口从子包/门面导出后，下面所有数值与等价关系必须逐位复现。
运行（program/ 下）：uv run python tests/solve_refactor_test.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import program as pm

from solve import q1, q2, q3
from solve import q3_proto as qp
from solve.q2_adaptive import AdaptiveWeightModel, simplex_grid, softmax
from solve.q4 import plan_horizon
from solve.q4_price_dyn import forecast_price as fp_dyn
from solve.q4_price_fit import forecast_price as fp_fit

# ---------------------------------------------------------------------------
# 固定问题（所有检查共用；与 2026-09-13 采集脚本一致）
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(0)
PRICE = 0.4 + 0.3 * np.sin(np.arange(q2.T) / 24 * 2 * np.pi) + 0.05 * _RNG.random(q2.T)
LOAD_KWH = (500 + 200 * _RNG.random(q2.T)) / 6.0
PV_KWH = np.clip(300 * np.sin(np.arange(q2.T) / 48 * np.pi), 0, None) / 6.0
E0 = 6000.0

GOLD_OBJ_Q1 = 1817.5053528737549
GOLD_X_SUM = 13801.814739472791
GOLD_EXEC = (11657.42226926288, 9442.512038102926, 0.0, 0.0, 6000.0)
GOLD_CACHE_KEY = "a76f418adfbe"


def check_lp_golden() -> None:
    """q1.build_lp ≡ q2.plan_day(eps=0)；q4.plan_horizon ≡ plan_day（多日版）。"""
    _, sol = q1.build_lp(PRICE, LOAD_KWH, PV_KWH)
    assert abs(float(PRICE @ sol["x"]) - GOLD_OBJ_Q1) < 1e-9

    x2, E2, _ = q2.plan_day(PRICE, LOAD_KWH, PV_KWH, E0, eps=0.0)
    assert abs(float(x2.sum()) - GOLD_X_SUM) < 1e-9
    assert abs(float(E2[-1]) - E0) < 1e-8
    assert np.allclose(sol["x"], x2, atol=1e-10)
    assert np.allclose(sol["E"], E2, atol=1e-10)

    for eps in (0.0, 1e-3):
        xh, Eh = plan_horizon(PRICE, LOAD_KWH, PV_KWH, E0, E0, eps=eps)
        xd, Ed, _ = q2.plan_day(PRICE, LOAD_KWH, PV_KWH, E0, eps=eps)
        assert np.allclose(xh, xd, atol=1e-10)
        assert np.allclose(Eh, Ed, atol=1e-10)


def check_exec_golden() -> None:
    """q2.exec_day ≡ qp.exec_segment_hindsight(t0=0, e_end=e_start)（事后下界 LP）。"""
    x, _, _ = q2.plan_day(PRICE, LOAD_KWH, PV_KWH, E0, eps=0.0)
    ex = q2.exec_day(PRICE, LOAD_KWH, PV_KWH, x, E0)
    ex_h = qp.exec_segment_hindsight(PRICE, LOAD_KWH, PV_KWH, x, E0, E0, 0)
    for key in ("c", "d", "s", "E", "e"):
        assert np.allclose(ex[key], ex_h[key], atol=1e-12), key
    sums = (
        float(ex["c"].sum()), float(ex["d"].sum()), float(ex["s"].sum()),
        float(ex["e"].sum()), float(ex["E"][-1]),
    )
    assert np.allclose(sums, GOLD_EXEC, atol=1e-9)


def check_slots_golden() -> None:
    """槽位映射（相位敏感）：hour_to_slots 与 fc_slots 的发布时刻对齐。"""
    y = 100.0 + 10.0 * np.arange(1, 25)
    h2s = q2._hour_to_slots(y)
    assert abs(float(h2s[0]) - 9.166666666666666) < 1e-12
    assert abs(float(h2s[35]) - 159.16666666666666) < 1e-12
    assert abs(float(h2s[71]) - 219.16666666666666) < 1e-12
    assert abs(float(h2s[143]) - 339.1666666666667) < 1e-12
    assert abs(float(h2s.sum()) - 31380.0) < 1e-9

    fc_gold = {
        6: (110.0, 110.0, 174.16666666666666, 279.1666666666667, 24510.0),
        12: (110.0, 110.0, 114.16666666666666, 219.16666666666669, 19470.0),
        18: (110.0, 110.0, 110.0, 159.16666666666669, 16590.0),
    }
    for pub, (g0, g40, g80, g143, gsum) in fc_gold.items():
        f = qp.fc_slots(y, pub)
        assert abs(float(f[0]) - g0) < 1e-12
        assert abs(float(f[40]) - g40) < 1e-12
        assert abs(float(f[80]) - g80) < 1e-12
        assert abs(float(f[143]) - g143) < 1e-12
        assert abs(float(f.sum()) - gsum) < 1e-9


def check_weights_golden() -> None:
    """softmax / simplex_grid / 因果策略版本 / Q2E 缓存哈希输入。"""
    s = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(s, [0.09003057317038046, 0.24472847105479764, 0.6652409557748218])
    grid = simplex_grid(0.2)
    assert grid.shape == (21, 3)
    assert np.allclose(grid[3], [0.0, 0.6, 0.4])
    assert np.allclose(grid[20], [1.0, 0.0, 0.0])
    assert abs(float(grid.sum()) - 21.0) < 1e-9

    assert q2.CAUSAL_POLICY_VERSION == "greedy-reachable-v1"

    m = AdaptiveWeightModel(W=1)
    m.L = np.arange(365 * 144.0).reshape(365, 144) % 50
    m.P = m.L * 0.5
    m.price = np.linspace(0.2, 0.9, 144)
    m.L_typ = m.L[0]
    m.P_typ = m.P[0]
    assert m._cache_key() == GOLD_CACHE_KEY


def check_price_golden() -> None:
    """电价预测唯一实现：q4_price_fit ≡ q4_price_dyn，且数值冻结。"""
    p4 = 0.3 + 0.4 * np.abs(np.sin(np.arange(365 * 144).reshape(365, 144) / 37.0))
    p_typ = p4[100] * 0.9
    v = np.array([0.2, 0.3, 0.5])
    a = fp_fit(30, v, p4, p_typ)
    b = fp_dyn(30, v, p4, p_typ)
    assert np.array_equal(a, b)
    assert abs(float(a.sum()) - 73.61137670353284) < 1e-9
    assert abs(float(a[0]) - 0.48820724638685353) < 1e-12


def check_forecast_golden() -> None:
    """hist_forecast（历史口E 权重）与 latest_forecast（按段最新预报）冻结。"""
    pv_act = np.clip(
        300 * np.sin(np.arange(365 * 144).reshape(365, 144) / 48.0 * np.pi), 0, None
    )
    pv_typ = np.clip(250 * np.sin(np.arange(144) / 48.0 * np.pi), 0, None)
    th = np.zeros((365, 6))
    th[:, 3:6] = np.array([0.3, 0.3, 0.4])
    h = qp.hist_forecast({"pv_act": pv_act, "pv_typ": pv_typ, "TH": th}, 30)
    assert abs(float(h.sum()) - 14289.678975896264) < 1e-9
    assert abs(float(h[0]) - 3.124641251856605e-12) < 1e-12
    assert abs(float(h[143]) - 12.138252243609397) < 1e-12

    data = {
        "fc0": np.tile(100.0 + np.arange(24.0), (365, 1)),
        "fc6": np.tile(200.0 + np.arange(24.0), (365, 1)),
        "fc12": np.tile(300.0 + np.arange(24.0), (365, 1)),
        "fc18": np.tile(400.0 + np.arange(24.0), (365, 1)),
    }
    lf = qp.latest_forecast(data, 10)
    assert abs(float(lf[10]) - 100.75) < 1e-12
    assert abs(float(lf[40]) - 200.0) < 1e-12
    assert abs(float(lf[80]) - 300.4166666666667) < 1e-12
    assert abs(float(lf[120]) - 401.0833333333333) < 1e-12
    assert abs(float(lf.sum()) - 36000.0) < 1e-9


def check_record_replace() -> None:
    """record() 语义：同名章节先删除再追加（临时根目录，不触碰真实报告）。"""
    tmp = Path(tempfile.mkdtemp(prefix="solve_refactor_"))
    pm.init(root=str(tmp))
    q3.record("金标章节", {"a": 1}, note="first-note")
    q3.record("金标章节", {"a": 2}, note="second-note")
    text = (pm.reports_dir() / "RESULTS_REPORT.md").read_text(encoding="utf-8")
    assert text.count("### 金标章节") == 1
    assert "second-note" in text and "first-note" not in text


def check_facade_api() -> None:
    """兼容面：q2 / q3_proto / q2_adaptive 必须保留这些公开名。"""
    from solve import q2_adaptive

    required_q2 = [
        "plan_day", "exec_day", "exec_day_causal", "exec_segment_causal",
        "load_all", "run_deterministic", "run_mc", "run_hedge", "write_result2",
        "verify", "_hour_to_slots", "_events", "_fmt_time",
        "CAUSAL_POLICY_VERSION", "EMERG_MULT", "EPS_THROUGHPUT",
        "N_DAY", "REPORT_START", "T",
    ]
    for name in required_q2:
        assert hasattr(q2, name), f"q2.{name} 缺失"

    required_qp = [
        "adjust_day", "adjust_day_hedge", "causal_residual_pool",
        "exec_segment_hindsight", "fc_slots", "forecast_residual", "hist_forecast",
        "load_extended", "make_smooth_u", "simulate_day", "simulate_day_rt",
        "simulate_day_rt_hedge", "latest_forecast", "perfect_day",
        "EPS_TH", "EMERG_MULT", "LAMBDA_GRID", "W_WINDOW", "pm",
    ]
    for name in required_qp:
        assert hasattr(qp, name), f"q3_proto.{name} 缺失"

    required_q2e = [
        "AdaptiveWeightModel", "softmax", "simplex_grid", "total_of", "record",
        "_verify_result2", "_init_worker", "N_DAY", "REPORT_START",
    ]
    for name in required_q2e:
        assert hasattr(q2_adaptive, name), f"q2_adaptive.{name} 缺失"


if __name__ == "__main__":
    check_lp_golden()
    check_exec_golden()
    check_slots_golden()
    check_weights_golden()
    check_price_golden()
    check_forecast_golden()
    check_record_replace()
    check_facade_api()
    print("solve_refactor_test: PASS")
