"""C 题因果执行与无前视残差池的轻量回归测试。

运行（program/ 下）：uv run python tests/solve_causal_test.py
"""
from __future__ import annotations

import numpy as np

from solve import q2
from solve import q3_proto as qp


def check_dispatch() -> None:
    # 完全平衡：不充放、不紧急购电
    ex = q2.exec_segment_causal(
        np.full(6, 10.0), np.zeros(6), np.full(6, 10.0), 6000.0, 6000.0
    )
    assert np.allclose(ex["E"], 6000.0)
    assert np.allclose(ex["c"], 0.0)
    assert np.allclose(ex["d"], 0.0)
    assert np.allclose(ex["e"], 0.0)

    # 先富余 100 kWh，再缺口 81 kWh；双向 0.9 效率后恰好回到起点
    ex = q2.exec_segment_causal(
        np.array([0.0, 81.0]), np.array([100.0, 0.0]), np.zeros(2), 6000.0, 6000.0
    )
    assert np.allclose(ex["c"], [100.0, 0.0], atol=1e-8)
    assert np.allclose(ex["d"], [0.0, 81.0], atol=1e-8)
    assert np.allclose(ex["e"], 0.0, atol=1e-8)
    assert abs(ex["E"][-1] - 6000.0) < 1e-8

    # 约束全回代：功率平衡、SOC 动态、终端、充放互斥
    load = np.array([1000.0, 1000.0])
    pv = np.zeros(2)
    buy = np.zeros(2)
    ex = q2.exec_segment_causal(load, pv, buy, 6000.0, 6000.0)
    balance = pv + buy + ex["d"] + ex["e"] - load - ex["c"] - ex["s"]
    dynamics = np.diff(np.r_[6000.0, ex["E"]]) - (
        q2.ETA * ex["c"] - ex["d"] / q2.ETA
    )
    assert np.max(np.abs(balance)) < 1e-8
    assert np.max(np.abs(dynamics)) < 1e-8
    assert abs(ex["E"][-1] - 6000.0) < 1e-8
    assert not np.any((ex["c"] > 1e-9) & (ex["d"] > 1e-9))

    # 随机场景回代（终端限制在 m 步可达域内，与真实使用一致）
    rng = np.random.default_rng(0)
    max_up, max_down = q2.ETA * q2.P_MAX_E, q2.P_MAX_E / q2.ETA
    for _ in range(100):
        m = int(rng.integers(1, 40))
        ld = rng.uniform(0, 4000, m)
        pv = rng.uniform(0, 4000, m)
        x = rng.uniform(0, 4000, m)
        e0 = rng.uniform(q2.E_MIN, q2.E_MAX)
        band = min(m * max_up, m * max_down) * 0.9
        e1 = float(np.clip(e0 + rng.uniform(-band, band), q2.E_MIN, q2.E_MAX))
        ex = q2.exec_segment_causal(ld, pv, x, e0, e1)
        bal = pv + x + ex["d"] + ex["e"] - ld - ex["c"] - ex["s"]
        dyn = np.diff(np.r_[e0, ex["E"]]) - (q2.ETA * ex["c"] - ex["d"] / q2.ETA)
        assert np.max(np.abs(bal)) < 1e-7 and np.max(np.abs(dyn)) < 1e-7
        assert ex["c"].max() <= q2.P_MAX_E + 1e-8
        assert ex["d"].max() <= q2.P_MAX_E + 1e-8
        assert abs(ex["E"][-1] - e1) < 1e-7


def check_causal_pool() -> None:
    months = np.repeat(np.arange(1, 13), [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    data = {"months": months}
    for D in (31, 45, 59, 120, 364):
        pool = qp.causal_residual_pool(data, D)
        assert len(pool) > 0
        assert int(pool.max()) < D
        assert int(pool.min()) >= max(14, D - 90)


def check_hedge_probability_weights() -> None:
    """Q2 等概率场景补救成本必须除以场景数。"""
    captured = {}
    original = q2.pm.optimize.solve_lp

    class FakeResult:
        success = True
        message = ""
        x = np.zeros(q2.T + 4 * 5 * q2.T)

    def fake_solve(c, **kwargs):
        captured["c"] = np.asarray(c, dtype=float)
        return FakeResult()

    q2.pm.optimize.solve_lp = fake_solve
    try:
        q2.hedge_day(
            np.ones(q2.T), np.zeros(q2.T), np.zeros(24),
            np.zeros((1, 24)), 6000.0,
            n_scen=4, rng=np.random.default_rng(0),
        )
    finally:
        q2.pm.optimize.solve_lp = original

    c = captured["c"]
    assert np.allclose(c[:q2.T], 1.0)
    for s in range(4):
        base = q2.T + s * 5 * q2.T
        assert np.allclose(c[base + 4 * q2.T:base + 5 * q2.T], 5.0 / 4.0)

    fc = np.full(24, 100.0)
    residual = np.full(24, 20.0)  # 预报比实际高 20
    assert np.allclose(q2._scenario_from_residual(fc, residual), 80.0)


def check_publication_residual() -> None:
    """6:00 残差不得拼入同一历史日 12:00/18:00 的预报。"""
    n = 20
    data = {
        "fc0": np.zeros((n, 24)),
        "fc6": np.full((n, 24), 6.0),
        "fc12": np.full((n, 24), 12.0),
        "fc18": np.full((n, 24), 18.0),
        "pv_act": np.zeros((n, q2.T)),
    }
    expected = qp.fc_slots(data["fc6"][15], 6) / 6.0
    actual = qp.forecast_residual(data, 15, 6, None)
    assert np.allclose(actual, expected)
    # 12:00 以后仍应保持 6:00 发布口径，而不是历史“最新预报”拼接口径。
    assert not np.allclose(actual[72:108], qp.latest_forecast(data, 15)[72:108] / 6.0)


if __name__ == "__main__":
    check_dispatch()
    check_causal_pool()
    check_hedge_probability_weights()
    check_publication_residual()
    print("solve_causal_test: PASS")
