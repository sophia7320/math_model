"""模型一致性守卫测试（唯一参数源 + 统一构造）。

校验：
1. ``consistency.py`` 的参数值即规范值；
2. ``q2_tune.seqs_ewma`` 与 ``consistency.ewma_weights_from_table`` 完全一致（EWMA 唯一源）；
3. ``q3_proto`` 的历史预测使用 EWMA 权重（与手工公式一致）；
4. 统一参数被各引擎以默认值引用（κ/m/情景数）。

运行（program/ 下）：uv run python tests/model_consistency_test.py
"""
from __future__ import annotations

import inspect

import numpy as np

import program as pm
from solve import consistency as cs
from solve import q2_tune, q3_proto
from solve.q2_adaptive import AdaptiveWeightModel
from solve.common import ROOT


def main() -> None:
    pm.init(seed=42, root=str(ROOT))

    # 1) 参数值 = 规范值
    assert cs.EWMA_HL == 5.0, cs.EWMA_HL
    assert cs.KAPPA == 1.02, cs.KAPPA
    assert cs.MARGIN == 25.0, cs.MARGIN
    assert cs.N_SCEN == 40, cs.N_SCEN
    assert cs.START_DAY == 31 and cs.PLAN_HORIZON == 288

    # 2) EWMA 序列唯一源
    model = AdaptiveWeightModel().load().build_table()
    seq = q2_tune.seqs_ewma(model, cs.EWMA_HL)
    table = cs.ewma_weights_from_table(model.C, model.grid, hl=cs.EWMA_HL)
    days = list(range(q2_tune.REPORT_START, q2_tune.q2.N_DAY))
    assert len(seq) == len(days)
    for i, d in enumerate(days):
        assert np.allclose(seq[i][0], table[d][0]) and np.allclose(seq[i][1], table[d][1])

    # 3) q3_proto 历史预测 = EWMA 权重手工公式（负荷/光伏两通道）
    data = q3_proto.load_extended()
    assert "EWMA_WU" in data
    for D in (60, 200):
        w, u = data["EWMA_WU"][D]
        pv = q3_proto.hist_forecast(data, D)
        manual_pv = np.clip(u[0] * data["pv_act"][D - 1] + u[1] * data["pv_act"][D - 2]
                            + u[2] * data["pv_typ"], 0.0, None)
        assert np.allclose(pv, manual_pv), D
        load = q3_proto.hist_load_forecast(data, D)
        manual_l = np.clip(w[0] * data["load"][D - 7] + w[1] * data["load"][D - 14]
                           + w[2] * data["load_typ"], 0.0, None)
        assert np.allclose(load, manual_l), D

    # 4) 默认参数来自唯一源
    sig = inspect.signature(q3_proto.plan_two_day)
    assert sig.parameters["kappa"].default == cs.KAPPA
    assert sig.parameters["margin"].default == cs.MARGIN
    sig2 = inspect.signature(q3_proto.simulate_day_rt_hedge)
    assert sig2.parameters["n_scen"].default == cs.N_SCEN
    assert sig2.parameters["kappa"].default == cs.KAPPA
    assert sig2.parameters["margin"].default == cs.MARGIN
    assert sig2.parameters["exec_policy"].default == cs.EXEC_POLICY
    assert cs.EXEC_POLICY == "free"

    print("model consistency test: PASS")


if __name__ == "__main__":
    main()
