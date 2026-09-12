"""临时诊断：q3_roll 单日模拟检查。用完即删。"""
import numpy as np

import program as pm
from solve import q3_proto as qp
from solve.common import ROOT
from solve.q3_roll import simulate_day_roll

pm.init(seed=42, root=str(ROOT))
data = qp.load_extended()
data["U_SMOOTH"] = qp.make_smooth_u(data, 0.1)

for D in (31, 32, 33):
    r = simulate_day_roll(data, D, 6000.0)
    print(
        f"D={D}: 计划 {r['plan_cost']:8.0f} 偏差 {r['dev']:7.0f} 紧急 {r['emerg']:7.0f} "
        f"总 {r['total']:8.0f} | E_start {r['E_start']:.0f} E_end {r['E_end']:.0f} "
        f"| 调整量 {r['adj_abs_kwh']:7.0f} | 场景 {r['scenario_count']}/{r['scenario_max_day']}"
    )
    # 约束自检
    bal = (data["pv_act"][D] / 6 + r["x_final"] + r["d"] + r["e"]
           - data["load"][D] / 6 - r["c"] - r["s"])
    print(f"    平衡残差 max={np.abs(bal).max():.2e} kWh；"
          f"SOC范围 [{r['E'].min():.1f}, {r['E'].max():.1f}]")
