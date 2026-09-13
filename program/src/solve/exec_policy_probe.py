"""执行器"段末硬目标"机制对照探针（小样本，供论文与验收引用）。

背景：原逐槽因果执行器要求 SOC 在 6:00/12:00/18:00/24:00 跟踪规划轨迹
（可达性投影），实测会迫使"紧急购电充能 / 扣留放电"，在段末小时形成紧急购电尖峰。
本探针在统一结构（2 日滚动 + EWMA + κ/m + 40 情景联合对冲）下，用小样本对比：

- ``target``：段末目标硬跟踪（旧口径）；
- ``dayend``：仅当日最后一段跟踪日末目标；
- ``free``  ：无任何段末目标，仅容量/功率约束（现行统一口径 EXEC_POLICY）。

样本（小数据，避免大仿真）：
- 夏季应力样本：2025-06-01 ~ 07-30（60 天）；
- 秋季应力样本：2025-10-16 ~ 12-01（47 天）。

结论（2026-09-13，末段修复后重跑）：段末目标分两段计损——中段目标
（target→dayend）夏季 +40.3 万/12.2%、秋季 +8.4 万/4.3%；末段目标
（dayend→free）再 +4.6 万/1.6%、+0.5 万/0.3%；合计 free 相对 target
−44.9 万/−13.6%、−8.9 万/−4.5%；SOC 仍全幅利用（[1200, 10800]）。

运行（program/ 下）：uv run python -m solve.exec_policy_probe
"""
from __future__ import annotations

import time

import numpy as np

import program as pm
from solve import consistency as cs
from solve import q3_proto as qp
from solve.common import ROOT

SAMPLES = {
    "夏季（2025-06-01~07-30）": list(range(151, 211)),
    "秋季（2025-10-16~12-01）": list(range(288, 335)),
}
POLICIES = ["target", "dayend", "free"]


def run_sample(data, days, policy, lam=0.7) -> dict:
    t0 = time.time()
    E = float(qp.E0)
    tot = plan = dev = em = spill = 0.0
    win_em = np.zeros(4)
    emin, emax = 1e9, -1e9
    for D in days:
        r = qp.simulate_day_rt_hedge(
            data, D, lam, adj_lam=lam, e_start=E, exec_policy=policy)
        tot += r["total"]
        plan += r["plan_cost"]
        dev += r["deviation_cost"]
        em += r["emerg"]
        spill += float(r["s"].sum())
        e = r["e"]
        win_em += np.array([e[0:36].sum(), e[36:72].sum(),
                            e[72:108].sum(), e[108:144].sum()])
        emin = min(emin, float(r["E"].min()))
        emax = max(emax, float(r["E"].max()))
        E = float(r["E_end"])
    return {"样本": days and data["dates"][days[0]], "策略": policy,
            "天数": len(days),
            "总费用/万元": round(tot / 1e4, 1),
            "计划购电费/万元": round(plan / 1e4, 1),
            "偏差费/万元": round(dev / 1e4, 1),
            "紧急购电费/万元": round(em / 1e4, 1),
            "弃电量/kWh": round(spill, 1),
            "紧急0-6h/万元": round(win_em[0] / 1e4, 1),
            "紧急6-12h/万元": round(win_em[1] / 1e4, 1),
            "紧急12-18h/万元": round(win_em[2] / 1e4, 1),
            "紧急18-24h/万元": round(win_em[3] / 1e4, 1),
            "SOC最小/kWh": round(emin, 0),
            "SOC最大/kWh": round(emax, 0),
            "用时/s": round(time.time() - t0, 0)}


def record(section: str, data_, note: str = "") -> None:
    path = pm.reports_dir() / "RESULTS_REPORT.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        marker = f"### {section}"
        pos = text.find(marker)
        if pos != -1:
            end = text.find("\n### ", pos + len(marker))
            text = text[:pos] if end == -1 else text[:pos] + text[end + 1:]
            path.write_text(text, encoding="utf-8")
    pm.record_result(section, data_, note=note)


def main() -> None:
    import pandas as pd

    pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()
    rows = []
    for name, days in SAMPLES.items():
        for policy in POLICIES:
            r = run_sample(data, days, policy)
            r["样本"] = name
            rows.append(r)
            print(f"  [{name}][{policy:>6}] 总 {r['总费用/万元']:6.1f} 万 "
                  f"（计划 {r['计划购电费/万元']:.1f} + 偏差 {r['偏差费/万元']:.1f} "
                  f"+ 紧急 {r['紧急购电费/万元']:.1f}）弃电 {r['弃电量/kWh']:.0f} kWh")
    df = pd.DataFrame(rows)
    pm.save_outputs(df, "exec_policy_probe")
    record(
        "执行器段末目标机制对照（小样本）",
        df,
        note=("统一结构下对比 target（段末硬目标）/ dayend（仅末段跟踪日末目标）/ "
              "free（完全无目标，现行统一口径 consistency.EXEC_POLICY）；"
              "末段修复后重跑：中段目标夏季 +40.3 万（12.2%）、秋季 +8.4 万（4.3%），"
              "末段目标再 +4.6 万（1.6%）、+0.5 万（0.3%）；"
              "完整探针见 code/outputs/exec_policy_probe.csv 与 "
              "reports/执行器段末目标实验.md。"),
    )
    print("完成。")


if __name__ == "__main__":
    main()
