"""C 题 问题四：实时波动电价下的重算（Q2 层 / Q3 层）。

口径：
- 电价 G（题面主口径）：未来电价已由附件 4 给出，决策直接使用对应日价格；
  H（仅历史价格）是额外的信息受限扩展，用滚动费用标定 v（W=7）+ β=0.1 动态修正。
- 负荷由目标日前历史数据预测；光伏用附件 3 的 0:00 官方预报，
  2 日视野中的次日用历史自适应预报（qp.hist_forecast，无前视）。
- 执行：逐槽因果（q2.exec_segment_causal，不读未来实际值）。
- 储能结构：日循环 vs 2 日滚动（跨日连续），按费用择优。

运行（program/ 下）：
    uv run python -m solve.q4 --probe        # 结构对比探针
    uv run python -m solve.q4 --result4-2    # 生成 result4-2.xlsx（待实现）
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import openpyxl
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

import program as pm
from solve import consistency as cs
from solve import q2
from solve import q3_proto as qp
from solve.common import DATA_C, RESULTS_DIR, ROOT, E0, E_MAX, E_MIN, ETA, P_MAX_E, T
from solve.q4_price_fit import forecast_price

N_DAY = 365
REPORT_START = 31       # 2025-02-01
BETA = 0.1              # 电价权重的动态修正（新旧混合）
W_V = 7                 # 滚动标定窗口（天）
EPS_PLAN = 1e-3


# ---------------------------------------------------------------------------
# 数据与电价预测
# ---------------------------------------------------------------------------
def load_q4_data() -> tuple[dict, np.ndarray, np.ndarray]:
    """附件 1/2/3 + 附件 4 电价 + 统一 EWMA h=5 权重（consistency.py）。"""
    data = qp.load_extended()
    df4 = pm.read_table(DATA_C / "附件4.xlsx")
    p4 = df4.iloc[:, 1:145].to_numpy(float)
    return data, p4, data["price"]


def rolling_v_seq() -> np.ndarray:
    """逐日电价三源权重：滚动费用标定（W=7）+ 动态修正 β=0.1（无前视）。

    复用 q4_price_fit 的逐日费用表（含因果执行口径）；D<31 预热用纯典型日。
    """
    z = np.load(ROOT / "code" / "outputs" / "q4_price_fit_daily.npz")
    daily_costs = z["daily_costs"]
    grid = [tuple(v) for v in z["grid"]]
    seq = np.tile(np.array([0.0, 0.0, 1.0]), (N_DAY, 1))
    prev = None
    for D in range(REPORT_START, N_DAY):
        hist = daily_costs[max(REPORT_START, D - W_V):D]
        v_new = (np.array(grid[int(np.argmin(hist.mean(axis=0)))])
                 if len(hist) else np.array([0.0, 0.0, 1.0]))
        prev = v_new if prev is None else BETA * v_new + (1.0 - BETA) * prev
        seq[D] = prev
    return seq


def price_forecast_d(D: int, v: np.ndarray, p4: np.ndarray, p_typ: np.ndarray) -> np.ndarray:
    """目标日 D 的电价预测（D≥7）；D<7 退化为典型日。"""
    return forecast_price(D, v, p4, p_typ) if D >= 7 else p_typ


def price_forecast_next(D: int, v: np.ndarray, p4: np.ndarray, p_typ: np.ndarray) -> np.ndarray:
    """day D+1 的预测（仅用 day D 可得信息）：P(D) 未知，用 P(D−1) 替代。"""
    d1 = min(D + 1, N_DAY - 1)
    return v[0] * p4[d1 - 2] + v[1] * p4[d1 - 7] + v[2] * p_typ


# ---------------------------------------------------------------------------
# LP：2 日滚动计划（跨日连续，末端回到 e_terminal）
# ---------------------------------------------------------------------------
def plan_horizon(price_h, load_h, pv_h, e_start, e_terminal=None, eps=EPS_PLAN):
    """多日计划 LP：min Σp·x + eps·Σ(c+d)；储能跨日连续 E(0)=e_start。

    ``e_terminal=None``（默认）时规划窗口末端完全自由（不锚定终值）；
    传入数值时锚定 E(H)=e_terminal（供结构对照）。
    """
    H = len(price_h)
    n = 5 * H
    c_obj = np.zeros(n)
    c_obj[0:H] = price_h
    c_obj[H:3 * H] = eps

    n_rows = 2 * H + (1 if e_terminal is not None else 0)
    A_eq = lil_matrix((n_rows, n))
    b_eq = np.zeros(n_rows)
    for t in range(H):
        A_eq[t, 0 * H + t] = 1.0
        A_eq[t, 1 * H + t] = -1.0
        A_eq[t, 2 * H + t] = 1.0
        A_eq[t, 3 * H + t] = -1.0
        b_eq[t] = load_h[t] - pv_h[t]
    for t in range(H):
        r = H + t
        A_eq[r, 4 * H + t] = 1.0
        if t:
            A_eq[r, 4 * H + t - 1] = -1.0
        A_eq[r, 1 * H + t] = -ETA
        A_eq[r, 2 * H + t] = 1.0 / ETA
        b_eq[r] = e_start if t == 0 else 0.0
    if e_terminal is not None:
        A_eq[2 * H, 4 * H + H - 1] = 1.0
        b_eq[2 * H] = float(e_terminal)

    bounds = (
        [(0.0, None)] * H
        + [(0.0, P_MAX_E)] * H
        + [(0.0, P_MAX_E)] * H
        + [(0.0, float(v)) for v in pv_h]
        + [(E_MIN, E_MAX)] * H
    )
    res = pm.optimize.solve_lp(c_obj, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"多日计划 LP 失败：{res.message}")
    return res.x[0:H], res.x[4 * H:5 * H]


# ---------------------------------------------------------------------------
# 全年模拟
# ---------------------------------------------------------------------------
def run_year(data, p4, p_typ, vseq, price_mode="H", storage="2day",
             start=REPORT_START, end=N_DAY) -> list[dict]:
    """全年滚动模拟（统一口径：EWMA 预测 + κ/m；2-1 起 E0 起步）。

    返回逐日记录（含 x/c/d/s/e/E 与费用，REPORT_START 起计费）。
    """
    load, pv_act = data["load"], data["pv_act"]
    E = E0
    recs: list[dict] = []
    for D in range(start, end):
        if price_mode == "G":
            p_d = p4[D]
            p_d1 = p4[min(D + 1, N_DAY - 1)]
        else:
            v = vseq[D]
            p_d = price_forecast_d(D, v, p4, p_typ)
            p_d1 = price_forecast_next(D, v, p4, p_typ) if D + 1 < N_DAY else p_d
        load_act = load[D] / 6.0
        load_fc_d = cs.kappa_load(qp.hist_load_forecast_asof(data, D, D)) / 6.0
        pv_fc_d = cs.margin_pv(qp.hist_forecast_asof(data, D, D)) / 6.0

        if storage == "daily":
            x_day, E_plan, _ = q2.plan_day(p_d, load_fc_d, pv_fc_d, E, eps=EPS_PLAN)
            e_end = float(np.clip(E_plan[-1], E_MIN, E_MAX))
        else:
            D1 = min(D + 1, N_DAY - 1)
            load_fc_d1 = cs.kappa_load(qp.hist_load_forecast_asof(data, D + 1, D)) / 6.0
            pv_fc_d1 = cs.margin_pv(qp.hist_forecast_asof(data, D + 1, D)) / 6.0
            xh, Eh = plan_horizon(
                np.concatenate([p_d, p_d1]),
                np.concatenate([load_fc_d, load_fc_d1]),
                np.concatenate([pv_fc_d, pv_fc_d1]),
                E, None,
            )
            x_day = xh[:T]
            e_end = float(np.clip(Eh[T - 1], E_MIN, E_MAX))

        # 统一执行策略（free）：无段末硬目标
        ex = q2.exec_segment_causal(load_act, pv_act[D] / 6.0, x_day, E, None)
        rec = {"D": D, "x": x_day, "c": ex["c"], "d": ex["d"], "s": ex["s"],
               "e": ex["e"], "E": ex["E"], "E_start": float(E)}
        if D >= REPORT_START:
            rec["plan_cost"] = float(p4[D] @ x_day)
            rec["emerg"] = float(q2.EMERG_MULT * (p4[D] @ ex["e"]))
            rec["total"] = rec["plan_cost"] + rec["emerg"]
        recs.append(rec)
        E = float(ex["E"][-1])
    return recs


def total_of(recs) -> dict:
    days = [r for r in recs if "total" in r]
    return {
        "plan": sum(r["plan_cost"] for r in days),
        "emerg": sum(r["emerg"] for r in days),
        "total": sum(r["total"] for r in days),
        "emerg_kwh": sum(r["e"].sum() for r in days),
        "days": len(days),
    }


# ---------------------------------------------------------------------------
# 校验 / 写表 / 图表 / 报告
# ---------------------------------------------------------------------------
def verify(recs, data, p4) -> dict:
    """约束回代：功率平衡、SOC 动态、区间、充放互斥。"""
    load, pv_act = data["load"], data["pv_act"]
    bal = dyn = 0.0
    emin, emax = np.inf, -np.inf
    overlap = 0
    for r in recs:
        D = r["D"]
        if D < REPORT_START:
            continue
        res = pv_act[D] / 6.0 + r["x"] + r["d"] + r["e"] - load[D] / 6.0 - r["c"] - r["s"]
        bal = max(bal, float(np.abs(res).max()))
        E = np.concatenate([[r["E_start"]], r["E"]])
        dyn = max(dyn, float(np.abs(np.diff(E) - (ETA * r["c"] - r["d"] / ETA)).max()))
        emin = min(emin, float(r["E"].min()))
        emax = max(emax, float(r["E"].max()))
        overlap += int(np.sum((r["c"] > 1e-7) & (r["d"] > 1e-7)))
    return {
        "功率平衡最大残差/kWh": bal,
        "储能动态最大残差/kWh": dyn,
        "储电量最小值/kWh": emin,
        "储电量最大值/kWh": emax,
        "约束区间检查": bool(emin >= E_MIN - 1e-6 and emax <= E_MAX + 1e-6),
        "同槽同时充放次数": overlap,
    }


def write_result4_2(data, p4, recs, out_name="result4-2.xlsx"):
    """按附件 5 模板生成 results/result4-2.xlsx（计划 / 充放电 / 紧急，334 天）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result4-2.xlsx")
    days = list(range(REPORT_START, N_DAY))

    ws = wb["计划购电量"]
    for i, D in enumerate(days, start=2):
        x = recs[D]["x"]
        ws.cell(row=i, column=1, value=pd.Timestamp(data["dates"][D]).to_pydatetime())
        for t in range(T):
            ws.cell(row=i, column=2 + t, value=float(x[t]))
        ws.cell(row=i, column=146, value=float(x.sum()))
        ws.cell(row=i, column=147, value=float(p4[D] @ x))

    ws = wb["充放电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for D in days:
        r = recs[D]
        for b in range(6):
            rr = row + b
            if b == 0:
                ws.cell(rr, 1, pd.Timestamp(data["dates"][D]).to_pydatetime())
            ws.cell(rr, 2, f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(rr, 3, float(r["c"][24 * b:24 * (b + 1)].sum()))
            ws.cell(rr, 4, float(r["d"][24 * b:24 * (b + 1)].sum()))
            if b == 0:
                ws.cell(rr, 5, "00:00")
                ws.cell(rr, 6, float(r["E_start"]))
            elif b == 1:
                ws.cell(rr, 5, "24:00")
                ws.cell(rr, 6, float(r["E"][-1]))
        row += 6

    ws = wb["紧急购电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for D in days:
        for j, (a, b, kwh) in enumerate(q2._events(recs[D]["e"])):
            if j == 0:
                ws.cell(row, 1, pd.Timestamp(data["dates"][D]).to_pydatetime())
            ws.cell(row, 2, f"{q2._fmt_time(a * 10)}-{q2._fmt_time((b + 1) * 10)}")
            ws.cell(row, 3, round(kwh, 4))
            row += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / out_name
    wb.save(p)
    wb.close()
    return p


def verify_result4_2(path, data, p4, recs) -> dict:
    """回读 result4-2 并与逐日记录勾稽。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    chk = {"sheets": "/".join(wb.sheetnames)}
    days = list(range(REPORT_START, N_DAY))

    ws = wb["计划购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    chk["计划表行数"] = len(rows)
    chk["计划购电量勾稽差/kWh"] = float(abs(
        sum(float(r[145]) for r in rows) - sum(float(recs[D]["x"].sum()) for D in days)))
    chk["计划购电费勾稽差/元"] = float(abs(
        sum(float(r[146]) for r in rows) - sum(float(p4[D] @ recs[D]["x"]) for D in days)))

    ws = wb["充放电量"]
    crows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None]
    chk["充放表行数"] = len(crows)
    chk["充放电量勾稽差/kWh"] = float(abs(
        sum(float(r[2]) for r in crows if r[2] is not None)
        - sum(float(recs[D]["c"].sum()) for D in days)))

    ws = wb["紧急购电量"]
    kw = sum(float(r[2]) for r in ws.iter_rows(min_row=2, values_only=True) if r[2] is not None)
    chk["紧急表电量勾稽差/kWh"] = float(abs(
        kw - sum(float(recs[D]["e"].sum()) for D in days)))
    wb.close()
    return chk


def make_figures(data, recs_h, recs_g, s_h, s_g, daily) -> None:
    pm.bar_group(
        ["计划购电费", "紧急购电费"],
        {"H·历史预测": [s_h["plan"] / 1e4, s_h["emerg"] / 1e4],
         "G·题面已知电价": [s_g["plan"] / 1e4, s_g["emerg"] / 1e4]},
        ylabel="334 天费用 / 万元", save="Q4_H与G费用对比",
    )
    days = list(range(REPORT_START, N_DAY))
    e_h = np.array([float(recs_h[D]["e"].sum()) for D in days])
    e_g = np.array([float(recs_g[D]["e"].sum()) for D in days])
    fig, ax = pm.line(
        np.arange(len(days)), [e_h, e_g],
        labels=["H·历史预测", "G·题面已知电价"],
        xlabel="日期（2025-02-01 起）", ylabel="紧急购电量 / kWh",
    )
    pm.save_fig(fig, "Q4_逐日紧急购电",
                data=pd.DataFrame({"日期": [data["dates"][D] for D in days],
                                   "H紧急量/kWh": e_h, "G紧急量/kWh": e_g}))


def record(section: str, data_, note: str = "") -> None:
    """写结果报告：先删同名旧章节再追加（pm.record_result 为追加模式）。"""
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


def run_result42() -> None:
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q4")
    t0 = time.time()
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    recs_h = run_year(data, p4, p_typ, vseq, price_mode="H", storage="2day")
    recs_g = run_year(data, p4, p_typ, vseq, price_mode="G", storage="2day")
    s_h, s_g = total_of(recs_h), total_of(recs_g)
    chk = verify(recs_g, data, p4)
    print(f"问题四 Q2 层：H {s_h['total']/1e4:.1f} 万（计划 {s_h['plan']/1e4:.1f} + "
          f"紧急 {s_h['emerg']/1e4:.1f}）；G {s_g['total']/1e4:.1f} 万；"
          f"电价信息价值 {100*(s_h['total']-s_g['total'])/s_h['total']:.2f}%（{time.time()-t0:.0f}s）")

    p = write_result4_2(data, p4, recs_g)
    excel = verify_result4_2(p, data, p4, recs_g)
    log.info("result4-2.xlsx -> {}", p)

    days = list(range(REPORT_START, N_DAY))
    daily = pd.DataFrame({
        "日期": [data["dates"][D] for D in days],
        "计划购电费/元": [recs_g[D]["plan_cost"] for D in days],
        "紧急购电量/kWh": [float(recs_g[D]["e"].sum()) for D in days],
        "紧急购电费/元": [recs_g[D]["emerg"] for D in days],
        "总费用/元": [recs_g[D]["total"] for D in days],
    })
    pm.save_outputs(daily, "q4_result42_daily")
    make_figures(data, recs_h, recs_g, s_h, s_g, daily)

    record(
        "问题四 结果：Q2 层（result4-2，2025-02-01 ~ 12-31）",
        {
            "题面口径 G·总费用/万元": round(s_g["total"] / 1e4, 1),
            "G·计划购电费/万元": round(s_g["plan"] / 1e4, 1),
            "G·紧急购电费/万元": round(s_g["emerg"] / 1e4, 1),
            "G·紧急购电量/kWh": float(s_g["emerg_kwh"]),
            "扩展口径 H·历史价格预测/万元": round(s_h["total"] / 1e4, 1),
            "电价信息价值（H 比 G 高）": f"{100 * (s_h['total'] - s_g['total']) / s_h['total']:.2f}%",
            "储能结构": "2 日滚动（跨日连续）",
            **excel,
        },
        note=(
            "题面口径 G：附件 4 已给未来电价，0:00 用历史负荷预测 + 附件 3 的 0:00 光伏预报；"
            "采用 2 日滚动计划、逐槽因果执行并按附件 4 结算。H 口径不读取目标日价格，"
            "仅作为信息受限扩展，不写入正式结果表。"
            "图 figures/Q4_H与G费用对比.pdf、Q4_逐日紧急购电.pdf；"
            "逐日表 code/outputs/q4_result42_daily.csv；结构探针 code/outputs/q4_structure_probe.csv。"
        ),
    )
    record("问题四 约束与一致性校验", chk,
           note="功率平衡、SOC 递推、区间与充放互斥全部回代；残差应为浮点误差量级。")


# ---------------------------------------------------------------------------
# Q3 层：0:00 计划 + 6/12/18 调整（+6/12 对冲），结算用附件 4 真实价
# ---------------------------------------------------------------------------
def simulate_day_q3(data, p4, p_typ, vseq, D, price_mode="H", storage="daily",
                    adj_hours=(6, 12, 18), hedge=True, n_scen=cs.N_SCEN, seed=7,
                    lam=0.7, adj_lam=0.7, kappa=cs.KAPPA, margin=cs.MARGIN,
                    e_start=E0, eps=qp.EPS_TH) -> dict:
    """Q4 Q3 层单日模拟（统一口径：EWMA 预测 + κ/m + （Δ负荷, Δ光伏）联合残差场景）。

    H 口径：决策价格 = 三源预测（滚动 v）；G：决策价格 = 当天真实价。
    storage='daily'：日循环（计划/调整末端回 E0）；'2day'：2 日滚动、跨日连续。
    结算一律用附件 4 真实价：买入 + 偏差费（0.5p|x−x⁰|）+ 5 倍紧急。
    """
    p_dec = p4[D] if price_mode == "G" else price_forecast_d(D, vseq[D], p4, p_typ)
    load_act_kwh = data["load"][D] / 6.0
    load_nom = cs.kappa_load(qp.hist_load_forecast_asof(data, D, D), kappa) / 6.0
    pv_act_kwh = data["pv_act"][D] / 6.0
    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = qp.hist_forecast(data, D)
    pv_plan = cs.margin_pv(np.clip(lam * f0 + (1.0 - lam) * ph, 0.0, None),
                           margin) / 6.0

    if storage == "daily":
        x_plan, E_plan, _ = q2.plan_day(p_dec, load_nom, pv_plan, e_start, eps=eps)
        e_day_end = E0
    else:
        D1 = min(D + 1, N_DAY - 1)
        p_dec1 = (p4[D1] if price_mode == "G"
                  else price_forecast_next(D, vseq[D], p4, p_typ))
        x_plan, E_plan, _ = qp.plan_two_day(
            data, D, lam, e_start, kappa=kappa, margin=margin,
            price2=np.concatenate([p_dec, p_dec1]))
        e_day_end = float(E_plan[-1])

    x_seq = x_plan.copy()
    E_target = E_plan.copy()
    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    s_all = np.zeros(T)
    E_exec = np.zeros(T)
    scenario_days: list[int] = []
    E_now = e_start
    t = 0
    for pub in (6, 12, 18):
        t1 = pub * 6
        # 统一执行策略（free）：无段末硬目标
        seg = q2.exec_segment_causal(
            load_act_kwh[t:t1], pv_act_kwh[t:t1], x_seq[t:t1], E_now, None,
        )
        e_total[t:t1] = seg["e"]
        c_all[t:t1] = seg["c"]
        d_all[t:t1] = seg["d"]
        s_all[t:t1] = seg["s"]
        E_exec[t:t1] = seg["E"]
        E_now = float(seg["E"][-1])
        if pub in adj_hours:
            off = qp.fc_slots(data[f"fc{pub}"][D], pub)
            src = off if adj_lam is None else adj_lam * off + (1.0 - adj_lam) * ph
            pv_new = cs.margin_pv(np.clip(src, 0.0, None), margin) / 6.0
            if pub in (6, 12) and hedge and D > 14:
                pool = qp.causal_residual_pool(data, D)
                rng = np.random.default_rng(np.random.SeedSequence([seed, D, pub]))
                sel = rng.choice(len(pool), size=max(0, n_scen - 1), replace=True)
                scenario_days.extend(int(pool[k]) for k in sel)
                RL, RP = qp.joint_residual_blocks(data, pool, pub, adj_lam, kappa, margin)
                load_scens = [load_nom[t1:]] + [
                    np.clip(load_nom[t1:] + RL[k][t1:], 0.0, None) for k in sel]
                pv_scens = [pv_new[t1:]] + [
                    np.clip(pv_new[t1:] + RP[k][t1:], 0.0, None) for k in sel]
                x_adj, E_adj = qp.adjust_day_hedge(
                    p_dec, load_nom, pv_scens, x_plan, E_now, t1,
                    e_terminal=e_day_end, load_scens=load_scens)
            else:
                x_adj, E_adj = qp.adjust_day(
                    p_dec, load_nom, pv_new, x_plan, E_now, t1,
                    e_terminal=e_day_end)
            x_seq[t1:] = x_adj
            E_target[t1:] = E_adj
        t = t1
    seg = q2.exec_segment_causal(
        load_act_kwh[108:144], pv_act_kwh[108:144], x_seq[108:144], E_now, None,
    )
    e_total[108:] = seg["e"]
    c_all[108:] = seg["c"]
    d_all[108:] = seg["d"]
    s_all[108:] = seg["s"]
    E_exec[108:] = seg["E"]

    x_final = x_seq.copy()
    buy = float(p4[D] @ x_final)
    dev = float((0.5 * p4[D] * np.abs(x_final - x_plan)).sum())
    emerg = float(q2.EMERG_MULT * (p4[D] @ e_total))
    return {
        "D": D, "x_plan": x_plan, "x_final": x_final,
        "c": c_all, "d": d_all, "s": s_all, "E": E_exec, "E_start": float(e_start),
        "e": e_total, "plan_cost": float(p4[D] @ x_plan),
        "buy": buy, "dev": dev, "emerg": emerg, "total": buy + dev + emerg,
        "E_end": float(E_exec[-1]),
        "scenario_max_day": max(scenario_days, default=-1),
    }


def run_year_q3(data, p4, p_typ, vseq, price_mode="H", storage="daily",
                adj_hours=(6, 12, 18), hedge=True, n_scen=cs.N_SCEN, seed=7,
                lam=0.7, adj_lam=0.7) -> list[dict]:
    E = E0
    recs = []
    for D in range(REPORT_START, N_DAY):
        r = simulate_day_q3(
            data, p4, p_typ, vseq, D, price_mode=price_mode, storage=storage,
            adj_hours=adj_hours, hedge=hedge, n_scen=n_scen, seed=seed,
            lam=lam, adj_lam=adj_lam,
            e_start=(E0 if storage == "daily" else E),
        )
        recs.append(r)
        E = r["E_end"] if storage == "2day" else E0
    return recs


def total_of_q3(recs) -> dict:
    days = [r for r in recs if r["D"] >= REPORT_START]
    return {
        "plan": sum(r["plan_cost"] for r in days),
        "buy": sum(r["buy"] for r in days),
        "dev": sum(r["dev"] for r in days),
        "emerg": sum(r["emerg"] for r in days),
        "total": sum(r["total"] for r in days),
        "emerg_kwh": sum(float(r["e"].sum()) for r in days),
        "adj_kwh": sum(float(np.abs(r["x_final"] - r["x_plan"]).sum()) for r in days),
        "days": len(days),
    }


def probe_q3() -> None:
    """Q4-3 结构探针：价格口径 × 储能结构 × 调整/对冲。"""
    pm.init(seed=42, root=str(ROOT))
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    cfgs = [
        ("H", "daily", (), False, 1.0, 1.0),
        ("H", "daily", (6, 12, 18), False, 1.0, 1.0),
        ("H", "daily", (6, 12, 18), True, 1.0, 1.0),
        ("H", "daily", (6, 12, 18), True, 0.7, 0.7),
        ("H", "2day", (6, 12, 18), True, 0.7, 0.7),
        ("G", "2day", (6, 12, 18), True, 0.7, 0.7),
    ]
    rows = []
    for mode, storage, adj, hedge, lam, adj_lam in cfgs:
        t0 = time.time()
        recs = run_year_q3(data, p4, p_typ, vseq, price_mode=mode,
                           storage=storage, adj_hours=adj, hedge=hedge,
                           lam=lam, adj_lam=adj_lam)
        s = total_of_q3(recs)
        tag = (f"{mode}·{storage}·{'三点' if adj else '无调整'}"
               f"{'+对冲' if hedge else ''}{'·组合' if lam < 0.9 else '·官方'}")
        print(f"  [{tag:<16}] 总 {s['total']/1e4:7.1f} 万（买入 {s['buy']/1e4:.1f} + "
              f"偏差 {s['dev']/1e4:.1f} + 紧急 {s['emerg']/1e4:.1f}）"
              f" 调整量 {s['adj_kwh']:,.0f} kWh（{time.time()-t0:.0f}s）")
        rows.append({"配置": tag, "计划费": s["plan"], "买入": s["buy"],
                     "偏差费": s["dev"], "紧急": s["emerg"], "总费用": s["total"]})
    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q4_q3_probe.csv",
                              index=False, encoding="utf-8-sig")


def verify_q3(recs, data, p4) -> dict:
    """Q3 层约束回代：功率平衡、SOC 动态、区间、充放互斥、场景无前视。"""
    load, pv_act = data["load"], data["pv_act"]
    bal = dyn = 0.0
    emin, emax = np.inf, -np.inf
    overlap = 0
    future_viol = 0
    for r in recs:
        D = r["D"]
        if D < REPORT_START:
            continue
        res = (pv_act[D] / 6.0 + r["x_final"] + r["d"] + r["e"]
               - load[D] / 6.0 - r["c"] - r["s"])
        bal = max(bal, float(np.abs(res).max()))
        E = np.concatenate([[r["E_start"]], r["E"]])
        dyn = max(dyn, float(np.abs(np.diff(E) - (ETA * r["c"] - r["d"] / ETA)).max()))
        emin = min(emin, float(r["E"].min()))
        emax = max(emax, float(r["E"].max()))
        overlap += int(np.sum((r["c"] > 1e-7) & (r["d"] > 1e-7)))
        if int(r.get("scenario_max_day", -1)) >= D:
            future_viol += 1
    return {
        "功率平衡最大残差/kWh": bal,
        "储能动态最大残差/kWh": dyn,
        "储电量最小值/kWh": emin,
        "储电量最大值/kWh": emax,
        "约束区间检查": bool(emin >= E_MIN - 1e-6 and emax <= E_MAX + 1e-6),
        "同槽同时充放次数": overlap,
        "场景池前视违规天数": future_viol,
    }


def write_result4_3(data, p4, recs, out_name="result4-3.xlsx"):
    """按附件 5 模板生成 results/result4-3.xlsx（计划/调整/充放电/紧急，334 天）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result4-3.xlsx")
    days = list(range(REPORT_START, N_DAY))

    for sheet, key in (("计划购电量", "x_plan"), ("调整购电量", "x_final")):
        ws = wb[sheet]
        for i, D in enumerate(days, start=2):
            x = recs[D][key]
            ws.cell(row=i, column=1, value=pd.Timestamp(data["dates"][D]).to_pydatetime())
            for t in range(T):
                ws.cell(row=i, column=2 + t, value=float(x[t]))
            ws.cell(row=i, column=146, value=float(x.sum()))
            if key == "x_plan":
                ws.cell(row=i, column=147, value=float(p4[D] @ x))
            else:
                xp = recs[D]["x_plan"]
                settle = float((p4[D] * x + 0.5 * p4[D] * np.abs(x - xp)).sum())
                ws.cell(row=i, column=147, value=settle)

    ws = wb["充放电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for D in days:
        r = recs[D]
        for b in range(6):
            rr = row + b
            if b == 0:
                ws.cell(rr, 1, pd.Timestamp(data["dates"][D]).to_pydatetime())
            ws.cell(rr, 2, f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(rr, 3, float(r["c"][24 * b:24 * (b + 1)].sum()))
            ws.cell(rr, 4, float(r["d"][24 * b:24 * (b + 1)].sum()))
            if b == 0:
                ws.cell(rr, 5, "00:00")
                ws.cell(rr, 6, float(r["E_start"]))
            elif b == 1:
                ws.cell(rr, 5, "24:00")
                ws.cell(rr, 6, float(r["E"][-1]))
        row += 6

    ws = wb["紧急购电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for D in days:
        for j, (a, b, kwh) in enumerate(q2._events(recs[D]["e"])):
            if j == 0:
                ws.cell(row, 1, pd.Timestamp(data["dates"][D]).to_pydatetime())
            ws.cell(row, 2, f"{q2._fmt_time(a * 10)}-{q2._fmt_time((b + 1) * 10)}")
            ws.cell(row, 3, round(kwh, 4))
            row += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / out_name
    wb.save(p)
    wb.close()
    return p


def verify_result4_3(path, data, p4, recs) -> dict:
    """回读 result4-3 并与逐日记录勾稽。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    chk = {"sheets": "/".join(wb.sheetnames)}
    days = list(range(REPORT_START, N_DAY))
    for sheet, key, tag in (("计划购电量", "x_plan", "计划"), ("调整购电量", "x_final", "调整")):
        rows = [r for r in wb[sheet].iter_rows(min_row=2, values_only=True) if r[0] is not None]
        chk[f"{tag}表行数"] = len(rows)
        chk[f"{tag}购电量勾稽差/kWh"] = float(abs(
            sum(float(r[145]) for r in rows) - sum(float(recs[D][key].sum()) for D in days)))
    rows = [r for r in wb["充放电量"].iter_rows(min_row=2, values_only=True) if r[1] is not None]
    chk["充放表行数"] = len(rows)
    chk["充放电量勾稽差/kWh"] = float(abs(
        sum(float(r[2]) for r in rows if r[2] is not None)
        - sum(float(recs[D]["c"].sum()) for D in days)))
    kw = sum(float(r[2]) for r in wb["紧急购电量"].iter_rows(min_row=2, values_only=True)
             if r[2] is not None)
    chk["紧急表电量勾稽差/kWh"] = float(abs(
        kw - sum(float(recs[D]["e"].sum()) for D in days)))
    wb.close()
    return chk


def make_figures_q3(data, recs_g, recs_h, s_g, s_h, daily_df) -> None:
    # 1) 同一正式策略下的电价信息口径对比
    pm.bar_group(
        ["买入费", "偏差费", "紧急购电费"],
        {"G·题面已知电价": [s_g["buy"] / 1e4, s_g["dev"] / 1e4, s_g["emerg"] / 1e4],
         "H·历史价格预测": [s_h["buy"] / 1e4, s_h["dev"] / 1e4, s_h["emerg"] / 1e4]},
        ylabel="334 天费用 / 万元", save="Q4_调整层策略对比",
    )
    # 2) 题面口径 G 的逐日紧急购电
    fig, ax = pm.line(
        np.arange(len(daily_df)), daily_df["紧急购电量/kWh"],
        xlabel="日期（2025-02-01 起）", ylabel="紧急购电量 / kWh",
    )
    pm.save_fig(fig, "Q4_调整层逐日紧急购电",
                data=daily_df[["日期", "紧急购电量/kWh"]])


def run_result43() -> None:
    """生成 result4-3：G·2 日·三点+40 场景对冲·组合；H 作扩展对照。"""
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q4")
    t0 = time.time()
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    kw = dict(storage="2day", adj_hours=(6, 12, 18), hedge=True, n_scen=cs.N_SCEN, seed=7,
              lam=0.7, adj_lam=0.7)
    recs_h = run_year_q3(data, p4, p_typ, vseq, price_mode="H", **kw)
    recs_g = run_year_q3(data, p4, p_typ, vseq, price_mode="G", **kw)
    s_h, s_g = total_of_q3(recs_h), total_of_q3(recs_g)
    chk = verify_q3(recs_g, data, p4)
    print(f"问题四 Q3 层：H {s_h['total']/1e4:.1f} 万（买入 {s_h['buy']/1e4:.1f} + "
          f"偏差 {s_h['dev']/1e4:.1f} + 紧急 {s_h['emerg']/1e4:.1f}）；G {s_g['total']/1e4:.1f} 万"
          f"（{time.time()-t0:.0f}s）")

    p = write_result4_3(data, p4, recs_g)
    excel = verify_result4_3(p, data, p4, recs_g)
    log.info("result4-3.xlsx -> {}", p)

    days = list(range(REPORT_START, N_DAY))
    daily = pd.DataFrame({
        "日期": [data["dates"][D] for D in days],
        "计划购电费/元": [recs_g[D]["plan_cost"] for D in days],
        "买入/元": [recs_g[D]["buy"] for D in days],
        "偏差费/元": [recs_g[D]["dev"] for D in days],
        "紧急购电费/元": [recs_g[D]["emerg"] for D in days],
        "总费用/元": [recs_g[D]["total"] for D in days],
        "紧急购电量/kWh": [float(recs_g[D]["e"].sum()) for D in days],
        "调整量/kWh": [float(np.abs(recs_g[D]["x_final"] - recs_g[D]["x_plan"]).sum())
                    for D in days],
    })
    pm.save_outputs(daily, "q4_result43_daily")
    make_figures_q3(data, recs_g, recs_h, s_g, s_h, daily)

    record(
        "问题四 结果：Q3 层（result4-3，2025-02-01 ~ 12-31）",
        {
            "题面口径 G·总费用/万元": round(s_g["total"] / 1e4, 1),
            "G·买入/万元": round(s_g["buy"] / 1e4, 1),
            "G·偏差费/万元": round(s_g["dev"] / 1e4, 1),
            "G·紧急购电费/万元": round(s_g["emerg"] / 1e4, 1),
            "G·紧急购电量/kWh": float(s_g["emerg_kwh"]),
            "G·调整量/kWh": float(s_g["adj_kwh"]),
            "扩展口径 H·历史价格预测/万元": round(s_h["total"] / 1e4, 1),
            "电价信息价值（H 比 G 高）": f"{100 * (s_h['total'] - s_g['total']) / s_h['total']:.2f}%",
            "储能结构": "2 日滚动（跨日连续）",
            **excel,
        },
        note=(
            "题面口径 G：0:00 计划使用附件 4 已知电价、历史负荷预测及组合光伏预测；"
            "6/12/18 更新光伏并调整，6/12 使用 40 个严格无前视残差情景对冲，逐槽因果执行；"
            "采用 2 日滚动跨日储能并按附件 4 结算。H 不读取目标日价格，仅作扩展对照。"
            "图 figures/Q4_调整层策略对比.pdf、Q4_调整层逐日紧急购电.pdf；"
            "逐日表 code/outputs/q4_result43_daily.csv；探针表 code/outputs/q4_q3_probe.csv。"
        ),
    )
    record("问题四 Q3 层约束与校验", chk,
           note="功率平衡、SOC 递推、区间、充放互斥与场景池无前视全部回代通过。")


def fig_weights() -> None:
    """电价预测权重演化图：滚动费用标定 v* + 动态修正 β=0.1（2025-02-01 起）。"""
    pm.init(seed=42, root=str(ROOT))
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    days = list(range(REPORT_START, N_DAY))
    df = pd.DataFrame({
        "日期": [data["dates"][D] for D in days],
        "v1(D−1)": vseq[days, 0], "v2(D−7)": vseq[days, 1], "v3(典型日)": vseq[days, 2],
    })
    fig, ax = pm.line(
        np.arange(len(days)), [vseq[days, 0], vseq[days, 1], vseq[days, 2]],
        labels=["$v_1$·P(D−1)", "$v_2$·P(D−7)", "$v_3$·典型日"],
        xlabel="日期（2025-02-01 起）", ylabel="电价预测权重",
    )
    pm.save_fig(fig, "Q4_电价权重演化", data=df)
    print("已生成 figures/Q4_电价权重演化.pdf")


# ---------------------------------------------------------------------------
# 结构探针
# ---------------------------------------------------------------------------
def probe() -> None:
    pm.init(seed=42, root=str(ROOT))
    t0 = time.time()
    data, p4, p_typ = load_q4_data()
    vseq = rolling_v_seq()
    print(f"数据就绪（{time.time()-t0:.0f}s），v 序列 D=31..364："
          f"v1 {vseq[31:,0].mean():.2f} / v2 {vseq[31:,1].mean():.2f} / v3 {vseq[31:,2].mean():.2f}")
    rows = []
    for price_mode in ("H", "G"):
        for storage in ("daily", "2day"):
            t1 = time.time()
            recs = run_year(data, p4, p_typ, vseq, price_mode=price_mode, storage=storage)
            s = total_of(recs)
            print(f"  [{price_mode} · {storage:5}] 总 {s['total']/1e4:7.1f} 万"
                  f"（计划 {s['plan']/1e4:.1f} + 紧急 {s['emerg']/1e4:.1f}）"
                  f" 紧急量 {s['emerg_kwh']:,.0f} kWh（{time.time()-t1:.0f}s）")
            rows.append({"价格口径": price_mode, "储能结构": storage, **s})
    pd.DataFrame(rows).to_csv(ROOT / "code" / "outputs" / "q4_structure_probe.csv",
                              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="储能/价格口径结构对比探针")
    ap.add_argument("--probe-q3", action="store_true", help="Q3 层结构探针")
    ap.add_argument("--result4-2", action="store_true", help="生成 result4-2.xlsx（Q2 层）")
    ap.add_argument("--result4-3", action="store_true", help="生成 result4-3.xlsx（Q3 层）")
    ap.add_argument("--fig-weights", action="store_true", help="生成电价权重演化图")
    args = ap.parse_args()
    if args.probe:
        probe()
    elif args.probe_q3:
        probe_q3()
    elif args.result4_2:
        run_result42()
    elif args.result4_3:
        run_result43()
    elif args.fig_weights:
        fig_weights()
    else:
        ap.print_help()
