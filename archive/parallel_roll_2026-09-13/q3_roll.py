"""C 题 问题三（滚动口径修订，2026-09-13）：2 日规划窗口 + 6/12 小时执行窗口。

修订动机同问题二：原口径把"每天 0:00/24:00 储电量相同（=6000 kWh）"作为储能终端
条件（计划层与各调整层均如此），题面并未要求；新口径改为滚动 MPC：

- 规划窗口：每个决策时刻（0:00 / 6:00 / 12:00 / 18:00）以"当前时刻 → 次日 24:00"
  为窗口（0:00 恰为 2 天 = 288 槽；其后为剩余当日 + 次日）求解最小费用 LP，末端自由；
- 执行窗口：两个决策时刻之间（6 h；可选 12 h 口径 = 只在 12:00 调整，执行 12 h）；
- 储电量跨日连续、不锚定 6000；当日执行的末端储电量即次日规划初值；
- 0:00 计划：光伏 = λ·官方 0:00 预报 + (1−λ)·历史自适应（λ=0.7），历史权重平滑 β=0.1；
  次日的预测只用决策时点可得信息（无前视，见 ``hist_forecast_next``）；
- 6/12/18 调整：用最新预报重优化 [t1, 次日 24:00)，当日剩余时段按 50%/150% 偏差费
  （与 x_plan 比较），次日段为暂定计划（正常价、无偏差费）；6:00/12:00 叠加无前视
  场景对冲（残差块仅取自目标日之前，10 情景）；
- 执行：逐槽因果（q2.exec_segment_causal，只读已实现值），段末跟踪规划轨迹；
  缺口按 5 倍交易时刻电价紧急购电；
- 结算：p·x_adj + 0.5·p·|x_plan − x_adj| + 5·p·e（题面 50%/150% 口径）；
- 状态：仿真自 2025-01-16 起（预热 15 天），报送 2025-02-01 ~ 12-31（334 天）；
  末日窗口的"次日"用当日复制填充。

运行（program/ 下，先设 $env:PYTHONIOENCODING='utf-8'）：
    uv run python -m solve.q3_roll --result3          # 官方 result3 + 对照 + 图 + 报告
    uv run python -m solve.q3_roll --probe            # 结构/执行窗口对比
"""
from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import openpyxl
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

import program as pm
from solve import q2
from solve import q3_proto as qp
from solve.common import E0, E_MAX, E_MIN, ETA, P_MAX_E, RESULTS_DIR, ROOT, T

LAM = 0.7               # 0:00 组合权重（官方占比）
ADJ_LAM = 0.7           # 调整层组合权重
BETA = 0.1              # 历史权重平滑系数
N_SCEN = 40             # 对冲情景数（3 种子 10/20/40 收敛检查后由 10 升级为 40）
SEED = 7
START_SIM = 15          # 滚动仿真起点（日序号；前 15 天为预测/权重预热）
N_DAY = q2.N_DAY
REPORT_START = q2.REPORT_START
REP = list(range(REPORT_START, N_DAY))


def record(section: str, data, note: str = "") -> None:
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
    pm.record_result(section, data, note=note)


# ---------------------------------------------------------------------------
# 次日预测（无前视）
# ---------------------------------------------------------------------------
def hist_forecast_next(data: dict, D: int) -> np.ndarray:
    """次日光伏预测（kW，144 槽）：P̂(D+1)=u1·P̂(D)+u2·P(D−1)+u3·P̄。

    只用 ≤D 可得信息（P(D) 未知，用当日预测 P̂(D) 递推替代）。
    """
    u = data["U_SMOOTH"][D]
    pv, typ = data["pv_act"], data["pv_typ"]
    p_hat = u[0] * pv[D - 1] + u[1] * pv[D - 2] + u[2] * typ
    return np.clip(u[0] * p_hat + u[1] * pv[D - 1] + u[2] * typ, 0.0, None)


def hist_load_forecast_next(data: dict, D: int) -> np.ndarray:
    """次日负荷预测（kW，144 槽，无前视）：同星期日/两周前/典型日三源。

    权重沿用当日可用的 Q2E 负荷权重（TH[D−1]），输入滞后前移一天：
    L̂(D+1)=w1·L(D−6)+w2·L(D−13)+w3·L̄。
    """
    j = max(D - 1, 13)
    w = qp._softmax(data["TH"][j, :3])
    load, typ = data["load"], data["load_typ"]
    return np.clip(w[0] * load[D - 6] + w[1] * load[D - 13] + w[2] * typ, 0.0, None)


# ---------------------------------------------------------------------------
# 扩展调整 LP（窗口 [t1, 次日 24:00)，末端自由）
# ---------------------------------------------------------------------------
def adjust_horizon(price_ext, load_ext, pv_ext, x_ref, e_init,
                   eps=qp.EPS_TH, e_terminal=None):
    """确定性调整 LP（2 日窗口）。

    x_ref：当日剩余时段的计划参考（长度 n_dev，偏差费 = 1.5p·adj − p·min(x_ref, adj)）；
    x_ref 之后的次日段按正常价 p 购电、无偏差费。
    变量块：0=adj(m) 1=c(m) 2=d(m) 3=s(m) 4=E(m) 5=y(n_dev)。
    返回 (adj, E)。
    """
    m = len(load_ext)
    nd = len(x_ref)
    n = 5 * m + nd

    def I(b, i):
        return b * m + i

    c_obj = np.zeros(n)
    c_obj[0:m] = np.where(np.arange(m) < nd, 1.5, 1.0) * price_ext
    c_obj[m:3 * m] = eps
    c_obj[5 * m:5 * m + nd] = -price_ext[:nd]

    rows = 2 * m + (1 if e_terminal is not None else 0)
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    for i in range(m):
        A_eq[i, I(0, i)] = 1.0
        A_eq[i, I(1, i)] = -1.0
        A_eq[i, I(2, i)] = 1.0
        A_eq[i, I(3, i)] = -1.0
        b_eq[i] = load_ext[i] - pv_ext[i]

        r = m + i
        A_eq[r, I(4, i)] = 1.0
        if i:
            A_eq[r, I(4, i - 1)] = -1.0
        A_eq[r, I(1, i)] = -ETA
        A_eq[r, I(2, i)] = 1.0 / ETA
        b_eq[r] = e_init if i == 0 else 0.0
    if e_terminal is not None:
        A_eq[2 * m, I(4, m - 1)] = 1.0
        b_eq[2 * m] = float(e_terminal)

    A_ub = lil_matrix((2 * nd, n))
    b_ub = np.zeros(2 * nd)
    for i in range(nd):
        A_ub[i, 5 * m + i] = 1.0
        b_ub[i] = x_ref[i]
        A_ub[nd + i, 5 * m + i] = 1.0
        A_ub[nd + i, I(0, i)] = -1.0

    bounds = (
        [(0.0, None)] * m
        + [(0.0, P_MAX_E)] * m
        + [(0.0, P_MAX_E)] * m
        + [(0.0, float(v)) for v in pv_ext]
        + [(E_MIN, E_MAX)] * m
        + [(0.0, None)] * nd
    )
    res = pm.optimize.solve_lp(c_obj, A_ub=csr_matrix(A_ub), b_ub=b_ub,
                               A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"扩展调整 LP 失败：{res.message}")
    return res.x[0:m], res.x[4 * m:5 * m]


def adjust_horizon_hedge(price_ext, load_ext, pv_scens, x_ref, e_init,
                         eps=qp.EPS_TH, e_terminal=None):
    """场景对冲调整 LP（2 日窗口）：adj 为第一阶段共享决策，各场景储能/紧急独立。

    pv_scens：list（第 0 个为名义预测）；每个元素长度 m。
    目标：min Σ[1.5p·adj − p·y]（当日段）+ Σ[p·adj]（次日段）
          + (1/S)Σ_s[ε(c_s+d_s) + 5p·e_s]。
    返回 (adj, 名义场景储能轨迹 E_nom)。
    """
    m = len(load_ext)
    nd = len(x_ref)
    S = len(pv_scens)
    n = m + nd + S * 5 * m
    off_y = m
    off_s = m + nd

    def A(i):
        return i

    def Y(i):
        return off_y + i

    def base(s):
        return off_s + s * 5 * m

    c_obj = np.zeros(n)
    c_obj[0:m] = np.where(np.arange(m) < nd, 1.5, 1.0) * price_ext
    c_obj[off_y:off_y + nd] = -price_ext[:nd]
    for s in range(S):
        b0 = base(s)
        c_obj[b0:b0 + m] = eps / S
        c_obj[b0 + m:b0 + 2 * m] = eps / S
        c_obj[b0 + 4 * m:b0 + 5 * m] = q2.EMERG_MULT * price_ext / S

    rows = S * (2 * m + (1 if e_terminal is not None else 0))
    A_eq = lil_matrix((rows, n))
    b_eq = np.zeros(rows)
    r = 0
    for s in range(S):
        b0 = base(s)
        for i in range(m):
            A_eq[r, A(i)] = 1.0
            A_eq[r, b0 + i] = -1.0
            A_eq[r, b0 + m + i] = 1.0
            A_eq[r, b0 + 2 * m + i] = -1.0
            A_eq[r, b0 + 4 * m + i] = 1.0
            b_eq[r] = load_ext[i] - pv_scens[s][i]
            r += 1
        for i in range(m):
            A_eq[r, b0 + 3 * m + i] = 1.0
            if i:
                A_eq[r, b0 + 3 * m + i - 1] = -1.0
            A_eq[r, b0 + i] = -ETA
            A_eq[r, b0 + m + i] = 1.0 / ETA
            b_eq[r] = e_init if i == 0 else 0.0
            r += 1
        if e_terminal is not None:
            A_eq[r, b0 + 3 * m + m - 1] = 1.0
            b_eq[r] = float(e_terminal)
            r += 1

    A_ub = lil_matrix((2 * nd, n))
    b_ub = np.zeros(2 * nd)
    for i in range(nd):
        A_ub[i, Y(i)] = 1.0
        b_ub[i] = x_ref[i]
        A_ub[nd + i, Y(i)] = 1.0
        A_ub[nd + i, A(i)] = -1.0

    bounds = [(0.0, None)] * m + [(0.0, None)] * nd
    for s in range(S):
        bounds += ([(0.0, P_MAX_E)] * m + [(0.0, P_MAX_E)] * m
                   + [(0.0, float(v)) for v in pv_scens[s]]
                   + [(E_MIN, E_MAX)] * m + [(0.0, None)] * m)
    res = pm.optimize.solve_lp(c_obj, A_ub=csr_matrix(A_ub), b_ub=b_ub,
                               A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds)
    if not res.success:
        raise RuntimeError(f"扩展对冲调整 LP 失败：{res.message}")
    E_nom = res.x[base(0) + 3 * m: base(0) + 4 * m]
    return res.x[0:m], E_nom


# ---------------------------------------------------------------------------
# 单日模拟
# ---------------------------------------------------------------------------
def simulate_day_roll(data: dict, D: int, E_now: float, lam: float = LAM,
                      adj_lam: float | None = ADJ_LAM, adj_hours=(6, 12, 18),
                      hedge: bool = True, n_scen: int = N_SCEN, seed: int = SEED,
                      pool_min_month: int = 14, pool_lookback: int = 90,
                      e_terminal=None, margin: float = 0.0) -> dict:
    """2 日窗口滚动单日模拟（执行窗口 = 相邻决策时刻之间）。

    E_now：当日 0:00 实际储电量（跨日连续）；e_terminal：窗口末端储电量
    （None = 自由；传入数值 = 锚定，供结构对照）。
    """
    price = data["price"]
    load_act = data["load"][D] / 6.0                 # 回测真值（执行/校验用）
    load_fc = qp.hist_load_forecast(data, D) / 6.0   # 因果负荷预测（计划输入）
    pv_act = data["pv_act"][D] / 6.0

    f0 = q2._hour_to_slots(data["fc0"][D])
    ph = qp.hist_forecast(data, D)
    pv0 = np.clip(lam * f0 + (1.0 - lam) * ph, 0.0, None) / 6.0
    if margin:
        pv0 = pv0 * (1.0 - margin)
    D1 = min(D + 1, N_DAY - 1)
    load1_fc = hist_load_forecast_next(data, D) / 6.0
    pv1 = hist_forecast_next(data, D) / 6.0

    x2, E2, _ = q2.plan_horizon(
        np.tile(price, 2), np.concatenate([load_fc, load1_fc]),
        np.concatenate([pv0, pv1]), E_now, e_terminal, eps=qp.EPS_TH)
    x_plan = x2[:T].copy()
    x_seq = x2[:T].copy()
    E_target = E2.copy()            # 长度 2T：窗口储能轨迹

    e_total = np.zeros(T)
    c_all = np.zeros(T)
    d_all = np.zeros(T)
    s_all = np.zeros(T)
    E_exec = np.zeros(T)
    scenario_days: list[int] = []
    E_cur = float(E_now)

    points = [0] + sorted(p * 6 for p in adj_hours) + [T]
    for a, b in zip(points[:-1], points[1:]):
        seg = q2.exec_segment_causal(
            load_act[a:b], pv_act[a:b], x_seq[a:b], E_cur,
            float(np.clip(E_target[b - 1], E_MIN, E_MAX)))
        e_total[a:b] = seg["e"]
        c_all[a:b] = seg["c"]
        d_all[a:b] = seg["d"]
        s_all[a:b] = seg["s"]
        E_exec[a:b] = seg["E"]
        E_cur = float(seg["E"][-1])

        pub = b // 6
        if b < T and pub in adj_hours:
            t1 = b
            m = 2 * T - t1
            nd = T - t1
            price_ext = np.concatenate([price[t1:], price])
            load_ext = np.concatenate([load_fc[t1:], load1_fc])
            off = qp.fc_slots(data[f"fc{pub}"][D], pub)
            pv_new_full = (off if adj_lam is None
                           else np.clip(adj_lam * off + (1.0 - adj_lam) * ph, 0.0, None))
            pv_rest = pv_new_full[t1:] / 6.0
            x_ref = x_plan[t1:]
            if hedge and pub in (6, 12) and D > 14:
                pool = qp.causal_residual_pool(data, D, min_same_month=pool_min_month,
                                               lookback=pool_lookback)
                rng = np.random.default_rng(np.random.SeedSequence([seed, D, pub]))
                idxs = rng.choice(pool, size=max(0, n_scen - 1), replace=True)
                scenario_days.extend(int(ix) for ix in idxs)
                scens = []
                for ix in [None] + list(idxs):
                    seg_d = pv_rest if ix is None else np.clip(
                        pv_rest - qp.forecast_residual(data, int(ix), pub, adj_lam)[t1:],
                        0.0, None)
                    scens.append(np.concatenate([seg_d, pv1]))
                x_adj_ext, E_adj_ext = adjust_horizon_hedge(
                    price_ext, load_ext, scens, x_ref, E_cur, eps=qp.EPS_TH,
                    e_terminal=e_terminal)
            else:
                pv_ext = np.concatenate([pv_rest, pv1])
                x_adj_ext, E_adj_ext = adjust_horizon(
                    price_ext, load_ext, pv_ext, x_ref, E_cur, eps=qp.EPS_TH,
                    e_terminal=e_terminal)
            x_seq[t1:] = x_adj_ext[:nd]
            E_target[t1:] = E_adj_ext

    x_final = x_seq.copy()
    buy = float(price @ x_final)
    dev = float((0.5 * price * np.abs(x_final - x_plan)).sum())
    emerg = float((q2.EMERG_MULT * price * e_total).sum())
    plan_cost = float(price @ x_plan)
    total = buy + dev + emerg
    return {
        "D": D, "x_plan": x_plan, "x_final": x_final,
        "c": c_all, "d": d_all, "s": s_all, "E": E_exec,
        "E_start": float(E_now), "E_end": float(E_cur),
        "e": e_total, "plan_cost": plan_cost, "buy": buy, "dev": dev,
        "emerg": emerg, "total": total, "adjust_net": buy + dev - plan_cost,
        "adj_abs_kwh": float(np.abs(x_final - x_plan).sum()),
        "scenario_max_day": max(scenario_days, default=-1),
        "scenario_count": len(scenario_days),
    }


def run_year_roll(data: dict, start: int = START_SIM, **kw) -> list[dict]:
    """全年滚动（储电量跨日连续；初始 6000 kWh）。"""
    E = E0
    recs = []
    for D in range(start, N_DAY):
        r = simulate_day_roll(data, D, E, **kw)
        recs.append(r)
        E = r["E_end"]
    return recs


def total_of(recs, rep=REP) -> dict:
    days = [r for r in recs if r["D"] in rep]
    return {
        "plan": float(sum(r["plan_cost"] for r in days)),
        "buy": float(sum(r["buy"] for r in days)),
        "dev": float(sum(r["dev"] for r in days)),
        "emerg": float(sum(r["emerg"] for r in days)),
        "total": float(sum(r["total"] for r in days)),
        "emerg_kwh": float(sum(float(r["e"].sum()) for r in days)),
        "adj_kwh": float(sum(r["adj_abs_kwh"] for r in days)),
        "days": len(days),
    }


def verify(recs, data) -> dict:
    """约束回代：平衡、SOC 动态、区间、充放互斥、跨日衔接、场景无前视。"""
    bal = dyn = 0.0
    emin, emax = np.inf, -np.inf
    overlap = 0
    future_viol = 0
    cont = 0.0
    for r in recs:
        D = r["D"]
        if D < REPORT_START:
            continue
        res = (data["pv_act"][D] / 6.0 + r["x_final"] + r["d"] + r["e"]
               - data["load"][D] / 6.0 - r["c"] - r["s"])
        bal = max(bal, float(np.abs(res).max()))
        E = np.concatenate([[r["E_start"]], r["E"]])
        dyn = max(dyn, float(np.abs(np.diff(E) - (ETA * r["c"] - r["d"] / ETA)).max()))
        emin = min(emin, float(r["E"].min()))
        emax = max(emax, float(r["E"].max()))
        overlap += int(np.sum((r["c"] > 1e-7) & (r["d"] > 1e-7)))
        if int(r.get("scenario_max_day", -1)) >= D:
            future_viol += 1
    for r0, r1 in zip(recs, recs[1:]):
        cont = max(cont, abs(r0["E_end"] - r1["E_start"]))
    return {
        "功率平衡最大残差/kWh": bal,
        "储能动态最大残差/kWh": dyn,
        "储电量最小值/kWh": round(emin, 3),
        "储电量最大值/kWh": round(emax, 3),
        "约束区间检查": bool(emin >= E_MIN - 1e-6 and emax <= E_MAX + 1e-6),
        "同槽同时充放次数": overlap,
        "跨日储电量衔接最大误差/kWh": cont,
        "场景池前视违规天数": future_viol,
    }


# ---------------------------------------------------------------------------
# result3 写出 / 回读
# ---------------------------------------------------------------------------
def write_result3(dates, price, rmap, out_name="result3.xlsx"):
    """按附件 5 模板生成 results/result3.xlsx（日序 31..364；rmap: day → 记录）。"""
    from solve.common import DATA_C

    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result3.xlsx")
    days = REP

    ws = wb["计划购电量"]
    for i, D in enumerate(days, start=2):
        xp = rmap[D]["x_plan"]
        ws.cell(row=i, column=1, value=pd.Timestamp(dates[D]).to_pydatetime())
        for t in range(T):
            ws.cell(row=i, column=2 + t, value=float(xp[t]))
        ws.cell(row=i, column=146, value=float(xp.sum()))
        ws.cell(row=i, column=147, value=float(price @ xp))

    ws = wb["调整购电量"]
    for i, D in enumerate(days, start=2):
        r = rmap[D]
        xa, xp = r["x_final"], r["x_plan"]
        ws.cell(row=i, column=1, value=pd.Timestamp(dates[D]).to_pydatetime())
        for t in range(T):
            ws.cell(row=i, column=2 + t, value=float(xa[t]))
        ws.cell(row=i, column=146, value=float(xa.sum()))
        settle = float((price * xa + 0.5 * price * np.abs(xa - xp)).sum())
        ws.cell(row=i, column=147, value=settle)

    ws = wb["充放电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for D in days:
        r = rmap[D]
        for b in range(6):
            rr = row + b
            if b == 0:
                ws.cell(row=rr, column=1, value=pd.Timestamp(dates[D]).to_pydatetime())
            ws.cell(row=rr, column=2, value=f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(row=rr, column=3, value=float(r["c"][24 * b:24 * (b + 1)].sum()))
            ws.cell(row=rr, column=4, value=float(r["d"][24 * b:24 * (b + 1)].sum()))
            if b == 0:
                ws.cell(row=rr, column=5, value="00:00")
                ws.cell(row=rr, column=6, value=float(r["E_start"]))
            elif b == 1:
                ws.cell(row=rr, column=5, value="24:00")
                ws.cell(row=rr, column=6, value=float(r["E_end"]))
        row += 6

    ws = wb["紧急购电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for D in days:
        for j, (a, b, kwh) in enumerate(q2._events(rmap[D]["e"])):
            if j == 0:
                ws.cell(row=row, column=1, value=pd.Timestamp(dates[D]).to_pydatetime())
            ws.cell(row=row, column=2,
                    value=f"{q2._fmt_time(a * 10)}-{q2._fmt_time((b + 1) * 10)}")
            ws.cell(row=row, column=3, value=round(kwh, 4))
            row += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / out_name
    wb.save(p)
    wb.close()
    return p


def verify_result3(path, dates, price, rmap) -> dict:
    """回读 result3.xlsx：行数、勾稽、SOC 端点与跨日衔接、紧急事件。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    chk = {"sheet 名": "/".join(wb.sheetnames)}

    ws = wb["计划购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    chk["计划表行数"] = len(rows)
    chk["计划购电量勾稽差/kWh"] = float(abs(sum(float(r[145]) for r in rows)
                                       - sum(float(rmap[D]["x_plan"].sum()) for D in REP)))
    chk["计划购电费勾稽差/元"] = float(abs(sum(float(r[146]) for r in rows)
                                     - sum(float(price @ rmap[D]["x_plan"]) for D in REP)))

    ws = wb["调整购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    chk["调整表行数"] = len(rows)
    chk["调整购电量勾稽差/kWh"] = float(abs(sum(float(r[145]) for r in rows)
                                       - sum(float(rmap[D]["x_final"].sum()) for D in REP)))

    ws = wb["充放电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None]
    chk["充放表行数"] = len(rows)
    soc_seq = [float(r[5]) for r in rows if r[4] in ("00:00", "24:00") and r[5] is not None]
    chk["储能端点最小值/kWh"] = round(min(soc_seq), 2)
    chk["储能端点最大值/kWh"] = round(max(soc_seq), 2)
    chk["跨日衔接最大误差/kWh"] = float(max(
        abs(soc_seq[2 * i + 1] - soc_seq[2 * i + 2]) for i in range(len(REP) - 1)))
    chk["表内0:00与记录差/kWh"] = float(max(
        abs(soc_seq[2 * i] - rmap[D]["E_start"]) for i, D in enumerate(REP)))

    ws = wb["紧急购电量"]
    kw = sum(float(r[2]) for r in ws.iter_rows(min_row=2, values_only=True) if r[2] is not None)
    chk["紧急表电量勾稽差/kWh"] = float(abs(kw - sum(float(rmap[D]["e"].sum()) for D in REP)))
    wb.close()
    return chk


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def make_figures(daily, base_noadj, base_off, rmap, dates, e_off_daily) -> None:
    import matplotlib.pyplot as plt

    names = ["无调整", "三点调整", "主方案（组合+对冲）"]
    plan = [base_noadj["plan"] / 1e4, base_off["plan"] / 1e4,
            sum(rmap[D]["plan_cost"] for D in REP) / 1e4]
    # 调整净额 = 买入 + 偏差 − 计划（口径与旧版一致）
    adj = [(base_noadj["buy"] + base_noadj["dev"] - base_noadj["plan"]) / 1e4,
           (base_off["buy"] + base_off["dev"] - base_off["plan"]) / 1e4,
           sum(rmap[D]["adjust_net"] for D in REP) / 1e4]
    em = [base_noadj["emerg"] / 1e4, base_off["emerg"] / 1e4,
          sum(rmap[D]["emerg"] for D in REP) / 1e4]
    pm.bar_group(names, {"计划购电费": plan, "调整净额": adj, "紧急购电费": em},
                 ylabel="334 天费用 / 万元", save="Q3_策略费用对比")

    e_main = np.array([float(rmap[D]["e"].sum()) for D in REP])
    e_off = np.asarray(e_off_daily, dtype=float)
    fig, ax = pm.line(np.arange(len(REP)), [e_main, e_off],
                      labels=["主方案（组合+对冲）", "三点调整（纯官方）"],
                      xlabel="日期（2025-02-01 起）", ylabel="紧急购电量 / kWh")
    pm.save_fig(fig, "Q3_逐日紧急购电",
                data=pd.DataFrame({"日期": [dates[D] for D in REP],
                                   "主方案紧急量/kWh": e_main, "三点官方紧急量/kWh": e_off}))

    dev = np.array([rmap[D]["adj_abs_kwh"] for D in REP])
    fig2, ax2 = pm.hist(dev, bins=30, xlabel="逐日调整量 |x_adj − x_plan| / kWh",
                        ylabel="天数")
    pm.save_fig(fig2, "Q3_调整量分布", data=pd.DataFrame({"逐日调整量_kWh": dev}))

    # 储电量轨迹
    e_lo = np.array([float(rmap[D]["E"].min()) for D in REP])
    e_hi = np.array([float(rmap[D]["E"].max()) for D in REP])
    fig3, ax3 = pm.line(np.arange(len(REP)), [e_lo, e_hi],
                        labels=["日内最低", "日内最高"],
                        xlabel="日期（2025-02-01 起）", ylabel="储电量 / kWh")
    ax3.axhline(E_MIN, color="grey", ls=":", lw=1)
    ax3.axhline(E_MAX, color="grey", ls=":", lw=1)
    pm.save_fig(fig3, "Q3_储能轨迹",
                data=pd.DataFrame({"日期": [dates[D] for D in REP],
                                   "日内最低/kWh": e_lo, "日内最高/kWh": e_hi}))


# ---------------------------------------------------------------------------
# 结构探针 / 官方主流程
# ---------------------------------------------------------------------------
def _run_config(data, tag, **kw):
    t0 = time.time()
    recs = run_year_roll(data, **kw)
    s = total_of(recs)
    print(f"  [{tag:<28}] 总 {s['total']/1e4:7.1f} 万（计划 {s['plan']/1e4:.1f} + "
          f"调整净额 {(s['buy']+s['dev']-s['plan'])/1e4:+.1f} + 紧急 {s['emerg']/1e4:.1f}）"
          f"用时 {time.time()-t0:.0f}s")
    return recs, s


def probe(data) -> pd.DataFrame:
    cfgs = [
        ("2 日滚动·6h·末端自由（新）", dict(e_terminal=None, adj_hours=(6, 12, 18))),
        ("2 日滚动·12h·末端自由", dict(e_terminal=None, adj_hours=(12,))),
        ("2 日滚动·6h·末端锚定 6000", dict(e_terminal=E0, adj_hours=(6, 12, 18))),
        ("2 日滚动·6h·无对冲·纯官方", dict(e_terminal=None, adj_hours=(6, 12, 18),
                                            hedge=False, lam=1.0, adj_lam=None)),
    ]
    rows = []
    for tag, kw in cfgs:
        recs, s = _run_config(data, tag, **kw)
        rmap = {r["D"]: r for r in recs}
        E_end = np.array([rmap[D]["E_end"] for D in REP])
        rows.append({
            "配置": tag,
            "计划购电费/万元": round(s["plan"] / 1e4, 1),
            "调整净额/万元": round((s["buy"] + s["dev"] - s["plan"]) / 1e4, 1),
            "紧急购电费/万元": round(s["emerg"] / 1e4, 1),
            "总费用/万元": round(s["total"] / 1e4, 1),
            "紧急购电量/kWh": round(s["emerg_kwh"], 1),
            "调整量/kWh": round(s["adj_kwh"], 1),
            "日末储电量均值/kWh": round(float(E_end.mean()), 1),
            "日末储电量最小/kWh": round(float(E_end.min()), 1),
            "日末储电量最大/kWh": round(float(E_end.max()), 1),
        })
    df = pd.DataFrame(rows)
    pm.save_outputs(df, "q3_roll_structure_probe")
    return df


def run_result3(data, e_terminal=None, adj_hours=(6, 12, 18)) -> dict:
    """官方 result3：主方案 + 两个基线对照 + 写出 + 校验 + 图 + 报告。"""
    pm.init(seed=42, root=str(ROOT))
    log = pm.get_logger("q3roll")
    t0 = time.time()

    print("基线对照（纯官方）……")
    recs_no, s_no = _run_config(data, "无调整（官方）", e_terminal=None, adj_hours=(),
                                hedge=False, lam=1.0, adj_lam=None)
    recs_off, s_off = _run_config(data, "三点调整（官方）", e_terminal=None,
                                  adj_hours=(6, 12, 18), hedge=False, lam=1.0,
                                  adj_lam=None)
    print("主方案（组合预测 + 对冲）……")
    recs, s = _run_config(data, "主方案", e_terminal=e_terminal, adj_hours=adj_hours,
                          hedge=True, lam=LAM, adj_lam=ADJ_LAM, n_scen=N_SCEN, seed=SEED)

    cur = RESULTS_DIR / "result3.xlsx"
    backup = pm.outputs_dir() / "result3_dailyCycle_backup.xlsx"
    if cur.exists() and not backup.exists():
        shutil.copy2(cur, backup)
        log.info("旧（日循环主方案）result3 备份 -> {}", backup)

    rmap = {r["D"]: r for r in recs}
    p = write_result3(data["dates"], data["price"], rmap)
    chk = verify_result3(p, data["dates"], data["price"], rmap)
    vchk = verify(recs, data)
    log.info("result3.xlsx -> {}", p)

    daily = pd.DataFrame({
        "日期": [data["dates"][D] for D in REP],
        "计划费/元": [rmap[D]["plan_cost"] for D in REP],
        "调整净额/元": [rmap[D]["adjust_net"] for D in REP],
        "紧急费/元": [rmap[D]["emerg"] for D in REP],
        "总费用/元": [rmap[D]["total"] for D in REP],
        "紧急量/kWh": [float(rmap[D]["e"].sum()) for D in REP],
        "调整量/kWh": [rmap[D]["adj_abs_kwh"] for D in REP],
        "场景最大日序号": [int(rmap[D]["scenario_max_day"]) for D in REP],
        "场景数": [int(rmap[D]["scenario_count"]) for D in REP],
    })
    pm.save_outputs(daily, "q3_roll_daily")
    e_off_daily = [float(r["e"].sum()) for r in recs_off if r["D"] >= REPORT_START]
    make_figures(daily, s_no, s_off, rmap, data["dates"], e_off_daily)

    record(
        "问题三 主方案结果（2 日滚动，2025-02-01 ~ 12-31，334 天）",
        {
            "计划购电费/万元": round(s["plan"] / 1e4, 1),
            "调整净额/万元": round((s["buy"] + s["dev"] - s["plan"]) / 1e4, 1),
            "紧急购电费/万元": round(s["emerg"] / 1e4, 1),
            "总费用/万元": round(s["total"] / 1e4, 1),
            "对照·无调整（纯官方）/万元": round(s_no["total"] / 1e4, 1),
            "对照·三点调整（纯官方）/万元": round(s_off["total"] / 1e4, 1),
            "主方案相对无调整": f"{100 * (s['total'] / s_no['total'] - 1):+.2f}%",
            "主方案相对三点官方": f"{100 * (s['total'] / s_off['total'] - 1):+.2f}%",
            "发生紧急购电天数": int(sum(1 for D in REP if rmap[D]["e"].sum() > 1e-3)),
            "全程紧急购电量/kWh": round(s["emerg_kwh"], 1),
            "全程调整量/kWh": round(s["adj_kwh"], 1),
            **chk, **vchk,
        },
        note=(
            "新结构（2 日滚动·末端自由）：每个决策时刻（0/6/12/18）以"
            "当前→次日 24:00 为优化窗口（0:00 即 48 h，末端不锚定 6000）；"
            "执行窗口 = 相邻决策时刻（6 h）；0:00 计划用组合预测"
            "（λ=0.7·官方 + 0.3·历史，历史权重平滑 β=0.1），6/12/18 用最新预报"
            "调整（50%/150% 偏差费、次日段无偏差费），6/12 叠加无前视场景对冲"
            "（残差块仅取自目标日之前，10 情景）；逐槽因果执行、缺口 5 倍紧急；"
            "储电量跨日连续（仿真自 2025-01-16 起，报送 2025-02-01 ~ 12-31）。"
            "旧日循环主方案备份 code/outputs/result3_dailyCycle_backup.xlsx（1327.8 万）。"
            "复现：uv run python -m solve.q3_roll --result3。"
            "图 figures/Q3_策略费用对比.pdf、Q3_逐日紧急购电.pdf、Q3_调整量分布.pdf、"
            "Q3_储能轨迹.pdf；逐日表 code/outputs/q3_roll_daily.csv。"
        ),
    )
    return {"recs": recs, "s": s, "s_no": s_no, "s_off": s_off,
            "checks": {**chk, **vchk}, "out": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result3", action="store_true", help="写官方 result3.xlsx")
    ap.add_argument("--probe", action="store_true", help="结构/执行窗口对比")
    ap.add_argument("--adj-hours", default="6,12,18",
                    help="调整时刻（逗号分隔；'12' = 12 h 执行窗口）")
    args = ap.parse_args()

    pm.init(seed=42, root=str(ROOT))
    data = qp.load_extended()
    data["U_SMOOTH"] = qp.make_smooth_u(data, BETA)

    if args.probe:
        df = probe(data)
        print(df.to_string(index=False))
        record("问题三 结构/执行窗口对比（2 日滚动）", df,
               note=("新结构为 2 日滚动·末端自由：每个决策时刻（0/6/12/18）优化窗口为"
                     "当前→次日 24:00，执行窗口 6 h（相邻决策时刻）；对照 12 h 执行窗口"
                     "（只保留 12:00 调整）、末端锚定 6000、以及无对冲纯官方；"
                     "0:00 组合预测与其余口径与主方案一致（除注明外）。"))
        return
    if args.result3:
        adj_hours = tuple(int(x) for x in args.adj_hours.split(",") if x.strip())
        run_result3(data, e_terminal=None, adj_hours=adj_hours)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
