"""官方结果表（result1~4）的写入与回读勾稽。

# ===========================================================================
# 附件 5 模板结构（4 张表，按问题选用）
#   计划购电量：行为日期（2 起始；result2/3 模板无日期列，result4-* 有），
#               第 2..145 列为 144 槽购电量，
#               第 146 列 = Σ_t x_t（日合计），第 147 列 = Σ_t p_t·x_t（日购电费）
#   调整购电量：同布局；第 147 列 = Σ_t [ p_t·x_adj,t + 0.5·p_t·|x_plan,t − x_adj,t| ]
#               （调整结算：调低 50% 违约金、调高 150% 购电价）
#   充放电量：每日 6 行（4 小时块），列 3/4 为块充/放电量；
#             块 0/1 的行在列 5/6 记 0:00（时间值，格式 h:mm，同模板）/
#             "24:00"（文本，格式 @）与两端储电量
#   紧急购电量：每个事件一行（日期、起止时间、电量 kWh），
#               事件由 core.slots.emergency_events 合并相邻紧急槽得到
#
# 统一实现：四类表由 _fill_plan_sheet / _fill_adjust_sheet /
# _fill_charge_sheet / _fill_emergency_sheet 参数化填充；
# 回读勾稽对 145/146 列合计、储能端点、充放电与紧急电量逐项与内存结果比对，
# 残差应为数值误差量级（写入采用 float，无舍入损失）。
# ===========================================================================
"""
from __future__ import annotations

from datetime import time as _time

import numpy as np
import openpyxl
import pandas as pd

from solve.common import DATA_C, N_DAY, REPORT_START, RESULTS_DIR, T
from solve.core.slots import emergency_events as _events
from solve.core.slots import fmt_time as _fmt_time


# ---------------------------------------------------------------------------
# 共享表填充器
# ---------------------------------------------------------------------------
def _fill_plan_sheet(ws, x_list, prices, dates=None) -> None:
    """计划购电量表：[(x, p)] 逐日写入；dates=None 时不写日期列（result2/3 模板）。

    每行：可选日期、144 槽、日合计（第 146 列）、日购电费（第 147 列）。
    """
    for i, (x, p) in enumerate(zip(x_list, prices)):
        row = 2 + i
        if dates is not None:
            ws.cell(row=row, column=1, value=dates[i])
        for t in range(T):
            ws.cell(row=row, column=2 + t, value=float(x[t]))
        ws.cell(row=row, column=146, value=float(x.sum()))
        if p is not None:
            ws.cell(row=row, column=147, value=float(p @ x))


def _fill_adjust_sheet(ws, xa_list, xp_list, prices, dates=None) -> None:
    """调整购电量表：结算费 = Σ[p·x_adj + 0.5·p·|x_plan − x_adj|]（调低 50% 违约金）。"""
    for i, (xa, xp, p) in enumerate(zip(xa_list, xp_list, prices)):
        row = 2 + i
        if dates is not None:
            ws.cell(row=row, column=1, value=dates[i])
        for t in range(T):
            ws.cell(row=row, column=2 + t, value=float(xa[t]))
        ws.cell(row=row, column=146, value=float(xa.sum()))
        settle = float((p * xa + 0.5 * p * np.abs(xa - xp)).sum())
        ws.cell(row=row, column=147, value=settle)


def _fill_charge_sheet(ws, entries) -> None:
    """充放电量表：entries = [(日期对象, c, d, E_start, E_end)]，每日 6 个 4 小时块。

    清行重写会丢失模板示例行的单元格格式，因此按模板显式恢复：
    日期列 mm-dd-yy；首块"时刻"= 时间值 0:00（h:mm）；次块 = 文本 "24:00"（@）。
    """
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for date_obj, c, d, e_start, e_end in entries:
        for b in range(6):
            rr = row + b
            if b == 0:
                cell = ws.cell(row=rr, column=1, value=date_obj)
                cell.number_format = "mm-dd-yy"
            ws.cell(row=rr, column=2, value=f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(row=rr, column=3, value=float(c[24 * b:24 * (b + 1)].sum()))
            ws.cell(row=rr, column=4, value=float(d[24 * b:24 * (b + 1)].sum()))
            if b == 0:
                cell = ws.cell(row=rr, column=5, value=_time(0, 0))
                cell.number_format = "h:mm"
                ws.cell(row=rr, column=6, value=float(e_start))
            if b == 1:
                cell = ws.cell(row=rr, column=5, value="24:00")
                cell.number_format = "@"
                ws.cell(row=rr, column=6, value=float(e_end))
        row += 6


def _fill_emergency_sheet(ws, entries) -> None:
    """紧急购电量表：entries = [(日期对象, e 槽序列)]，相邻紧急槽合并为一个事件。

    日期列按模板恢复为 mm-dd-yy（清行重写会丢失示例行格式）。
    """
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for date_obj, e in entries:
        for j, (a, b, kwh) in enumerate(_events(e)):
            if j == 0:
                cell = ws.cell(row=row, column=1, value=date_obj)
                cell.number_format = "mm-dd-yy"
            ws.cell(row=row, column=2,
                    value=f"{_fmt_time(a * 10)}-{_fmt_time((b + 1) * 10)}")
            ws.cell(row=row, column=3, value=round(kwh, 4))
            row += 1


def _to_dt(ts) -> object:
    """ISO 日期字符串/pandas 时间戳 → datetime（openpyxl 可写）。"""
    return pd.Timestamp(ts).to_pydatetime()


# ---------------------------------------------------------------------------
# result2：计划 / 充放电 / 紧急（口径 D/E）
# ---------------------------------------------------------------------------
def write_result2(dates, price, x_plans, recs, out_name="result2.xlsx"):
    """按附件 5 模板生成 result2.xlsx（计划购电量 / 充放电量 / 紧急购电量）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result2.xlsx")
    rep = list(range(REPORT_START, N_DAY))

    _fill_plan_sheet(wb["计划购电量"], [x_plans[d] for d in rep], [price] * len(rep))
    _fill_charge_sheet(wb["充放电量"], [
        (_to_dt(dates[d]), recs[d]["c"], recs[d]["d"], recs[d]["E_start"], recs[d]["E_end"])
        for d in rep
    ])
    _fill_emergency_sheet(wb["紧急购电量"],
                          [(_to_dt(dates[d]), recs[d]["e"]) for d in rep])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / out_name
    wb.save(out)
    wb.close()
    return out


def verify_result2(path, df: pd.DataFrame) -> dict:
    """回读 result2.xlsx 并与逐日表勾稽（返回校验指标 dict）。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    checks = {"sheet 名": "/".join(wb.sheetnames)}

    ws = wb["计划购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    checks["计划表行数"] = len(rows)
    checks["全天购电量勾稽差/kWh"] = float(abs(
        sum(float(r[145]) for r in rows) - df["计划购电量/kWh"].sum()))
    checks["全天购电费勾稽差/元"] = float(abs(
        sum(float(r[146]) for r in rows) - df["计划购电费/元"].sum()))

    ws = wb["充放电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None]
    checks["充放表行数"] = len(rows)
    e0 = [float(rows[6 * i][5]) for i in range(len(rows) // 6)]
    e24 = [float(rows[6 * i + 1][5]) for i in range(len(rows) // 6)]
    checks["跨日SOC衔接最大误差/kWh"] = (
        float(max(abs(e24[i] - e0[i + 1]) for i in range(len(e0) - 1)))
        if len(e0) > 1 else 0.0)
    checks["日末SOC非固定取值数"] = len(set(round(v, 3) for v in e24))

    ws = wb["紧急购电量"]
    kw = sum(float(r[2]) for r in ws.iter_rows(min_row=2, values_only=True)
             if r[2] is not None)
    checks["紧急表电量勾稽差/kWh"] = float(abs(kw - df["紧急购电量/kWh"].sum()))
    wb.close()
    return checks


# ---------------------------------------------------------------------------
# result3：计划 / 调整 / 充放电 / 紧急（Q3 主方案）
# ---------------------------------------------------------------------------
def write_result3(dates, price, out):
    """按附件 5 模板生成 results/result3.xlsx（计划/调整/充放电/紧急，334 天）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result3.xlsx")
    i0 = REPORT_START
    n_q3 = N_DAY - REPORT_START
    rep = range(n_q3)

    _fill_plan_sheet(wb["计划购电量"], [out["xp"][i] for i in rep], [price] * n_q3)
    _fill_adjust_sheet(wb["调整购电量"], [out["xa"][i] for i in rep],
                       [out["xp"][i] for i in rep], [price] * n_q3)
    _fill_charge_sheet(wb["充放电量"], [
        (_to_dt(dates[i0 + i]), out["c"][i], out["d"][i],
         out["E_start"][i], out["E"][i, -1])
        for i in rep
    ])
    _fill_emergency_sheet(wb["紧急购电量"],
                          [(_to_dt(dates[i0 + i]), out["e"][i]) for i in rep])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "result3.xlsx"
    wb.save(p)
    wb.close()
    return p


def verify_result3(path, dates, price, out):
    """回读校验：行数、端点、勾稽、紧急事件。"""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    chk = {"sheets": "/".join(wb.sheetnames)}
    n_q3 = N_DAY - REPORT_START

    ws = wb["计划购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    chk["计划表行数"] = len(rows)
    chk["计划购电量勾稽差/kWh"] = float(abs(sum(float(r[145]) for r in rows) - out["xp"].sum()))
    chk["计划购电费勾稽差/元"] = float(abs(sum(float(r[146]) for r in rows)
                                       - sum(float(price @ out["xp"][i]) for i in range(n_q3))))

    ws = wb["调整购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    chk["调整表行数"] = len(rows)
    chk["调整购电量勾稽差/kWh"] = float(abs(sum(float(r[145]) for r in rows) - out["xa"].sum()))

    ws = wb["充放电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None]
    chk["充放表行数"] = len(rows)
    e0 = [float(rows[6 * i][5]) for i in range(n_q3)]
    e24 = [float(rows[6 * i + 1][5]) for i in range(n_q3)]
    chk["跨日SOC衔接最大误差/kWh"] = float(
        max(abs(e24[i] - e0[i + 1]) for i in range(n_q3 - 1))
    )
    chk["日末SOC非固定取值数"] = int(len(set(round(v, 3) for v in e24)))

    ws = wb["紧急购电量"]
    kw = sum(float(r[2]) for r in ws.iter_rows(min_row=2, values_only=True) if r[2] is not None)
    chk["紧急表电量勾稽差/kWh"] = float(abs(kw - out["e"].sum()))
    wb.close()
    return chk


# ===========================================================================
# Q4：result4-2（Q2 层）与 result4-3（Q3 层），
#     结算一律用附件 4 真实电价：买入 + 0.5p|x_adj−x_plan| + 5p·紧急。
# ===========================================================================
def write_result4_2(data, p4, recs, out_name="result4-2.xlsx"):
    """按附件 5 模板生成 results/result4-2.xlsx（计划 / 充放电 / 紧急，334 天）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result4-2.xlsx")
    days = list(range(REPORT_START, N_DAY))

    _fill_plan_sheet(wb["计划购电量"], [recs[D]["x"] for D in days],
                     [p4[D] for D in days], dates=[_to_dt(data["dates"][D]) for D in days])
    _fill_charge_sheet(wb["充放电量"], [
        (_to_dt(data["dates"][D]), recs[D]["c"], recs[D]["d"],
         recs[D]["E_start"], recs[D]["E"][-1])
        for D in days
    ])
    _fill_emergency_sheet(wb["紧急购电量"],
                          [(_to_dt(data["dates"][D]), recs[D]["e"]) for D in days])

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


def write_result4_3(data, p4, recs, out_name="result4-3.xlsx"):
    """按附件 5 模板生成 results/result4-3.xlsx（计划/调整/充放电/紧急，334 天）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result4-3.xlsx")
    days = list(range(REPORT_START, N_DAY))
    dts = [_to_dt(data["dates"][D]) for D in days]

    _fill_plan_sheet(wb["计划购电量"], [recs[D]["x_plan"] for D in days],
                     [p4[D] for D in days], dates=dts)
    _fill_adjust_sheet(wb["调整购电量"], [recs[D]["x_final"] for D in days],
                       [recs[D]["x_plan"] for D in days], [p4[D] for D in days],
                       dates=dts)
    _fill_charge_sheet(wb["充放电量"], [
        (_to_dt(data["dates"][D]), recs[D]["c"], recs[D]["d"],
         recs[D]["E_start"], recs[D]["E"][-1])
        for D in days
    ])
    _fill_emergency_sheet(wb["紧急购电量"],
                          [(_to_dt(data["dates"][D]), recs[D]["e"]) for D in days])

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
