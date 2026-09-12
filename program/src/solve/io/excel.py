"""官方结果表（result1~4）的写入与回读勾稽。

# ===========================================================================
# 附件 5 模板结构（4 张表，按问题选用）
#   计划购电量：行为日期（2 起始），第 2..145 列为 144 槽购电量，
#               第 146 列 = Σ_t x_t（日合计），第 147 列 = Σ_t p_t·x_t（日购电费）
#   调整购电量：同布局；第 147 列 = Σ_t [ p_t·x_adj,t + 0.5·p_t·|x_plan,t − x_adj,t| ]
#               （调整结算：调低 50% 违约金、调高 150% 购电价）
#   充放电量：每日 6 行（4 小时块），列 3/4 为块充/放电量；
#             块 0/1 的行在列 5/6 记 "00:00"/"24:00" 与两端储电量
#   紧急购电量：每个事件一行（日期、起止时间、电量 kWh），
#               事件由 core.slots.emergency_events 合并相邻紧急槽得到
# 回读勾稽：对 145/146 列合计、储能端点、充放电与紧急电量逐项与内存结果比对，
#           残差应为数值误差量级（写入采用 float，无舍入损失）。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd

from solve.common import DATA_C, N_DAY, REPORT_START, RESULTS_DIR, T
from solve.core.slots import emergency_events as _events
from solve.core.slots import fmt_time as _fmt_time


def write_result2(dates, price, x_plans, recs, out_name="result2.xlsx"):
    """按附件 5 模板生成 result2.xlsx（计划购电量 / 充放电量 / 紧急购电量）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result2.xlsx")
    rep = list(range(REPORT_START, N_DAY))

    ws = wb["计划购电量"]
    for i, d in enumerate(rep):
        row = 2 + i
        for t in range(T):
            ws.cell(row=row, column=2 + t, value=float(x_plans[d, t]))
        ws.cell(row=row, column=146, value=float(x_plans[d].sum()))
        ws.cell(row=row, column=147, value=float(price @ x_plans[d]))

    ws = wb["充放电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for d in rep:
        r = recs[d]
        for b in range(6):
            rr = row + b
            if b == 0:
                ws.cell(row=rr, column=1, value=pd.Timestamp(dates[d]).to_pydatetime())
            ws.cell(row=rr, column=2, value=f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(row=rr, column=3, value=float(r["c"][24 * b:24 * (b + 1)].sum()))
            ws.cell(row=rr, column=4, value=float(r["d"][24 * b:24 * (b + 1)].sum()))
            if b == 0:
                ws.cell(row=rr, column=5, value="00:00")
                ws.cell(row=rr, column=6, value=r["E_start"])
            if b == 1:
                ws.cell(row=rr, column=5, value="24:00")
                ws.cell(row=rr, column=6, value=r["E_end"])
        row += 6

    ws = wb["紧急购电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for d in rep:
        for j, (i0, i1, kwh) in enumerate(_events(recs[d]["e"])):
            if j == 0:
                ws.cell(row=row, column=1, value=pd.Timestamp(dates[d]).to_pydatetime())
            ws.cell(row=row, column=2,
                    value=f"{_fmt_time(i0 * 10)}-{_fmt_time((i1 + 1) * 10)}")
            ws.cell(row=row, column=3, value=round(kwh, 4))
            row += 1

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
    e_vals = [float(r[5]) for r in rows if r[4] in ("00:00", "24:00") and r[5] is not None]
    checks["充放表行数"] = len(rows)
    checks["储能端点最大偏离6000/kWh"] = (
        float(max(abs(v - 6000.0) for v in e_vals)) if e_vals else -1.0)

    ws = wb["紧急购电量"]
    kw = sum(float(r[2]) for r in ws.iter_rows(min_row=2, values_only=True)
             if r[2] is not None)
    checks["紧急表电量勾稽差/kWh"] = float(abs(kw - df["紧急购电量/kWh"].sum()))
    wb.close()
    return checks


def write_result3(dates, price, out):
    """按附件 5 模板生成 results/result3.xlsx（计划/调整/充放电/紧急，334 天）。"""
    wb = openpyxl.load_workbook(DATA_C / "附件5" / "result3.xlsx")
    i0 = REPORT_START
    n_q3 = N_DAY - REPORT_START

    ws = wb["计划购电量"]
    for i in range(n_q3):
        row = 2 + i
        for t in range(T):
            ws.cell(row=row, column=2 + t, value=float(out["xp"][i, t]))
        ws.cell(row=row, column=146, value=float(out["xp"][i].sum()))
        ws.cell(row=row, column=147, value=float(price @ out["xp"][i]))

    ws = wb["调整购电量"]
    for i in range(n_q3):
        row = 2 + i
        xa, xp = out["xa"][i], out["xp"][i]
        for t in range(T):
            ws.cell(row=row, column=2 + t, value=float(xa[t]))
        ws.cell(row=row, column=146, value=float(xa.sum()))
        # 调整结算费 = Σ [p·x_adj + 0.5·p·|x_plan − x_adj|]（调低 50% 违约金）
        settle = float((price * xa + 0.5 * price * np.abs(xa - xp)).sum())
        ws.cell(row=row, column=147, value=settle)

    ws = wb["充放电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for i in range(n_q3):
        for b in range(6):
            rr = row + b
            if b == 0:
                ws.cell(row=rr, column=1, value=pd.Timestamp(dates[i0 + i]).to_pydatetime())
            ws.cell(row=rr, column=2, value=f"{4 * b}:00-{4 * (b + 1)}:00")
            ws.cell(row=rr, column=3, value=float(out["c"][i, 24 * b:24 * (b + 1)].sum()))
            ws.cell(row=rr, column=4, value=float(out["d"][i, 24 * b:24 * (b + 1)].sum()))
            if b == 0:
                ws.cell(row=rr, column=5, value="00:00")
                ws.cell(row=rr, column=6, value=6000.0)
            if b == 1:
                ws.cell(row=rr, column=5, value="24:00")
                ws.cell(row=rr, column=6, value=float(out["E"][i, -1]))
        row += 6

    ws = wb["紧急购电量"]
    while ws.max_row > 1:
        ws.delete_rows(2)
    row = 2
    for i in range(n_q3):
        for j, (a, b, kwh) in enumerate(_events(out["e"][i])):
            if j == 0:
                ws.cell(row=row, column=1, value=pd.Timestamp(dates[i0 + i]).to_pydatetime())
            ws.cell(row=row, column=2, value=f"{_fmt_time(a * 10)}-{_fmt_time((b + 1) * 10)}")
            ws.cell(row=row, column=3, value=round(kwh, 4))
            row += 1

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
    ev = [float(r[5]) for r in rows if r[4] in ("00:00", "24:00") and r[5] is not None]
    chk["储能端点最大偏离6000/kWh"] = float(max(abs(v - 6000.0) for v in ev))

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
        for j, (a, b, kwh) in enumerate(_events(recs[D]["e"])):
            if j == 0:
                ws.cell(row, 1, pd.Timestamp(data["dates"][D]).to_pydatetime())
            ws.cell(row, 2, f"{_fmt_time(a * 10)}-{_fmt_time((b + 1) * 10)}")
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
        for j, (a, b, kwh) in enumerate(_events(recs[D]["e"])):
            if j == 0:
                ws.cell(row, 1, pd.Timestamp(data["dates"][D]).to_pydatetime())
            ws.cell(row, 2, f"{_fmt_time(a * 10)}-{_fmt_time((b + 1) * 10)}")
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
