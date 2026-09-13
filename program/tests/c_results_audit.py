"""C 题五份官方结果文件的结构审计（独立于写出脚本，提交前固定检查）。

校验内容：sheet 名与顺序、日期连续性、行数、数值非负/有限、全天购电量与
源电价逐槽费用重算、储能端点/跨日连续性与 SOC 界、时间标签。

注意：官方模板（附件 5）把 24:00 储电量放在 4:00-8:00 行，本脚本按模板校验。

运行（program/ 下）：uv run python tests/c_results_audit.py
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DATA_C = ROOT / "program" / "data" / "C"

EXPECTED = {
    "result1.xlsx": (["计划购电量", "充放电量"], 1, 6),
    "result2.xlsx": (["计划购电量", "充放电量", "紧急购电量"], 334, 2004),
    "result3.xlsx": (["计划购电量", "调整购电量", "充放电量", "紧急购电量"], 334, 2004),
    "result4-2.xlsx": (["计划购电量", "充放电量", "紧急购电量"], 334, 2004),
    "result4-3.xlsx": (["计划购电量", "调整购电量", "充放电量", "紧急购电量"], 334, 2004),
}


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _prices() -> tuple[np.ndarray, dict[date, np.ndarray]]:
    wb = openpyxl.load_workbook(DATA_C / "附件1.xlsx", read_only=True, data_only=True)
    static = np.asarray([r[1] for r in list(wb.active.values)[1:145]], dtype=float)
    wb.close()
    wb = openpyxl.load_workbook(DATA_C / "附件4.xlsx", read_only=True, data_only=True)
    dynamic = {
        _as_date(r[0]): np.asarray(r[1:145], dtype=float)
        for r in list(wb.active.values)[1:] if r[0] is not None
    }
    wb.close()
    return static, dynamic


def audit_workbook(name: str, sheets: list[str], n_days: int, n_charge: int) -> None:
    path = RESULTS / name
    assert path.exists(), f"缺少 {path}"
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    assert wb.sheetnames == sheets, (name, wb.sheetnames)

    # 计划/调整购电量：日行 × 144 槽 + 全天电量/费用汇总列
    if name == "result1.xlsx":
        rows = [r for r in wb["计划购电量"].iter_rows(min_row=2, values_only=True)
                if r[0] is not None]
        assert len(rows) == 144, (name, "计划购电量", len(rows))
        values = np.asarray([r[1] for r in rows], dtype=float)
        assert np.isfinite(values).all() and (values >= -1e-7).all()
    static_price, dynamic_price = _prices()
    purchase_rows: dict[str, list[tuple]] = {}
    for sheet in ("计划购电量", "调整购电量"):
        if sheet not in sheets or name == "result1.xlsx":
            continue
        rows = [r for r in wb[sheet].iter_rows(min_row=2, values_only=True) if r[0] is not None]
        purchase_rows[sheet] = rows
        assert len(rows) == n_days, (name, sheet, len(rows))
        dates = [_as_date(r[0]) for r in rows]
        expected = [date(2025, 2, 1) + timedelta(days=i) for i in range(n_days)]
        assert dates == expected, (name, sheet, "日期不连续或范围错误", dates[:1], dates[-1:])
        for i, r in enumerate(rows):
            slots = np.asarray(r[1:145], dtype=float)
            assert np.isfinite(slots).all() and (slots >= -1e-7).all()
            assert abs(float(r[145]) - float(slots.sum())) < 1e-4, (name, sheet, "勾稽")
            assert np.isfinite(float(r[146])) and float(r[146]) >= -1e-7
            price = dynamic_price[dates[i]] if name.startswith("result4-") else static_price
            if sheet == "调整购电量":
                plan = np.asarray(purchase_rows["计划购电量"][i][1:145], dtype=float)
                expected_fee = float(price @ slots + (0.5 * price * np.abs(slots - plan)).sum())
            else:
                expected_fee = float(price @ slots)
            assert abs(float(r[146]) - expected_fee) < 1e-4, (
                name, sheet, "费用列不能由源电价重算", float(r[146]), expected_fee)

    # 充放电量：0:00/24:00 行（官方模板位于第 1/2 块）标签与 SOC
    charge = [r for r in wb["充放电量"].iter_rows(min_row=2, values_only=True)
              if r[1] is not None]
    assert len(charge) == n_charge, (name, "充放电量", len(charge))
    cc, dc, ec = (1, 2, 4) if name == "result1.xlsx" else (2, 3, 5)
    e0 = e24 = None
    first_block = charge[0]
    second_block = charge[1] if len(charge) > 1 else None
    if name == "result1.xlsx":
        assert first_block[3] == "0:00" and first_block[ec] is not None
        assert second_block[3] == "24:00" and second_block[ec] is not None
        e0, e24 = float(first_block[ec]), float(second_block[ec])
    for r in charge:
        assert float(r[cc]) >= -1e-7 and float(r[dc]) >= -1e-7
        if r[ec] is not None:
            assert 1200.0 - 1e-6 <= float(r[ec]) <= 10800.0 + 1e-6, (name, r[ec])
    if name == "result1.xlsx":
        assert abs(e0 - e24) < 1e-6, (name, "0:00/24:00 储电量不相等", e0, e24)
    else:
        plan_dates = [_as_date(r[0]) for r in purchase_rows["计划购电量"]]
        e0s, e24s, charge_dates = [], [], []
        for i in range(n_days):
            block = charge[6 * i:6 * (i + 1)]
            t0 = block[0][4]
            assert ((t0 == "00:00") or
                    (hasattr(t0, "strftime") and t0.strftime("%H:%M") == "00:00")) \
                and block[1][4] == "24:00", (name, i, "0:00/24:00 标签错误")
            charge_dates.append(_as_date(block[0][0]))
            e0s.append(float(block[0][ec]))
            e24s.append(float(block[1][ec]))
        assert charge_dates == plan_dates, (name, "充放表日期与计划表不一致")
        assert max(abs(e24s[i] - e0s[i + 1]) for i in range(n_days - 1)) < 1e-6, (
            name, "跨日 SOC 不连续")

    # 紧急购电量：电量非负
    if "紧急购电量" in sheets:
        for r in wb["紧急购电量"].iter_rows(min_row=2, values_only=True):
            if r[2] is not None:
                assert float(r[2]) >= -1e-7, (name, r[:3])

    # 时间标签：与附件 5 模板一致（附件与模板标签相差一个 10 分钟刻度，
    # 一律按行序号对齐；重写文件时不得改动模板标签）
    tpl = openpyxl.load_workbook(DATA_C / "附件5" / name, read_only=True, data_only=True)
    t_rows = list(tpl["计划购电量"].iter_rows(values_only=True))
    hdr = next(wb["计划购电量"].iter_rows(min_row=1, max_row=1, values_only=True))

    def _norm(v):
        return None if v is None else str(v)

    assert [_norm(v) for v in hdr] == [_norm(v) for v in t_rows[0]], (
        name, "计划购电量表头/时段标签与附件 5 模板不一致")
    if name == "result1.xlsx":
        labels = [_norm(r[0]) for r in wb["计划购电量"].iter_rows(min_row=2, values_only=True)
                  if r[0] is not None]
        assert labels == [_norm(r[0]) for r in t_rows[1:]], (
            name, "逐行时段标签与附件 5 模板不一致")
    label_col = 0 if name == "result1.xlsx" else 1   # 充放电量表的“时间段”列
    blocks = [str(r[label_col]) for r in wb["充放电量"].iter_rows(min_row=2, values_only=True)
              if r[label_col] is not None]
    expected_blocks = ["0:00-4:00", "4:00-8:00", "8:00-12:00",
                       "12:00-16:00", "16:00-20:00", "20:00-24:00"]
    assert blocks and all(b == expected_blocks[i % 6] for i, b in enumerate(blocks)), (
        name, "充放电量块标签错误", blocks[:6])
    tpl.close()
    wb.close()


# ---------------------------------------------------------------------------
# 物理与总量复核（2026-09-13 补充，独立于写出脚本）
#   - 物理平衡（隐含弃光 ≥ 0）：xa/x + P + d + e = L + c + s
#   - 块级充放功率限（4 h 块 ≤ 24×5000/6 kWh）
#   - SOC 逐日递推：E24 = E0 + η·Σc − Σd/η
#   - 紧急费包络：事件内价格上下界 × 电量（真值必落在包络内），
#     并冻结报告口径的紧急费（万元）
#   - 总量冻结：计划费 / 买入 / 偏差 / 调整净额（万元，报告口径）
# ---------------------------------------------------------------------------
ETA = 0.9
DAY0 = date(2025, 2, 1)
START_DAY = 31
BLOCK_LIMIT = 24 * 5000.0 / 6.0  # 4 小时块充电/放电上限（kWh）

EXPECTED_TOTALS = {  # 万元（RESULTS_REPORT v1.4 冻结口径）
    "result1.xlsx": {"kwh": 59482.7, "cost": 35126.95},
    "result2.xlsx": {"plan": 1316.0, "emerg": 79.6},
    "result3.xlsx": {"plan": 1290.5, "net": 25.2, "emerg": 31.4},
    "result4-2.xlsx": {"plan": 1367.0, "emerg": 92.8},
    "result4-3.xlsx": {"buy": 1333.5, "dev": 36.3, "emerg": 40.5},
}


def _load_a2() -> tuple[np.ndarray, np.ndarray]:
    wb = openpyxl.load_workbook(DATA_C / "附件2.xlsx", read_only=True, data_only=True)
    load = np.asarray([[v for v in r[1:145]] for r in list(wb.active.values)[1:]], float)
    pv = np.asarray([[v for v in r[1:145]] for r in list(wb["光伏发电实际功率"].values)[1:]],
                    float)
    wb.close()
    return load, pv


def _read_plan(ws) -> dict:
    out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is None:
            continue
        out[_as_date(r[0])] = np.asarray(r[1:145], float)
    return out


def _read_charge(ws, first: bool) -> dict:
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None]
    cc, dc, ec = (1, 2, 4) if first else (2, 3, 5)
    out = {}
    for i in range(len(rows) // 6):
        b = rows[6 * i:6 * (i + 1)]
        key = i if first else _as_date(b[0][0])
        out[key] = (np.asarray([x[cc] for x in b], float),
                    np.asarray([x[dc] for x in b], float),
                    float(b[0][ec]), float(b[1][ec]))
    return out


def _parse_events(ws) -> dict:
    """{日期: [(起始槽, 结束槽, kWh)]}；结束槽为开区间，跨日写法按当日末处理。"""
    out: dict = {}
    cur = None
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0] is not None:
            cur = _as_date(r[0])
        if r[1] is None:
            continue
        m = re.match(r"(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})(\+1)?", str(r[1]))
        assert m, (str(r[1]),)
        a = int(m.group(1)) * 60 + int(m.group(2))
        b = int(m.group(3)) * 60 + int(m.group(4))
        if m.group(5) or b <= a:  # 跨日写法（如 0:00+1）
            b = 24 * 60
        out.setdefault(cur, []).append((a // 10, b // 10, float(r[2])))
    return out


def audit_physics(name: str) -> None:
    static_price, dynamic_price = _prices()
    wb = openpyxl.load_workbook(RESULTS / name, read_only=True, data_only=True)
    exp = EXPECTED_TOTALS[name]

    if name == "result1.xlsx":
        wb1 = openpyxl.load_workbook(DATA_C / "附件1.xlsx", read_only=True, data_only=True)
        rows = list(wb1.active.values)[1:145]
        P1 = np.asarray([r[3] for r in rows], float) / 6.0
        L1 = np.asarray([r[2] for r in rows], float) / 6.0
        x = np.asarray([r[1] for r in wb["计划购电量"].iter_rows(min_row=2, values_only=True)
                        if r[0] is not None], float)
        c, d, e0, e24 = _read_charge(wb["充放电量"], first=True)[0]
        assert abs(x.sum() - exp["kwh"]) < 0.05, (name, x.sum())
        cost = float(static_price @ x)
        assert abs(cost - exp["cost"]) < 0.05, (name, cost)
        s_impl = float(x.sum() + P1.sum() + d.sum() - L1.sum() - c.sum())
        assert -1e-6 <= s_impl <= 1e-3, (name, "隐含弃光", s_impl)
        assert abs(e24 - (e0 + ETA * c.sum() - d.sum() / ETA)) < 1e-6, (name, "SOC 递推")
        assert max(c.max(), d.max()) <= BLOCK_LIMIT + 1e-6, (name, "块级功率限")
        wb.close()
        return

    load2, pv2 = _load_a2()
    plan = _read_plan(wb["计划购电量"])
    adj = _read_plan(wb["调整购电量"]) if "调整购电量" in wb.sheetnames else None
    chg = _read_charge(wb["充放电量"], first=False)
    ev = _parse_events(wb["紧急购电量"])
    days = sorted(plan)
    assert len(days) == 334, len(days)

    plan_cost = buy = dev = 0.0
    lo_all = hi_all = 0.0
    worst_s = np.inf
    max_soc = 0.0
    max_block = 0.0
    for dt in days:
        D = (dt - DAY0).days + START_DAY
        price = dynamic_price[dt] if name.startswith("result4-") else static_price
        x = plan[dt]
        plan_cost += float(price @ x)
        if adj is not None:
            xa = adj[dt]
            buy += float(price @ xa)
            dev += float((0.5 * price * np.abs(xa - x)).sum())
            x_day = xa
        else:
            x_day = x
        c, d, e0, e24 = chg[dt]
        max_block = max(max_block, c.max(), d.max())
        max_soc = max(max_soc, abs(e24 - (e0 + ETA * c.sum() - d.sum() / ETA)))
        e_day = 0.0
        for a, b, k in ev.get(dt, []):
            assert 0 <= a < b <= 144, (name, dt, a, b)
            seg = price[a:b]
            lo_all += 5.0 * seg.min() * k
            hi_all += 5.0 * seg.max() * k
            e_day += k
        Ld, Pd = load2[D] / 6.0, pv2[D] / 6.0
        worst_s = min(worst_s, float(x_day.sum() + Pd.sum() + d.sum() + e_day
                                     - Ld.sum() - c.sum()))

    assert worst_s >= -1e-3, (name, "隐含弃光为负", worst_s)
    assert max_soc < 1e-6, (name, "SOC 递推", max_soc)
    assert max_block <= BLOCK_LIMIT + 1e-6, (name, "块级功率限", max_block)
    wan = 1e4
    if "plan" in exp:
        assert abs(plan_cost / wan - exp["plan"]) < 0.1, (name, plan_cost / wan)
    if "net" in exp:
        assert abs((buy + dev - plan_cost) / wan - exp["net"]) < 0.1, (
            name, (buy + dev - plan_cost) / wan)
    if "buy" in exp:
        assert abs(buy / wan - exp["buy"]) < 0.1, (name, buy / wan)
        assert abs(dev / wan - exp["dev"]) < 0.1, (name, dev / wan)
    e_seg = exp["emerg"] * wan
    assert lo_all - 1e-6 <= e_seg <= hi_all + 1e-6, (
        name, "紧急费包络", lo_all / wan, e_seg / wan, hi_all / wan)
    wb.close()


def main() -> None:
    for name, (sheets, n_days, n_charge) in EXPECTED.items():
        audit_workbook(name, sheets, n_days, n_charge)
        audit_physics(name)
        print(f"ok - {name}")
    print("C result audit: PASS")


if __name__ == "__main__":
    main()
