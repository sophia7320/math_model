"""C 题五份官方结果文件的结构审计（独立于写出脚本，提交前固定检查）。

校验内容：sheet 名与顺序、行数、数值非负/有限、全天购电量与购电费勾稽、
储能端点与 SOC 界、时间标签（0:00/24:00 按官方模板位于前两块）。

注意：官方模板（附件 5）把 24:00 储电量放在 4:00-8:00 行，本脚本按模板校验。

运行（program/ 下）：uv run python tests/c_results_audit.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

EXPECTED = {
    "result1.xlsx": (["计划购电量", "充放电量"], 1, 6),
    "result2.xlsx": (["计划购电量", "充放电量", "紧急购电量"], 334, 2004),
    "result3.xlsx": (["计划购电量", "调整购电量", "充放电量", "紧急购电量"], 334, 2004),
    "result4-2.xlsx": (["计划购电量", "充放电量", "紧急购电量"], 334, 2004),
    "result4-3.xlsx": (["计划购电量", "调整购电量", "充放电量", "紧急购电量"], 334, 2004),
}


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
    for sheet in ("计划购电量", "调整购电量"):
        if sheet not in sheets or name == "result1.xlsx":
            continue
        rows = [r for r in wb[sheet].iter_rows(min_row=2, values_only=True) if r[0] is not None]
        assert len(rows) == n_days, (name, sheet, len(rows))
        for r in rows:
            slots = np.asarray(r[1:145], dtype=float)
            assert np.isfinite(slots).all() and (slots >= -1e-7).all()
            assert abs(float(r[145]) - float(slots.sum())) < 1e-4, (name, sheet, "勾稽")
            assert np.isfinite(float(r[146])) and float(r[146]) >= -1e-7

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

    # 紧急购电量：电量非负
    if "紧急购电量" in sheets:
        for r in wb["紧急购电量"].iter_rows(min_row=2, values_only=True):
            if r[2] is not None:
                assert float(r[2]) >= -1e-7, (name, r[:3])
    wb.close()


def main() -> None:
    for name, (sheets, n_days, n_charge) in EXPECTED.items():
        audit_workbook(name, sheets, n_days, n_charge)
        print(f"ok - {name}")
    print("C result audit: PASS")


if __name__ == "__main__":
    main()
