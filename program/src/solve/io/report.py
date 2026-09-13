"""结果报告写入：RESULTS_REPORT.md 章节级去重追加。

报告是论文数值的唯一来源；同名章节重跑时必须先删后写，避免重复堆积。
"""
from __future__ import annotations

import program as pm


def record(section: str, data, note: str = "") -> None:
    """写结果报告：先删除同名旧章节再追加（pm.record_result 为追加模式）。

    删除规则：定位 "### {section}" 起始，截到下一个 "\\n### " 之前；
    若为最后一节则截到文件末尾。
    """
    path = pm.reports_dir() / "RESULTS_REPORT.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        marker = f"### {section}"
        pos = text.find(marker)
        if pos != -1:
            end = text.find("\n### ", pos + len(marker))
            text = text[:pos] if end == -1 else text[:pos] + text[end + 1 :]
            path.write_text(text, encoding="utf-8")
    pm.record_result(section, data, note=note)
