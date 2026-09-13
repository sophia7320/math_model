"""共用绘图辅助：月份刻度。

# ===========================================================================
# month_axis(ax, dates)
#     输入：与横轴一一对应的日期字符串列表（ISO，如 "2025-03-01"）
#     作用：以每月 1 日的位置设刻度，标签为 "N月"
#     等价于旧代码中手写的
#         pos = [i for i, s in enumerate(dates) if s.endswith("-01")]
#         ax.set_xticks(pos); ax.set_xticklabels([dates[i][5:7] + "月" for i in pos])
# ===========================================================================
"""
from __future__ import annotations


def month_axis(ax, dates) -> None:
    """在 ax 上设置月份刻度（dates 与横轴索引一一对应）。"""
    pos = [i for i, s in enumerate(dates) if s.endswith("-01")]
    ax.set_xticks(pos)
    ax.set_xticklabels([dates[i][5:7] + "月" for i in pos])
