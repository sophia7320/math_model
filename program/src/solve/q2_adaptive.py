"""C 题 问题二 口径 E：历史自适应加权预测 + 网格定位 + 梯度精化（兼容门面）。

实现已迁至 ``solve.models.adaptive``（类 :class:`AdaptiveWeightModel`）；
本文件保留原命令行入口与全部公开名（``softmax`` / ``simplex_grid`` /
``record`` / ``_verify_result2`` / N_DAY / REPORT_START），供旧脚本与测试导入。

用法：
    from solve.q2_adaptive import AdaptiveWeightModel

    model = AdaptiveWeightModel(W=1).load()
    result = model.build_table().run()          # 主结果（含对照/图/报告）
    model.window_sensitivity()                  # 标定窗口敏感性

命令行：uv run python -m solve.q2_adaptive   （在 program/ 目录；首次约 8 分钟）
"""
from __future__ import annotations

from solve.common import N_DAY, REPORT_START
from solve.io.excel import verify_result2 as _verify_result2
from solve.io.report import record
from solve.models.adaptive import (
    AdaptiveWeightModel,
    _init_worker,
    _worker_day,
    hist_forecast,
    total_of,
)
from solve.models.weights import simplex_grid, softmax

if __name__ == "__main__":
    import sys

    _model = AdaptiveWeightModel().load().build_table()
    if "--result2" in sys.argv:
        _model.write_result2()
    else:
        _model.run()
