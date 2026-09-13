"""program —— 数学建模便利工具包（CUMCM / 华为杯 / MCM 通用）。

十一个模块覆盖数学建模全流程：

============================  ==================================================
模块                            用途
============================  ==================================================
:mod:`program.config`         全局配置：中文字体、论文风格、输出目录、随机种子
:mod:`program.utils`          随机种子、计时、日志、结果缓存、运行环境、格式化
:mod:`program.dataio`         数据读写（csv/xlsx/json/pkl）、缺失处理、概览
:mod:`program.plotting`       论文级绘图：PDF 矢量输出 + 常用图型
:mod:`program.stats`          统计检验、相关、回归、置信区间（中文结论）
:mod:`program.fitting`        曲线拟合、多项式、插值、平滑
:mod:`program.metrics`        回归 / 分类 / 聚类评估指标
:mod:`program.optimize`       LP / MILP / NLP / 全局优化 / 背包 / 指派 / TSP
:mod:`program.ml`             机器学习快捷流程（训练 + 交叉验证 + 评估）
:mod:`program.ode`            常微分方程求解与参数反演
:mod:`program.report`         RESULTS_REPORT 记录、LaTeX 表格、结果文件输出
============================  ==================================================

标准脚本骨架::

    import program as pm

    pm.init()                                   # 中文字体 + 风格 + 随机种子
    df = pm.read_table("data/附件1.xlsx")        # 数据
    ...                                         # 建模
    pm.record_result("问题一结果", {"最优值": 42})  # 结果写进 reports/RESULTS_REPORT.md
    pm.line(x, y, xlabel="t", ylabel="y", save="q1_curve")   # 图存 figures/q1_curve.pdf

完整说明见 ``program/AGENTS.md``。
"""

from __future__ import annotations

from . import (  # noqa: F401  （统一导入，便于 pm.stats.xxx 访问）
    config,
    dataio,
    fitting,
    metrics,
    ml,
    ode,
    optimize,
    plotting,
    report,
    stats,
    utils,
)

# 高频函数提升到顶层，便于 pm.read_table / pm.save_fig 直接使用
from .config import (  # noqa: F401
    find_cjk_font,
    figures_dir,
    init,
    outputs_dir,
    project_root,
    reports_dir,
    set_project_root,
)
from .dataio import read_sheets, read_table, save_table, summarize  # noqa: F401
from .plotting import (  # noqa: F401
    bar,
    bar_group,
    box,
    corr_heatmap,
    dual_axis,
    errorbar,
    heatmap,
    hist,
    line,
    pie,
    radar,
    roc,
    save_fig,
    scatter,
    scatter_fit,
    surface3d,
)
from .report import record, record_environment, record_result, save_outputs, to_latex  # noqa: F401
from .utils import DataCache, Timer, fmt, get_logger, markdown_table, set_seed  # noqa: F401

__version__ = "0.2.0"

__all__ = [
    # 子模块
    "config", "dataio", "fitting", "metrics", "ml", "ode", "optimize",
    "plotting", "report", "stats", "utils",
    # 配置与工具
    "init", "set_seed", "get_logger", "Timer", "DataCache",
    "project_root", "set_project_root", "figures_dir", "reports_dir", "outputs_dir",
    "find_cjk_font", "fmt", "markdown_table",
    # 数据
    "read_table", "read_sheets", "save_table", "summarize",
    # 绘图
    "save_fig", "line", "scatter", "scatter_fit", "bar", "bar_group", "hist",
    "box", "heatmap", "corr_heatmap", "radar", "dual_axis", "pie", "errorbar",
    "surface3d", "roc",
    # 报告
    "record", "record_result", "record_environment", "save_outputs", "to_latex",
]


def main() -> None:
    """CLI 入口（``uv run program``）：打印包信息与模块清单。"""
    print(f"program —— 数学建模便利工具包 v{__version__}")
    print()
    print("模块：")
    for name in (
        "config", "utils", "dataio", "plotting", "stats", "fitting",
        "metrics", "optimize", "ml", "ode", "report",
    ):
        mod = __import__(f"program.{name}", fromlist=[name])
        doc = (mod.__doc__ or "").strip().splitlines()[0]
        print(f"  program.{name:<10} {doc}")
    print()
    print("快速开始：")
    print("  import program as pm")
    print("  pm.init()")
    print("  df = pm.read_table('data.xlsx')")
    print("  pm.record_result('问题一结果', {'目标值': 1.23})")
    print("  pm.line(x, y, save='fig1')")
    print()
    print("详细文档：program/AGENTS.md")
