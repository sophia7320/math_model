"""C 题求解入口（微网与外部电网电力调控策略）。

在 ``program/`` 目录下运行：

    uv run python -m solve      # 运行问题一（问题二~四后续在此扩展）

输出约定（工作区根目录）：
    figures/q1_*.pdf        论文图（矢量 PDF）
    reports/RESULTS_REPORT.md   结果报告（论文数值唯一来源）
    code/outputs/*.csv      中间结果与图表数据
    results/result1.xlsx    官方格式结果文件
"""
from __future__ import annotations


def main() -> None:
    # 延迟导入：避免 `python -m solve.q1` 先经包初始化再执行子模块时
    # 触发 runpy 的 "found in sys.modules" 警告。
    from solve.q1 import run_q1

    run_q1()


if __name__ == "__main__":
    main()
