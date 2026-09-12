"""C 题求解公共部分：路径解析、储能参数（附录 1）、附件读取。

运行入口见 ``python -m solve``（在 program/ 目录下执行）。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

import program as pm

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parents[2]   # program/
DATA_C = PKG_DIR / "data" / "C"                 # 附件 1~5 所在目录


def workspace_root() -> Path:
    """工作区根目录（figures / reports / code / results 的输出位置）。

    优先级：环境变量 ``MATHMODEL_ROOT`` > ``program/`` 的上一级（含 AGENTS.md）> 当前目录。
    """
    env = os.environ.get("MATHMODEL_ROOT")
    if env:
        return Path(env).resolve()
    cand = PKG_DIR.parent
    if (cand / "AGENTS.md").exists():
        return cand
    return Path.cwd()


ROOT = workspace_root()
RESULTS_DIR = ROOT / "results"                  # 官方结果文件（result1.xlsx 等）

# ---------------------------------------------------------------------------
# 附录 1：储能设备参数（全文共用；口径假设见 q1.py 文档）
# ---------------------------------------------------------------------------
T = 144                     # 一天 144 个 10 分钟时段
DT_H = 1.0 / 6.0            # 每时段小时数（10 min）
ETA = 0.9                   # 充放电效率（充、放双向计损口径）
E0 = 6000.0                 # 初始储电量 kWh（2025-01-01 0:00）
E_MIN, E_MAX = 1200.0, 10800.0
P_MAX_KW = 5000.0           # 最大充/放电功率 kW
P_MAX_E = P_MAX_KW * DT_H   # 每时段最大充/放电量 kWh（= 833.33…）


def read_attachment(name: str, sheet: int | str = 0) -> pd.DataFrame:
    """读取 ``program/data/C`` 下的附件（如 ``"附件1.xlsx"``）。"""
    return pm.read_table(DATA_C / name, sheet=sheet)
