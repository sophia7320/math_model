# -*- coding: utf-8 -*-
"""Q3 深度学习试验：数据准备（无前视特征）。

输入（预测目标日 D）：
- 过去 7 天实际光伏（7 × 144，kW）
- 当天 0:00 官方预报（24 整点 → 插值 144 槽）
- 小时 sin/cos（2 × 144）
- 年内日 sin/cos（2 × 144）
共 12 通道 × 144 槽；目标 = 当天 144 槽实际光伏。
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROGRAM = Path(__file__).resolve().parents[2]
_SRC = _PROGRAM / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np

from solve import q2

N_DAY, T = 365, 144
LOOKBACK = 7
SCALE = 10000.0
TRAIN_END, VAL_END = 279, 299   # train: 7..279, val: 280..299, test: 300..364


def load_raw() -> dict:
    return q2.load_all()


def build_dataset(data: dict):
    """返回 X (N,12,T), Y (N,T), days (N,)，全部为归一化后的 float32。"""
    pv = data["pv_act"] / SCALE
    fc0s = np.stack([q2._hour_to_slots(data["fc0"][d]) / SCALE for d in range(N_DAY)])

    hours = (np.arange(T) + 0.5) / 6.0
    hfeat = np.stack([np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)], 0)

    days = np.arange(LOOKBACK, N_DAY)
    X = np.zeros((len(days), 12, T), np.float32)
    Y = np.zeros((len(days), T), np.float32)
    for i, d in enumerate(days):
        doy = 2 * np.pi * d / 365
        df = np.stack([np.full(T, np.sin(doy)), np.full(T, np.cos(doy))], 0)
        hist = pv[d - LOOKBACK:d][::-1]                 # 最近一天在前
        X[i] = np.concatenate([hist, fc0s[d][None], hfeat, df], 0)
        Y[i] = pv[d]
    return X, Y, days


def split(days: np.ndarray):
    tr = days <= TRAIN_END
    va = (days > TRAIN_END) & (days <= VAL_END)
    te = days > VAL_END
    return tr, va, te
