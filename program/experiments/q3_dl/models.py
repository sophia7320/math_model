# -*- coding: utf-8 -*-
"""Q3 深度学习试验：模型定义（TCN / Transformer）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class TCNNet(nn.Module):
    """膨胀卷积（TCN 风格）：输入 (B,12,144) → 输出 (B,144)。"""

    def __init__(self, cin: int = 12, ch: int = 64, layers: int = 5, drop: float = 0.1):
        super().__init__()
        blocks = []
        c = cin
        for i in range(layers):
            d = 2 ** i
            blocks += [
                nn.Conv1d(c, ch, 5, padding=2 * d, dilation=d),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            c = ch
        self.body = nn.Sequential(*blocks)
        self.head = nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)


class PVTransformer(nn.Module):
    """每个 10 分钟槽作为一个 token 的 Transformer 编码器：输入 (B,12,144) → 输出 (B,144)。"""

    def __init__(self, cin: int = 12, d: int = 64, nhead: int = 4,
                 layers: int = 4, drop: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(cin, d)
        self.pos = nn.Parameter(torch.zeros(1, 144, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=d * 2,
            dropout=drop, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = x.transpose(1, 2)          # (B,144,12)
        h = self.proj(h) + self.pos
        h = self.enc(h)
        return self.head(h).squeeze(-1)
