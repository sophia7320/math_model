# -*- coding: utf-8 -*-
"""Q3 DL 试验：滚动重训（扩展窗口）生成全年预测。

块划分（每块：train 7..cut−20，val 最后 20 天，预测未来 60 天）：
    cut=120 → predict 121..180
    cut=180 → predict 181..240
    cut=240 → predict 241..300
    cut=300 → predict 301..364
覆盖 121~364 共 244 天；每块从零训练（不继承）。

运行（program/ 下）：uv run python experiments/q3_dl/train_rolling.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import SCALE, build_dataset, load_raw
from models import PVTransformer
from train import SEED, set_seed, train_one

OUT = _DIR / "outputs"
BLOCKS = [(120, 180), (180, 240), (240, 300), (300, 364)]
VAL_DAYS = 20


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    data = load_raw()
    X, Y, days = build_dataset(data)
    act = data["pv_act"]

    all_days, all_pred, val_maes = [], [], []
    for cut, pr_end in BLOCKS:
        tr = (days >= 7) & (days <= cut - VAL_DAYS)
        va = (days > cut - VAL_DAYS) & (days <= cut)
        pr = (days > cut) & (days <= pr_end)
        print(f"\n[block cut={cut}] train {int(tr.sum())} / val {int(va.sum())} / predict {int(pr.sum())}")
        set_seed(SEED)
        model = PVTransformer(d=64, nhead=4, layers=4, drop=0.1).to(device)
        loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])),
                            batch_size=args.batch, shuffle=True)
        va_x = torch.from_numpy(X[va]).to(device)
        va_y = torch.from_numpy(Y[va]).to(device)
        t0 = time.time()
        _, val_mae = train_one(model, loader, va_x, va_y, args.epochs, args.lr, device, f"cut{cut}")
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X[pr]).to(device)).cpu().numpy() * SCALE
        d_pr = days[pr]
        mae = float(np.abs(pred - act[d_pr]).mean())
        print(f"    val MAE {val_mae:.1f} kW | 预测块 MAE {mae:.1f} kW（{len(d_pr)} 天，{time.time() - t0:.0f}s）")
        all_days.append(d_pr)
        all_pred.append(pred)
        val_maes.append(val_mae)

    all_days = np.concatenate(all_days)
    all_pred = np.concatenate(all_pred)
    np.savez_compressed(OUT / "dl_pred_full.npz", days=all_days, pred=all_pred,
                        val_mae=np.array(val_maes), blocks=np.array(BLOCKS))
    print(f"\n已保存 {OUT / 'dl_pred_full.npz'}（{len(all_days)} 天）")


if __name__ == "__main__":
    main()
