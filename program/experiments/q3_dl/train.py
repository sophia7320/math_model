# -*- coding: utf-8 -*-
"""Q3 深度学习试验：训练 + 评估（GPU）。

运行（program/ 下）：uv run python experiments/q3_dl/train.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import SCALE, T, build_dataset, load_raw, split
from models import PVTransformer, TCNNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = _DIR / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42


def set_seed(s: int):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_one(model, loader, va_x, va_y, epochs, lr, device, name):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.L1Loss()
    best = {"val": 1e9, "state": None, "epoch": -1}
    hist = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        model.eval()
        with torch.no_grad():
            val = float(lossf(model(va_x), va_y).item())
        hist["train"].append(tot / n)
        hist["val"].append(val)
        if val < best["val"] - 1e-6:
            best = {"val": val, "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep % 50 == 0 or ep == 1:
            print(f"    [{name}] epoch {ep:4d}/{epochs}  train {tot / n:.5f}  "
                  f"val {val:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best["state"])
    print(f"    [{name}] best epoch {best['epoch']}  "
          f"val MAE {best['val'] * SCALE:.1f} kW  ({time.time() - t0:.0f}s)")
    return hist, best["val"] * SCALE


def metrics(pred, act):
    e = np.abs(pred - act)
    day = act >= 500
    return {
        "MAE_kW": float(e.mean()),
        "MAE_光照样本_kW": float(e[day].mean()) if day.any() else float("nan"),
        "MAE_正午_kW": float(e[:, 54:90].mean()),
        "RMSE_kW": float(np.sqrt(((pred - act) ** 2).mean())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    import solve.q2 as _q2
    import solve.q3_proto as _qp

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"device = {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("device = cpu（未检测到 CUDA）")

    data = load_raw()
    X, Y, days = build_dataset(data)
    tr, va, te = split(days)
    print(f"样本：train {int(tr.sum())} / val {int(va.sum())} / test {int(te.sum())}")

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(Y[tr])),
                           batch_size=args.batch, shuffle=True)
    va_x = torch.from_numpy(X[va]).to(device)
    va_y = torch.from_numpy(Y[va]).to(device)

    results, preds, hists = {}, {}, {}
    models = ((TCNNet, "TCN", dict(ch=64, layers=5, drop=0.1)),
              (PVTransformer, "Transformer", dict(d=64, nhead=4, layers=4, drop=0.1)))
    for cls, name, kw in models:
        model = cls(**kw).to(device)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n[{name}] 参数量 {n_par / 1e3:.0f}k")
        hist, best_val = train_one(model, tr_loader, va_x, va_y, args.epochs, args.lr, device, name)
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(X[te]).to(device)).cpu().numpy() * SCALE
        preds[name], hists[name] = pred, hist
        results[name] = {"params": n_par, "val_MAE_kW": best_val}
        torch.save(model.state_dict(), OUT / f"model_{name}.pt")

    # ---- 测试集对比 ----
    act = data["pv_act"][days[te]]
    fc0 = np.stack([_q2._hour_to_slots(data["fc0"][int(d)]) for d in days[te]])
    ext = _qp.load_extended()
    hist = np.stack([_qp.hist_forecast(ext, int(d)) for d in days[te]])
    dl = 0.5 * (preds["TCN"] + preds["Transformer"])

    print("\n===== 测试集（65 天）预测误差 =====")
    rows = {"官方0:00": fc0, "历史口E": hist, "TCN": preds["TCN"],
            "Transformer": preds["Transformer"], "DL均值": dl}
    for name, p in rows.items():
        m = metrics(p, act)
        print(f"  {name:<12} 全天 MAE {m['MAE_kW']:7.1f}  光照 {m['MAE_光照样本_kW']:7.1f}  "
              f"正午 {m['MAE_正午_kW']:7.1f}  RMSE {m['RMSE_kW']:7.1f} kW")
        results.setdefault("test", {})[name] = m

    # ---- 无前视组合：前 30 天拟合三源权重，后 35 天评估 ----
    k = 30
    best = None
    for w1 in np.arange(0, 1.0001, 0.05):
        for w2 in np.arange(0, 1.0001 - w1 + 1e-9, 0.05):
            w3 = 1.0 - w1 - w2
            mae = np.abs(w1 * fc0[:k] + w2 * hist[:k] + w3 * dl[:k] - act[:k]).mean()
            if best is None or mae < best[0]:
                best = (mae, w1, w2, w3)
    _, w1, w2, w3 = best
    combo = w1 * fc0 + w2 * hist + w3 * dl
    m_combo = metrics(combo[k:], act[k:])
    m_off = metrics(fc0[k:], act[k:])
    print(f"\n三源组合（fit 前30天→评估后35天）：w=({w1:.2f} 官方, {w2:.2f} 历史, {w3:.2f} DL)")
    print(f"  官方  全天 MAE {m_off['MAE_kW']:.1f} kW → 组合 {m_combo['MAE_kW']:.1f} kW "
          f"（{100 * (m_combo['MAE_kW'] / m_off['MAE_kW'] - 1):+.1f}%）")
    results["combo"] = {"w_official": float(w1), "w_hist": float(w2), "w_dl": float(w3),
                        "MAE_official": m_off["MAE_kW"], "MAE_combo": m_combo["MAE_kW"]}

    # 月度分解（测试段）
    months = np.array([int(data["dates"][int(d)][5:7]) for d in days[te]])
    print("\n测试段月度 MAE / kW（全天）:")
    print("  月份   官方0:00   历史E    DL均值")
    for mo in sorted(set(months.tolist())):
        mm = months == mo
        print(f"  {mo:>4}   {np.abs(fc0[mm] - act[mm]).mean():8.1f}  "
              f"{np.abs(hist[mm] - act[mm]).mean():7.1f}  {np.abs(dl[mm] - act[mm]).mean():7.1f}")
    results["monthly"] = {int(mo): {
        "official": float(np.abs(fc0[months == mo] - act[months == mo]).mean()),
        "hist": float(np.abs(hist[months == mo] - act[months == mo]).mean()),
        "dl": float(np.abs(dl[months == mo] - act[months == mo]).mean()),
    } for mo in sorted(set(months.tolist()))}

    mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f"\nCUDA 峰值显存占用 {mem:.0f} MB")
    results["device"] = str(device)
    results["cuda_peak_MB"] = mem

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in hists:
        ax.plot(np.array(hists[name]["train"]) * SCALE, lw=1.2, label=f"{name} train")
        ax.plot(np.array(hists[name]["val"]) * SCALE, lw=1.2, label=f"{name} val")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE / kW"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, p in (("官方", fc0), ("历史E", hist), ("DL均值", dl)):
        ax.plot(np.abs(p - act).mean(axis=0), lw=1.1, label=name)
    ax.set_xlabel("时段（10 分钟）"); ax.set_ylabel("MAE / kW")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mae_by_slot.png", dpi=150); plt.close(fig)

    j = int(np.argmax(act.max(axis=1)))
    d_day = days[te][j]
    fig, ax = plt.subplots(figsize=(7, 4))
    hh = (np.arange(T) + 0.5) / 6.0
    ax.plot(hh, act[j], lw=1.4, label="实际")
    ax.plot(hh, fc0[j], lw=1.0, label="官方0:00")
    ax.plot(hh, hist[j], lw=1.0, label="历史E")
    ax.plot(hh, dl[j], lw=1.0, label="DL均值")
    ax.set_xlabel("时刻 / h"); ax.set_ylabel("光伏功率 / kW")
    ax.set_title(f"测试集样例日 day={d_day}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "sample_day.png", dpi=150); plt.close(fig)

    m = act >= 500
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(act[m], dl[m], s=4, alpha=0.3)
    lim = [0, act.max() * 1.02]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("实际 / kW"); ax.set_ylabel("DL 预测 / kW")
    ax.set_title("光照样本散点（测试集）")
    fig.tight_layout(); fig.savefig(OUT / "scatter_dl.png", dpi=150); plt.close(fig)

    with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果与图已保存到 {OUT}")


if __name__ == "__main__":
    main()
