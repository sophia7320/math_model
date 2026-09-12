# -*- coding: utf-8 -*-
"""Q3 深度学习试验：把 DL 预测接入 Q3 决策，比较费用（测试段）。

口径：实时执行 + 全开调整（调整层用官方最新预报，各配置一致）。
0:00 计划的预报源对比：官方 / 历史口E / 组合 / DL / 官方+DL 混合。

运行（program/ 下）：uv run python experiments/q3_dl/eval_q3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

import numpy as np
import torch

import solve.q3_proto as qp
from common import SCALE, build_dataset, load_raw, split
from models import PVTransformer, TCNNet

OUT = _DIR / "outputs"


def load_preds(device):
    data = load_raw()
    X, Y, days = build_dataset(data)
    tr, va, te = split(days)
    preds = {}
    for cls, name, kw in ((TCNNet, "TCN", dict(ch=64, layers=5, drop=0.1)),
                          (PVTransformer, "Transformer", dict(d=64, nhead=4, layers=4, drop=0.1))):
        model = cls(**kw).to(device)
        model.load_state_dict(torch.load(OUT / f"model_{name}.pt", map_location=device))
        model.eval()
        with torch.no_grad():
            preds[name] = model(torch.from_numpy(X[te]).to(device)).cpu().numpy() * SCALE
    return data, days[te], preds


def run(data_ext, test_days, override=None, lam=1.0):
    d2 = dict(data_ext)
    if override is not None:
        d2["PV_OVERRIDE"] = {int(d): override[i] for i, d in enumerate(test_days)}
    tot = plan = adj = em = 0.0
    for d in test_days:
        r = qp.simulate_day_rt(d2, int(d), lam=lam)
        tot += r["total"]; plan += r["plan_cost"]
        adj += r["adjust_net"]; em += r["emerg"]
    return {"total": tot, "plan": plan, "adj": adj, "emerg": em}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, test_days, preds = load_preds(device)
    dl = preds["Transformer"]
    ext = qp.load_extended()

    configs = [
        ("官方 0:00（λ=1）", ext, None, 1.0),
        ("历史口E（λ=0）", ext, None, 0.0),
        ("组合 0.5（λ=0.5）", ext, None, 0.5),
        ("DL（λ=0）", ext, dl, 0.0),
        ("官方+DL（λ=0.3）", ext, dl, 0.3),
        ("官方+DL（λ=0.5）", ext, dl, 0.5),
    ]
    rows = []
    print(f"测试段 {len(test_days)} 天（{test_days[0]} ~ {test_days[-1]}），实时执行 + 全开调整：")
    print(f"  {'配置':<20}{'总费用/万元':>12}{'计划费/万元':>12}{'调整/万元':>10}{'紧急/万元':>10}")
    for name, dd, ov, lam in configs:
        r = run(dd, test_days, override=ov, lam=lam)
        rows.append({"配置": name, "总费用/万元": round(r["total"] / 1e4, 2),
                     "计划费/万元": round(r["plan"] / 1e4, 2),
                     "调整/万元": round(r["adj"] / 1e4, 2),
                     "紧急/万元": round(r["emerg"] / 1e4, 2)})
        print(f"  {name:<20}{r['total'] / 1e4:>12.2f}{r['plan'] / 1e4:>12.2f}"
              f"{r['adj'] / 1e4:>10.2f}{r['emerg'] / 1e4:>10.2f}")

    base = rows[0]["总费用/万元"]
    print("\n相对官方 0:00：")
    for r in rows[1:]:
        print(f"  {r['配置']:<20}{100 * (r['总费用/万元'] / base - 1):+.2f}%")

    with open(OUT / "q3_cost_compare.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {OUT / 'q3_cost_compare.json'}")


if __name__ == "__main__":
    main()
