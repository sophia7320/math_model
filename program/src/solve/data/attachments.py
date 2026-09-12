"""附件数据装载：附件 1（典型日）、附件 2（实际曲线）、附件 3（预报）、附件 4（电价）。

# ===========================================================================
# 关键派生量
#   1) 时段对齐：一天 144 个 10 分钟槽；附件中的"时间标签"为时段右端点，
#      第 k 个整点（k 时）对应槽序号 6k−1（即 6k−1 号槽末）。
#   2) 典型日矩阵 T[k, m]：第 m 月、第 k 个整点在实际曲线上的均值，
#          T[k, m] = mean_{d ∈ 月 m} A[d, 6k−1],  k = 1..24
#   3) 误差块 Z[d, k]：0:00 预报相对期望出力的标准化误差
#          z_dk = [ (F_dk − A_dk)/T[k,m_d] − μ(k) ] / σ(k)   （T > 100 kW 的槽）
#   4) 口径 E 历史权重 TH：由 code/outputs/cache/q2e/table_*.npz 提供，
#      hist_forecast 使用 TH[D−1] 生成第 D 天光伏预测（无前视）。
# ===========================================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import program as pm
from solve.common import DATA_C, N_DAY, ROOT
from solve.models.errors import mu, sigma
from solve.models.price import BETA
from solve.models.weights import make_smooth_u


def load_all() -> dict:
    """读取附件 1/2/3，返回数组与辅助量。"""
    # ---- 附件 2：实际负荷与光伏（365 天 × 144 槽）----
    sheets = pm.read_sheets(DATA_C / "附件2.xlsx")
    ldf = sheets["小区负载"]
    pdf = sheets["光伏发电实际功率"]
    dates = pd.to_datetime(ldf.iloc[:, 0]).dt.strftime("%Y-%m-%d").tolist()
    load = ldf.iloc[:, 1:145].to_numpy(float)
    pv_act = pdf.iloc[:, 1:145].to_numpy(float)

    # ---- 附件 1：典型日电价（逐槽）----
    a1 = pm.read_table(DATA_C / "附件1.xlsx")
    price = a1["电价"].to_numpy(float)

    # ---- 附件 3：0:00 发布的 24 点光伏预报，按日期块归并到天 ----
    fc3 = pm.read_table(DATA_C / "附件3.xlsx")
    fc0 = np.zeros((N_DAY, 24))
    day = -1
    for _, r in fc3.iterrows():
        cell = r["日期"]
        if isinstance(cell, str) and cell.strip():
            day += 1
        if int(str(r["预报时刻"]).split(":")[0]) == 0:
            fc0[day] = r.iloc[2:26].astype(float).to_numpy()

    # ---- 月份标签与 (整点, 月) 典型出力矩阵 T[k, m] ----
    months = np.array([int(x[5:7]) for x in dates])
    typ_hm = np.zeros((25, 13))
    for k in range(1, 25):
        col = pv_act[:, 6 * k - 1]
        for m in range(1, 13):
            typ_hm[k, m] = col[months == m].mean()

    # ---- 标准化误差日块 Z[d, k]（0:00 预报，k=1..24） ----
    Z = np.zeros((N_DAY, 24))
    ks = np.arange(1, 25)
    for i in range(N_DAY):
        m = months[i]
        typ = typ_hm[1:25, m]
        err = fc0[i] - pv_act[i, 6 * ks - 1]
        mask = typ > 100
        zz = np.zeros(24)
        zz[mask] = (err[mask] / typ[mask] - mu(ks[mask])) / sigma(ks[mask])
        Z[i] = zz

    return {
        "dates": dates, "load": load, "pv_act": pv_act, "price": price,
        "fc0": fc0, "months": months, "typ_hm": typ_hm, "Z": Z,
    }


def load_extended() -> dict:
    """q2 基础数据 + 6/12/18 预报 + 典型日光伏 + Q2E 历史权重。

    Q2E 权重表选最新且 tag == "q2e" 的缓存（变体表在其它 tag 目录，避免误用）。
    """
    data = load_all()

    # ---- 附件 3：6/12/18 发布的预报（与 0:00 相同解析口径）----
    fc3 = pm.read_table(DATA_C / "附件3.xlsx")
    fc = {h: np.zeros((N_DAY, 24)) for h in (6, 12, 18)}
    day = -1
    for _, r in fc3.iterrows():
        cell = r["日期"]
        if isinstance(cell, str) and cell.strip():
            day += 1
        h = int(str(r["预报时刻"]).split(":")[0])
        if h in fc:
            fc[h][day] = r.iloc[2:26].astype(float).to_numpy()
    data["fc6"], data["fc12"], data["fc18"] = fc[6], fc[12], fc[18]

    # ---- 附件 1：典型日光伏（列名兼容两种模板）----
    a1 = pm.read_table(DATA_C / "附件1.xlsx")
    pv_col = "光伏发电预测功率" if "光伏发电预测功率" in a1.columns else a1.columns[3]
    data["pv_typ"] = a1[pv_col].to_numpy(float)

    # ---- 口径 E 历史权重 TH（取最新 q2e 表；无 tag 兼容旧表）----
    cache_dir = ROOT / "code" / "outputs" / "cache" / "q2e"
    files = sorted(cache_dir.glob("table_*.npz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    npz_path = None
    for p in files:
        try:
            with np.load(p) as z:
                tag = str(z["tag"]) if "tag" in z.files else "q2e"
        except Exception:
            continue
        if tag == "q2e":
            npz_path = p
            break
    if npz_path is None:
        npz_path = files[0]
    data["TH"] = np.load(npz_path)["TH"]
    return data


def load_q4_data() -> tuple[dict, np.ndarray, np.ndarray]:
    """附件 1/2/3 + 附件 4 电价 + 平滑历史权重。

    返回 (data, p4, p_typ)：p4 为附件 4 逐槽实时电价，p_typ 为典型日电价。
    """
    data = load_extended()
    # 光伏历史权重参数平滑 β=0.1（Q3/Q4 主方案口径）
    data["U_SMOOTH"] = make_smooth_u(data, BETA)
    df4 = pm.read_table(DATA_C / "附件4.xlsx")
    p4 = df4.iloc[:, 1:145].to_numpy(float)
    return data, p4, data["price"]
