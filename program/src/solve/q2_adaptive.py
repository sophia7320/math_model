"""C 题 问题二 口径 E：历史自适应加权预测 + 网格定位 + 梯度精化。

预测（每天 0:00，只用历史实际与典型日，无前视）：
    负荷 L̂(d,t) = w1·L(d-7,t) + w2·L(d-14,t) + w3·L̄(t)
    光伏 P̂(d,t) = u1·P(d-1,t) + u2·P(d-2,t) + u3·P̄(t)
w、u 为凸权重（softmax 参数化），每天一组、144 个时段共用；典型日取附件 1。

标定（目标：窗口内实际总费用 = 计划费 + 紧急购电费）：
    1) 每个日历日 k 在权重网格（步长 grid_step）上算成本表 C[k, i, j]；
    2) 目标日 d 的权重 = 窗口 [d-W, d-1] 平均成本最小的格子（查表）；
    3) W=1 主结果再用 torch.optim.Adam + 中心差分数值梯度在该窗口上精化，
       保留更优者。成本表并行预计算并缓存。

执行沿用问题二口径（见 q2.py）：计划购电 take-or-pay、储能日循环 E=6000、
缺口按 5 倍交易时刻电价紧急购电。result2.xlsx 仍由 q2.py 负责。

用法：
    from solve.q2_adaptive import AdaptiveWeightModel

    model = AdaptiveWeightModel(W=1).load()
    result = model.build_table().run()          # 主结果（含对照/图/报告）
    model.window_sensitivity()                  # 标定窗口敏感性

命令行：uv run python -m solve.q2_adaptive   （在 program/ 目录；首次约 8 分钟）
"""

from __future__ import annotations

import hashlib
import os
import time

import numpy as np
import pandas as pd
import torch

import program as pm
from solve import q2
from solve.common import DATA_C, E0, RESULTS_DIR, ROOT, T

N_DAY = q2.N_DAY  # 全年天数（365）
REPORT_START = q2.REPORT_START  # 报送起始日 2025-02-01（0 基 31）

_WORKER: "AdaptiveWeightModel | None" = None  # 子进程中的模型副本


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def softmax(theta) -> np.ndarray:
    """logit 向量 → 凸权重（非负、和=1）。"""
    e = np.exp(theta - theta.max())
    return e / e.sum()


def simplex_grid(step: float) -> np.ndarray:
    """3 维凸权重网格（非负、和=1），组数 (n+1)(n+2)/2。"""
    n = int(round(1.0 / step))
    return np.array(
        [
            (i / n, j / n, (n - i - j) / n)
            for i in range(n + 1)
            for j in range(n + 1 - i)
        ]
    )


def total_of(df: pd.DataFrame) -> float:
    """逐日表的年度总费用（计划 + 紧急）。"""
    return float(df["计划购电费/元"].sum() + df["紧急购电费/元"].sum())


def record(section: str, data, note: str = "") -> None:
    """写结果报告：先删除同名旧章节再追加（pm.record_result 为追加模式）。"""
    path = pm.reports_dir() / "RESULTS_REPORT.md"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        marker = f"### {section}"
        pos = text.find(marker)
        if pos != -1:
            end = text.find("\n### ", pos + len(marker))
            text = text[:pos] if end == -1 else text[:pos] + text[end + 1 :]
            path.write_text(text, encoding="utf-8")
    pm.record_result(section, data, note=note)


def _init_worker(model: "AdaptiveWeightModel") -> None:
    global _WORKER
    _WORKER = model


def _worker_day(d: int):
    return _WORKER._one_day(d)


def _verify_result2(path, df: pd.DataFrame) -> dict:
    """回读 result2.xlsx 并与逐日表勾稽（返回校验指标 dict）。"""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    checks = {"sheet 名": "/".join(wb.sheetnames)}

    ws = wb["计划购电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    checks["计划表行数"] = len(rows)
    checks["全天购电量勾稽差/kWh"] = float(abs(
        sum(float(r[145]) for r in rows) - df["计划购电量/kWh"].sum()))
    checks["全天购电费勾稽差/元"] = float(abs(
        sum(float(r[146]) for r in rows) - df["计划购电费/元"].sum()))

    ws = wb["充放电量"]
    rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[1] is not None]
    checks["充放表行数"] = len(rows)
    e0 = [float(rows[6 * i][5]) for i in range(len(rows) // 6)]
    e24 = [float(rows[6 * i + 1][5]) for i in range(len(rows) // 6)]
    checks["跨日SOC衔接最大误差/kWh"] = (
        float(max(abs(e24[i] - e0[i + 1]) for i in range(len(e0) - 1)))
        if len(e0) > 1 else 0.0)
    checks["日末SOC非固定取值数"] = len(set(round(v, 3) for v in e24))

    ws = wb["紧急购电量"]
    kw = sum(float(r[2]) for r in ws.iter_rows(min_row=2, values_only=True)
             if r[2] is not None)
    checks["紧急表电量勾稽差/kWh"] = float(abs(kw - df["紧急购电量/kWh"].sum()))
    wb.close()
    return checks


# ---------------------------------------------------------------------------
# 口径 E 模型
# ---------------------------------------------------------------------------
class AdaptiveWeightModel:
    """问题二口径 E：三源自适应加权预测 + 网格/梯度标定 + 全年滚动模拟。

    Parameters
    ----------
    W : int
        标定窗口（天）：目标日 d 用 [d-W, d-1] 的实际总费用标定。主结果 1。
    grid_step : float
        权重网格步长（0.2 → 21×21 = 441 组/天）。
    steps, lr : int, float
        Adam 精化步数与学习率。
    eps_grad : float
        中心差分步长（θ 空间）。
    workers : int | None
        并行进程数，默认 min(8, CPU 核数)。
    """

    START = 14  # 最早可用于标定的日历日：需要 d-14 的历史负荷
    LAG_L = (7, 14)  # 负荷来源滞后（同星期几，周周期强）
    LAG_P = (1, 2)  # 光伏来源滞后（短期持续性）
    TYP_L = "小区负载"  # 附件 1 典型日负荷列名
    TYP_P = "光伏发电预测功率"  # 附件 1 典型日光伏列名
    EPS_PLAN = 1e-3  # 计划 LP 正则项（唯一化最优方案）

    def __init__(
        self,
        W: int = 1,
        grid_step: float = 0.2,
        steps: int = 10,
        lr: float = 0.05,
        eps_grad: float = 0.1,
        workers: int | None = None,
    ):
        self.W = W
        self.grid_step = grid_step
        self.steps = steps
        self.lr = lr
        self.eps_grad = eps_grad
        self.workers = workers or min(8, os.cpu_count() or 1)
        self.grid = simplex_grid(grid_step)

        self.dates = self.L = self.P = self.price = None
        self.L_typ = self.P_typ = None
        self.data = None
        self.C = self.TH = self.imp = None

    # ------------------------------------------------------------------
    # 数据
    # ------------------------------------------------------------------
    def load(self, data: dict | None = None) -> "AdaptiveWeightModel":
        """读取附件 2 实际序列 + 附件 1 典型日（负荷/光伏，kW）。"""
        if data is None:
            data = q2.load_all()
            a1 = pm.read_table(DATA_C / "附件1.xlsx")
            data["load_typ"] = a1[self.TYP_L].to_numpy(float)
            data["pv_typ"] = a1[self.TYP_P].to_numpy(float)
        self.dates = data["dates"]
        self.L = data["load"]
        self.P = data["pv_act"]
        self.price = data["price"]
        self.L_typ = data["load_typ"]
        self.P_typ = data["pv_typ"]
        self.data = data
        return self

    # ------------------------------------------------------------------
    # 预测与单日成本
    # ------------------------------------------------------------------
    def forecast(self, w, u, d: int):
        """第 d 天预测（kWh/时段）：三源凸组合，裁剪非负。"""
        l_kw = (
            w[0] * self.L[d - self.LAG_L[0]]
            + w[1] * self.L[d - self.LAG_L[1]]
            + w[2] * self.L_typ
        )
        p_kw = (
            u[0] * self.P[d - self.LAG_P[0]]
            + u[1] * self.P[d - self.LAG_P[1]]
            + u[2] * self.P_typ
        )
        return np.clip(l_kw, 0.0, None) / 6.0, np.clip(p_kw, 0.0, None) / 6.0

    def day_cost(self, d: int, w, u) -> float:
        """第 d 天实际总费用：按预测制定计划（take-or-pay）+ 按实际执行 + 紧急购电。

        执行口径与正式统一口径一致（``consistency.EXEC_POLICY="free"``）：
        无段末硬目标，仅容量/功率约束。
        """
        l_kwh, p_kwh = self.forecast(w, u, d)
        x, _E, _ = q2.plan_day(self.price, l_kwh, p_kwh, E0, eps=self.EPS_PLAN)
        ex = q2.exec_segment_causal(
            self.L[d] / 6.0, self.P[d] / 6.0, x, E0, None
        )
        return float(self.price @ x + q2.EMERG_MULT * (self.price @ ex["e"]))

    def window_days(self, d: int, W: int | None = None):
        """目标日 d 的标定窗口 [d-W, d-1]（截断到最早可用日 START）。"""
        W = self.W if W is None else W
        return list(range(max(self.START, d - W), d))

    # ------------------------------------------------------------------
    # 梯度精化（softmax + 数值梯度 + torch.optim.Adam）
    # ------------------------------------------------------------------
    def window_cost(self, theta, days) -> float:
        """窗口平均实际总费用（标定目标；θ 为 6 维：负荷 3 + 光伏 3）。"""
        w, u = softmax(theta[:3]), softmax(theta[3:])
        return float(np.mean([self.day_cost(d, w, u) for d in days]))

    def calibrate(self, d: int, theta, W: int | None = None) -> np.ndarray:
        """在窗口 [d-W, d-1] 上做 steps 步 Adam（中心差分梯度），返回 6 维 θ。"""
        days = self.window_days(d, W)
        th = torch.tensor(theta, dtype=torch.float64)
        opt = torch.optim.Adam([th], lr=self.lr)
        for _ in range(self.steps):
            base = th.detach().numpy()
            grad = np.empty(6)
            for i in range(6):
                plus, minus = base.copy(), base.copy()
                plus[i] += self.eps_grad
                minus[i] -= self.eps_grad
                grad[i] = (
                    self.window_cost(plus, days) - self.window_cost(minus, days)
                ) / (2 * self.eps_grad)
            th.grad = torch.from_numpy(grad)
            opt.step()
            opt.zero_grad()
        return th.detach().numpy()

    # ------------------------------------------------------------------
    # 成本表（并行预计算 + 缓存）
    # ------------------------------------------------------------------
    def _one_day(self, d: int):
        """日历日 d：网格成本行 + 从网格最优出发的 Adam 精化权重（供次日使用）。"""
        n = len(self.grid)
        row = np.empty((n, n))
        best = (np.inf, 0, 0)
        for i, w in enumerate(self.grid):
            for j, u in enumerate(self.grid):
                c = self.day_cost(d, w, u)
                row[i, j] = c
                if c < best[0]:
                    best = (c, i, j)
        theta0 = np.concatenate(
            [np.log(self.grid[best[1]] + 1e-9), np.log(self.grid[best[2]] + 1e-9)]
        )
        if d + 1 < N_DAY:  # 精化“用于次日”的权重：目标日 d+1，标定在日 d
            theta = self.calibrate(d + 1, theta0, W=1)
            w, u = softmax(theta[:3]), softmax(theta[3:])
            polished = self.day_cost(d, w, u)
            if polished < best[0]:
                return d, row, theta, float(best[0]), float(polished)
        return d, row, theta0, float(best[0]), float(best[0])

    def _cache_key(self) -> str:
        h = hashlib.md5()
        for arr in (self.L, self.P, self.price, self.L_typ, self.P_typ):
            h.update(np.ascontiguousarray(arr, dtype=float).tobytes())
        h.update(
            np.asarray(
                [self.grid_step, self.steps, self.lr, self.eps_grad], float
            ).tobytes()
        )
        h.update(q2.CAUSAL_POLICY_VERSION.encode("utf-8"))
        return h.hexdigest()[:12]

    def build_table(self, force: bool = False) -> AdaptiveWeightModel:
        """计算/加载成本表：C[k,i,j] 日 k 网格费用，TH[k] 日 k 精化 θ，imp 精化收益。"""
        pm.init(root=str(ROOT))
        tag = getattr(self, "CACHE_TAG", "q2e")   # 子类变体隔离缓存（如回归源模型）
        cache = pm.outputs_dir() / "cache" / tag
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"table_{self._cache_key()}.npz"
        if path.exists() and not force:
            z = np.load(path)
            self.C, self.TH, self.imp = z["C"], z["TH"], z["imp"]
            return self

        days = list(range(self.START, N_DAY))
        t0 = time.time()
        if self.workers > 1:
            from multiprocessing import Pool

            with Pool(self.workers, initializer=_init_worker, initargs=(self,)) as pool:
                results = pool.map(_worker_day, days)
        else:
            results = [self._one_day(d) for d in days]

        n = len(self.grid)
        self.C = np.zeros((N_DAY, n, n))
        self.TH = np.zeros((N_DAY, 6))
        self.imp = np.zeros(N_DAY)
        for d, row, theta, c_grid, c_pol in results:
            self.C[d], self.TH[d], self.imp[d] = row, theta, max(0.0, c_grid - c_pol)
        np.savez_compressed(path, C=self.C, TH=self.TH, imp=self.imp, tag=tag)
        print(
            f"[q2e] 成本表完成：{len(days)} 天 × {n * n} 组，用时 {time.time() - t0:.0f}s"
        )
        return self

    # ------------------------------------------------------------------
    # 权重选择
    # ------------------------------------------------------------------
    def weights(self, W: int | None = None, use_gd: bool = True):
        """目标日 d 的权重：W=1 用精化 θ，其余用窗口成本表最小格。"""
        W = self.W if W is None else W
        out = []
        for d in range(REPORT_START, N_DAY):
            if use_gd and W == 1:
                theta = self.TH[d - 1]
                out.append((softmax(theta[:3]), softmax(theta[3:])))
            else:
                lo = max(self.START, d - W)
                i, j = np.unravel_index(
                    np.argmin(self.C[lo:d].mean(axis=0)),
                    (len(self.grid), len(self.grid)),
                )
                out.append((self.grid[i].copy(), self.grid[j].copy()))
        return out

    def oracle_weights(self):
        """事后最优权重：每个目标日直接取该日成本最小的网格格（仅作下界参考）。"""
        out = []
        for d in range(REPORT_START, N_DAY):
            i, j = np.unravel_index(
                np.argmin(self.C[d]), (len(self.grid), len(self.grid))
            )
            out.append((self.grid[i].copy(), self.grid[j].copy()))
        return out

    def const_weights(self, w, u):
        """固定权重序列（对照基线）。"""
        return [(np.asarray(w, float), np.asarray(u, float))] * (N_DAY - REPORT_START)

    # ------------------------------------------------------------------
    # 全年模拟
    # ------------------------------------------------------------------
    def simulate(self, weights, detail: bool = False):
        """全年滚动：计划（按预测）→ 执行（按实际）→ 紧急购电。

        detail=False 返回逐日表；detail=True 额外返回 (x_plans, recs)，
        供官方 result2 填表（recs[d] 含 c/d/e/E/E_start/E_end）。
        """
        rows = []
        x_plans = np.zeros((N_DAY, T)) if detail else None
        recs = [None] * N_DAY if detail else None
        for i, d in enumerate(range(REPORT_START, N_DAY)):
            w, u = weights[i]
            l_kwh, p_kwh = self.forecast(w, u, d)
            x, _E, _ = q2.plan_day(self.price, l_kwh, p_kwh, E0, eps=self.EPS_PLAN)
            ex = q2.exec_day_causal(
                self.price, self.L[d] / 6.0, self.P[d] / 6.0, x, E0
            )
            if detail:
                x_plans[d] = x
                recs[d] = {"c": ex["c"], "d": ex["d"], "s": ex["s"], "e": ex["e"],
                           "E": ex["E"], "E_start": E0, "E_end": float(ex["E"][-1])}
            residual = (
                self.P[d] / 6.0
                + x
                + ex["d"]
                + ex["e"]
                - self.L[d] / 6.0
                - ex["c"]
                - ex["s"]
            )
            rows.append(
                {
                    "日期": self.dates[d],
                    "w1_L(d-7)": w[0],
                    "w2_L(d-14)": w[1],
                    "w3_典型日": w[2],
                    "u1_P(d-1)": u[0],
                    "u2_P(d-2)": u[1],
                    "u3_典型日": u[2],
                    "计划购电量/kWh": float(x.sum()),
                    "计划购电费/元": float(self.price @ x),
                    "紧急购电量/kWh": float(ex["e"].sum()),
                    "紧急购电费/元": float(q2.EMERG_MULT * (self.price @ ex["e"])),
                    "平衡残差/kWh": float(np.abs(residual).max()),
                    "储电量最小/kWh": float(ex["E"].min()),
                    "储电量最大/kWh": float(ex["E"].max()),
                }
            )
        df = pd.DataFrame(rows)
        return (df, x_plans, recs) if detail else df

    def write_result2(self, W: int = 7, out_name: str = "result2.xlsx"):
        """用口径 E（默认 W=7）生成官方 ``results/result2.xlsx``（三张表）。

        写出前把现有 result2 备份到 ``code/outputs/result2_D_backup.xlsx``，
        写出后回读校验并记录到 RESULTS_REPORT.md。
        """
        import shutil

        pm.init(root=str(ROOT))
        log = pm.get_logger("q2e")
        if self.C is None:
            self.build_table()
        df, x_plans, recs = self.simulate(self.weights(W=W, use_gd=False), detail=True)

        plan = float(df["计划购电费/元"].sum())
        emerg = float(df["紧急购电费/元"].sum())
        cur = RESULTS_DIR / out_name
        if cur.exists():
            backup = pm.outputs_dir() / "result2_D_backup.xlsx"
            shutil.copy2(cur, backup)
            log.info("旧 result2 已备份 -> {}", backup)
        out = q2.write_result2(self.dates, self.price, x_plans, recs, out_name=out_name)
        checks = _verify_result2(out, df)
        pm.save_outputs(df, f"q2e_w{W}_daily")
        record(
            f"问题二 口径E（W={W}）官方 result2 结果与校验",
            {
                "计划购电费/万元": round(plan / 1e4, 1),
                "紧急购电费/万元": round(emerg / 1e4, 1),
                "总费用/万元": round((plan + emerg) / 1e4, 1),
                "天数": len(df),
                **checks,
            },
            note=(
                f"官方 results/{out_name} 由口径 E（W={W}，7 天窗口成本表 argmin）生成；"
                "旧口径 D 版备份在 code/outputs/result2_D_backup.xlsx。"
                "复现：uv run python -m solve.q2_adaptive --result2。"
            ),
        )
        log.info("result2.xlsx（口径 E，W={}）写出完成：{}，总费用 {:.1f} 万元",
                 W, out, (plan + emerg) / 1e4)
        return out

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def run(self, force_table: bool = False) -> dict:
        """口径 E 主流程：成本表 → 权重 → 模拟 → 对照 → 校验 → 出图 → 记录。"""
        import matplotlib.pyplot as plt

        pm.init(root=str(ROOT))
        log = pm.get_logger("q2e")
        if self.C is None:
            self.build_table(force=force_table)
        W = self.W

        df_e = self.simulate(self.weights(W=W, use_gd=True))  # 主结果
        df_g = self.simulate(self.weights(W=1, use_gd=False))  # 仅网格
        df_b = self.simulate(self.const_weights([0, 0, 1], [0, 0, 1]))  # 典型日
        df_q = self.simulate(self.const_weights([1 / 3] * 3, [1 / 3] * 3))  # 等权
        _x_d, recs_d = q2.run_deterministic(self._as_data())  # 口径 D
        rep = list(range(REPORT_START, N_DAY))
        df_d = pd.DataFrame(
            {
                "日期": [self.dates[d] for d in rep],
                "计划购电费/元": [recs_d[d]["plan_cost"] for d in rep],
                "紧急购电量/kWh": [recs_d[d]["emerg_kwh"] for d in rep],
                "紧急购电费/元": [recs_d[d]["emerg_cost"] for d in rep],
            }
        )
        df_o = self.simulate(self.oracle_weights())  # 事后最优下界

        # ---- 对照表 ----
        names = [
            "典型日（B）",
            "等权组合",
            "网格选择（W=1）",
            f"网格+梯度（E, W={W}）",
            "0:00 预报（D）",
            "事后最优（下界）",
        ]
        dfs = [df_b, df_q, df_g, df_e, df_d, df_o]
        table = pd.DataFrame(
            {
                "口径": names,
                "计划购电费/万元": [
                    round(d["计划购电费/元"].sum() / 1e4, 1) for d in dfs
                ],
                "紧急购电费/万元": [
                    round(d["紧急购电费/元"].sum() / 1e4, 1) for d in dfs
                ],
            }
        )
        table["总费用/万元"] = (
            table["计划购电费/万元"] + table["紧急购电费/万元"]
        ).round(1)
        record(
            "问题二 口径E 结果对照（2025-02-01 ~ 12-31）",
            table,
            note=(
                "预测式：负荷 w1·L(d-7)+w2·L(d-14)+w3·典型日；光伏 u1·P(d-1)+u2·P(d-2)+u3·典型日。"
                f"权重每天在窗口 W={W} 上标定后用于次日：先步长 {self.grid_step} 网格全局定位，"
                f"再做 {self.steps} 步 Adam（中心差分）精化并保留更优者。"
                "对照：典型日、等权、纯网格、附件 3 的 0:00 预报（口径 D）、事后最优（网格下界）。"
                "执行口径与问题二一致。"
            ),
        )
        pm.save_outputs(table, "q2e_compare")
        pm.save_outputs(df_e, "q2e_daily")

        # ---- 一致性校验 ----
        w_cols = ["w1_L(d-7)", "w2_L(d-14)", "w3_典型日"]
        u_cols = ["u1_P(d-1)", "u2_P(d-2)", "u3_典型日"]
        n_imp = int((self.imp > 1e-6).sum())
        record(
            "问题二 口径E 一致性与梯度精化校验",
            {
                "平衡最大残差/kWh": float(df_e["平衡残差/kWh"].max()),
                "负荷权重和最小值": float(df_e[w_cols].sum(axis=1).min()),
                "负荷权重和最大值": float(df_e[w_cols].sum(axis=1).max()),
                "光伏权重和最小值": float(df_e[u_cols].sum(axis=1).min()),
                "光伏权重和最大值": float(df_e[u_cols].sum(axis=1).max()),
                "储电量最小值/kWh": float(df_e["储电量最小/kWh"].min()),
                "储电量最大值/kWh": float(df_e["储电量最大/kWh"].max()),
                "发生紧急购电天数": int((df_e["紧急购电量/kWh"] > 1e-3).sum()),
                "梯度精化改善天数": n_imp,
                "梯度精化日均改善/元": float(self.imp[self.imp > 0].mean())
                if n_imp
                else 0.0,
                "梯度精化年改善/元": float(self.imp[REPORT_START:].sum()),
            },
            note=(
                "权重由 softmax 生成、和恒为 1；平衡残差为数值误差量级即通过。"
                f"梯度精化相对网格最优的改善：{n_imp} 天，"
                f"年累计 {self.imp[REPORT_START:].sum():.0f} 元。"
            ),
        )

        # ---- 图：权重演化 ----
        dates = df_e["日期"].tolist()
        x = np.arange(len(dates))
        month_starts = [i for i, s in enumerate(dates) if s.endswith("-01")]
        month_labels = [dates[i][5:7] + "月" for i in month_starts]

        fig, axes = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
        groups = [
            (axes[0], w_cols, "负荷权重", ["d-7", "d-14", "典型日"]),
            (axes[1], u_cols, "光伏权重", ["d-1", "d-2", "典型日"]),
        ]
        for ax, cols, ylabel, labels in groups:
            ax.stackplot(
                x, [df_e[c].to_numpy() for c in cols], labels=labels, alpha=0.9
            )
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper center", ncol=3, fontsize=8, framealpha=0.9)
        axes[1].set_xticks(month_starts)
        axes[1].set_xticklabels(month_labels)
        axes[1].set_xlabel("日期")
        pm.save_fig(fig, "Q2E_权重演化", data=df_e[["日期"] + w_cols + u_cols])

        # ---- 图：费用对照 ----
        pm.bar_group(
            names[:5],
            {
                "计划购电费": table["计划购电费/万元"].tolist()[:5],
                "紧急购电费": table["紧急购电费/万元"].tolist()[:5],
            },
            ylabel="年度费用 / 万元",
            rot=10,
            save="Q2E_费用对比",
        )

        # ---- 图：逐日紧急购电量 ----
        fig3, ax3 = pm.line(
            x,
            [df_e["紧急购电量/kWh"].to_numpy(), df_d["紧急购电量/kWh"].to_numpy()],
            labels=["口径E（网格+梯度）", "口径D（0:00预报）"],
            xlabel="日期",
            ylabel="紧急购电量 / kWh",
        )
        ax3.set_xticks(month_starts)
        ax3.set_xticklabels(month_labels)
        ax3.legend()
        pm.save_fig(
            fig3,
            "Q2E_紧急购电对比",
            data=pd.DataFrame(
                {
                    "日期": dates,
                    "口径E_紧急购电量_kWh": df_e["紧急购电量/kWh"],
                    "口径D_紧急购电量_kWh": df_d["紧急购电量/kWh"],
                }
            ),
        )

        log.info(
            "口径 E（W={}）总费用 {:.1f} 万元（计划 {:.1f} + 紧急 {:.1f}）",
            W,
            total_of(df_e) / 1e4,
            df_e["计划购电费/元"].sum() / 1e4,
            df_e["紧急购电费/元"].sum() / 1e4,
        )
        return {
            "daily": df_e,
            "compare": table,
            "C": self.C,
            "TH": self.TH,
            "imp": self.imp,
        }

    def _as_data(self) -> dict:
        """还原 q2 所需的完整 data 字典（供口径 D 对照）。"""
        return self.data

    # ------------------------------------------------------------------
    # 窗口敏感性
    # ------------------------------------------------------------------
    def window_sensitivity(
        self, W_list=(1, 3, 7, 14, 30), force_table=False
    ) -> pd.DataFrame:
        """标定窗口 W 敏感性：查成本表选择权重后全年模拟。"""
        pm.init(root=str(ROOT))
        log = pm.get_logger("q2e")
        if self.C is None:
            self.build_table(force=force_table)
        rows = []
        for W in W_list:
            t0 = time.time()
            df = self.simulate(self.weights(W=W, use_gd=False))
            rows.append(
                {
                    "窗口W/天": W,
                    "计划购电费/万元": round(float(df["计划购电费/元"].sum()) / 1e4, 1),
                    "紧急购电费/万元": round(float(df["紧急购电费/元"].sum()) / 1e4, 1),
                    "总费用/万元": round(total_of(df) / 1e4, 1),
                }
            )
            log.info(
                "W={} 总费用 {:.1f} 万元（{:.0f}s）",
                W,
                rows[-1]["总费用/万元"],
                time.time() - t0,
            )
        out = pd.DataFrame(rows)
        pm.save_outputs(out, "Q2E_窗口敏感性")
        record(
            "问题二 口径E 标定窗口敏感性",
            out,
            note="主结果 W=1（含梯度精化）；本表为纯网格选择，W 越大权重越平滑。",
        )
        pm.line(
            out["窗口W/天"],
            out["总费用/万元"],
            xlabel="标定窗口 W / 天",
            ylabel="年度总费用 / 万元",
            marker="o",
            save="Q2E_窗口敏感性",
        )
        return out


if __name__ == "__main__":
    import sys

    _model = AdaptiveWeightModel().load().build_table()
    if "--result2" in sys.argv:
        _model.write_result2()
    else:
        _model.run()
