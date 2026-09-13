"""项目级模型一致性规范：**唯一参数源**与统一构造（2026-09-13 定稿）。

本模块是整个项目口径参数的唯一定义处；Q1–Q4 的求解脚本必须从这里 import，
不得在各自模块中另写常量（由 ``tests/model_consistency_test.py`` 守卫）。

统一口径条款（全文见 ``reports/模型一致性规范.md``）：

1. 预测：三源自适应凸权重（负荷：d−7 / d−14 / 典型日；光伏：d−1 / d−2 / 典型日），
   权重按 EWMA 半衰期 ``EWMA_HL`` 由 Q2E 成本表逐日标定（目标日 d 只用 ≤ d−1 信息）。
2. 风险裕度：负荷抬升 ``KAPPA``、光伏折减 ``MARGIN`` kW（时间留出选定：
   2–6 月开发期选参，7–12 月冻结验证，不参与选择）。
3. 场景：``N_SCEN`` 个等概率情景；**（Δ负荷, Δ光伏）同月整日联合残差块**重采样，
   池严格取目标日之前（同月 ≥ ``SCEN_MIN_SAME_MONTH`` 天，否则回看 ``SCEN_LOOKBACK`` 天）。
4. 结构：2 日（288 槽）滚动规划、**规划窗口末端完全自由**；Q2 执行 1 天、
   Q3/Q4 在 0/6/12/18 决策、执行窗 6 h；储电量跨日连续；逐槽因果执行。
5. 起点：2025-02-01（日序号 ``START_DAY``）0:00 以 E0=6000 kWh 起步；1 月仅作预测预热。
6. 信息集：Q1 用附件 1；Q2 用附件 1/2（历史自适应预测，不用附件 3 预报）；
   Q3/Q4 可用附件 3 预报（0:00 计划用组合预测 λ·官方 + (1−λ)·历史；调整层用最新发布）；
   Q4 电价：G = 附件 4 已知（题面主口径），H = 历史预测扩展。
7. Q1 豁免：题面明确单日且要求 0:00 = 24:00，保持日循环；仅共用物理常数
   （见 ``solve.common``）。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 统一参数（唯一来源，调研与选择过程见 reports/ 专题）
# ---------------------------------------------------------------------------
EWMA_HL = 5.0            # 预测权重标定 EWMA 半衰期（天）
KAPPA = 1.02             # 负荷预测抬升系数（全局定稿：Q2 留出冠军，Q2+Q3 联合核验）
MARGIN = 50.0            # 光伏预测折减（kW）（同上）
N_SCEN = 40              # 对冲/评估等概率情景数
SCEN_MIN_SAME_MONTH = 14  # 情景池同月最少天数
SCEN_LOOKBACK = 90        # 同月不足时的回看窗口（天）
START_DAY = 31            # 2025-02-01（0 基）；仿真与报送起点
PLAN_HORIZON = 288        # 规划窗口槽数（2 日 × 144）
EXEC_HOURS_Q2 = 24        # Q2 执行窗口（小时）
EXEC_HOURS_Q3 = 6         # Q3/Q4 执行窗口（小时），决策点 0/6/12/18
EXEC_POLICY = "free"      # 执行器策略：无段末硬目标（含末段），仅容量/功率约束
#   依据（reports/执行器段末目标实验.md，末段修复后重跑）：段末目标会迫使
#   "紧急购电充能/扣留放电"——中段目标 60 天夏季 +40.3 万（12.2%）、47 天秋季
#   +8.4 万（4.3%）；末段目标再 +4.6 万（1.6%）/ +0.5 万（0.3%）；
#   合计 free 相对 target：−13.6% / −4.5%。
HOLDOUT_DEV_START = 31    # 开发期起点（2–1）
HOLDOUT_VAL_START = "2025-07-01"   # 冻结验证期起点
WARM_FROM = 14            # EWMA 预热起始日（d−14 与成本表最早可用日）

# 历史对照参数（仅用于复现旧口径，禁止进入正式结果）
LEGACY = {
    "EWMA_W7": 7.0,      # 旧官方标定窗口
    "BETA_SMOOTH": 0.1,  # 旧参数平滑（Q3 原型）
    "N_SCEN_OLD": 10,    # 旧情景数
    "KAPPA_OLD": 1.02,   # 旧调优参数
    "MARGIN_OLD": 50.0,
}


# ---------------------------------------------------------------------------
# 统一构造工具
# ---------------------------------------------------------------------------
def ewma_weights_from_table(C, grid, hl: float = EWMA_HL,
                            first_target: int = START_DAY,
                            warm_from: int = WARM_FROM,
                            n_days: int = 365) -> dict:
    """由 Q2E 成本表 ``C`` 生成逐日 EWMA 标定权重（无前视）。

    目标日 d 的记分矩阵 M = 指数衰减加权 [warm_from, d−1] 的成本表行；
    返回 ``{d: (w, u)}``（w 为负荷权重、u 为光伏权重，均凸、和=1）。
    """
    decay = float(np.exp(-np.log(2.0) / hl))
    M = None
    for k in range(warm_from, first_target - 1):
        M = C[k].copy() if M is None else decay * M + (1.0 - decay) * C[k]
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for d in range(first_target, n_days):
        M = C[d - 1].copy() if M is None else decay * M + (1.0 - decay) * C[d - 1]
        i, j = np.unravel_index(np.argmin(M), M.shape)
        out[d] = (grid[i].copy(), grid[j].copy())
    return out


def joint_residuals(load_act, pv_act, load_nom, pv_nom):
    """（Δ负荷, Δ光伏）整日联合残差块：实际 − 名义（同单位，kWh/槽）。"""
    RL = np.asarray(load_act, dtype=float) - np.asarray(load_nom, dtype=float)
    RP = np.asarray(pv_act, dtype=float) - np.asarray(pv_nom, dtype=float)
    return RL, RP


def scenario_indices(months, target_day: int,
                     pool_min_same_month: int = SCEN_MIN_SAME_MONTH,
                     lookback: int = SCEN_LOOKBACK) -> np.ndarray:
    """无前视情景池（同月优先；不足回看 ``lookback`` 天）。所有日序号 < target_day。"""
    months = np.asarray(months)
    past = np.arange(max(WARM_FROM, target_day - lookback), target_day, dtype=int)
    same = past[months[past] == months[target_day]]
    pool = same if len(same) >= pool_min_same_month else past
    if not len(pool) or int(pool.max()) >= target_day:
        raise RuntimeError(f"情景池因果性校验失败：D={target_day}, pool={pool.tolist()}")
    return pool


def kappa_load(load_kw: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """负荷名义预测：κ·L̂（裁剪非负）。"""
    return np.clip(np.asarray(load_kw, dtype=float) * kappa, 0.0, None)


def margin_pv(pv_kw: np.ndarray, margin: float = MARGIN) -> np.ndarray:
    """光伏名义预测：max(P̂ − m, 0)。"""
    return np.clip(np.asarray(pv_kw, dtype=float) - margin, 0.0, None)
