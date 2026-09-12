"""专题/探索脚本（保留勿删；不属于正式结果生成链）。

运行（program/ 下）：
    uv run python -m solve.experiments.q2_arima        # ARIMA 预测对照
    uv run python -m solve.experiments.q2_bias         # 有偏预测 vs 场景对冲
    uv run python -m solve.experiments.q2_bias_roll    # δ 滚动标定
    uv run python -m solve.experiments.q2_regsrc       # 回归参考量探索（已搁置）
    uv run python -m solve.experiments.q2e_smooth      # Q2E 平滑回测（负结果）
    uv run python -m solve.experiments.q2e_structure   # 误差分布结构分析
    uv run python -m solve.experiments.q3_proto_seg    # Q3 分段/规则原型
    uv run python -m solve.experiments.q3_proto_abl    # Q3 消融+分解+MC
    uv run python -m solve.experiments.q3_proto_rt     # Q3 实时对比
    uv run python -m solve.experiments.q4_price_eda    # 附件 4 电价 EDA
    uv run python -m solve.experiments.q4_price_dyn    # β 动态修正验证
说明：正式脚本仍在 solve/ 顶层（q1~q4、q2_tune、q2_adaptive、q3_sensitivity、
q3_ablation、q4_price_fit、q3_proto 门面）；本包脚本复用顶层门面与子包实现。
"""
