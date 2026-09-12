# Q3 深度学习试验（GPU）

> 目的：用深度学习（TCN / Transformer）做光伏预测，检验它能否作为
> "官方预报 + 历史预测"组合的第三源，替代或增强现有预测层。
>
> 设备：NVIDIA RTX 5060（torch 2.14.0+cu130）
> 运行（`program/` 目录下）：
> ```powershell
> uv run python experiments/q3_dl/train.py            # 训练 + 评估（约 3 分钟）
> ```
>
> 设计（无前视）：
> - 任务：预测目标日 D 的 144 个 10 分钟槽光伏功率
> - 输入（1812 维 → 12 通道 × 144 槽）：过去 7 天实际光伏（7×144）、
>   当天 0:00 官方预报（插值 144）、小时 sin/cos、年内日 sin/cos
> - 划分：train 7~279，val 280~299，test 300~364（时间顺序）
> - 损失：MAE（归一化尺度），早停；随机种子 42

## 文件

| 文件 | 说明 |
| --- | --- |
| `common.py` | 数据加载与特征构造 |
| `models.py` | TCNNet / PVTransformer |
| `train.py` | 训练、测试集评估、与官方/历史对比、组合权重（无前视）、出图 |
| `outputs/` | 训练曲线、误差剖面、样例预测、指标 JSON |
