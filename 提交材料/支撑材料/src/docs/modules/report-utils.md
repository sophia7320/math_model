# 报告与工具（report · utils · config）

工具链的「后勤模块」：把结果写进报告、把图和数据落盘、把过程变成可复现的脚本。

## report：一切结果进 `RESULTS_REPORT.md`

论文阶段的铁律——**论文里出现的每个数字都必须能从报告中找到**。
`record_result` 就是写报告的标准动作：

```python
import program as pm

# 指标字典 → 自动生成两列表格
pm.record_result(
    "问题一结果",
    {"最大利润": 36.0, "产品1产量": 2.0, "产品2产量": 6.0},
    note="求解器：HiGHS，全部约束满足，最大违反量 0。",
)

# DataFrame → 自动转 Markdown 表
pm.record_result("模型对比", compare_table)

# 也可以写整段正文
pm.record("## 灵敏度分析\n\n产量上限 ±10% 时目标值变化如下……")
```

写入的内容长这样（`reports/RESULTS_REPORT.md`）：

````markdown
### 问题一结果

求解器：HiGHS，全部约束满足，最大违反量 0。

| 指标 | 数值 |
| --- | --- |
| 最大利润 | 36 |
| 产品1产量 | 2 |
| 产品2产量 | 6 |
````

**其他报告工具**：

```python
pm.record_environment()                       # 写入运行环境（各库版本）
pm.save_outputs(table, "q1_sensitivity")      # 结果落盘 code/outputs/q1_sensitivity.csv
tex = pm.to_latex(table, caption="问题一结果", label="tab:q1")   # 论文 LaTeX 表格
```

## utils：种子、计时、缓存、格式化

```python
import program as pm

pm.set_seed(42)                               # random / numpy / torch 全部固定

with pm.Timer("求解"):
    result = heavy_solve()                    # 自动打印耗时

cache = pm.DataCache("q1")                    # 耗时计算只跑一次
params = cache.get_or_compute("params", expensive_fit, x, y)

print(pm.fmt(0.000012345))                    # '1.234e-05'（论文友好）
print(pm.markdown_table({"指标": ["R²"], "值": [0.9614]}))
```

日志（重要步骤留痕，便于排查与复现）：

```python
log = pm.get_logger("q1", log_file="code/outputs/q1.log")
log.info("开始求解，参数 = {}", params)
```

## config：初始化与路径

```python
pm.init(seed=42)              # 常用默认；backend="Agg" 无窗口稳定出图
pm.init(root="..")            # 输出目录改到工作区根
pm.init(seed=None)            # 不设种子（仅特殊场景）

pm.figures_dir()              # .../figures
pm.reports_dir()              # .../reports
pm.outputs_dir()              # .../code/outputs
pm.project_root()             # 当前项目根
```

`pm.init()` 返回实际生效的配置，可打印确认：

```python
{'root': 'C:\\...\\math_model',
 'font': 'Microsoft YaHei',
 'backend': 'agg',
 'seed': 42}
```

## 推荐的工作节奏

每个子问题按下面 5 步走，这同时也是 `3coding-visual` 阶段对代码的要求：

1. `pm.read_table` 读数据 → `pm.summarize` 概览进报告；
2. 建模求解（optimize / ml / stats / ode…），随机过程设种子；
3. 约束与结果校验 → `pm.record_result("约束校验", {...})`；
4. 出图（≥2 张）→ `save=` 自动进 `figures/`；
5. `pm.record_result("问题X结果", ...)` 把关键数值写进报告。

!!! warning "不要踩的线"
    - 论文里引用 **未记录过** 的数字（编辑后程序改了参数，报告没更新）；
    - 只在训练集上报指标；优化结果不做可行性检验；
    - 图表文件名随意起（推荐 `q1_xxx`、`q2_xxx`，方便论文引用与验收）。
