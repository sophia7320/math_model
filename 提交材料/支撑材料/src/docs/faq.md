# 常见问题

## 中文变方框 / 负号乱码

**原因**：没有调用 `pm.init()`。

```python
import program as pm
pm.init()          # 必须放在 import matplotlib / 绘图之前
```

如果确实不想调 `init`，绘图模块会做一次兜底字体设置，但风格参数（字号、
网格、配色）不会生效，不推荐。

## 图/报告跑到别的目录去了

输出路径 = **运行脚本时的当前工作目录**（项目根）+ 固定子目录：

```
项目根/
├── figures/          图
├── reports/          RESULTS_REPORT.md
└── code/outputs/     中间数据、图表数据
```

从别处运行时显式指定：

```python
pm.init(root=r"C:\...\math_model")     # 或环境变量 MATHMODEL_ROOT
```

## 论文里怎么引用图和表

- 图：`figures/q1_curve.pdf` → Typst `image("../../figures/q1_curve.pdf")`，
  LaTeX `\includegraphics{../../figures/q1_curve.pdf}`；
- 表：用 `pm.to_latex(df, caption="...", label="tab:...")` 直接生成；
- **所有数字**引用 `reports/RESULTS_REPORT.md` 中记录过的值，不要手打。

## 结果数怎么保证和论文一致

纪律：程序改了就重新跑，`RESULTS_REPORT.md` 是唯一数值来源。
`record_result` 每条都带方法说明，写作阶段（5writing）只从这里取数。

## 换台电脑/重装环境能复现吗

能，前提是：

1. 依赖都在 `pyproject.toml` / `uv.lock` 中（用了新包记得 `uv add` 并提交）；
2. 随机种子统一由 `pm.init(seed=...)` 或函数 `seed=` 参数控制；
3. 运行方式固定：`uv run --project program python program/src/program/xxx.py`。

## 缺依赖怎么办

```powershell
uv add <包名>            # 会自动更新 pyproject.toml + uv.lock
```

**不要**用 `pip install`，也不要在代码里 `os.system("pip ...")`。

## 报错排查顺序

| 症状 | 先查 |
| --- | --- |
| `ModuleNotFoundError: program` | 是否在 program/ 或其父目录下用 `uv run` 运行 |
| 中文方框 | `pm.init()` 是否调用、是否在绘图前调用 |
| `KeyError` 读表失败 | `pm.read_table` 的 `sheet=` 参数、列名是否被 `clean_columns` 改过 |
| ODE 报参数个数错误 | 函数签名应为 `f(t, y, *params)` |
| 优化结果违反直觉 | 是否忘了 `maximize=True`、变量默认非负 |
| 交叉验证波动大 | 样本太少 / 没分层（分类用 `stratify=True` 默认已开） |

## 关于目录里的 `code/outputs/`

设计目的：**可追溯**。每个图对应 `figure_data/同名.csv`，答辩或复查时能回答
「这个曲线是用什么数据画的」。验收阶段（6verity）会检查。

## LaTeX / Typst 编译相关

- LaTeX 模板在 `.agents/skills/5writing/templates/zh/<赛事>-latex/`，
  编译：`xelatex main.tex`（**跑两遍**）；
- **Windows 用户注意**：把 `main.tex` 里的 `fontset=mac` 改为 `fontset=windows`；
- Typst 模板在 `.../zh/<赛事>/`，编译：`typst compile main.typ`；
- 两个引擎都已在本机就绪，选定后不要混用。

## 这个手册和 `AGENTS.md` 什么关系

| 文档 | 读者 | 内容 |
| --- | --- | --- |
| `program/docs/`（本手册） | 人 | 教程、示例、食谱、FAQ |
| `program/AGENTS.md` | AI 助手 | 紧凑的 API 速查与硬性约定 |

两者内容互补，改代码后请同步更新。

## 有功能想加 / 发现 bug

工具包在 `program/` 仓库中维护：

1. 改 `src/program/` 下对应模块；
2. 在 `tests/smoke_test.py` 加断言并跑通 `uv run python tests/smoke_test.py`；
3. 更新本手册与 `AGENTS.md` 对应条目；
4. `git commit`。
