CUMCM 2026 C 题 支撑材料说明
========================================
目录结构
  src/           完整可运行源程序（Python/uv 工程）+ 运行说明
  data/          题目附件 1–5（含结果模板）及数据来源与口径说明
  results/       五份正式结果表、中间输出（outputs/）、论文图表（figures/）；
                 各数据文件用途见 results/数据说明.md
  AI工具使用详情.pdf
  README.txt     本文件

软件环境与依赖
  Python 3.14；依赖由 uv 管理，版本锁定见 src/pyproject.toml 与 src/uv.lock。
  主要依赖：numpy、scipy、pandas、matplotlib、openpyxl；仅深度学习原型使用 torch（未进入正式方案）。

一键复现步骤（工作目录必须为 src/）
  1) 安装 uv：https://docs.astral.sh/uv/
  2) cd src && uv sync
  3) 依次运行：
     uv run python -m solve.q1                    # 问题一：典型日调度
     uv run python -m solve.q2_tune --result2     # 问题二：官方 result2
     uv run python -m solve.q3                    # 问题三：正式无对冲方案
     uv run python -m solve.q4_price_fit          # 问题四前置：电价逐日费用表
     uv run python -m solve.q4 --result4-2        # 问题四：Q2 层（G 主口径）
     uv run python -m solve.q4 --result4-3        # 问题四：Q3 层（G 主口径）
     uv run python -m solve.q3_ablation           # 问题三消融（论文检验章节）
  运行输出写入工程上一级目录的 figures/、reports/、code/outputs/，与随附 results/ 一致。

随机种子
  全局初始化 seed=42；Q3 场景对冲 seed=7（灵敏度另用 17/27）；详见 src/src/solve/consistency.py 与各脚本。

一致性声明
  1) 本目录 src/ 与论文附录“源程序代码”一致；
  2) 运行结果与 results/ 中五份正式结果表一致；
  3) 历史探索脚本与深度学习原型仅留存备考，不参与正式结果（正式链路见上述命令）。
