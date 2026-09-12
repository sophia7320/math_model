"""全局配置与初始化（中文字体、论文风格、随机种子、输出目录）。

用法::

    import program as pm

    pm.init()                       # 一键初始化：中文字体 + 论文风格 + 随机种子
    pm.init(seed=0, root=r"D:\\case")  # 自定义随机种子与项目根目录

输出目录约定（全部从 :func:`project_root` 派生，默认为当前工作目录，
可用环境变量 ``MATHMODEL_ROOT`` 或 :func:`set_project_root` 覆盖）：

- ``figures/``         论文图表（:func:`program.plotting.save_fig` 默认输出）
- ``reports/``         ``RESULTS_REPORT.md`` 等报告（:func:`program.report.record_result`）
- ``code/outputs/``    中间数据、图表数据、log（:func:`program.report.save_outputs`）
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 论文配色：seaborn "deep" 风格 10 色循环，对色盲友好、印刷清晰
# ---------------------------------------------------------------------------
PALETTE: tuple[str, ...] = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
)

# 中文字体候选（按平台常见度排序）
CJK_FONTS: tuple[str, ...] = (
    "Microsoft YaHei", "SimHei", "SimSun", "KaiTi", "FangSong",   # Windows
    "PingFang SC", "Hiragino Sans GB", "Songti SC", "Heiti SC",   # macOS
    "Noto Sans CJK SC", "Noto Serif CJK SC", "Source Han Sans SC",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",                   # Linux
)

_ROOT: Path | None = None


# ---------------------------------------------------------------------------
# 项目根目录与输出目录
# ---------------------------------------------------------------------------
def set_project_root(path: str | Path) -> Path:
    """显式指定项目根目录（覆盖环境变量 ``MATHMODEL_ROOT`` 与当前工作目录）。"""
    global _ROOT
    _ROOT = Path(path).expanduser().resolve()
    return _ROOT


def project_root() -> Path:
    """项目根目录：显式设置 > 环境变量 ``MATHMODEL_ROOT`` > 当前工作目录。"""
    if _ROOT is not None:
        return _ROOT
    env = os.environ.get("MATHMODEL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd()


def figures_dir() -> Path:
    """论文图表目录 ``<root>/figures``。"""
    return project_root() / "figures"


def reports_dir() -> Path:
    """报告目录 ``<root>/reports``。"""
    return project_root() / "reports"


def outputs_dir() -> Path:
    """中间数据目录 ``<root>/code/outputs``。"""
    return project_root() / "code" / "outputs"


def figure_data_dir() -> Path:
    """图表对应的数据目录 ``<root>/code/outputs/figure_data``。"""
    return outputs_dir() / "figure_data"


# ---------------------------------------------------------------------------
# 中文字体探测
# ---------------------------------------------------------------------------
def find_cjk_font() -> str | None:
    """返回本机第一个可用的中文字体名，找不到返回 ``None``。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            return name
    return None


# ---------------------------------------------------------------------------
# 一键初始化
# ---------------------------------------------------------------------------
def init(
    seed: int | None = 42,
    root: str | Path | None = None,
    chinese: bool = True,
    backend: str | None = "Agg",
    figsize: tuple[float, float] = (7.0, 4.3),
    verbose: bool = True,
) -> dict:
    """初始化绘图环境与随机种子，脚本开头调用一次即可。

    Parameters
    ----------
    seed : int | None
        全局随机种子（random / numpy / torch / PYTHONHASHSEED），``None`` 表示不设置。
    root : str | Path | None
        项目根目录，传给 :func:`set_project_root`。
    chinese : bool
        是否自动探测并启用中文字体。
    backend : str | None
        matplotlib 后端，默认 ``"Agg"``（无窗口、稳定保存文件）；
        需要交互式窗口时传 ``None`` 或 ``"TkAgg"``。
    figsize : (float, float)
        默认图幅尺寸（英寸），论文单栏常用 7x4.3。
    verbose : bool
        是否打印初始化摘要。

    Returns
    -------
    dict
        实际生效的配置摘要（根目录、字体、后端、种子）。
    """
    if root is not None:
        set_project_root(root)

    if seed is not None:
        from .utils import set_seed

        set_seed(seed)

    # 后端必须在 pyplot 首次导入前设置
    import matplotlib

    if backend:
        try:
            matplotlib.use(backend, force=True)
        except Exception:  # 后端不可用时静默回退
            pass

    import matplotlib.pyplot as plt
    from cycler import cycler

    font_name = find_cjk_font() if chinese else None

    rc: dict = {
        # 图幅与分辨率
        "figure.figsize": figsize,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # 字号
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        # 网格与边框
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # 线条与配色
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "legend.frameon": False,
        "axes.prop_cycle": cycler(color=PALETTE),
        # 负号与数学字体
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
    }
    if font_name:
        rc["font.family"] = "sans-serif"
        rc["font.sans-serif"] = [font_name, "DejaVu Sans", "Arial"]
    plt.rcParams.update(rc)

    summary = {
        "root": str(project_root()),
        "font": font_name or "（未找到中文字体，中文可能显示为方框）",
        "backend": matplotlib.get_backend(),
        "seed": seed,
    }
    if verbose:
        from loguru import logger

        logger.info(
            "program 初始化完成 | root={root} | 中文字体={font} | 后端={backend} | seed={seed}",
            **summary,
        )
    return summary
