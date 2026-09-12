"""结果记录与论文素材输出：RESULTS_REPORT 追加、LaTeX 表格、结果文件。

典型用法（每个子问题算完后）::

    import program as pm

    pm.report.record_result(
        "问题一结果",
        {"最优成本": 1234.5, "总利润": 6789.0},
        note="求解器：HiGHS，约束全部满足。",
    )
    pm.report.record_result("问题一灵敏度", sensitivity_df)
    pm.report.save_outputs(sensitivity_df, "q1_sensitivity")
    print(pm.report.to_latex(result_table, caption="问题一结果", label="tab:q1"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import reports_dir
from .utils import fmt, markdown_table

REPORT_NAME = "RESULTS_REPORT.md"


# ---------------------------------------------------------------------------
# 报告写入
# ---------------------------------------------------------------------------
def report_path(path: str | Path | None = None) -> Path:
    """RESULTS_REPORT 路径（默认 ``<root>/reports/RESULTS_REPORT.md``）。"""
    return Path(path) if path is not None else reports_dir() / REPORT_NAME


def ensure_report(title: str = "# 计算结果", path: str | Path | None = None) -> Path:
    """确保报告文件存在（不存在时写入标题），返回路径。"""
    p = report_path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(title.rstrip() + "\n\n", encoding="utf-8")
    return p


def record(text: str, path: str | Path | None = None) -> Path:
    """向 RESULTS_REPORT 追加一段 Markdown（自动建目录/文件）。"""
    p = ensure_report(path=path)
    with p.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")
    return p


def record_result(
    title: str,
    data: Any,
    *,
    note: str | None = None,
    path: str | Path | None = None,
    level: int = 3,
) -> Path:
    """把一段结果以 ``### 标题 + 表格`` 的形式追加进 RESULTS_REPORT。

    Parameters
    ----------
    title : str
        小节标题，如 ``"问题一结果"``。
    data : dict | DataFrame | list | str
        - ``dict[str, 标量]`` → 两列「指标 / 数值」表格；
        - ``dict[str, 序列]`` 或 DataFrame → 普通表格；
        - ``str`` → 原样写入正文。
    note : str | None
        表格前的补充说明（方法、约束校验、结论等）。
    level : int
        Markdown 标题级别，默认 3（``###``）。
    """
    import pandas as pd

    parts = [f"{'#' * level} {title}"]
    if note:
        parts.append(note)
    if isinstance(data, str):
        parts.append(data)
    elif isinstance(data, pd.DataFrame):
        parts.append(markdown_table(data))
    elif isinstance(data, dict):
        scalars = all(np.isscalar(v) or v is None or isinstance(v, str) for v in data.values())
        if scalars:
            rows = [{"指标": k, "数值": fmt(v)} for k, v in data.items()]
            parts.append(markdown_table(rows))
        else:
            parts.append(markdown_table(data))
    else:
        parts.append(markdown_table(data))
    return record("\n\n".join(parts), path=path)


def record_environment(path: str | Path | None = None) -> Path:
    """把运行环境（Python 与库版本）写入报告。"""
    from .utils import env_markdown

    return record("## 运行环境\n\n" + env_markdown(), path=path)


# ---------------------------------------------------------------------------
# 论文素材
# ---------------------------------------------------------------------------
def to_latex(
    df,
    *,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "%.4f",
    index: bool = False,
    escape: bool = True,
    column_format: str | None = None,
) -> str:
    """DataFrame → LaTeX 表格字符串（booktabs 风格，可直接贴进论文）。

    注意：``float_format`` 为 printf 风格字符串；含下划线的列名请传 ``escape=False``
    或改用 ``\\textbackslash`` 转义。
    """
    kwargs: dict = dict(
        index=index,
        escape=escape,
        float_format=float_format,
        caption=caption,
        label=label,
    )
    if column_format:
        kwargs["column_format"] = column_format
    try:
        return df.to_latex(**kwargs)
    except TypeError:
        # 兼容新旧 pandas 对 float_format 类型的差异
        import pandas as pd

        kwargs["float_format"] = lambda v: f"{float(v):.4f}"
        latex = df.to_latex(**kwargs)
        _ = pd  # 保持引用避免误删导入
        return latex


def save_outputs(
    data: Any,
    name: str,
    *,
    fmt: str | None = None,
    outdir: str | Path | None = None,
) -> Path:
    """把结果数据保存到 ``code/outputs/<name>.<ext>`` 并返回文件路径。

    - DataFrame → ``.csv``（``utf-8-sig``，Excel 直接打开不乱码）
    - dict / list → ``.json``
    - ndarray → ``.csv``（1D 为单列，2D 带列名 col0、col1…）
    - str → ``.md``
    - 其他 → ``.pkl``
    """
    from .config import outputs_dir

    outdir = Path(outdir) if outdir is not None else outputs_dir()
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    if fmt is None:
        if isinstance(data, pd.DataFrame):
            fmt = "csv"
        elif isinstance(data, (dict, list, tuple)):
            fmt = "json"
        elif isinstance(data, np.ndarray):
            fmt = "csv"
        elif isinstance(data, str):
            fmt = "md"
        else:
            fmt = "pkl"
    fmt = fmt.lower().lstrip(".")

    p = outdir / f"{name}.{fmt}"
    if fmt == "csv":
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                data = pd.DataFrame({name: data})
            else:
                data = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data.to_csv(p, index=False, encoding="utf-8-sig")
    elif fmt == "json":
        from .dataio import save_json

        save_json(data, p)
    elif fmt in {"md", "txt"}:
        p.write_text(str(data), encoding="utf-8")
    else:
        import pickle

        with p.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p
