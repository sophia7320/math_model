"""通用工具：随机种子、计时、日志、缓存、运行环境、格式化。

本模块是纯工具层，不依赖绘图与建模库，可在任何脚本中安全导入。
"""

from __future__ import annotations

import os
import pickle
import platform
import random
import sys
import time
from contextlib import ContextDecorator
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 随机种子
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> int:
    """一次性固定 random / numpy / torch（若已安装）的随机种子。

    所有涉及随机过程的脚本（采样、优化、神经网络、bootstrap）必须先调用本函数，
    保证结果可复现。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:  # 不主动导入 torch（加载慢），已导入才设置
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------
class Timer(ContextDecorator):
    """计时器：上下文管理器兼装饰器。

    用法::

        with Timer("求解"):
            heavy_work()

        @Timer("训练")
        def train(): ...
    """

    def __init__(self, label: str = "", logger: Any = None):
        self.label = label
        self.logger = logger
        self.elapsed: float = 0.0

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.elapsed = time.perf_counter() - self._start
        self._log(f"{self.label or '耗时'}：{self.elapsed:.3f} s")
        return False


def timeit(func: Callable) -> Callable:
    """装饰器：运行函数并打印耗时，原样返回函数结果。"""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时：{time.perf_counter() - t0:.3f} s")
        return result

    wrapper.__name__ = getattr(func, "__name__", "wrapper")
    return wrapper


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def get_logger(name: str = "mathmodel", log_file: str | Path | None = None):
    """返回 loguru 日志器（带模块名绑定）。

    Parameters
    ----------
    name : str
        模块/脚本名，出现在每条日志的 ``module=`` 字段里。
    log_file : str | Path | None
        可选日志文件路径；传入后同时写入文件。

    用法::

        log = get_logger("q1")
        log.info("开始求解")
    """
    from loguru import logger

    log = logger.bind(module=name)
    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        # loguru 的 add 返回 sink id；重复调用同一文件会重复写，用属性去重
        added = getattr(get_logger, "_files", set())
        if str(p.resolve()) not in added:
            logger.add(p, encoding="utf-8", level="DEBUG", enqueue=False)
            added.add(str(p.resolve()))
            get_logger._files = added
    return log


# ---------------------------------------------------------------------------
# 结果缓存
# ---------------------------------------------------------------------------
class DataCache:
    """某次计算的中间结果缓存（pickle 序列化）。

    默认目录 ``code/outputs/cache/<name>/``。用途：避免重复训练/重复求解，
    同时把关键中间结果沉淀为文件（RESULTS_REPORT 要求记录数据来源）。

    用法::

        cache = DataCache("q1")
        params = cache.get_or_compute("params", heavy_fit, x, y)
    """

    def __init__(self, name: str, root: str | Path | None = None):
        from .config import outputs_dir

        base = Path(root) if root is not None else outputs_dir() / "cache"
        self.dir = base / name
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        """缓存文件路径（``<key>.pkl``）。"""
        return self.dir / f"{key}.pkl"

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def save(self, key: str, obj: Any) -> Path:
        """写入缓存，返回文件路径。"""
        p = self.path(key)
        with p.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    def load(self, key: str) -> Any:
        """读取缓存；不存在则抛 ``FileNotFoundError``。"""
        p = self.path(key)
        if not p.exists():
            raise FileNotFoundError(f"缓存不存在：{p}")
        with p.open("rb") as f:
            return pickle.load(f)

    def get_or_compute(self, key: str, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """存在则读缓存，否则调用 ``func(*args, **kwargs)`` 并缓存结果。"""
        if self.exists(key):
            return self.load(key)
        result = func(*args, **kwargs)
        self.save(key, result)
        return result

    def clear(self) -> int:
        """清空缓存目录，返回删除的文件数。"""
        n = 0
        for p in self.dir.glob("*.pkl"):
            p.unlink()
            n += 1
        return n


# ---------------------------------------------------------------------------
# 运行环境
# ---------------------------------------------------------------------------
def env_info() -> dict[str, str]:
    """返回运行环境关键信息（平台、Python 与主要库版本）。"""
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    for pkg in (
        "numpy", "scipy", "pandas", "matplotlib", "scikit-learn",
        "statsmodels", "sympy", "torch", "networkx", "pulp", "openpyxl", "loguru",
    ):
        try:
            info[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return info


def env_markdown() -> str:
    """运行环境的 Markdown 片段（直接可贴进 RESULTS_REPORT）。"""
    info = env_info()
    lines = ["| 项目 | 版本 |", "| --- | --- |"]
    labels = {"platform": "操作系统", "python": "Python"}
    for k, v in info.items():
        lines.append(f"| {labels.get(k, k)} | {v} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------
def fmt(x: Any, nd: int = 4) -> str:
    """把数值格式化成论文/报告友好的字符串。

    - ``None`` / NaN → ``"—"`` / ``"NaN"``
    - 绝对值过大或过小 → 科学计数法（3 位有效数字）
    - 其余 → 至多 ``nd`` 位有效数字
    """
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isnan(f):
        return "NaN"
    if np.isinf(f):
        return "∞" if f > 0 else "−∞"
    if f == 0:
        return "0"
    a = abs(f)
    if a >= 1e5 or a < 1e-3:
        return f"{f:.3e}"
    return f"{f:.{nd}g}"


def markdown_table(
    data: Any,
    headers: Sequence[str] | None = None,
    nd: int = 4,
    index: bool = False,
) -> str:
    """把表格数据转成 Markdown 表格字符串。

    Parameters
    ----------
    data : DataFrame | list[dict] | list[list] | dict[str, list]
        表格数据。
    headers : list[str] | None
        自定义表头；``list[list]`` / ``DataFrame`` 默认取第一行或列名。
    nd : int
        数字保留的有效位数。
    index : bool
        是否输出 DataFrame 索引列。
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):
        df = data.reset_index() if index else data
        cols = [str(c) for c in df.columns]
        rows = df.itertuples(index=False, name=None)
        body = [[fmt(v, nd) for v in row] for row in rows]
    elif isinstance(data, dict):
        cols = list(headers) if headers else [str(k) for k in data]
        body = [[fmt(v, nd) for v in data[c]] for c in data]
    elif isinstance(data, (list, tuple)):
        seq = list(data)
        if not seq:
            return "（空表）"
        if isinstance(seq[0], dict):
            cols = list(headers) if headers else [str(k) for k in seq[0]]
            body = [[fmt(item.get(c, ""), nd) for c in cols] for item in seq]
        else:
            cols = list(headers) if headers else [str(i) for i in range(len(seq[0]))]
            body = [[fmt(v, nd) for v in row] for row in seq]
    else:
        raise TypeError(f"markdown_table 不支持的类型：{type(data)!r}")

    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---" for _ in cols) + " |"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in body]
    return "\n".join(out)


def ensure_dir(path: str | Path) -> Path:
    """创建目录（存在则跳过），返回 ``Path``。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def flatten(items: Iterable) -> list:
    """展平一层嵌套序列。"""
    out: list = []
    for it in items:
        out.extend(it)
    return out
