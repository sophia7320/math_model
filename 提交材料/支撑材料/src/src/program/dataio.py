"""数据读取、清洗与概览。

覆盖数学建模常见数据格式（csv / tsv / txt / xlsx / json / pkl），
自动处理中文编码（UTF-8 / GBK），并提供缺失值、数据概览等清洗工具。

用法::

    import program as pm

    df = pm.read_table("data/附件1.xlsx", sheet="Sheet1")
    print(pm.dataio.summarize(df))            # 数据概览（可直接贴进报告）
    df = pm.dataio.fill_missing(df)           # 数值列均值 / 类别列众数填充
    X_train, X_test, y_train, y_test = pm.dataio.train_test_split_df(
        df, target="销量", test_size=0.2, seed=42
    )
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .utils import markdown_table


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------
def _read_csv_smart(
    path: Path,
    sep: str | None,
    encoding: str | None,
    **kwargs: Any,
) -> pd.DataFrame:
    if sep is None:
        kwargs.setdefault("engine", "python")
    encodings = [encoding] if encoding else ["utf-8-sig", "gb18030"]
    last_err: Exception | None = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc, **kwargs)
        except UnicodeDecodeError as err:  # 中文附件常见 GBK 编码
            last_err = err
    raise last_err if last_err else ValueError(f"无法读取 {path}")


def read_table(
    path: str | Path,
    *,
    sheet: str | int | None = 0,
    sep: str | None = None,
    encoding: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """读取数据表，按扩展名自动分派，返回 DataFrame。

    - ``.csv``：默认逗号分隔，优先 UTF-8，失败自动尝试 GB18030（GBK 超集）
    - ``.tsv``：制表符分隔
    - ``.txt``：自动嗅探分隔符
    - ``.xlsx / .xls / .xlsm``：Excel，``sheet`` 指定工作表（默认第一个）
    - ``.json``：尝试 ``pd.read_json``，失败则按记录列表解析
    - ``.pkl / .pickle``：pickle

    其余 ``**kwargs`` 透传给底层 ``pandas`` 读取函数。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"数据文件不存在：{p.resolve()}")
    suf = p.suffix.lower()

    if suf == ".csv":
        return _read_csv_smart(p, "," if sep is None else sep, encoding, **kwargs)
    if suf == ".tsv":
        return _read_csv_smart(p, "\t" if sep is None else sep, encoding, **kwargs)
    if suf == ".txt":
        return _read_csv_smart(p, sep, encoding, **kwargs)
    if suf in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0, **kwargs)
    if suf == ".json":
        try:
            return pd.read_json(p, **kwargs)
        except ValueError:
            data = load_json(p)
            if isinstance(data, dict):
                data = list(data.values())[0] if data else []
            return pd.DataFrame(data)
    if suf in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    raise ValueError(f"暂不支持的文件类型：{suf}（支持 csv/tsv/txt/xlsx/json/pkl）")


def read_sheets(path: str | Path, **kwargs: Any) -> dict[str, pd.DataFrame]:
    """读取 Excel 全部工作表，返回 ``{表名: DataFrame}``。"""
    return pd.read_excel(path, sheet_name=None, **kwargs)


# ---------------------------------------------------------------------------
# 写出
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """把 DataFrame 写出到 csv / xlsx / json / pkl（按扩展名分派）。

    写 csv 时使用 ``utf-8-sig`` 编码，Excel 打开不乱码。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower()
    if suf == ".csv":
        df.to_csv(p, index=index, encoding="utf-8-sig")
    elif suf in {".xlsx", ".xls"}:
        df.to_excel(p, index=index)
    elif suf == ".json":
        save_json(df.to_dict(orient="records"), p)
    elif suf in {".pkl", ".pickle"}:
        df.to_pickle(p)
    else:
        raise ValueError(f"暂不支持的写出格式：{suf}")
    return p


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (datetime.date, datetime.datetime, pd.Timestamp)):
        return o.isoformat()
    raise TypeError(f"JSON 无法序列化类型：{type(o)!r}")


def save_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    """保存 JSON（自动处理 numpy 标量/数组、路径、日期）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, ensure_ascii=False, indent=indent, default=_json_default),
        encoding="utf-8",
    )
    return p


def load_json(path: str | Path) -> Any:
    """读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame, *, lower: bool = False) -> pd.DataFrame:
    """规整列名：去除首尾空格、把「Unnamed」空列名替换为 ``col{i}``。"""
    df = df.copy()
    new_cols: list[str] = []
    for i, c in enumerate(df.columns):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed"):
            name = f"col{i}"
        if lower:
            name = name.lower()
        new_cols.append(name)
    df.columns = new_cols
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """每列缺失情况统计表（缺失数 / 缺失占比 / 非空数）。"""
    n = len(df)
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "列名": [str(c) for c in df.columns],
            "类型": [str(t) for t in df.dtypes],
            "非空数": (n - missing).to_numpy(),
            "缺失数": missing.to_numpy(),
            "缺失占比": (missing / max(n, 1)).to_numpy(),
        }
    )


def fill_missing(
    df: pd.DataFrame,
    strategy: str = "auto",
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """缺失值填充，返回新 DataFrame（不修改原表）。

    Parameters
    ----------
    strategy : str
        - ``"auto"``：数值列均值、类别列众数（推荐）
        - ``"mean" / "median" / "mode" / "zero" / "ffill" / "bfill"``
        - ``"drop"``：删除含缺失的行
    cols : 序列 | None
        只处理指定列，默认全部列。
    """
    if strategy == "drop":
        return df.dropna(subset=list(cols) if cols else None).copy()

    out = df.copy()
    target_cols = list(cols) if cols else list(out.columns)
    for c in target_cols:
        s = out[c]
        if not s.isna().any():
            continue
        if strategy == "auto":
            st = "mean" if pd.api.types.is_numeric_dtype(s) else "mode"
        else:
            st = strategy
        if st == "mean":
            out[c] = s.fillna(s.mean())
        elif st == "median":
            out[c] = s.fillna(s.median())
        elif st == "mode":
            m = s.mode()
            out[c] = s.fillna(m.iloc[0] if len(m) else "未知")
        elif st == "zero":
            out[c] = s.fillna(0)
        elif st == "ffill":
            out[c] = s.ffill()
        elif st == "bfill":
            out[c] = s.bfill()
        else:
            raise ValueError(f"未知填充策略：{strategy!r}")
    return out


# ---------------------------------------------------------------------------
# 概览与转换
# ---------------------------------------------------------------------------
def column_report(df: pd.DataFrame) -> pd.DataFrame:
    """每列概况：类型 / 非空 / 缺失 / 唯一值数 / 最小值 / 最大值 / 均值。"""
    rows = []
    for c in df.columns:
        s = df[c]
        row: dict[str, Any] = {
            "列名": str(c),
            "类型": str(s.dtype),
            "非空数": int(s.notna().sum()),
            "缺失数": int(s.isna().sum()),
            "唯一值": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            row.update(
                最小值=s.min(),
                最大值=s.max(),
                均值=s.mean(),
            )
        else:
            row.update(最小值=None, 最大值=None, 均值=None)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n: int = 5, title: str = "数据概览") -> str:
    """生成 DataFrame 的 Markdown 概览（形状 + 列信息 + 前 n 行）。

    直接把返回值交给 ``report.record`` 即可写入 RESULTS_REPORT。
    """
    parts = [
        f"**{title}**：{df.shape[0]} 行 × {df.shape[1]} 列，"
        f"重复行 {int(df.duplicated().sum())}，缺失单元格 {int(df.isna().sum().sum())}"
    ]
    parts.append(markdown_table(column_report(df)))
    parts.append(f"**前 {n} 行**\n")
    parts.append(markdown_table(df.head(n)))
    return "\n\n".join(parts)


def split_xy(
    df: pd.DataFrame,
    target: str,
    drop: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """按目标列拆分特征矩阵 X 与标签 y（y 从 X 中移除，``drop`` 可再排除列）。"""
    if target not in df.columns:
        raise KeyError(f"目标列不存在：{target}；现有列：{list(df.columns)}")
    y = df[target]
    drop_cols = [target] + list(drop or [])
    X = df.drop(columns=drop_cols)
    return X, y


def to_numpy(df: pd.DataFrame, *, numeric_only: bool = True) -> np.ndarray:
    """DataFrame → float ndarray（默认只取数值列）。"""
    d = df.select_dtypes(include="number") if numeric_only else df
    return np.asarray(d, dtype=float)


def train_test_split_df(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = False,
) -> tuple:
    """DataFrame 版的训练/测试划分（内部固定随机种子）。

    - ``target=None``：返回 ``(train_df, test_df)``
    - ``target="列名"``：返回 ``(X_train, X_test, y_train, y_test)``，保持 DataFrame/Series
    """
    from sklearn.model_selection import train_test_split

    if target is None:
        return train_test_split(df, test_size=test_size, random_state=seed)
    X, y = split_xy(df, target)
    return train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if stratify else None,
    )
