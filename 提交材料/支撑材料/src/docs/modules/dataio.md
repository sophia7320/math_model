# 数据读写 dataio

赛题附件永远是第一道坎：Excel 里多个工作表、CSV 是 GBK 编码、表头有空格、
关键列有缺失……这个模块把这些事一次做顺。

## 快速上手

```python
import program as pm

# 读取：格式自动识别，中文编码自动回退
df = pm.read_table("data/附件1.xlsx", sheet="Sheet1")   # Excel 指定工作表
df = pm.read_table("data/附件2.csv")                     # CSV（UTF-8/GBK 都行）
sheets = pm.read_sheets("data/附件3.xlsx")               # 一次读全部工作表 → dict

# 看一眼：直接得到可贴进报告的 Markdown 概览
print(pm.summarize(df))
pm.record("## 数据理解\n\n" + pm.summarize(df))          # 顺手写入 RESULTS_REPORT

# 清洗
df = pm.dataio.clean_columns(df)                         # 列名去空格、修空列名
df = pm.dataio.fill_missing(df, strategy="auto")         # 数值列均值、类别列众数
df.to_csv("data/cleaned.csv", index=False)               # 或 pm.save_table(df, ...)
```

`pm.summarize(df)` 的输出长这样：

```text
**数据概览**：200 行 × 5 列，重复行 0，缺失单元格 3

| 列名 | 类型 | 非空数 | 缺失数 | 唯一值 | 最小值 | 最大值 | 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 温度 | float64 | 200 | 0 | 198 | -5.2 | 38.7 | 15.9 |
| 湿度 | float64 | 197 | 3 | 199 | 21 | 98 | 63.4 |
...
```

## 函数速查

| 函数 | 说明 |
| --- | --- |
| `read_table(path, sheet=0, sep=None)` | 自动识别 `.csv/.tsv/.txt/.xlsx/.json/.pkl`；CSV 优先 UTF-8，失败自动试 GB18030 |
| `read_sheets(path)` | Excel 全部工作表 → `{表名: DataFrame}` |
| `save_table(df, path)` | 按扩展名写出；CSV 用 `utf-8-sig`，Excel 打开不乱码 |
| `save_json(obj, path)` / `load_json(path)` | JSON 读写，自动处理 numpy 类型与日期 |
| `summarize(df, n=5)` | 形状 + 列信息 + 前 n 行，Markdown 格式 |
| `column_report(df)` | 逐列：类型 / 非空 / 缺失 / 唯一值 / 最值 / 均值 |
| `missing_report(df)` | 缺失情况专项统计 |
| `fill_missing(df, strategy="auto", cols=None)` | `auto / mean / median / mode / zero / ffill / bfill / drop` |
| `clean_columns(df, lower=False)` | 规整列名 |
| `split_xy(df, target)` | → `(X, y)` |
| `train_test_split_df(df, target=None, test_size=0.2, seed=42, stratify=False)` | 划分数据；带 `target` 时返回 `X_train, X_test, y_train, y_test` |
| `to_numpy(df)` | 只取数值列 → float ndarray |

## 小贴士

!!! tip "编码问题一劳永逸"
    读表格时优先用 `pm.read_table`：内部先试 UTF-8、再试 GB18030（GBK 超集）。
    如果某个文件有特殊编码，也可显式传 `encoding="...")`。

!!! tip "缺失值先诊断再填充"
    先 `pm.dataio.missing_report(df)` 看缺失集中在哪几列，再决定策略；
    对时间序列用 `strategy="ffill"` 往往比均值更合理。

!!! warning "写 CSV 给 Excel 看"
    用 `pm.save_table`（内部 `utf-8-sig`），不要直接 `df.to_csv()`，否则
    Windows Excel 打开中文列名会乱码。
