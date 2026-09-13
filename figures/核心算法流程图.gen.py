# -*- coding: utf-8 -*-
"""核心算法流程图生成器（scibox-diagram 路径 B：显式坐标手写 XML）。

用途：论文正文单栏图「统一核心算法流程（Q1–Q4 共用引擎）」，覆盖
预测 → 风险裕度 → 计划 LP → 日内调整 → 因果执行 → 结算，含核心公式与四问信息集差异。
口径依据：program/src/solve/consistency.py（v1.4）+ paper/sections/5–8 + reports/模型一致性规范.md。

运行：& program\\.venv\\Scripts\\python.exe figures\\核心算法流程图.gen.py
输出：figures/核心算法流程图.drawio（可直接用 draw.io 编辑；改内容请改本脚本重生成）
"""
from __future__ import annotations

import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).with_name("核心算法流程图.drawio")

W, H = 700, 990                       # 画布
MX = 30                               # 主列左边界
BW = 600                              # 主列盒宽（30..630）
CX = MX + BW // 2                     # 纵向箭头 x = 330
FONT = "Microsoft YaHei,PingFang SC,Helvetica"


def sw(line: str, fs: float) -> float:
    """与 check_layout.py 一致的中文字宽模型：全角=字号，半角=字号/2。"""
    return sum(fs if unicodedata.east_asian_width(c) in ("W", "F") else fs / 2 for c in line)


def plain(value: str) -> str:
    import re
    v = re.sub(r"<br\s*/?>", "\n", value)
    v = re.sub(r"<[^>]+>", "", v)
    return (v.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"'))


def precheck(bid: str, value: str, fs: float, w: float) -> None:
    for ln in plain(value).split("\n"):
        usable = w - (8 if len(ln) > 1 else 2)
        if sw(ln, fs) > usable:
            print(f"  ! {bid}: 溢出 {sw(ln, fs):.0f} > {usable:.0f}px :: {ln}")


# ---------------------------------------------------------------------------
# 内容（逐行；标题 17px、正文 15px、口径 13px）
# ---------------------------------------------------------------------------
T = '<b><font style="font-size:17px;" color="#1f3f6b">{}</font></b>'


def tag(s: str) -> str:
    return f'<font style="font-size:13px;" color="#8a5a2b">口径：{s}</font>'


boxes: list[dict] = [
    dict(id="b0_input", y=14, h=90, fill="#f2f4f6", stroke="#8a97a3", body="<br>".join([
        T.format("输入数据（附件 1–4）"),
        "负荷 L、光伏 P、电价 p、光伏预报 f",
        "储能：η = 0.9，E ∈ [1200, 10800] kWh，充放 ≤ 833.3 kWh/槽（Δt = 10 min）",
        tag("Q1 用附件 1；Q2 用 1/2；Q3 用 1/2/3；Q4 用 1/2/4"),
    ])),
    dict(id="b1_pred", y=134, h=128, fill="#eef6fd", stroke="#3b547f", body="<br>".join([
        T.format("① 多源组合预测（0:00 发布 · 严格因果）"),
        "L̂<sub>d,t</sub> = w<sub>1</sub>L<sub>d−7,t</sub> + w<sub>2</sub>L<sub>d−14,t</sub> + w<sub>3</sub>L̄<sub>t</sub>",
        "P̂<sub>d,t</sub> = u<sub>1</sub>P<sub>d−1,t</sub> + u<sub>2</sub>P<sub>d−2,t</sub> + u<sub>3</sub>P̄<sub>t</sub>",
        "权重 (w, u)：单纯形网格搜索，按历史实现费用 EWMA 打分",
        "M<sub>d</sub> = ρM<sub>d−1</sub> + (1−ρ)C<sub>d−1</sub>，ρ = 2<sup>−1/5</sup>（h = 5 天，只用 ≤ d−1 信息）",
        tag("Q1 不预测（完全信息）；Q3/Q4 叠加官方预报 λf + (1−λ)P̂（λ = 0.7）"),
    ])),
    dict(id="b2_margin", y=292, h=90, fill="#fcead9", stroke="#c08b5c", body="<br>".join([
        T.format("② 风险裕度（时间留出选定 · Q2–Q4 共用）"),
        "L̃ = κL̂（κ = 1.02）；P̃ = max(P̂ − mΔt, 0)（m = 25 kW）",
        "方向依据：5× 紧急价 ⇒ 报童临界比 0.8（≈ 残差 q<sub>80</sub>）",
        tag("Q1 免裕度（确定性）；Q3/Q4 对组合预报同样折减"),
    ])),
    dict(id="b3_plan", y=412, h=109, fill="#cfe0f5", stroke="#3b547f", body="<br>".join([
        T.format("③ 两日滚动计划 LP（每日 0:00 · 288 槽 · 末端自由）"),
        "min Σ<sub>t</sub> [p<sub>t</sub>x<sub>t</sub> + ε(c<sub>t</sub> + q<sub>t</sub>)]",
        "s.t. x<sub>t</sub> + q<sub>t</sub> + P̃<sub>t</sub> = L̃<sub>t</sub> + c<sub>t</sub> + s<sub>t</sub>，E<sub>t</sub> = E<sub>t−1</sub> + ηc<sub>t</sub> − q<sub>t</sub>/η",
        "0 ≤ c<sub>t</sub>, q<sub>t</sub> ≤ 833.3；0 ≤ s<sub>t</sub> ≤ P̃<sub>t</sub>；E<sub>t</sub> ∈ [1200, 10800]；E<sub>0</sub> = 上一日实际末值",
        tag("Q1 单日 144 槽、E<sub>0</sub> = E<sub>144</sub> = 6000；Q2–Q4 只执行首日，次日滚动重解"),
    ])),
    dict(id="b4_adjust", y=551, h=109, fill="#fddecd", stroke="#c08b5c", body="<br>".join([
        T.format("④ 日内调整 LP（Q3/Q4：0/6/12/18 决策）"),
        "min Σ<sub>t</sub> (1.5p<sub>t</sub>x<sup>a</sup><sub>t</sub> − p<sub>t</sub>y<sub>t</sub>)，y = min(x<sup>0</sup><sub>t</sub>, x<sup>a</sup><sub>t</sub>) ≥ 0",
        "等价费用：g = p x<sup>a</sup> + 0.5p|x<sup>a</sup> − x<sup>0</sup>|（调低 50% · 调高 150%）",
        "连接条件：E<sup>adj</sup><sub>d,144</sub> = E<sup>plan</sup><sub>d,144</sub>（日末参考，执行不强制到达）",
        tag("Q1/Q2 无调整层；各决策点用最新预报 + 实际 SOC 重算剩余时段"),
    ])),
    dict(id="b5_exec", y=690, h=128, fill="#dbeef4", stroke="#4f8f8b", body="<br>".join([
        T.format("⑤ 逐槽因果执行（free · 每槽只读当前已实现值）"),
        "n<sub>t</sub> = L<sub>t</sub> − P<sub>t</sub> − x<sub>t</sub>（正为缺口，负为富余）",
        "n<sub>t</sub> ≥ 0：q<sub>t</sub> = min{n<sub>t</sub>, 833.3, η(E<sub>t−1</sub>−1200)}，e<sub>t</sub> = n<sub>t</sub> − q<sub>t</sub>",
        "n<sub>t</sub> &lt; 0：c<sub>t</sub> = min{−n<sub>t</sub>, 833.3, (10800−E<sub>t−1</sub>)/η}，s<sub>t</sub> = −n<sub>t</sub> − c<sub>t</sub>",
        "未列变量为 0；SOC 递推同 ③；跨日连续 E<sub>d+1,0</sub> = E<sup>exec</sup><sub>d,T</sub>",
        tag("Q2 执行窗 24 h；Q3/Q4 窗 6 h；缺口 5× 紧急、富余弃光（不售电）"),
    ])),
    dict(id="b6_settle", y=848, h=71, fill="#e5dfeb", stroke="#7f5faf", body="<br>".join([
        T.format("⑥ 费用结算（按实际实现回放）"),
        "C = Σ<sub>t</sub> [p<sub>t</sub>x<sup>a</sup><sub>t</sub> + 0.5p<sub>t</sub>|x<sup>a</sup><sub>t</sub> − x<sup>0</sup><sub>t</sub>|] + 5Σ<sub>t</sub> p<sub>t</sub>e<sub>t</sub>",
        tag("Q1 仅 Σp x；Q2 为 Σp x<sup>0</sup> + 5Σp e；Q4 电价取附件 4（G 主口径）"),
    ])),
]

note = dict(id="note", value="<br>".join([
    "注：正式口径 v1.4——EWMA h = 5，κ = 1.02，m = 25 kW，λ = 0.7；正式方案不含场景对冲。",
    "计划 LP 的 ε 项仅唯一化充放、不计入报送费用；决策只读历史与已发布预报，未来实际值仅用于回放结算。",
]))

# ---------------------------------------------------------------------------
# 生成 XML
# ---------------------------------------------------------------------------
mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "agent": "opencode", "version": "24.7.17"})
diagram = ET.SubElement(mxfile, "diagram", {"id": "core_algorithm", "name": "核心算法流程图"})
model = ET.SubElement(diagram, "mxGraphModel", {
    "dx": str(W), "dy": str(H), "grid": "1", "gridSize": "10", "guides": "1",
    "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
    "pageScale": "1", "pageWidth": str(W), "pageHeight": str(H), "math": "0", "shadow": "0",
})
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", {"id": "0"})
ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

print("字宽预检：")
for b in boxes:
    precheck(b["id"], b["body"], 15, BW)
precheck(note["id"], note["value"], 13, 640)

for b in boxes:
    style = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
             f"fillColor={b['fill']};strokeColor={b['stroke']};strokeWidth=1.5;"
             f"fontSize=15;fontColor=#262626;fontFamily={FONT};"
             "align=center;verticalAlign=middle;")
    c = ET.SubElement(root, "mxCell", {"id": b["id"], "value": b["body"], "style": style,
                                       "vertex": "1", "parent": "1"})
    ET.SubElement(c, "mxGeometry", {"x": str(MX), "y": str(b["y"]),
                                    "width": str(BW), "height": str(b["h"]), "as": "geometry"})

cn = ET.SubElement(root, "mxCell", {
    "id": note["id"], "value": note["value"],
    "style": ("text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
              f"whiteSpace=wrap;fontSize=13;fontColor=#5f6f7f;fontFamily={FONT};"),
    "vertex": "1", "parent": "1"})
ET.SubElement(cn, "mxGeometry", {"x": str(MX), "y": "935", "width": "640", "height": "40",
                                 "as": "geometry"})


def add_edge(eid, pts, style, value="", label_offset=(0, 0)):
    e = ET.SubElement(root, "mxCell", {"id": eid, "value": value, "style": style,
                                       "edge": "1", "parent": "1"})
    g = ET.SubElement(e, "mxGeometry", {"relative": "1", "as": "geometry"})
    if label_offset != (0, 0):
        ET.SubElement(g, "mxPoint", {"as": "offset", "x": str(label_offset[0]),
                                     "y": str(label_offset[1])})
    ET.SubElement(g, "mxPoint", {"x": str(pts[0][0]), "y": str(pts[0][1]), "as": "sourcePoint"})
    if len(pts) > 2:
        arr = ET.SubElement(g, "Array", {"as": "points"})
        for x, y in pts[1:-1]:
            ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})
    ET.SubElement(g, "mxPoint", {"x": str(pts[-1][0]), "y": str(pts[-1][1]), "as": "targetPoint"})


ARROW = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=5;"
         "strokeColor=#1f3f6b;strokeWidth=2;")
order = ["b0_input", "b1_pred", "b2_margin", "b3_plan", "b4_adjust", "b5_exec", "b6_settle"]
byid = {b["id"]: b for b in boxes}
for i in range(len(order) - 1):
    a, b = byid[order[i]], byid[order[i + 1]]
    add_edge(f"e{i}", [(CX, a["y"] + a["h"] + 1), (CX, b["y"] - 1)], ARROW)

loop = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;endFill=1;endSize=5;"
        "strokeColor=#c08b5c;strokeWidth=1.5;dashed=1;dashPattern=6 4;"
        f"fontSize=13;fontColor=#8a5a2b;labelBackgroundColor=#ffffff;fontFamily={FONT};")
add_edge("e_loop", [(631, 754), (660, 754), (660, 466), (631, 466)], loop,
         value="跨日滚动", label_offset=(0, 82))

ET.indent(mxfile, space="  ")
OUT.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(mxfile, encoding="unicode") + "\n",
               encoding="utf-8")
print(f"✓ {OUT}  ({W}×{H})")
