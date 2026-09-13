# C题论文写作记录（v1.3）

日期：2026-09-13。分支：gpt。计算基线：合并提交01eeaf5，远端结果提交fd1ac20。
用户确认沿用LaTeX与既有CUMCM模板；采用5writing撰写、6verity验收。

## 交付物

- 入口：paper/main.tex；摘要：paper/abstract.tex。
- 正文：paper/sections/（10个正文章节与附录）。
- 指定日期表：paper/tables/q1.tex、q2.tex、q3.tex。
- 文献：paper/references.tex。
- 成品：paper/main.pdf；根目录同内容副本：论文_CUMCM2026_C题.pdf。
- 验收：reports/VERIFY_REPORT.md。

既有cumcmthesis.cls和cumcm_official_final.sty继续使用；不复制覆盖已确认的模板。
原版论文可从Git基线01eeaf5恢复。本轮仅更新论文及写作相关文档，没有重跑或修改求解模型。

## 正式数值映射

| 论文内容 | 使用来源 | 数值/说明 |
| --- | --- | --- |
| Q1 | results/result1.xlsx；RESULTS_REPORT的Q1节 | 35126.95元，59482.7 kWh，节省26.9% |
| Q2 | q2e_tuned_daily.csv、result2.xlsx | 1395.7万元；开发期选定h=5、κ=1.02、m=25 |
| Q3 | q3_daily.csv、result3.xlsx | 1349.2万元 |
| Q4-2 G | q4_result42_daily.csv、result4-2.xlsx | 1459.8万元 |
| Q4-3 G | q4_result43_daily.csv、result4-3.xlsx | 1412.6万元 |
| H扩展 | RESULTS_REPORT当前Q4节及Q4图表数据 | 1465.9、1420.5万元 |
| Q2留出表 | code/outputs/q2e_tune_2day_holdout.csv | m=25是开发期冠军，验证排名5 |
| Q3消融 | code/outputs/q3_ablation.csv | 无对冲1347.1，主方案1349.2；不声称对冲降本 |
| Q3灵敏度 | code/outputs/q3_sensitivity.csv | 独立参考1349.3；保留与正式值的0.1差异说明 |
| Q3独立选参 | code/outputs/q3e_tune_unified_coarse/fine.csv | 独立开发冠军1.01/0对应1345.7 |
| 指定日期表 | result1/2/3.xlsx | Q2/Q3购电、储能与全部紧急事件直接提取 |

注意：RESULTS_REPORT末尾存在旧灵敏度重复章节（参考约1460万元），未用于本文；
ANALYSIS_MODELING_REPORT中的旧日循环、实际负荷已知等叙述由代码与v1.3修正口径覆盖。

## 图表规划与使用

| PDF | 所属章节 | 用途与数据 |
| --- | --- | --- |
| Q1_调度时序.pdf | 问题一 | 典型日时序；q1_schedule.csv |
| Q2_储能轨迹.pdf | 问题二 | 自由日末与跨日连续；figure_data同名CSV |
| Q2_总费用分布.pdf | 问题二 | 固定计划联合残差MC；figure_data同名CSV |
| Q3_策略费用对比.pdf | 问题三 | 当前三策略费用分解；figure_data同名CSV |
| Q4_调整层策略对比.pdf | 问题四 | G/H调整层费用分解；figure_data同名CSV |

所有数据图直接在对应正文章节引用。旧技术路线图与Q2流程图含过时机制，
本轮采用信息集对照表和公式说明，未将这些图插入新论文。
不再引用已删除的Q2_对冲费用分布.pdf。

## 正文中准确披露的实现细节

1. Q2只用附件1/2历史输入，不使用附件3预报；Q3才引入官方预报。
2. 两日规划末端自由；日内调整LP仍以当天计划的动态日末SOC为参考；
   实际执行不追赶此目标。依据q3_proto.adjust_day/adjust_day_hedge/run_exec。
3. day_cost虽已切换free执行，但规划仍调用plan_day；论文称其为单日代理标定。
4. 1月预热、2月1日以6000 kWh重置是当前代码事实，论文明确是条件回测。
5. 开发期费用不是独立验证；验证期不用于改选，整体334天是混合评价区间。
6. Q3场景内补救不是实际在线可提前知情；本文所有主费用来自因果执行。
7. Q3调整表最后一列是买入加偏差费，不是单纯买入费。
8. 对冲的未来尾部风险收益尚未验证，不复用旧对冲MC降本数字。

## 文献核实

- Mayne等：作者机构出版目录 https://sites.engineering.ucsb.edu/~jbraw/publications.html
  与DOI 10.1016/S0005-1098(99)00214-9。
- Huangfu与Hall：出版社 https://link.springer.com/article/10.1007/s12532-017-0130-5 。
- Rockafellar与Uryasev：作者出版目录 https://uryasev.github.io/publications/
  与期刊目录 https://www.risk.net/journal-of-risk/volume-2-number-3-spring-2000 ；
  卷2期3，21–41，DOI 10.21314/JOR.2000.038。

## 输入文件SHA256

| 文件 | SHA256 |
| --- | --- |
| result1.xlsx | 5351de53ac07eb497cbb630a9ba12c45b220c5ec950aae125f5f4bddce57a762 |
| result2.xlsx | c88e35025e382efd76786938ddbd4a4e6c45077a21f709eda9463cbb37ff0791 |
| result3.xlsx | 04a0cf0843ff0936da93443a36174331845d75c91e15f75e82471f1b921904c0 |
| result4-2.xlsx | 1ab9a14b2b08e4f319b9d4b3ccbb71377ee08a3a92cab6ac19aabe47c6c9d150 |
| result4-3.xlsx | d59610d440d7cb2d257ebb0cc803b765d6b0fb5f35cc344cb71a6ff300ac8d70 |
