# Stoa — Make Blueprints Talk（Junction 2026）

来源：https://hackjunction.app/participate/45/challenges/335 （整理日期 2026-11-13）

## 两句话版本

> Build a program that answers natural language questions about construction drawings and building information models accurately and verifiably. In brief, answering to questions to a construction person like "How many square meters of tiling should i order for the toilets".

做一个程序，用自然语言准确、可核验地回答关于建筑施工图纸和建筑信息模型（BIM）的问题。简单说，就是回答工地人员这类问题：“卫生间要订多少平方米的瓷砖？”

---

## 1. 挑战说明（官方原文要点）

- **任务**：做一个程序，用自然语言回答关于建筑施工图纸的问题。
- **痛点**：工地工程师每天要花好几个小时找信息；通用 AI 模型分析复杂技术图纸时经常出错。官方称这是建筑科技领域最难的未解决问题之一。
- **数据**：官方提供真实技术图纸的数据集，格式多样、语言多样，附示例问题。
- **好答案的标准**：答案正确，并且指出出自图纸的哪个位置，让人几秒内就能核验。
- **方向偏好**：想要**明年的前沿模型出来后也不会过时的方案**，例如把前沿模型和计算机视觉、3D、几何或小型微调模型结合起来的混合方案。

## 2. 数据集

| 版本 | 大小 | 链接 |
|---|---|---|
| 预览版 | 0.5 GB | https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset-preview.zip |
| 完整版 | 10 GB | https://fwjydgxpttajiqpfmqtj.supabase.co/storage/v1/object/public/stoa-junction-2026/junction-dataset.zip |

内容是真实的建筑图纸和建筑模型。具体格式（PDF、DWG、IFC 等）、问题数量、有没有标准答案，**还没下载确认**。

## 3. 评审标准（四项各 25%）

| 评审项 | 官方描述 | 应对思路 |
|---|---|---|
| 准确率 | Accuracy of answers | 基于示例问题自建评测集，持续量化迭代 |
| 推理可视化 | How well the solution visualizes its reasoning so a human can verify the answer | 在图纸上高亮依据区域，展示推理链和引用 |
| 创意 | Creativity | 混合方案：版面和符号检测、OCR、矢量和几何解析、BIM 结构化查询，再用 LLM 编排 |
| 推理成本低 | Low inference cost | 先路由、只裁关键区域给大模型、缓存预处理结果、用本地小模型 |

要点：
- 成本单独占 25%。全靠最贵的多模态大模型硬读整张图会在这一项丢分。
- 页面没提排行榜或隐藏测试集，准确率可能由评委现场提问来测。

## 4. 奖项

| 名次 | 奖励 |
|---|---|
| 第一名 | €1000 + Stoa 工作试用 |
| 第二名 | €750 |
| 第三名 | €250 |

## 5. 公司和导师

- **Stoa**（https://stoabuild.com/）：赫尔辛基公司，在做“面向建筑业的 AI 原生操作系统”。建筑业是 15 万亿美元的行业，约占全球 GDP 的 13%。
- 团队来自 Supercell、AWS、McKinsey、Slush、Wave Ventures、SRV、Sensofusion，已经和北欧几家最大的建筑公司合作。
- 投资方：Lifeline Ventures，Wolt、Supercell、SiloAI 的创始人，两位前 Y Combinator 合伙人等。他们称这是芬兰史上最大的种子前轮融资。
- **导师**：Stoa 的创始技术团队（每天都在做建筑 AI 的工程师），加一位有多年大型建筑项目经验的建筑工程师。

## 6. 示例问题怎么拆：“卫生间要订多少平方米瓷砖？”

这类问题需要多步推理，不能只靠 OCR 查一下：
1. **找到卫生间**：在平面图里识别房间名或房间编号（可能是芬兰语 WC、KPH、PESUH 等），或者在 BIM 里按房间类型筛选。
2. **取面积和尺寸**：地面面积来自房间面积标注或多边形几何；墙面面积 = 周长 × 贴砖高度 − 门窗洞口。
3. **查做法和材料**：从房间做法表或材料说明里确认哪些面贴瓷砖、贴到多高。
4. **汇总**：所有卫生间加起来，再按行业习惯加 5–10% 损耗。
5. **给出依据**：在每个房间的位置、尺寸标注和做法表上高亮，并列出计算过程。

## 7. 推荐技术路线

1. **离线预处理（一次性、低成本）**
   - 矢量 PDF：直接提取文字、线条、图层和坐标（PyMuPDF 等）。
   - 扫描图或光栅图：版面分析（标题栏、图例、剖面、表格）+ OCR + 符号检测。
   - IFC/BIM：用 ifcopenshell 解析成结构化数据（房间、构件、属性、几何）。
   - 建索引：每个文字片段、表格和图块都记录页码和坐标框。
2. **在线问答**
   - 问题分类和路由：能用结构化查询或几何计算回答的，就不调大模型。
   - 检索相关图块，只把裁好的区域交给多模态模型。
   - 数值类问题交给代码或工具来算（面积、数量），LLM 只负责编排。
   - 输出答案、坐标框、计算过程和本次花费。
3. **前端演示**：问题 → 答案 → 高亮了依据区域的可缩放图纸 → 推理步骤 → 每次问答的成本。

## 8. 可叠报的技术挑战

- **Gemini API**、**Nemotron**（Nemotron 3.5 Super VL 能读图和 PDF）
- **Verda**（如果预处理或微调需要 GPU）

## 9. 待确认事项

- [ ] 下载预览数据集：确认文件格式、语言、示例问题和有没有标准答案
- [ ] 去现场问 Stoa：准确率怎么测（评委现场提问还是隐藏测试集）、成本怎么算
- [ ] 确认能否同时报多个主挑战
