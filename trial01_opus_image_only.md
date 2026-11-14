# Trial 01 — Opus 5.5 只看图 + 问题的基线（2026-11-13）

## 设置

- 模型：Opus 5.5，effort medium；每题一个独立子代理，互不共享上下文。
- 输入：仅 `question.txt` 和 `drawing.png`（匿名文件名，放在数据集目录之外）。PDF 以最高 300 dpi 渲染成 PNG，长边上限 6000 像素。
- 约束：禁止读其他文件、跑脚本、联网。事后审计转录，50 个子代理都只读了这两个文件。
- 抽样：从完整版 `questions.jsonl`（127,044 题）中抽 50 题，随机种子 20261113，按来源分层：cubicasa5k 13、swiss_dwellings 13、ifc_models 12（仅针对渲染 PDF 的题）、opening_design 12；来源内按题型轮流抽。
- 未覆盖：ifc_bench 和所有针对 `.ifc` 文件的题（没有图像）。
- 评分：官方规则（计数完全一致；面积、长度、体积误差 2% 以内）。4 道文本题无官方规则，人工比对。
- 已知不纯之处：子代理启动时自动带有用户全局 `CLAUDE.md`。

## 结果

总计 **36 / 50（72%）**：数值题 32 / 46，文本题 4 / 4（人工判）。

| 来源 | 正确 / 总数 |
|---|---|
| opening_design | 12 / 12 |
| cubicasa5k | 9 / 13 |
| swiss_dwellings | 9 / 13 |
| ifc_models | 6 / 12 |

| 题型（数值题） | 正确 / 总数 |
|---|---|
| count | 19 / 26 |
| area | 9 / 14 |
| length | 2 / 3 |
| volume | 2 / 3 |

| 模型自报置信度（数值题） | 正确 / 总数 |
|---|---|
| high | 15 / 15 |
| medium | 8 / 10 |
| low | 9 / 21 |

成本：50 个子代理共约 236 万 token（平均每题约 4.7 万，含子代理自带的系统提示和工具定义，不等于直接调 API 的成本）。

## 观察

- 答案印在图上的题（opening_design 的门表、装修表）全对；需要对着比例尺估像素的题不稳，单个小房间面积题 5 道只对 1 道。
- 数量多的计数题会漏数（窗 10 答 6、门 18 答 15、墙 70 答 61）；答案在 3 以内的基本都对。
- 体积题对的 2 道是假设了典型板厚后碰巧落进 2%，图上并没有厚度信息。
- 样本小，分项比例只能看趋势。

## 逐题

| id | 来源 | 题型 | 原始文件 | 问题 | 标准答案 | 模型答案 | 对错 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| q01 | swiss_dwellings | count | `data/swiss_dwellings/site1060_bldg2175_floor9833.pdf` | How many bathtubs are drawn on this floor plan? | 3 | 3 | ✓ | medium |
| q02 | opening_design | text | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what is the height of the four doors of type B2 that serve the Fitness Center (118.A1-118.B2)? | 7'-10 1/4" | 7'-10 1/4" | ✓ | high |
| q03 | ifc_models | length | `data/ifc_models/bsc-molio_molio-with-uris_plan-09-6-sal.pdf` | What is the total length of all walls on '6. sal', in metres? | 69.444 | 69.5 | ✓ | low |
| q04 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-10429.png` | How many doors are shown on this floor plan? Count doorways fitted with a door; open doorways without a door leaf do not count. | 6 | 6 | ✓ | medium |
| q05 | opening_design | area | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what is the area of room 122 (Wrestling) in the finish room schedule? | 3814 | 3814 | ✓ | high |
| q06 | swiss_dwellings | area | `data/swiss_dwellings/site30_bldg262_floor1025.pdf` | What is the net floor area of room 4.23 ('Bad') in m²? | 3.64 | 3.8 | ✗ | low |
| q07 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-10495.png` | Kuinka monta WC-istuinta pohjapiirustukseen on piirretty? | 1 | 1 | ✓ | high |
| q08 | opening_design | count | `data/opening_design/CTR/Most Current Set/Full size sheets/A700 - SCHEDULES.pdf` | On sheet A700 of the Cannery Trail Residences drawings (1750 N Oxford Ave, Eau Claire, WI), how many Vistagrande fiberglass entry doors are scheduled on the 1st floor? | 19 | 19 | ✓ | medium |
| q09 | ifc_models | volume | `data/ifc_models/bsc-gymzaal-amersfoort_example-project-location_plan-01-00-begane-grond.pdf` | Wat is het totale netto volume van alle vloeren op '00 Begane grond', in m³? | 94.221 | 108 | ✗ | low |
| q10 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-3963.png` | How many doors are shown on this floor plan? Count doorways fitted with a door; open doorways without a door leaf do not count. | 10 | 10 | ✓ | low |
| q11 | swiss_dwellings | count | `data/swiss_dwellings/site288_bldg533_floor15119.pdf` | Combien de baignoires sont dessinées sur ce plan ? | 2 | 2 | ✓ | high |
| q12 | swiss_dwellings | area | `data/swiss_dwellings/site428_bldg1088_floor4764.pdf` | Qual è la superficie netta dell'appartamento D in m² (somma di tutti i locali, esclusi balconi, logge, terrazze e altri spazi esterni)? | 104.9 | 102 | ✗ | low |
| q13 | opening_design | count | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, how many doors in the door schedule carry note B (opens 180°)? | 4 | 4 | ✓ | high |
| q14 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-14630.png` | How many kitchens (including kitchenettes, open kitchens and utility kitchens) are on this floor plan? | 1 | 1 | ✓ | high |
| q15 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-12424.png` | Kuinka monta kylpy- tai pesuhuonetta (esim. KH, KPH, PH, PESUH, SH) pohjapiirustuksessa on, erillisiä WC-tiloja lukuun ottamatta? | 1 | 1 | ✓ | high |
| q16 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-1241.png` | How many windows are shown on this floor plan? | 4 | 3 | ✗ | medium |
| q17 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-1231.png` | How many showers and bathtubs are drawn on this floor plan in total? | 1 | 2 | ✗ | low |
| q18 | ifc_models | count | `data/ifc_models/kit-smiley-west_ac-20-smiley-west-10-bldg_plan-04-2-og.pdf` | Wie viele Wände sind im Grundriss „2.OG“ dargestellt? | 70 | 61 | ✗ | low |
| q19 | ifc_models | area | `data/ifc_models/bsc-esplanades_1807-ep-ar-v18_plan-05-5-korrus.pdf` | What is the net floor area of the room 'ESIK (72-2)' on '5.korrus', in m²? | 3.01 | 2.6 | ✗ | low |
| q20 | swiss_dwellings | count | `data/swiss_dwellings/site2138_bldg2981_floor13589.pdf` | How many rooms labelled 'Wohnen/Essen' are there on this floor? | 1 | 1 | ✓ | high |
| q21 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-5693.png` | How many balconies (parveke, including glazed balconies) are on this floor plan? | 1 | 1 | ✓ | medium |
| q22 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-3621.png` | Kuinka monta erillistä WC-tilaa pohjapiirustuksessa on? | 1 | 1 | ✓ | medium |
| q23 | ifc_models | volume | `data/ifc_models/kit-institute_ac20-institute-var-2_plan-03-1-obergeschoss.pdf` | Wie groß ist das gesamte Netto-Volumen aller Wände im Geschoss „1. Obergeschoss“ in m³? | 161.958 | 160 | ✓ | low |
| q24 | swiss_dwellings | count | `data/swiss_dwellings/site110_bldg1_floor11271.pdf` | Wie viele Türen (Innentüren und Eingangstüren zusammen) sind auf diesem Grundriss eingezeichnet? | 18 | 15 | ✗ | low |
| q25 | opening_design | text | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what door material is specified for door 115.A? | AL-2/GL-2 | AL-2/GL-2 (aluminum storefront with thermal brake, Tubelite 14000, with tempered and insulated glazing) | ✓ | medium |
| q26 | swiss_dwellings | count | `data/swiss_dwellings/site1119_bldg2262_floor10576.pdf` | Wie viele Duschen sind auf diesem Grundriss eingezeichnet? | 2 | 4 | ✗ | low |
| q27 | opening_design | area | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what is the combined area of the women's and men's locker rooms? | 1181 | 1181 | ✓ | high |
| q28 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-4795.png` | Kuinka monta parveketta (myös lasitetut parvekkeet) pohjapiirustuksessa on? | 0 | 0 | ✓ | medium |
| q29 | swiss_dwellings | area | `data/swiss_dwellings/site2285_bldg3364_floor14968.pdf` | Wie gross ist die Nettofläche von Raum 3.05 («Küche») in m²? | 9.52 | 9.4 | ✓ | low |
| q30 | ifc_models | count | `data/ifc_models/bimspeed-vitoria_ald26-before_plan-03-p2.pdf` | How many stairs are shown on the floor plan of 'P2'? | 1 | 1 | ✓ | medium |
| q31 | swiss_dwellings | area | `data/swiss_dwellings/site4165_bldg6902_floor27817.pdf` | What is the total net area in m² of all rooms labelled 'Schlafzimmer' on this floor? | 29.92 | 29.6 | ✓ | low |
| q32 | opening_design | count | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, how many doors in the door schedule are door type A1 (swing door)? | 14 | 14 | ✓ | medium |
| q33 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-11931.png` | Kuinka monta ovea pohjapiirustuksessa on? Laske oviaukot, joissa on ovi; pelkkiä oviaukkoja ilman ovilehteä ei lasketa. | 6 | 8 | ✗ | medium |
| q34 | cubicasa5k | count | `data/cubicasa5k/high_quality_architectural-13185.png` | How many windows are shown on this floor plan? | 10 | 6 | ✗ | low |
| q35 | opening_design | text | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what door type is door 121.A, and what kind of door is that? | B3 (aluminium sliding door) | Door type B3, an aluminum sliding door | ✓ | medium |
| q36 | opening_design | text | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, how many rooms have floor finish F0 (unfinished concrete slab), and what is their total area? | 3 rooms, 4541 SF | 3 rooms (120 C. Vest, 121 Clinic, 130 Trash Enclosure); total 4541 SF | ✓ | medium |
| q37 | ifc_models | area | `data/ifc_models/bimspeed-vitoria_ald26-before_plan-02-p1.pdf` | What is the net floor area of the room '1DER-BAÑ (01.28)' on 'P1', in m²? | 4.34 | 4.1 | ✗ | low |
| q38 | ifc_models | length | `data/ifc_models/kit-smiley-west_ac-20-smiley-west-10-bldg_plan-03-1-og.pdf` | What is the total length of all walls on '1.OG', in metres? | 415.734 | 410 | ✓ | low |
| q39 | opening_design | area | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what is the area of room 121 (Clinic) in the finish room schedule? | 2470 | 2470 | ✓ | high |
| q40 | ifc_models | volume | `data/ifc_models/kit-institute_ac20-institute-var-2_plan-02-erdgeschoss.pdf` | What is the total net volume of all slabs on 'Erdgeschoss', in m³? | 156.42 | 157 | ✓ | low |
| q41 | swiss_dwellings | area | `data/swiss_dwellings/site610_bldg1324_floor50793.pdf` | What is the total net area in m² of all rooms labelled 'Pièce' on this floor? | 173.02 | 173.02 | ✓ | high |
| q42 | swiss_dwellings | area | `data/swiss_dwellings/site885_bldg1887_floor8359.pdf` | Wie gross ist die gesamte Nettogeschossfläche in m² aller nummerierten Innenräume dieses Geschosses (ohne Balkone, Loggien, Terrassen und andere Aussenräume, Schächte und Lufträume)? | 321.86 | 317 | ✓ | low |
| q43 | opening_design | count | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, how many rooms are listed in the finish room schedule? | 29 | 29 | ✓ | high |
| q44 | ifc_models | area | `data/ifc_models/bsc-esplanades_1807-ep-ar-v18_plan-04-4-korrus.pdf` | Kui suur on ruumi „ELUTUBA+KÖÖK (52-1)“ netopõrandapind korrusel „4.korrus“ (m²)? | 25.25 | 20 | ✗ | low |
| q45 | cubicasa5k | count | `data/cubicasa5k/high_quality-6694.png` | Kuinka monta kerrosta pohjapiirustuksessa on esitetty? | 2 | 2 | ✓ | high |
| q46 | swiss_dwellings | area | `data/swiss_dwellings/site412_bldg1050_floor4569.pdf` | What is the total net floor area in m² of all numbered indoor rooms on this floor (excluding balconies, loggias, terraces and other outdoor spaces, shafts and voids)? | 341.81 | 335 | ✓ | low |
| q47 | opening_design | area | `data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf` | On sheet A-801 of the Xcel Sports Complex drawings, what is the total area of rooms with floor finish F4 (resilient flooring)? | 4769 | 4769 | ✓ | high |
| q48 | ifc_models | count | `data/ifc_models/bsc-molio_molio-with-uris_plan-02-klder.pdf` | How many rooms (spaces) are shown on the floor plan of 'Kælder'? | 14 | 14 | ✓ | high |
| q49 | swiss_dwellings | count | `data/swiss_dwellings/site1211_bldg2381_floor11046.pdf` | How many rooms labelled 'Zimmer' are there on this floor? | 8 | 8 | ✓ | high |
| q50 | ifc_models | length | `data/ifc_models/kit-institute_ac20-institute-var-2_plan-05-dachgeschoss.pdf` | Wie lang sind alle Wände im Geschoss „Dachgeschoss“ zusammen, in Metern? | 133.6 | 53 | ✗ | low |

## 错题的模型推理

- **q06**（swiss_dwellings / area，标准答案 3.64，模型答 3.8）：The drawing has no area label for room 4.23, so I measured it against the scale bar. On the displayed image, 0 to 5 m on the bar spans about 238 px, which is about 47.6 px per metre. The inside of Bad 4.23 runs from about x 884 to 967 (about 1.75 m wide) and from about y 958 to 1060 (about 2.15 m deep). That gives roughly 1.75 × 2.15 ≈ 3.8 m². This is a pixel estimate, so it could be off by a few tenths of a square metre.
- **q09**（ifc_models / volume，标准答案 94.221，模型答 108）：The drawing does not show slab thickness, so this is a rough estimate. Using the 5 m scale bar (about 44 px/m), I measured the building outline with the shoelace formula and got about 540 m² gross. Subtracting the walls leaves about 520-540 m² of floor. I assumed a typical ground-floor slab of about 0.2 m, which gives about 108 m³.
- **q12**（swiss_dwellings / area，标准答案 104.9，模型答 102）：The drawing shows no room areas, so I measured each room against the 1:200 scale bar (10 m ≈ 238 px in the displayed image). From the room numbering, apartment D has rooms 3.23 to 3.31. Loggia 3.29 is outside, so it is left out. Rooms 3.21 and 3.22 are numbered with apartment C, so I left them out. This is a judgement call: by mirror symmetry with apartment A they could belong to D, which would raise the total. Approximate areas: living/dining 3.31 plus open kitchen 3.30 about 54.8 m²; corridor 3.25 about 9 m²; room 3.23 about 13.0 m²; room 3.28 about 12.3 m²; bathroom 3.24 about 3.5 m²; bathroom 3.27 about 7.5 m²; small room 3.26 about 2.2 m². The total is about 102 m².
- **q16**（cubicasa5k / count，标准答案 4，模型答 3）：I counted the thin-line openings in the exterior walls of apartment B3. There is one in the top wall of the living room (OH), next to the balcony door. There is one in the top wall of the bedroom (MH). There is one in the bottom wall of the kitchen (KK). The door arcs are left out of the count. The neighboring units, which the plan cuts off, are also left out.
- **q17**（cubicasa5k / count，标准答案 1，模型答 2）：There are no bathtubs on the plan. The answer of 2 counts two showers. One is the dashed shower area in the PE/WC washroom. The other is a shower-head symbol just right of the sauna (S), which seems to be across the wall in the neighbouring unit; it is in the image, so I counted it. Confidence is low: if only this apartment counts, or if the dashed square in PE/WC is really a floor drain, the answer would be 1.
- **q18**（ifc_models / count，标准答案 70，模型答 61）：My estimate is 61 walls, but I'm not confident in it. The 2.OG plan shows 10 terraced units side by side. I assumed each unit has 5 walls of its own. Those are the top exterior wall, the bottom exterior wall, the partition between the two Zimmer, the wall between the Zimmer row and the Flur/Bad row, and the Flur/Bad partition. That gives 50. I added 11 cross walls, counting the 9 party walls between units plus the 2 end walls. The total is 61. Two things could change this count. The exterior walls may be drawn as split segments at each party wall. The thick party walls may also be modelled as two walls each.
- **q19**（ifc_models / area，标准答案 3.01，模型答 2.6）：The drawing has no printed area for ESIK 72-2, so this is an estimate measured from the image. The scale bar is about 32 px per metre in the displayed image. The entry hall lies between the outer wall and the partition next to TUALETTRUUM 72-3, which are about 1.7 m apart. It runs from the edge of ELUTUBA+KÖÖK 72-1 to the corridor wall, about 1.5 m. That gives roughly 2.5–2.6 m².
- **q24**（swiss_dwellings / count，标准答案 18，模型答 15）：I counted the door swing arcs on the 4th-floor plan. I found 15: Zimmer 4.01, Zimmer 4.02, Bad 4.05, the double door between Küche 4.03 and Wohnen 4.04 (counted as one), the Korridor door of Küche 4.03, the entrance from the stair to apartment A, Bad 4.07, Bad 4.08, Bad 4.09, the entrance to apartment B, Küche 4.12, the balcony door from Wohnen 4.13, a second arc on Balkon 4.15, Zimmer 4.11 and Zimmer 4.14. Some arcs are small and hard to read, so the exact count is uncertain.
- **q26**（swiss_dwellings / count，标准答案 2，模型答 4）：I looked for the legend's Dusche symbol, a small square with a dot, in each bathroom of the six flats. I read showers in Bad 4.08 and Bad 4.47 (mirror images of each other), and probably in Bad 4.18 and Bad 4.37, which also mirror each other. The other bathrooms (4.09, 4.17, 4.20, 4.35, 4.36, 4.48) appear to hold bathtubs instead. Rooms 4.24 and 4.33 look like small WCs with only a toilet and washbasin. The symbols are very small at this resolution, so this count is uncertain.
- **q33**（cubicasa5k / count，标准答案 6，模型答 8）：I counted every door drawn with a swing arc (a door leaf). There are 6 in the main apartment: the balcony door at the top, the kitchen (KK) door, the bathroom (KH) door, the bedroom (MH) door, the closet door in the entrance hall (VH), and the main entrance door (PO-c1/4). There are 2 more entrance doors (PO-c1/4) at the bottom edge, which belong to the neighbouring units. That gives 8. If only the main apartment counts, the answer would be 6.
- **q34**（cubicasa5k / count，标准答案 10，模型答 6）：I counted the thin double-line openings in the outer walls: one in the living room (OH) top wall, one in the kitchen (KEITTIÖ) top wall, one in the washroom (PESU) top wall, one in the bottom wall of each of the two bedrooms (MH), and a small one in the toilet (WC) bottom wall. Some breaks in the wall are hard to make out, so the count is uncertain.
- **q37**（ifc_models / area，标准答案 4.34，模型答 4.1）：No area value is printed for room 01.28, so I measured it. The 5 m scale bar spans about 365 px, which is about 73 px per metre. The room's interior runs from about x=333 to x=537 px, which is about 2.8 m. It runs from about y=479 to y=586 px, which is about 1.47 m. That gives about 4.1 m².
- **q44**（ifc_models / area，标准答案 25.25，模型答 20）：The drawing has no area labels. I measured the room against the 5 m scale bar, which is about 162 px at display size, or about 32.4 px per metre. Room 52-1 is about 5.0 m wide inside, measured between the party walls along the building's long axis. The full unit is about 6.4 m deep inside. The entrance hall (ESIK 52-2) and toilet (TUALETTRUUM 52-3) take up about 2.5 m of that depth on the corridor side, which leaves about 3.9–4.0 m for the room. 5.0 m × 4.0 m gives roughly 20 m². This is an estimate from pixel measurements, not a value read off the drawing.
- **q50**（ifc_models / length，标准答案 133.6，模型答 53）：I measured the walls with the scale bar, where 5 m is about 215 px, or 43 px/m. The central stair core (Flur 3.OG Treppe) measures about 6.0 x 14.0 m on its outside faces. Its walls are about 0.3 m thick, which gives a centreline perimeter of about 2 x (5.7 + 13.7) = 38.8 m. The two end walls of the attic rooms (Dachboden-1 and Dachboden-2) are about 7.0 m each. The total is about 38.8 + 14 = 52.8 m, rounded to 53 m.
