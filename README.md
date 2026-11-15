# Blueprint Answers

**Ask a construction drawing a question. Models decide and point; code counts, sums and measures, and every number points back to the sheet.**

Entry for the Stoa challenge "Make Blueprints Talk", Junction 2026.

- Demo (stored answers, runs in the browser): https://ryenhails.github.io/blueprint-answers/
- Live demo, upload your own drawing: https://huggingface.co/spaces/Ryenhails/blueprint-answers
- Two-minute video: https://www.youtube.com/watch?v=AjSBnnXv8GM

## The idea

A frontier vision model given only the image and the question got 36 of 50 dataset questions right in our baseline
(`trial01_opus_image_only.md`). Its mistakes were almost never about understanding the drawing. It miscounted
and mismeasured. So the system gives the model the jobs it is good at and keeps the rest in code:

1. **Read every drawing once, into facts.** A parser is chosen from what the file is, never from where it came from.
   Each fact keeps its place on the sheet.
2. **Answer on the facts, with the cheapest step that can.** A stronger step runs only when the one before has no answer.
3. **Show the work.** Every answer comes with the shapes it rests on, the calculation, the steps that ran and what each cost.

| Drawing | How it is read | Facts |
|---|---|---|
| Vector plan PDF | code reads the PDF's own lines and text (`src/swiss_adapter.py`) | rooms, doors, windows, fixtures with exact geometry |
| BIM model (IFC), plans rendered from one | IfcOpenShell (`src/ifc_adapter.py`, `src/ifc_agent.py`) | one row per element: storey, type, materials, layers, quantities, properties |
| Schedule sheet | tables lifted from the text layer with cell positions (`src/table_adapter.py`) | tables with named columns |
| Floor-plan image | a trained detector, a segmentation network and a vision model; the middle count wins (`src/raster_ingest.py`) | counts with boxes and an agreement flag |
| Scanned floor plan | a model lists the rooms and points at each; code measures them and counts objects piece by piece (`src/scan_facts.py`, `src/measure.py`, `src/tile_count.py`) | the same floor-plan table, approximate |
| Any other sheet (piping, structural, details) | read per question; code measures and counts on request (`src/vision.py`) | none stored |

The ladder a question climbs (`src/system.py`):

| Step | What happens | Model | Typical cost |
|---|---|---|---|
| 0 Known question form | 177 question forms in a dozen languages map straight to a query; code computes | none | $0 |
| 1 Decision model | a new phrasing becomes a query, or is marked complex, or "not in the facts" | `gemini-3.1-flash-lite` | $0.0004 |
| 2 Model writes code on the facts | a few lines of Python over the fact tables, run in a sandbox, naming its evidence | `gemini-4-hydrogen`, `gemini-3.8-flash` for schedules | $0.01–0.06 |
| 3 Model reads the sheet | sees the image and the facts; may ask code to measure a room or count a symbol piece by piece | `gemini-4-hydrogen` | $0.01–0.20 |

Two things worth a look:

- **The model points, code measures** (`src/measure.py`). On a scan the model only says where: a spot inside a room, the
  corners of a zone, the two ends of a run. Code turns the printed scale and the sheet size into millimetres per pixel,
  checks that against the scale bar, fills the room out to its walls and returns net area and perimeter.
- **Counting piece by piece** (`src/tile_count.py`). A sheet shrunk to one image hides door swings. The sheet is cut into
  overlapping tiles, the model boxes the symbol in each, code merges the boxes, and every counted object is drawn on the sheet.

## Validation

1,234 questions with reference answers from the challenge dataset; 1,170 of them were never used while the system was built.
Full per-question results: `data/val_results.jsonl`. The same tables are in the demo under "Validation".

| Drawing kind | Questions | Result | API cost per question |
|---|---|---|---|
| Vector plan PDFs (4 languages) | 300 | 299 / 300 (99.7%) | $0 |
| BIM models, fixed question forms (8 languages) | 150 | 150 / 150 (100%) | $0 |
| Floor-plan images (English, Finnish) | 300 | 258 / 300 (86.0%) | $0 per question; about $0.036 once per image when it is first read |
| Scanned sheets (PNG twins of vector plans) | 220 | 145 / 220 (65.9%) | about $0.07 once per sheet to read it, then $0.012 |
| **All unseen numeric questions** | **970** | **852 / 970 (87.8%)** | **$0.016** including reading the scans |
| BIM models, open questions (IFC-Bench, graded by a model) | 200 | 98 correct, 37 partly correct, 65 wrong | $0.088 |
| Schedule sheets (all 64 dataset questions; prompts were tuned on them) | 64 | 46 / 48 numeric; text: 15 correct, 1 partly correct | $0.006 |

Scanned sheets by question form (20 questions each): apartments 20, rooms with a label 20, washbasins 18, toilets 17,
bathtubs 17, area of rooms with a label 16, area of one room 11, area of an apartment 10, total floor area 9, windows 7, doors 0.
Door counts on scans come out 10–35% low on every drawing. Areas printed on the sheet are read exactly; areas that have to be
measured are within 2% for about a quarter of the rooms at this resolution (checked room by room against the vector files of six sheets).
Before scans were read into the fact table (each question answered by reading the sheet), the same 220 questions scored 127.

The whole run cost $34 in API calls. For comparison, the development set (500 questions, used for tuning) scores 406 / 434 numeric.

How the set is built (`src/valset.py`, fixed seed): none of the questions is in the 500-question set used during
development; the floor-plan images are from the official CubiCasa5K test split, which the detector never saw; the scans are
the PNG twins of vector plans that were not used while the measuring step was tuned, asked the same questions with the
same reference answers. Counts must be exact; areas, lengths and volumes within 2%, as the dataset README says.

What the numbers do not cover:

- Open BIM questions have free-text reference answers. They were graded by a language model against the reference; no person checked the grades.
- The dataset has 64 schedule-sheet questions in total and the prompts were tuned on them, so they are reported apart.
- Scanned sheets are scored on one drawing style (the dataset's apartment plans as PNG, about 2,500 pixels wide, 17–34 mm per pixel). Doors on scans are undercounted on every drawing, and we did not find the cause. Measured areas of small rooms crowded with fixtures come out too small.
- The vector-plan parser relies on the drawing style of this dataset. Another CAD style needs new rules, or falls back to the scan route.
- The floor-plan detector and the median rule are fitted to CubiCasa5K drawings and its counting conventions; other images use the vision model alone.

## Run it

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # or put the key in api.txt (git-ignored)
export STOA_DATA=/path/to/junction-dataset   # the challenge dataset; default data/junction-dataset

python src/demo_server.py serve --gemini     # http://localhost:8765 : pick an example or upload a PDF, image or IFC
# hosting it for others (Docker, Hugging Face Space): see deploy/README.md
python src/system.py ask <file> "<question>" # one answer as JSON
```

Rebuild what is not in the repository (nothing derived from the dataset's drawings is committed):

```bash
python src/system.py build-cache             # question forms -> queries (src/plan_cache.json is committed)
python src/ifc_adapter.py build              # IFC summaries for the fixed question forms
python src/ifc_agent.py build                # element tables of the IFC models (minutes, several GB of memory)
python src/raster_ingest.py dev              # read the development images once with the vision model
python src/valset.py make && python src/valset.py run && python src/valset.py report
python src/demo_server.py export docs/index.html --gemini    # the published demo page
python video/make_video.py docs/index.html out/              # the pitch video, from the system's own outputs
```

The floor-plan detector (`src/cubi_yolo.py`, YOLO11s trained on the CubiCasa5K training split) and the segmentation
reader are optional. Without their stored readings every image is read by the vision model.

## Repository

| Path | What |
|---|---|
| `src/system.py` | entry point: routing, the ladder, the answer-path trace |
| `src/swiss_adapter.py`, `src/qa.py` | vector plans: parsing, queries, evidence |
| `src/ifc_adapter.py`, `src/ifc_agent.py` | BIM: fixed forms; open questions with generated code |
| `src/table_adapter.py` | schedule sheets |
| `src/raster_*.py`, `src/plan_facts.py`, `src/cubi_*.py` | floor-plan images: readers, the unified floor-plan fact table, detector training |
| `src/scan_facts.py`, `src/vision.py`, `src/measure.py`, `src/tile_count.py` | scans into fact tables; reading a sheet; measuring; counting |
| `src/llm_plan.py`, `src/agent.py`, `src/gemini_backend.py` | decision model, code-writing step, API calls and prices |
| `src/demo_server.py`, `src/demo.html` | the interface (live server and exported page) |
| `src/valset.py`, `src/scan_eval.py`, `src/evalkit.py` | validation sets and scoring |
| `data/val1200.jsonl`, `data/val_results.jsonl`, `data/dev500.jsonl` | the validation questions, every answer of the reported run, the development set |
| `docs/index.html` | the published demo page |
| `SYSTEM.md`, `technical_route.md`, `raster_*.md`, `handoff_raster_counts.md`, `trial01_opus_image_only.md` | working notes and experiment records (in Chinese) |
| `video/make_video.py` | builds the pitch video and deck from the system's outputs |

## Credits and licences

Code: AGPL-3.0 (the optional detector training uses Ultralytics YOLO, which is AGPL-3.0).

The drawings shown in the demo and the video come from the Junction construction-drawings dataset and keep their own
licences: Swiss Dwellings (Archilyse AG, CC BY 4.0), CubiCasa5K (CC BY-NC-SA 4.0), open IFC models (per file, see the
dataset's `LICENSE.md` files), Opening Design drawing sets (CC BY-SA 4.0). The dataset itself is not part of this
repository. IFC-Bench v2 questions are used for the open BIM questions.

Models: Google Gemini API (`gemini-3.1-flash-lite`, `gemini-3.8-flash`, `gemini-4-hydrogen`, `gemini-3.8-flash-tts` for the video narration).
