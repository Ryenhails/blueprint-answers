"""Demo front end.

  python demo_server.py serve [port] --gemini          all model steps through the Gemini API (key in GEMINI_API_KEY or api.txt)
  python demo_server.py serve [port] [--plan-model ID] [--table-model ID]
                                             live demo on http://localhost:8765. Without the model options it needs no GPU:
                                             known question forms, IFC and vector plans are answered exactly, and
                                             model-backed answers come from data/answer_cache.json
  python demo_server.py export <out.html>    one self-contained HTML file with precomputed examples (works offline)

The page (demo.html) is the same in both modes; in the exported file the data is embedded.
"""
import base64
import collections
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalkit  # noqa: E402
import ifc_adapter  # noqa: E402
import swiss_adapter  # noqa: E402
import system  # noqa: E402

ANSWER_CACHE = os.path.join(evalkit.ROOT, "data", "answer_cache.json")  # model-backed answers computed on a GPU node

EXAMPLES = [
    ("swiss", "Apartment floor, 6 units (French)", "data/swiss_dwellings/site492_bldg1175_floor5187.pdf"),
    ("swiss", "Apartment floor (German)", "data/swiss_dwellings/site1088_bldg2216_floor10132.pdf"),
    ("swiss", "Apartment floor (Italian)", "data/swiss_dwellings/site339_bldg710_floor3024.pdf"),
    ("ifc", "FZK-Haus, ground floor (BIM model)", "data/ifc_models/kit-fzk-haus_ac20-fzk-haus_plan-01-erdgeschoss.pdf"),
    ("ifc", "Esplanades housing, 2nd floor (BIM model)", "data/ifc_models/bsc-esplanades_1807-ep-ar-v18_plan-02-2-korrus.pdf"),
    ("raster", "Finnish flat, colour scan (raster)", "data/cubicasa5k/colorful-12002.png"),
    ("raster", "Finnish house, line drawing (raster)", "data/cubicasa5k/high_quality_architectural-13086.png"),
    ("table", "Sports complex, door and finish schedules", "data/opening_design/Sports_Complex/Sheets - Current Set/A-801 - Door and Finish Schedules & Wall Types.pdf"),
    ("table", "Cannery Trail Residences, schedules", "data/opening_design/CTR/Most Current Set/Full size sheets/A700 - SCHEDULES.pdf"),
]
SCAN = "data/swiss_dwellings/site1000_bldg2082_floor9385.png"
EXAMPLES += [
    ("raster", "Apartment floor as a scanned image (PNG)", SCAN),
    ("ifc", "Duplex apartment, IFC model (open questions)", "data/ifc_bench/duplex_arc.ifc"),
]
# Sheets the organisers posted as examples of harder questions. Shown in the live demo when the folder is there;
# left out of the exported page unless asked for (their licences are not known to us).
ORG = os.path.join(evalkit.ROOT, "extrainfo")
ORGANISER = [
    ("raster", "School plan, level 00 (organiser example, scan)", "uk_dfe_sen-dfe-xx-00-d-x-0201-special_school_plan_level_00-a-c01_p001.png",
     ["What is the floor area of the Multi-Purpose Hall, in m²?", "What is the total floor area of all Secondary Classrooms, in m²?",
      "How many linear metres of baseboard are needed for the Library and the Fitness Room, leaving out the door openings?", "How many doors and how many windows are on this floor?"]),
    ("raster", "Foundation and slab plan (organiser example, scan)", "us_aw_078-24-33_p056.png",
     ["What is the slab area by slab type (thickness and reinforcing bars), in m²?", "What is the total length of trenches shown on this plan, in metres?"]),
    ("raster", "Piping plan (organiser example, scan)", "us_aw_078-24-28_march_2020_p096.png",
     ["How many fuelling pumps and check valves are shown, and what are the sizes of the suction header and the pump discharge header?", "How many pressure gauges (PG) are shown on the piping plan?"]),
    ("raster", "Dormitory, second floor (organiser example, scan)", "us_daf_dorm_sd_drawings_april_2026_p003.png",
     ["What is the floor area of each wing and of the common space, and how many dorm rooms are on this floor?"]),
]
CUSTOM = {
    "data/ifc_bench/duplex_arc.ifc": ["What type of insulation is used in the external walls, and how thick is it?", "Which room in the building is the largest?",
                                      "Which door types are installed in the building, with their quantities?", "Combien de chambres y a-t-il, et quelle est leur surface totale ?"],
    SCAN: ["What is the net floor area of room 2.16, which has no area printed?", "Which apartment is the largest, and how big is it?"],
}
WITH_ORGANISER = True
EXTRA = os.path.join(evalkit.ROOT, "data", "demo_extra.json")  # optional: more examples (e.g. raster) added by other modules

_q = None
_cache = None
BACKENDS = {}  # filled by `serve --plan-model X --table-model Y` when a GPU is available


def questions_for(path, limit=12):
    """Dataset questions for a drawing, spread over question forms and languages."""
    global _q
    if _q is None:
        _q = collections.defaultdict(list)
        for r in evalkit.load_questions():
            _q[r["path"]].append(r)
    seen, out = set(), []
    rows = _q.get(path, [])
    for r in rows:
        t = system.normalise_q(re.sub(r"\d+", "#", r["question"]))[0]
        if t in seen:
            continue
        seen.add(t)
        out.append({"q": r["question"], "gold": r["answer"], "unit": r["unit"], "type": r["type"]})
    step = max(1, len(out) // limit)
    return out[::step][:limit]


def answer_cache():
    global _cache
    if _cache is None:
        _cache = json.load(open(ANSWER_CACHE)) if os.path.exists(ANSWER_CACHE) else {}
    return _cache


def _full(path):
    return path if os.path.isabs(path) else os.path.join(evalkit.DATA, path)


def examples():
    ex = [{"kind": k, "title": t, "path": p, "questions": questions_for(p, 4 if "ifc_bench" in p else 12)} for k, t, p in EXAMPLES]
    for e in ex:
        if e["path"] == SCAN:  # the scan is asked the dataset questions of its vector twin: same gold answers
            e["questions"] = [q for q in questions_for(SCAN[:-4] + ".pdf", 40) if q["q"].split()[0] in ("How", "What")][:9]
            e["twin"] = SCAN[:-4] + ".pdf"
        e["questions"] += [{"q": q, "gold": None, "unit": None, "type": "other", "model": True} for q in CUSTOM.get(e["path"], [])]
    if WITH_ORGANISER:
        for k, t, name, qs in ORGANISER:
            if os.path.exists(os.path.join(ORG, name)):
                ex.append({"kind": k, "title": t, "path": os.path.join(ORG, name), "questions": [{"q": q, "gold": None, "unit": None, "type": "other", "model": True} for q in qs]})
    for key, v in answer_cache().items():  # questions that were answered by a model (paraphrases etc.)
        p, q = key.split("\n", 1)
        for e in ex:
            if e["path"] == p and not any(x["q"] == q for x in e["questions"]):
                e["questions"].append({"q": q, "gold": v.get("gold"), "unit": v.get("unit"), "type": v.get("type"), "model": True})
    if os.path.exists(EXTRA):
        ex += json.load(open(EXTRA)).get("examples", [])
    return ex


def _b64(data, mime):
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())


def drawing(path, width=1700, page=1):
    """-> {"mode": "image", "clip": [x0,y0,x1,y1], "image": dataURL} or {"mode": "svg", "svg": text}.
    Evidence coordinates returned by ask() are in the same coordinate system as `clip`."""
    import fitz

    if os.path.exists(EXTRA):
        extra = json.load(open(EXTRA)).get("drawings", {})
        if path in extra:
            return extra[path]
    full = _full(path)
    kind = system.kind_of(path)
    if kind == "ifc":
        svg = re.sub(r"\.(pdf|png)$", ".svg", full)
        if svg.endswith(".svg") and os.path.exists(svg):
            return {"mode": "svg", "svg": open(svg, encoding="utf-8").read()}
    if path.lower().endswith(".pdf"):
        doc = fitz.open(full)
        n_pages, page_no = len(doc), min(max(1, int(page)), len(doc))
        page = doc[page_no - 1]
        clip = page.rect
        if kind == "swiss":
            words = page.get_text("words")
            leg = [w[1] for w in words if swiss_adapter.LEGEND_RE.match(w[4])]
            ytop = (min(leg) - 12) if leg else page.rect.height * 0.82
            xs = [w for w in words if w[1] < ytop]
            clip = fitz.Rect(min(w[0] for w in xs) - 50, min(w[1] for w in xs) - 50, max(w[2] for w in xs) + 50, max(w[3] for w in xs) + 50) & page.rect
        z = width / clip.width
        pix = page.get_pixmap(clip=clip, matrix=fitz.Matrix(z, z))
        if kind == "table":
            return {"mode": "image", "clip": list(clip), "image": _b64(pix.tobytes("jpeg", jpg_quality=72), "image/jpeg"), "pages": n_pages, "page": page_no}
        return {"mode": "image", "clip": list(clip), "image": _b64(pix.tobytes("png"), "image/png"), "pages": n_pages, "page": page_no}
    if path.lower().endswith(system.RASTER):
        import io

        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(full)
        W, H = im.size
        if max(W, H) <= 2600 and path.lower().endswith((".png", ".jpg", ".jpeg")):
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            return {"mode": "image", "clip": [0, 0, W, H], "image": _b64(open(full, "rb").read(), mime)}
        im = im.convert("RGB")  # a full scanned sheet: shown smaller, coordinates stay those of the original
        im.thumbnail((3400, 3400), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return {"mode": "image", "clip": [0, 0, W, H], "image": _b64(buf.getvalue(), "image/jpeg")}
    return {"mode": "none"}


def ask(path, question, page=None):
    key = path + "\n" + question
    res = system.ask(path, question, page=page, **BACKENDS)
    if res.get("answer") is None and key in answer_cache():
        res = dict(answer_cache()[key])
        res["precomputed"] = True
    for e in res.get("evidence", []):
        if e.get("bbox") is not None:
            e["bbox"] = [round(float(v), 2) for v in e["bbox"]]
        if e.get("poly"):
            e["poly"] = [[round(float(x), 2), round(float(y), 2)] for x, y in e["poly"]]
    gpath = SCAN[:-4] + ".pdf" if path == SCAN else path
    try:
        gold = next((x for x in questions_for(gpath, 10 ** 6) if x["q"] == question), None)
    except OSError:  # the dataset is not on this machine (hosted demo): no reference answers
        gold = None
    path = gpath
    if gold is None and _q is not None:
        gold = next(({"gold": r["answer"], "type": r["type"]} for r in _q.get(path, []) if r["question"] == question), None)
    if gold:
        res["gold"] = gold["gold"]
        res["correct"] = None if gold["type"] == "text" else evalkit.is_correct(res["answer"], gold["gold"], gold["type"]) if res.get("answer") is not None else False
    elif res.get("gold") is not None and res.get("type"):  # a re-phrased question: compared with the dataset question it restates
        res["correct"] = evalkit.is_correct(res["answer"], res["gold"], res["type"]) if res.get("answer") is not None else False
    return json.loads(json.dumps(res, ensure_ascii=False, default=str))


def facts(path, page=1):
    """What the system read from a drawing, as tables for the page: groups of rows, each row with its place on the sheet.
    Reads the drawing now if it has not been read yet (this is the one-time ingestion)."""
    full = _full(path)
    kind = system.kind_of(path)
    if full.lower().endswith(".pdf") and int(page) > 1:
        full = system.one_page(full, int(page))
    out = {"kind": kind, "summary": [], "groups": [], "note": "", "cost_usd": 0.0}

    def rows_of(items, cells, geo, limit=400):
        return [{"cells": cells(x), "bbox": (geo.get(x["id"]) or {}).get("bbox"), "poly": (geo.get(x["id"]) or {}).get("poly")} for x in items[:limit]]

    if kind == "ifc":
        import ifc_agent

        t = ifc_agent.tables_for(ifc_adapter.model_path(full) if not full.lower().endswith(".ifc") else full)
        by = collections.Counter(r["class"] for r in t["elements"])
        out["summary"] = [["elements", len(t["elements"])], ["storeys", len(t["model"]["storeys"])], ["classes", len(by)]]
        out["note"] = "Building model: every element is one row with its storey, type, materials, quantities and properties. Exact."
        out["groups"] = [{"name": "Elements by class", "columns": ["class", "elements"], "rows": [{"cells": [k, v]} for k, v in by.most_common(40)]},
                         {"name": "Storeys", "columns": ["storey", "elevation m"], "rows": [{"cells": [s["name"], s["elevation_m"]]} for s in t["model"]["storeys"]]}]
        return out
    if kind == "table":
        import table_adapter

        raw = table_adapter.parse_safe(full)
        ts = table_adapter.structured(raw)
        out["summary"] = [["tables", len(ts)], ["rows", sum(len(t["rows"]) for t in ts)]]
        out["note"] = "Schedules lifted from the PDF's text layer. Exact text; anything drawn on the sheet is read per question." if ts else ("Table extraction took too long on this sheet (very many vector paths), so it is read per question." if raw.get("skipped") else "No schedules on this sheet: it is read per question, with code measuring and counting.")
        for ti, t in enumerate(ts[:12]):
            bbs = raw["tables"][ti].get("row_bboxes", [])
            out["groups"].append({"name": (t.get("title") or "Table %d" % (ti + 1))[:60], "columns": [str(c)[:22] for c in t["columns"][:8]],
                                  "rows": [{"cells": [str(r.get(c, ""))[:40] for c in t["columns"][:8]], "bbox": bbs[j] if j < len(bbs) else None} for r, j in list(zip(t["rows"], t["src"]))[:300]]})
        return out
    tables = geo = None
    if kind == "swiss":
        import agent
        import qa

        tables, geo = agent.view(qa.facts_for(full))
        out["note"] = "Vector plan: parsed by code from the PDF's own lines. Geometry and areas are exact."
    elif kind == "raster" and system.big_sheet(full):
        import scan_facts

        rec, fresh = scan_facts.ensure(full)
        out["cost_usd"] = rec.get("usd", 0.0) if fresh else 0.0
        if "tables" in rec:
            tables, geo = rec["tables"], rec["geo"]
            out["note"] = "Scanned sheet: a model listed the rooms, code measured them (%s) and counted objects piece by piece. Approximate. Read once for $%.2f." % (rec.get("how", ""), rec.get("usd", 0))
        else:
            out["note"] = "%s. No fact table: this sheet is read per question, with code measuring and counting." % rec["outlier"].capitalize()
    elif kind == "raster":
        import plan_facts

        tables, geo, tok, usd = plan_facts.image_view(full)
        out["cost_usd"] = usd
        out["note"] = "Floor-plan image: rooms listed by a model, objects from the image readers. Approximate."
    if tables:
        rooms = tables["rooms"]
        fx = [k for k in ("toilets", "washbasins", "sinks", "bathtubs", "showers", "showers_baths") if any(k in r for r in rooms)]
        out["summary"] = [["rooms", len(rooms)], ["doors", len(tables["doors"])], ["windows", len(tables["windows"])], ["fixtures", len(tables["fixtures"])]]
        num = lambda v: "" if v is None else ("%.2f" % v if isinstance(v, float) else str(v))
        out["groups"] = [
            {"name": "Rooms", "columns": ["no.", "label", "m²", "unit", "doors", "win."] + [k[:6] for k in fx],
             "rows": rows_of(rooms, lambda r: [r.get("number") or r["id"], r.get("label") or r.get("type") or "", num(r.get("area_m2")), r.get("apartment") or "", len(r["doors"]), len(r["windows"])] + [r.get(k, 0) for k in fx], geo)},
            {"name": "Doors", "columns": ["id", "rooms it connects"], "rows": rows_of(tables["doors"], lambda d: [d["id"], ", ".join(d["rooms"])], geo)},
            {"name": "Windows", "columns": ["id", "room"], "rows": rows_of(tables["windows"], lambda w: [w["id"], ", ".join(w["rooms"])], geo)},
            {"name": "Fixtures", "columns": ["id", "type", "room"], "rows": rows_of(tables["fixtures"], lambda x: [x["id"], x["type"], x.get("room") or ""], geo)}]
    return json.loads(json.dumps(out, default=float))


def gemini_backends():
    """Every model-backed step through the Google API; no GPU needed."""
    import agent
    import llm_plan
    import raster_ingest
    import table_adapter
    import vision

    BACKENDS["llm"] = llm_plan.make_llm(os.environ.get("STOA_PLAN_MODEL", "gemini-3.1-flash-lite"))  # decides: simple query, complex, or not in the plan
    BACKENDS["agent"] = agent.make_agent()  # complex questions: code on the fact tables
    BACKENDS["table_llm"] = table_adapter.make_table_llm(os.environ.get("STOA_TABLE_MODEL", "gemini-3.8-flash"))
    BACKENDS["raster"] = raster_ingest.answer  # images: one read per drawing, then answers from the stored facts
    BACKENDS["vision"] = vision.answer  # last resort: read the sheet for this one question


_updir = None


def upload(name, data):
    """Keep an uploaded drawing in a temporary folder (outside the project) and describe it like an example."""
    global _updir
    import tempfile
    from urllib.parse import unquote

    name = re.sub(r"[^A-Za-z0-9._ -]", "_", os.path.basename(unquote(name)))[-80:]
    if not name.lower().endswith((".pdf", ".ifc") + system.RASTER):
        raise ValueError("Upload a PDF, an image (PNG, JPG, TIFF) or an IFC model.")
    if len(data) > 40 * 1024 * 1024:
        raise ValueError("The file is larger than 40 MB.")
    if _updir is None:
        _updir = tempfile.mkdtemp(prefix="stoa_upload_")
    full = os.path.join(_updir, name)
    with open(full, "wb") as f:
        f.write(data)
    return {"kind": system.kind_of(full), "title": name, "path": full, "questions": [], "uploaded": True}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json; charset=utf-8", code=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        u = urlparse(self.path)
        try:
            if u.path == "/":
                self._send(page_html(HOSTED, live=HOSTED is not None), "text/html; charset=utf-8")
            elif u.path == "/api/examples":
                self._send(json.dumps(examples(), ensure_ascii=False))
            elif u.path == "/api/facts":
                p = parse_qs(u.query)["path"][0]
                if HOSTED is not None and p in (HOSTED.get("facts") or {}):
                    self._send(json.dumps(HOSTED["facts"][p], ensure_ascii=False))
                elif HOSTED is not None and (ACCESS_CODE and self.headers.get("X-Access-Code", "") != ACCESS_CODE):
                    self._send(json.dumps({"error": "Enter the access code to read an uploaded drawing."}))
                elif SPENT[0] >= BUDGET:
                    self._send(json.dumps({"error": "This demo has used up its API budget for now."}))
                elif limited(self, "ask"):
                    self._send(json.dumps({"error": "Too many requests from this address in the last hour. Try again later."}))
                else:
                    f = facts(p, page=int(parse_qs(u.query).get("page", ["1"])[0]))
                    SPENT[0] += f.get("cost_usd") or 0
                    self._send(json.dumps(f, ensure_ascii=False))
            elif u.path == "/api/validation":
                self._send(json.dumps(validation(), ensure_ascii=False))
            elif u.path == "/api/drawing":
                self._send(json.dumps(drawing(parse_qs(u.query)["path"][0], page=int(parse_qs(u.query).get("page", ["1"])[0]))))
            else:
                self._send("{}", code=404)
        except Exception as e:
            self._send(json.dumps({"error": repr(e)}), code=500)

    def do_POST(self):
        if ACCESS_CODE and self.headers.get("X-Access-Code", "") != ACCESS_CODE:
            self._send(json.dumps({"error": "Enter the access code (it is in the submission notes) to upload or ask live.", "answer": None}), code=200)
            return
        if SPENT[0] >= BUDGET:
            self._send(json.dumps({"error": "This demo has used up its API budget for now. The stored examples still work.", "answer": None}), code=200)
            return
        wait = limited(self, "upload" if self.path.startswith("/api/upload") else "ask")
        if wait:
            self._send(json.dumps({"error": wait, "answer": None}))
            return
        if self.path.startswith("/api/upload"):
            try:
                self._send(json.dumps(upload(self.headers.get("X-Filename", "upload"), self.rfile.read(int(self.headers.get("Content-Length", 0)))), ensure_ascii=False))
            except Exception as e:
                self._send(json.dumps({"error": str(e)}), code=400)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if HOSTED is not None and not (_updir and os.path.realpath(str(body["path"])).startswith(os.path.realpath(_updir) + os.sep)):
                raise ValueError("live questions are answered for uploaded drawings")
            res = ask(body["path"], body["question"], page=body.get("page"))
            SPENT[0] += res.get("cost_usd") or 0
            self._send(json.dumps(res, ensure_ascii=False))
        except Exception as e:
            self._send(json.dumps({"error": repr(e)}), code=500)


def validation():
    try:
        import valset

        return valset.summary()
    except Exception:
        return None


HOSTED = None  # hosted mode: the examples of an exported page (stored answers) plus live answers for uploads
ACCESS_CODE = os.environ.get("STOA_ACCESS_CODE", "")
BUDGET = float(os.environ.get("STOA_BUDGET_USD", "50"))  # the server stops calling models after this much API spend
SPENT = [0.0]
LIMIT = {"ask": (int(os.environ.get("STOA_ASKS_PER_HOUR", "800")), 3600), "upload": (int(os.environ.get("STOA_UPLOADS_PER_HOUR", "200")), 3600)}  # per visitor address
_seen = collections.defaultdict(list)
_rl = __import__("threading").Lock()


def limited(handler, kind):
    """Rate limit per visitor (hosted mode only). -> message when the visitor has to wait, else None."""
    if HOSTED is None:
        return None
    import time

    who = (handler.headers.get("X-Forwarded-For") or handler.client_address[0]).split(",")[0].strip()
    n, window = LIMIT[kind]
    now = time.time()
    with _rl:
        q = _seen[(who, kind)] = [t for t in _seen[(who, kind)] if now - t < window]
        if len(q) >= n:
            return "This demo allows %d %s per hour from one address. Try again in %d minutes; the stored examples keep working." % (n, "uploads" if kind == "upload" else "live questions", int((window - (now - q[0])) / 60) + 1)
        q.append(now)
    return None


def page_html(static, live=False):
    html = open(os.path.join(HERE, "demo.html"), encoding="utf-8").read()
    payload = "null" if static is None else json.dumps(static, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("/*__STATIC__*/null", payload)
    if live:
        html = html.replace("/*__LIVE__*/false", "true").replace("/*__CODE__*/false", "true" if ACCESS_CODE else "false")
    if static is None or live:  # served by this program: needs a full document
        html = "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'></head><body>" + html + "</body></html>"
    return html


def export(out):
    ex = examples()
    static = {"examples": ex, "drawings": {}, "answers": {}, "facts": {}, "validation": validation()}
    old = {}
    if "--reuse" in sys.argv:  # keep the answers of an earlier export (they cost API calls); only missing or empty ones are asked again
        h = open(sys.argv[sys.argv.index("--reuse") + 1], encoding="utf-8").read()
        i, j = h.index("const STATIC = ") + 15, h.index(";\nconst LIVE" if ";\nconst LIVE" in h else ";\nconst $ =")
        old = {k: v for k, v in json.loads(h[i:j].replace("<\\/", "</"))["answers"].items() if v.get("answer") is not None and v.get("trace")
               and not (v.get("kind") == "raster" and v.get("plan_source") == "vision" and "--refresh-scans" in sys.argv)}
    for e in ex:
        static["drawings"][e["path"]] = drawing(e["path"], width=1500)
        try:
            static["facts"][e["path"]] = facts(e["path"])
        except Exception as err:
            static["facts"][e["path"]] = {"kind": e["kind"], "summary": [], "groups": [], "note": "The facts of this drawing could not be listed (%s)." % str(err)[:80]}
        for q in e["questions"]:
            key = e["path"] + "\n" + q["q"]
            static["answers"][key] = old[key] if key in old else ask(e["path"], q["q"])
    html = page_html(static)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    n = sum(len(e["questions"]) for e in ex)
    ok = sum(1 for a in static["answers"].values() if a.get("correct"))
    usd = sum(a.get("cost_usd") or 0 for a in static["answers"].values())
    print("API cost of the answers in this export: $%.2f" % usd)
    print("wrote %s: %.1f MB, %d drawings, %d questions (%d match the dataset answer)" % (out, os.path.getsize(out) / 1e6, len(ex), n, ok))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["serve"]:
        port = int(a[1]) if len(a) > 1 and a[1].isdigit() else 8765
        if "--plan-model" in a:
            import llm_plan
            BACKENDS["llm"] = llm_plan.make_llm(a[a.index("--plan-model") + 1])
        if "--gemini" in a:
            gemini_backends()
        if "--table-model" in a:
            import table_adapter
            BACKENDS["table_llm"] = table_adapter.make_table_llm(a[a.index("--table-model") + 1])
        if "raster" not in BACKENDS:
            try:
                import raster_backend
                BACKENDS["raster"] = raster_backend.answer
            except Exception:
                pass
        host = "127.0.0.1"
        if "--hosted" in a:  # python demo_server.py serve --gemini --hosted docs/index.html : public server for judges
            h = open(a[a.index("--hosted") + 1], encoding="utf-8").read()
            i, j = h.index("const STATIC = ") + 15, h.index(";\nconst LIVE")
            HOSTED = json.loads(h[i:j].replace("<\\/", "</"))
            host, port = "0.0.0.0", int(os.environ.get("PORT", port))
        print("demo on http://%s:%d  (Ctrl-C to stop)" % (host, port), flush=True)
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    elif a[:1] == ["export"] and len(a) >= 2:
        WITH_ORGANISER = "--with-organiser" in a
        if "--gemini" in a:
            gemini_backends()
        export(a[1])
    else:
        print(__doc__)
