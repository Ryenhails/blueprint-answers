"""Unified entry point: any drawing + any question -> answer, evidence, calculation, cost.

  python system.py build-cache            add the IFC question forms to src/plan_cache.json
  python system.py ask <path> "<question>"
  python system.py dev <preds.jsonl> [--gemini]   answer the whole dev set with whatever backends are available, then score
  python system.py eval-ifc               every ifc_models question, all languages

Routing (by what the file is, not by which dataset folder it sits in):
  swiss   vector apartment plan with a symbol legend      -> swiss_adapter + qa      (exact geometry)
  ifc     an IFC model, or a plan rendered from one       -> ifc_adapter             (model query)
  table   a sheet that is mostly schedules                -> table_adapter + model   (generated code on tables)
  raster  scanned / raster drawings                       -> raster backend          (plugged in separately)
"""
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import ifc_adapter  # noqa: E402
import qa  # noqa: E402

LET = qa.LET
_QUOTED = re.compile(r"«\s*(.*?)\s*»|„(.*?)[“”]|“(.*?)”|\"([^\"]*)\"|(?<![%s])'(.+?)'(?=[\s,?.)!:;]|$)" % LET)


def normalise_q(q):
    """-> (template with <Q> for every quoted string, [quoted values])"""
    vals = []

    def f(m):
        vals.append(next(g for g in m.groups() if g is not None))
        return "<Q>"

    t = _QUOTED.sub(f, q)
    return re.sub(r"\s+", " ", t).strip(), vals


RASTER = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")
_pages = {}


def pick_page(full, question):
    """A drawing set with several sheets: one cheap call picks the sheet the question is about.
    -> (path of a one-page PDF, page number from 1, tokens, usd); the file itself when it has one page."""
    import fitz

    doc = fitz.open(full)
    if len(doc) <= 1:
        return full, 1, 0, 0.0
    lines = []
    for i, pg in enumerate(doc):
        if i >= 80:
            break
        t = re.sub(r"\s+", " ", pg.get_text()).strip()
        lines.append("page %d: %s" % (i + 1, (t[:200] + " … " + t[-300:]) if len(t) > 500 else t or "(no text layer)"))
    n, tok, usd = 1, 0, 0.0
    try:
        import gemini_backend as gb

        g = gb.Gemini(gb.DEFAULT_PLAN)
        texts, a, b = g.generate([[{"role": "system", "content": "A PDF holds several sheets of a construction drawing set. For each page you get the start and the end of its text. "
            "Reply with the number of the one page that most likely answers the question (the sheet it names, or the sheet whose content fits). Reply with the number only."},
            {"role": "user", "content": "\n".join(lines) + "\n\nQuestion: " + question}]], 10)
        m = re.search(r"\d+", texts[0] or "")
        if m and 1 <= int(m.group(0)) <= len(doc):
            n = int(m.group(0))
        tok, usd = a + b, g.usd
    except Exception:
        pass
    key = (full, n)
    if key not in _pages:
        import tempfile

        one = fitz.open()
        one.insert_pdf(doc, from_page=n - 1, to_page=n - 1)
        fd, out = tempfile.mkstemp(prefix="stoa_page_", suffix="_p%d.pdf" % n)
        os.close(fd)
        one.save(out)
        _pages[key] = out
    return _pages[key], n, tok, usd


def big_sheet(full):
    """A scanned sheet (thousands of pixels across) rather than a small picture of one plan."""
    try:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        if max(Image.open(full).size) <= 2200:
            return False
        import raster_detector
        import raster_ingest

        return raster_ingest._key(full) not in raster_detector.load()  # dataset floor plans keep their stored readings
    except Exception:
        return False


def same_drawing(full):
    """CAD and vector files are read through the same drawing in a format we parse (the dataset keeps them side by side)."""
    stem = os.path.splitext(full)[0]
    for ext in (".pdf", ".png", ".jpg", ".jpeg"):
        if os.path.exists(stem + ext):
            return stem + ext
    return None


def kind_of(path):
    p = path.lower()
    if p.endswith(".ifc"):
        return "ifc"
    if p.endswith(RASTER):
        return "raster"
    if p.endswith((".pdf", ".svg")):
        if ifc_adapter.facts_for(ifc_adapter.model_path(path)) and re.search(r"_plan(-[^/]*)?\.(pdf|svg)$", p):
            return "ifc"
        if p.endswith(".pdf"):
            import fitz

            full = path if os.path.isabs(path) else os.path.join(evalkit.DATA, path)
            page = fitz.open(full)[0]
            txt = page.get_text()
            if "Swiss Dwellings" in txt and re.search(r"Légende|Legende|Legend|Legenda", txt):
                return "swiss"
            return "table"
    return "unknown"


# ---- IFC plans: learned from the English twin of every translated question ------------------


def _ifc_sig(plan):
    return json.dumps(plan, sort_keys=True)


def build_ifc_cache():
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "ifc_models"]
    cache = json.load(open(qa.CACHE_PATH)) if os.path.exists(qa.CACHE_PATH) else dict(qa.SEED)
    votes = collections.defaultdict(collections.Counter)
    n_en = 0
    for a, b in zip([None] + rows, rows):
        plan, _ = ifc_adapter.en_plan(b["question"])
        t, _ = normalise_q(b["question"])
        if plan is not None:
            cache[t] = dict(plan, domain="ifc")
            n_en += 1
            continue
        if a is None:
            continue
        pa, _ = ifc_adapter.en_plan(a["question"])
        if pa is not None and a["path"] == b["path"] and a["answer"] == b["answer"]:
            votes[t][_ifc_sig(pa)] += 1
    n_tr = 0
    for t, c in votes.items():
        sig, n = c.most_common(1)[0]
        if n / sum(c.values()) > 0.95:
            cache[t] = dict(json.loads(sig), domain="ifc")
            n_tr += 1
    with open(qa.CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    qa._cache = None
    print("plan cache now has %d templates (ifc: %d English question instances seen, %d translated templates learned)" % (len(cache), n_en, n_tr))


def _ifc_slots(vals, facts):
    """Assign quoted strings to slots by what they match in the model (order differs between languages)."""
    storeys = {ifc_adapter._n(s["name"]): s["name"] for s in facts["storeys"]}
    rooms = {ifc_adapter._n(ifc_adapter.space_display(s)) for s in facts["spaces"]}
    slots = {}
    for v in vals:
        n = ifc_adapter._n(v)
        if n in storeys and "storey" not in slots:
            slots["storey"] = storeys[n]
        elif n in rooms and "room" not in slots:
            slots["room"] = v
    return slots


def _default_raster():
    try:
        import raster_backend

        return raster_backend.answer
    except Exception:
        return None


STEP = ["Known question form", "Decision model", "Model writes code on the facts", "Model reads the sheet"]


def _step(res, level, detail=None, answered=None):
    """Record what one rung of the ladder did, for the answer-path display."""
    tr = res.setdefault("trace", [])
    spent = sum(t["cost_usd"] for t in tr)
    tr.append({"level": level, "name": STEP[level], "answered": (res.get("answer") is not None) if answered is None else answered,
               "detail": detail, "cost_usd": round(max(0.0, res.get("cost_usd", 0.0) - spent), 6)})


def _first(res, known, decision="it is not a single count or sum"):
    """Steps of the first route's result: a known form, or the decision model, or straight to code."""
    src = res.get("plan_source")
    if src in ("cache", "ingested", "keyword"):
        _step(res, 0, known)
    else:
        _step(res, 0, "not a question form seen before", answered=False)
        if src == "llm":
            op = (res.get("plan") or {}).get("op") or (res.get("plan") or {}).get("kind")
            _step(res, 1, "turned into the query '%s'; code computed the answer" % op if res.get("answer") is not None else "judged '%s': %s" % (op, decision))
        elif src == "code":
            _step(res, 2, res.get("note"))


def _escalate(res, more):
    """Replace a missing answer with the next route's result, keeping the running token and cost totals."""
    tok, usd = res.get("llm_tokens", 0) + more.get("llm_tokens", 0), res.get("cost_usd", 0.0) + more.get("cost_usd", 0.0)
    for k in ("note", "calc", "plan", "plan_source", "evidence", "reliability"):
        res.pop(k, None)
    res.update(more)
    res.update({"llm_tokens": tok, "cost_usd": round(usd, 6)})


def facts_text(kind, full):
    """The parsed facts of a drawing as text, for the sheet-reading step. Uses only what is already parsed or stored.
    -> (text or None, "exact" | "approximate")"""
    try:
        if kind == "swiss":
            import agent

            return agent.describe(agent.view(qa.facts_for(full))[0]), "exact"
        if kind == "table":
            import table_adapter

            t = table_adapter.render_tables(table_adapter.parse(full))
            return (t, "exact") if t.strip() else (None, "exact")
        if kind == "raster":
            import agent
            import plan_facts
            import raster_ingest

            if big_sheet(full):
                import scan_facts

                rec = raster_ingest._load(scan_facts.STORE).get(raster_ingest._key(full))
                return (agent.describe(rec["tables"]), "approximate") if rec and "tables" in rec else (None, "exact")
            if raster_ingest._key(full) in raster_ingest._load(plan_facts.STORE):
                return agent.describe(plan_facts.image_view(full)[0]), "approximate"
    except Exception:
        pass
    return None, "exact"


def ask(path, question, llm=None, table_llm=None, raster=None, agent=None, vision=None):
    """path: relative to the dataset root (as in questions.jsonl) or absolute.

    Cheapest route first, a stronger one only when the cheaper one has no answer:
      known question form (no model) -> model writes code on the parsed facts (agent) -> model reads the sheet (vision)."""
    t0 = time.time()
    if raster is None:
        raster = _default_raster()
    full = path if os.path.isabs(path) else os.path.join(evalkit.DATA, path)
    extra, ptok, pusd = {}, 0, 0.0
    if full.lower().endswith((".dxf", ".dwg")) or (full.lower().endswith(".svg") and kind_of(path) != "ifc"):
        twin = same_drawing(full)
        if twin is None and not full.lower().endswith(".svg"):
            return {"answer": None, "kind": "unknown", "evidence": [], "llm_tokens": 0, "cost_usd": 0.0, "seconds": 0.0,
                    "note": "CAD files (DWG, DXF) are not read directly. Export the sheet to PDF or an image and ask again."}
        if twin:
            extra["read_from"] = os.path.basename(twin)
            path = full = twin
    if full.lower().endswith(".pdf") and vision is not None:  # several sheets in one PDF: work on the sheet the question is about
        full, page, ptok, pusd = pick_page(full, question)
        if page > 1 or ptok:
            path, extra["page"] = full, page
    kind = kind_of(path)
    res = {"answer": None, "kind": kind, "evidence": [], "calc": None, "plan": None, "plan_source": None, "llm_tokens": 0, "cost_usd": 0.0}
    res.update(extra)
    if kind == "swiss":
        res.update(qa.ask(full, question, llm=llm))
        # no cached form and the cheap model could not express it as one count / one area: write code on the fact tables
        _first(res, "query from the plan cache, computed by code on the vector geometry")
        if res["answer"] is None and agent is not None and (llm is None or (res.get("plan") or {}).get("op") in (None, "complex")):
            _escalate(res, agent(full, question))
            _step(res, 2, res.get("note"))
        res["kind"] = "swiss"
    elif kind == "ifc":
        facts = ifc_adapter.facts_for(ifc_adapter.model_path(path))
        if qa._cache is None:
            qa.plan_for("")  # loads the cache
        t, vals = normalise_q(question)
        plan = qa._cache.get(t)
        if plan and plan.get("domain") == "ifc" and facts and "error" not in facts:
            slots = _ifc_slots(vals, facts)
            res.update(ifc_adapter.execute(plan, slots, facts))
            res.update({"plan": plan, "slots": slots, "plan_source": "cache"})
        _first(res, "query from the plan cache, computed by code on the building model")
        if res["answer"] is None and agent is not None:  # not one of the known forms: code on the model's element table
            ifc_full = os.path.join(evalkit.DATA, ifc_adapter.model_path(path)) if not os.path.isabs(path) else ifc_adapter.model_path(path)
            hint = "" if path.lower().endswith(".ifc") else "\n(The question is asked on a floor plan rendered from this model: file '%s'.)" % os.path.basename(path)
            _escalate(res, agent(ifc_full, question + hint))
            _step(res, 2, res.get("note") or "code on the model's element table")
        elif res["answer"] is None:
            res["note"] = "no plan for this question form" if facts else "model not parsed"
    elif kind == "table" and table_llm is not None:
        res.update(table_llm(full, question))
        if res.get("plan_source") == "code":
            _step(res, 2, "code on the schedules extracted from the sheet" if res["answer"] is not None else "the schedules do not hold this")
        else:
            _step(res, 2, "no schedules on this sheet", answered=False)
    elif kind == "raster" and big_sheet(full):
        # a full scanned sheet: read once into the floor-plan fact table; sheets that are not floor plans stay with the reading step
        import scan_facts

        try:
            rec, fresh = scan_facts.ensure(full)
        except Exception as e:
            rec, fresh = {"outlier": "the sheet could not be read into facts (%s)" % str(e)[:60], "usd": 0.0, "tokens": 0}, False
        if fresh:
            res.update({"cost_usd": rec["usd"], "llm_tokens": rec["tokens"]})
        if "tables" in rec:
            _step(res, 0, ("the scan was read into the floor-plan fact table just now" if fresh else "the scan's fact table is stored") + "; not a question form seen before", answered=False)
            if agent is not None:
                _escalate(res, agent(full, question))
                _step(res, 2, res.get("note") or "code on the fact table read from the scan")
        else:
            _step(res, 0, rec["outlier"] + ": no fact table, read per question", answered=False)
    elif kind == "raster" and raster is not None:
        res.update(raster(full, question))
        if res.get("kind_from") == "decision model":
            res["plan_source"] = "llm"
        _first(res, "count stored when the image was first read", "it needs more than one stored count")
        if res.get("answer") is not None and res.get("kind_from") == "decision model":
            res["trace"][-1]["detail"] = "recognised as the stored count '%s'" % (res.get("plan") or {}).get("kind")
        if res["answer"] is None and agent is not None and (res.get("plan") or {}).get("op") != "other":
            _escalate(res, agent(full, question))  # a complex question: code on the image's fact tables
            _step(res, 2, res.get("note"))
    else:
        res["note"] = "no backend available for drawing kind '%s' in this run" % kind
    a = res["answer"]
    if a in ("", [], {}) or (isinstance(a, str) and re.match(r"\s*(none|null|n/?a|unknown|not (found|available|specified|listed)\b.*)\s*$", a, re.I)):
        res["answer"] = None  # the code ran but found nothing: not an answer
    if res["answer"] is None and vision is not None and not full.lower().endswith(".ifc"):
        facts, trust = facts_text(kind, full)  # nothing computed an answer: the model reads the sheet, with the facts we have
        _escalate(res, vision(full, question, facts=facts, trust=trust))
        _step(res, 3, (res.get("plan") or {}).get("op"))
    res.update({"llm_tokens": res.get("llm_tokens", 0) + ptok, "cost_usd": round(res.get("cost_usd", 0.0) + pusd, 6), "seconds": round(time.time() - t0, 3)})
    return res


def eval_ifc():
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "ifc_models"]
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        res = ask(r["path"], r["question"])
        ok = res["answer"] is not None and evalkit.is_correct(res["answer"], r["answer"], r["type"])
        lang = "en" if re.match(r"(How|What) ", r["question"]) else "other"
        src = "file=" + r["path"].rsplit(".", 1)[-1]
        for k in ("ALL", "lang=" + lang, "type=" + r["type"], src, "plan=" + str(res["plan_source"])):
            agg[k][0] += ok
            agg[k][1] += 1
    for k, (a, b) in sorted(agg.items()):
        print("  %-16s %5d / %5d  %.1f%%" % (k, a, b, 100 * a / b))


def dev(pred_path, **backends):
    n = collections.Counter()
    cpath = os.path.join(evalkit.ROOT, "data", "answer_cache.json")
    cache = json.load(open(cpath)) if os.path.exists(cpath) else {}
    with open(evalkit.DEV) as f, open(pred_path, "w") as out:
        for l in f:
            r = json.loads(l)
            res = ask(r["path"], r["question"], **backends)
            if res["answer"] is None and cache.get(r["path"] + "\n" + r["question"], {}).get("answer") is not None:
                # model-backed routes: answers computed earlier on a GPU node (gpu_fill.py)
                c = cache[r["path"] + "\n" + r["question"]]
                res.update({"answer": c["answer"], "llm_tokens": c.get("llm_tokens", 0), "cost_usd": c.get("cost_usd", 0), "kind": res["kind"] + "(precomputed)"})
            n[res["kind"] + ("" if res["answer"] is not None else ":unanswered")] += 1
            ans = res["answer"]
            out.write(json.dumps({"id": r["id"], "answer": ans, "cost_usd": res.get("cost_usd", 0), "tokens": res.get("llm_tokens", 0)}, ensure_ascii=False, default=str) + "\n")
    print("routed:", dict(n))
    evalkit.score(pred_path)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["build-cache"]:
        qa.build_cache()
        build_ifc_cache()
    elif a[:1] == ["ask"] and len(a) == 3:
        r = ask(a[1], a[2])
        r["evidence"] = [{k: v for k, v in e.items() if k != "poly"} for e in r.get("evidence", [])][:8]
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    elif a[:1] == ["eval-ifc"]:
        eval_ifc()
    elif a[:1] == ["dev"] and len(a) >= 2:
        if "--gemini" in a:  # raster drawings answered from the facts stored at ingestion (reads any drawing not yet ingested)
            import raster_ingest
            dev(a[1], raster=raster_ingest.answer)
        else:
            dev(a[1])
    else:
        print(__doc__)
