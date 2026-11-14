"""Step 2: question -> query plan -> executed on the fact table -> answer + evidence.

  python qa.py build-cache                 learn the plan cache (src/plan_cache.json) from questions.jsonl
  python qa.py ask <pdf> "<question>" [out.png]
  python qa.py eval-swiss [n_files|all]    every swiss_dwellings question, all languages
  python qa.py dev <preds.jsonl>           answer the swiss part of dev500 and score it

Pipeline
  1. normalise the question into a template + slots (quoted label, room number, apartment letter)
  2. template -> plan: plan cache first (free); cache miss -> LLM backend (not configured yet)
  3. execute the plan with plain code on the fact table produced by swiss_adapter
  4. return answer, the entities used (with page coordinates), the calculation, and the cost
"""
import collections
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import swiss_adapter  # noqa: E402

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_cache.json")
WHITE, CIRC = (1.0, 1.0, 1.0), (0.96, 0.96, 0.96)
LET = "A-Za-zÀ-ÿ"

# ---- 1. normalisation -------------------------------------------------------------------------


def normalise(q):
    """-> (template, slots). Language independent: only punctuation / token shapes are used."""
    slots = {}

    def grab(name):
        def f(m):
            slots.setdefault(name, m.group(0))
            return "<%s>" % name
        return f

    t = re.sub(r"«.*?»|„.*?[“”]|“.*?”|\"[^\"]*\"", grab("Q"), q)
    t = re.sub(r"(?<![%s])'.*'(?![%s])" % (LET, LET), grab("Q"), t)
    t = re.sub(r"\b[A-Z]?\d+\.\d+\b", grab("R"), t)
    t = re.sub(r"(?<![%s<])[A-Z](?![%s>'’])" % (LET, LET), grab("L"), t)
    if "Q" in slots:
        slots["Q"] = re.sub(r"^[«„“\"'\s]+|[»“”\"'\s]+$", "", slots["Q"])
    return re.sub(r"\s+", " ", t).strip(), slots


# ---- 2. plans ---------------------------------------------------------------------------------

# The 12 query kinds, seeded from the English templates. Everything else is learned / translated.
SEED = {
    "How many rooms labelled <Q> are there on this floor?": {"op": "count", "entity": "room", "by": "label"},
    "How many apartments (residential units) are shown on this floor plan?": {"op": "count", "entity": "apartment"},
    "How many doors (interior doors and entrance doors together) are drawn on this floor plan?": {"op": "count", "entity": "door"},
    "How many windows are drawn on this floor plan?": {"op": "count", "entity": "window"},
    "How many toilets (WC) are drawn on this floor plan?": {"op": "count", "entity": "toilet"},
    "How many washbasins are drawn on this floor plan?": {"op": "count", "entity": "washbasin"},
    "How many bathtubs are drawn on this floor plan?": {"op": "count", "entity": "bathtub"},
    "How many showers are drawn on this floor plan?": {"op": "count", "entity": "shower"},
    "What is the net floor area of room <R> (<Q>) in m²?": {"op": "area", "entity": "room", "by": "number"},
    "What is the total net area in m² of all rooms labelled <Q> on this floor?": {"op": "area", "entity": "room", "by": "label"},
    "What is the net floor area of apartment <L> in m², summing all its rooms except balconies, loggias, terraces and other outdoor spaces?": {"op": "area", "entity": "room", "by": "apartment"},
    "What is the total net floor area in m² of all numbered indoor rooms on this floor (excluding balconies, loggias, terraces and other outdoor spaces, shafts and voids)?": {"op": "area", "entity": "room", "by": "floor"},
}


def build_cache():
    """Translated questions directly follow their English twin (same drawing, same answer):
    use that to map every non-English template to the plan of its English template."""
    rows = evalkit.load_questions()
    votes = collections.defaultdict(collections.Counter)
    for a, b in zip(rows, rows[1:]):
        ta, _ = normalise(a["question"])
        tb, _ = normalise(b["question"])
        if ta in SEED and tb not in SEED and a["path"] == b["path"] and a["answer"] == b["answer"]:
            votes[tb][ta] += 1
    cache = dict(SEED)
    for t, c in votes.items():
        en, n = c.most_common(1)[0]
        if n >= 5 and n / sum(c.values()) > 0.95:
            cache[t] = SEED[en]
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    print("plan cache: %d templates -> %s" % (len(cache), CACHE_PATH))
    return cache


_cache = None


def plan_for(question, llm=None):
    """-> (plan, slots, source) where source is 'cache', 'llm' or None."""
    global _cache
    if _cache is None:
        _cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else dict(SEED)
    template, slots = normalise(question)
    if template in _cache:
        return _cache[template], slots, "cache"
    if llm is not None:
        plan = llm(question, template, slots)  # expected to return a plan dict (and may update the cache)
        if plan:
            _cache[template] = plan
            return plan, slots, "llm"
    return None, slots, None


# ---- 3. execution -----------------------------------------------------------------------------


def _norm(s):
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


def execute(plan, slots, facts):
    """-> dict(answer, unit, evidence=[{kind, name, bbox, value}], calc)."""
    rooms = facts["rooms"]
    ev = []

    def room_ev(r):
        b = r["poly"].bounds if r["poly"] is not None else (r["c"][0] - 5, r["c"][1] - 5, r["c"][0] + 5, r["c"][1] + 5)
        return {"kind": "room", "name": "%s %s" % (r["no"], r["label"] or ""), "bbox": b,
                "poly": list(r["poly"].exterior.coords) if r["poly"] is not None and r["poly"].geom_type == "Polygon" else None,
                "value": None if r["area_m2"] is None else round(r["area_m2"], 2)}

    if plan["op"] == "count":
        ent = plan["entity"]
        if ent == "room":
            sel = [r for r in rooms if r["label"] and _norm(r["label"]) == _norm(slots.get("Q", ""))]
            ev = [room_ev(r) for r in sel]
            calc = "rooms whose label is '%s'" % slots.get("Q")
        elif ent == "apartment":
            ev = [{"kind": "apartment", "name": u["name"], "bbox": u["bbox"], "value": None} for u in facts["unit_names"]]
            calc = "apartment labels on the plan"
        else:
            key = {"door": "doors", "window": "windows", "toilet": "toilets", "washbasin": "washbasins", "bathtub": "bathtubs", "shower": "showers"}[ent]
            ev = [{"kind": ent, "name": ent, "bbox": b, "value": None} for b in facts[key]]
            calc = "%s symbols found in the vector drawing" % ent
        return {"answer": len(ev), "unit": "count", "evidence": ev, "calc": "count of " + calc}

    by = plan["by"]
    if by == "number":
        sel = [r for r in rooms if r["no"] == slots.get("R")]
        calc = "polygon area of room %s" % slots.get("R")
    elif by == "label":
        sel = [r for r in rooms if r["label"] and _norm(r["label"]) == _norm(slots.get("Q", ""))]
        calc = "sum of rooms labelled '%s'" % slots.get("Q")
    elif by == "apartment":
        sel = [r for r in rooms if r["unit"] and r["unit"].split()[-1] == slots.get("L") and r["fill"] == WHITE]
        calc = "sum of indoor rooms inside apartment %s" % slots.get("L")
    else:
        sel = [r for r in rooms if r["fill"] in (WHITE, CIRC)]
        calc = "sum of all numbered indoor rooms (outdoor spaces excluded)"
    sel = [r for r in sel if r["area_m2"] is not None]
    ev = [room_ev(r) for r in sel]
    if not sel:
        return {"answer": None, "unit": "m2", "evidence": [], "calc": calc + ": nothing matched"}
    total = sum(r["area_m2"] for r in sel)
    terms = " + ".join("%.2f" % r["area_m2"] for r in sel[:12]) + (" + ..." if len(sel) > 12 else "")
    return {"answer": round(total, 2), "unit": "m2", "evidence": ev, "calc": "%s: %s = %.2f m²" % (calc, terms, total)}


# ---- 4. ask + evidence rendering -------------------------------------------------------------

_facts = {}


def facts_for(pdf):
    if pdf not in _facts:
        _facts[pdf] = swiss_adapter.parse(pdf)
    return _facts[pdf]


def render(pdf, evidence, out_png, width=2000):
    """Draw the evidence on the drawing and save a crop around it + the plan."""
    import fitz

    doc = fitz.open(pdf)
    page = doc[0]
    colors = {"room": (0.1, 0.45, 0.9), "apartment": (0.8, 0.2, 0.2)}
    for i, e in enumerate(evidence):
        col = colors.get(e["kind"], (0.9, 0.3, 0.0))
        sh = page.new_shape()
        if e.get("poly"):
            sh.draw_polyline([fitz.Point(*p) for p in e["poly"]])
            sh.finish(color=col, fill=col, fill_opacity=0.25, width=1.2, closePath=True)
        else:
            b = e["bbox"]
            sh.draw_rect(fitz.Rect(b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2))
            sh.finish(color=col, width=1.4, fill=col, fill_opacity=0.2)
        sh.commit()
        b = e["bbox"]
        if e["kind"] != "room":
            page.insert_text(fitz.Point(b[0] - 2, b[1] - 3), str(i + 1), fontsize=5, color=col)
    words = [w for w in page.get_text("words")]
    leg = [w[1] for w in words if swiss_adapter.LEGEND_RE.match(w[4])]
    ytop = (min(leg) - 12) if leg else page.rect.height * 0.82
    xs = [w for w in words if w[1] < ytop]
    clip = fitz.Rect(min(w[0] for w in xs) - 60, min(w[1] for w in xs) - 60, max(w[2] for w in xs) + 60, max(w[3] for w in xs) + 60) & page.rect
    z = width / clip.width
    page.get_pixmap(clip=clip, matrix=fitz.Matrix(z, z)).save(out_png)


def ask(pdf, question, out_png=None, llm=None):
    t0 = time.time()
    plan, slots, src = plan_for(question)
    tokens = 0
    if plan is None and llm is not None:
        # unseen question form: the model translates it, given the labels found on this drawing
        plan, slots, tokens = llm(facts_for(pdf), question)
        src = "llm"
    if plan is None:
        return {"answer": None, "plan": None, "plan_source": None, "llm_tokens": tokens,
                "note": "This question form is new and no translation model is loaded in this run."}
    if plan.get("op") == "complex":
        return {"answer": None, "plan": plan, "plan_source": src, "llm_tokens": tokens, "cost_usd": round(getattr(llm, "last_usd", 0.0), 6),
                "note": "This question needs more than one count or one area sum; it is passed on to the code-writing model."}
    if plan.get("op") == "unsupported":
        return {"answer": None, "plan": plan, "plan_source": src, "llm_tokens": tokens,
                "note": "The model judged this question to be outside what the drawing data can answer."}
    res = execute(plan, slots, facts_for(pdf))
    res.update({"plan": plan, "slots": slots, "plan_source": src, "llm_tokens": tokens,
                "cost_usd": round(getattr(llm, "last_usd", 0.0), 6) if src == "llm" else 0.0, "seconds": round(time.time() - t0, 3)})
    if out_png:
        render(pdf, res["evidence"], out_png)
    return res


# ---- CLI ------------------------------------------------------------------------------------


def eval_swiss(n):
    import random

    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "swiss_dwellings"]
    files = sorted({r["path"] for r in rows})
    if n != "all":
        random.Random(7).shuffle(files)
        files = files[: int(n)]
    files = set(files)
    agg = collections.defaultdict(lambda: [0, 0])
    src = collections.Counter()
    for r in rows:
        if r["path"] not in files:
            continue
        res = ask(evalkit.DATA + "/" + r["path"], r["question"])
        src[res["plan_source"]] += 1
        ok = res["answer"] is not None and evalkit.is_correct(res["answer"], r["answer"], r["type"])
        lang = "en" if re.match(r"(How|What) ", r["question"]) else "other"
        kind = "-" if res["plan"] is None else res["plan"]["op"] + ":" + res["plan"].get("by", res["plan"]["entity"])
        for k in ("ALL", "lang=" + lang, "type=" + r["type"], kind):
            agg[k][0] += ok
            agg[k][1] += 1
    print("files", len(files), "| plan source", dict(src))
    for k, (a, b) in sorted(agg.items()):
        print("  %-22s %6d / %6d  %.1f%%" % (k, a, b, 100 * a / b))


def dev(pred_path):
    n = 0
    with open(evalkit.DEV) as f, open(pred_path, "w") as out:
        for l in f:
            r = json.loads(l)
            if r["source"] != "swiss_dwellings":
                continue
            res = ask(evalkit.DATA + "/" + r["path"], r["question"])
            out.write(json.dumps({"id": r["id"], "answer": res["answer"], "cost_usd": res.get("cost_usd", 0), "tokens": res.get("llm_tokens", 0)}) + "\n")
            n += 1
    print("answered", n, "swiss questions ->", pred_path)
    evalkit.score(pred_path)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["build-cache"]:
        build_cache()
    elif a[:1] == ["ask"] and len(a) >= 3:
        r = ask(a[1], a[2], a[3] if len(a) > 3 else None)
        for e in r.get("evidence", []):
            e.pop("poly", None)
            e["bbox"] = [round(v, 1) for v in e["bbox"]]
        print(json.dumps(r, ensure_ascii=False, indent=1))
    elif a[:1] == ["eval-swiss"]:
        eval_swiss(a[1] if len(a) > 1 else "100")
    elif a[:1] == ["dev"] and len(a) == 2:
        dev(a[1])
    else:
        print(__doc__)
