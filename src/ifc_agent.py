"""Free-form questions on an IFC model: the model writes Python over an element table of the building model.

ifc_adapter answers the dataset's fixed question forms from a small summary. Questions written by people ask about
anything the model holds (materials, wall types, thicknesses, equipment, the authoring software). For those, every
element of the IFC is stored once as one row (class, name, type, storey, materials, quantities, properties), and the
answer is computed by code the language model writes over those rows. The model may look at the data first.

  python ifc_agent.py build [source ...]     parse the IFC files that have questions -> data/facts_ifc_elements.zip (one file)
  python ifc_agent.py show <ifc>             the summary the language model sees
  python ifc_agent.py ask <ifc> "<question>" [model]
  python ifc_agent.py test [n] [model]       n IFC-Bench questions per model file, judged against the reference answers
"""
import collections
import json
import math
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylibs"))  # IfcOpenShell, if it is not installed in the environment
import evalkit  # noqa: E402
import table_adapter  # noqa: E402

DEFAULT_MODEL = os.environ.get("STOA_AGENT_MODEL", "gemini-4-hydrogen")
JUDGE_MODEL = "gemini-3.8-flash"
STORE = os.path.join(evalkit.ROOT, "data", "facts_ifc_elements.zip")
SKIP = ("IfcOpeningElement", "IfcVirtualElement")  # openings are holes, not things
ROUNDS = 3  # the model may look at the data twice before it must answer


# ---- parsing ---------------------------------------------------------------------------------


def _scalar(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return round(v, 4) if isinstance(v, float) else v
    if isinstance(v, str):
        return v if len(v) <= 120 else None
    return None


def _materials(ue, e, to_mm):
    """-> (material names, layers [{material, thickness_mm}])"""
    m = ue.get_material(e, should_inherit=True)
    if m is None:
        return [], []
    if m.is_a("IfcMaterialLayerSetUsage"):
        m = m.ForLayerSet
    if m.is_a("IfcMaterialProfileSetUsage"):
        m = m.ForProfileSet
    names, layers = [], []
    if m.is_a("IfcMaterial"):
        names = [m.Name]
    elif m.is_a("IfcMaterialLayerSet"):
        for l in m.MaterialLayers or []:
            n = l.Material.Name if l.Material else None
            layers.append({"material": n, "thickness_mm": round(float(l.LayerThickness) * to_mm, 1)})
            names.append(n)
    elif m.is_a("IfcMaterialConstituentSet"):
        names = [c.Material.Name for c in m.MaterialConstituents or [] if c.Material]
    elif m.is_a("IfcMaterialProfileSet"):
        names = [p.Material.Name for p in m.MaterialProfiles or [] if p.Material]
    elif m.is_a("IfcMaterialList"):
        names = [x.Name for x in m.Materials or []]
    return [n for n in dict.fromkeys(names) if n], layers


def inventory(path):
    """One IFC file -> {"model": {...}, "elements": [row, ...]}. Lengths in metres, thicknesses in mm."""
    import ifcopenshell
    import ifcopenshell.util.element as ue
    import ifcopenshell.util.unit as uu

    import ifc_adapter

    f = ifcopenshell.open(path)
    try:
        scale = float(uu.calculate_unit_scale(f))
    except Exception:
        scale = 1.0
    proj = f.by_type("IfcProject")
    apps = f.by_type("IfcApplication")
    model = {
        "schema": f.schema,
        "project": (proj[0].LongName or proj[0].Name) if proj else None,
        "software": [{"name": a.ApplicationFullName, "version": a.Version, "developer": a.ApplicationDeveloper.Name if a.ApplicationDeveloper else None} for a in apps],
        "exported_by": list(getattr(f.header.file_name, "originating_system", "") and [f.header.file_name.originating_system]) or [],
        "file_description": [str(x) for x in (f.header.file_description.description or [])][:6],
        "site": [s.Name or s.LongName for s in f.by_type("IfcSite")],
        "buildings": [b.Name or b.LongName for b in f.by_type("IfcBuilding")],
        "length_unit_in_metres": scale,
        "storeys": sorted(({"name": s.Name, "elevation_m": round(float(s.Elevation or 0) * scale, 3)} for s in f.by_type("IfcBuildingStorey")), key=lambda s: s["elevation_m"]),
    }
    rows = []
    for e in list(f.by_type("IfcElement")) + list(f.by_type("IfcSpace")):
        if e.is_a() in SKIP:
            continue
        t = ue.get_type(e)
        agg = ue.get_aggregate(e)
        names, layers = _materials(ue, e, scale * 1000)
        qty, props = {}, {}
        try:
            for ps, d in ue.get_psets(e, should_inherit=True).items():
                isq = ps.startswith(("Qto_", "BaseQuantities")) or "Quantit" in ps
                for k, v in d.items():
                    if k == "id":
                        continue
                    v = _scalar(v)
                    if v is None or v == "":
                        continue
                    if isq and isinstance(v, (int, float)) and not isinstance(v, bool):
                        qty[k] = v
                    else:
                        props[k if k not in props else ps + "." + k] = v
        except Exception:
            pass
        row = {"id": e.GlobalId, "class": e.is_a(), "name": e.Name, "type": t.Name if t is not None else getattr(e, "ObjectType", None),
               "predefined": str(getattr(e, "PredefinedType", None) or "") or None, "storey": ifc_adapter._storey(ue, e),
               "part_of": agg.is_a() if agg is not None and agg.is_a("IfcElement") else None,
               "external": props.get("IsExternal") if isinstance(props.get("IsExternal"), bool) else None,
               "materials": names, "layers": layers, "qty": qty, "props": props}
        if e.is_a("IfcSpace"):
            row["long_name"] = e.LongName
        if e.is_a("IfcDoor") or e.is_a("IfcWindow"):
            for k, a in (("width_m", "OverallWidth"), ("height_m", "OverallHeight")):
                v = getattr(e, a, None)
                row[k] = round(float(v) * scale, 3) if v else None
        rows.append(row)
    # room areas: the summary parser already works them out (quantity set, else geometry footprint)
    known = ifc_adapter.facts_for(os.path.relpath(path, evalkit.DATA)) if path.startswith(evalkit.DATA) else None
    areas = {s["guid"]: s for s in (known or {}).get("spaces", [])} if known and "error" not in known else None
    if areas is None and any(r["class"] == "IfcSpace" for r in rows):
        try:
            areas = {s["guid"]: s for s in ifc_adapter.parse(path)["spaces"]}
        except Exception:
            areas = {}
    for r in rows:
        if r["class"] == "IfcSpace":
            a = (areas or {}).get(r["id"], {})
            r["area_m2"] = round(a["area_m2"], 2) if a.get("area_m2") else None
            r["area_source"] = a.get("area_source")
    return {"model": model, "elements": rows}


def _member(path):
    return (os.path.relpath(path, evalkit.DATA) if path.startswith(evalkit.DATA) else os.path.basename(path)) + ".json"


_mem = {}


def tables_for(full):
    """Stored element table of a dataset model, or parse now (an uploaded file) and keep it in memory."""
    if full in _mem:
        return _mem[full]
    t = None
    if os.path.exists(STORE):
        with zipfile.ZipFile(STORE) as z:
            if _member(full) in z.namelist():
                t = json.loads(z.read(_member(full)))
    if t is None:
        if os.path.getsize(full) > 40e6 and not os.environ.get("STOA_PARSE_BIG"):  # minutes of work and gigabytes of memory: an ingestion job, not a question
            raise RuntimeError("this model is large and has not been ingested yet (run: python ifc_agent.py build)")
        t = inventory(full)
    _mem.clear()  # one model in memory at a time: the big ones are hundreds of MB
    _mem[full] = t
    return t


def build(sources=("ifc_bench", "ifc_models")):
    paths = sorted({r["path"] for r in evalkit.load_questions() if evalkit.source(r["path"]) in sources and r["path"].endswith(".ifc")})
    done = set(zipfile.ZipFile(STORE).namelist()) if os.path.exists(STORE) else set()
    for i, p in enumerate(paths):
        full = os.path.join(evalkit.DATA, p)
        if _member(full) in done:
            continue
        try:
            t = inventory(full)
        except Exception as e:
            print("[%d/%d] %s: ERROR %r" % (i + 1, len(paths), p, e), flush=True)
            continue
        data = json.dumps(t, ensure_ascii=False)
        with zipfile.ZipFile(STORE, "a", zipfile.ZIP_DEFLATED) as z:
            z.writestr(_member(full), data)
        print("[%d/%d] %s: %d elements, %.1f MB json" % (i + 1, len(paths), p, len(t["elements"]), len(data) / 1e6), flush=True)


# ---- what the language model sees ------------------------------------------------------------


def _top(counter, n):
    items = counter.most_common(n)
    return ", ".join("%s (%d)" % (json.dumps(k, ensure_ascii=False) if not isinstance(k, str) else k, v) for k, v in items) + (" … +%d more" % (len(counter) - n) if len(counter) > n else "")


def describe(t, max_chars=45000):
    m = t["model"]
    L = ["MODEL: " + json.dumps({k: v for k, v in m.items() if k != "storeys"}, ensure_ascii=False),
         "storeys (lowest first): " + json.dumps(m["storeys"], ensure_ascii=False), "",
         "ELEMENT CLASSES (rows in `elements`):"]
    by = collections.defaultdict(list)
    for r in t["elements"]:
        by[r["class"]].append(r)
    for cls, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        L.append("- %s: %d rows" % (cls, len(rs)))
        L.append("    storeys: " + _top(collections.Counter(r["storey"] for r in rs), 12))
        ty = collections.Counter(r["type"] for r in rs if r["type"])
        if ty:
            L.append("    type: " + _top(ty, 10))
        na = collections.Counter(r["name"] for r in rs if r["name"])
        if na and cls == "IfcSpace":
            L.append("    name / long_name examples: " + "; ".join("%s / %s" % (r["name"], r.get("long_name")) for r in rs[:8]))
        elif na and len(na) <= 40 and not ty:
            L.append("    name: " + _top(na, 8))
        pd = collections.Counter(r["predefined"] for r in rs if r["predefined"] and r["predefined"] != "NOTDEFINED")
        if pd:
            L.append("    predefined: " + _top(pd, 6))
        ex = collections.Counter(r["external"] for r in rs if r["external"] is not None)
        if ex:
            L.append("    external: " + _top(ex, 3))
        ma = collections.Counter(n for r in rs for n in r["materials"])
        if ma:
            L.append("    materials: " + _top(ma, 10))
        if any(r["layers"] for r in rs):
            L.append("    layers: present on %d rows, e.g. %s" % (sum(1 for r in rs if r["layers"]), json.dumps(next(r["layers"] for r in rs if r["layers"])[:4], ensure_ascii=False)))
        qk = collections.Counter(k for r in rs for k in r["qty"])
        if qk:
            L.append("    qty keys: " + _top(qk, 14))
        pk = collections.Counter(k for r in rs for k in r["props"])
        if pk:
            L.append("    props keys: " + _top(pk, 28))
    text = "\n".join(L)
    return text if len(text) <= max_chars else text[:max_chars] + "\n… (summary cut; use `inspect` to see more)"


SYSTEM = """You answer questions about a building from its IFC model (BIM). You do not see the file. Every element of the model is one row in a table, and you write Python that computes the answer from the rows. The code is executed for you.

Variables available to your code:
- model: dict with schema, project, software, exported_by, site, buildings, storeys (name, elevation_m)
- elements: list of dicts, one per element, with
    id, class (IFC class such as "IfcWall", "IfcDoor", "IfcSpace"), name, type (name of its type object), predefined, storey (name of the storey it belongs to, or None),
    part_of (class of the parent element when this row is a part of another element, e.g. a stair flight inside a stair),
    external (True / False / None), materials (list of names), layers (list of {material, thickness_mm}; walls, slabs, roofs),
    qty (dict of the model's own quantities, lengths in the model's unit; the summary lists the keys), props (dict of property values; the summary lists the keys)
    IfcSpace rows (rooms) also have long_name and area_m2; IfcDoor and IfcWindow rows have width_m and height_m.
- collections, math, re

You may look before you answer. Each reply is one ```python block that sets EITHER
- inspect: any small value you want to see (for example the distinct values of a property, or a few rows). It is shown to you, cut to 3000 characters, and you then write the next block. At most two looks.
OR, for the final reply, all of
- answer: a number, or a short text answer. For questions that ask for a list or breakdown, a compact multi-line string with the items and their numbers. None if the model does not hold the information.
- unit: "count", "m2", "m", "m3", "mm" or "" for text
- evidence: list of the ids of the elements the answer rests on (at most 200)
- steps: one or two plain sentences, in the language of the question, saying how the answer was computed, with the key numbers; say so if the model holds only part of what was asked.

Rules:
- Names in the model are in the modeller's language, whatever the language of the question. Match by meaning and list the matching names explicitly in the code.
- "Exterior" means external is True; if external is None everywhere for that class, look for another property that says it, and say what you used.
- Count things once: rows with part_of set are parts of another row (count stairs, curtain walls and roofs by the parent row).
- Report lengths in metres, areas in m², volumes in m³; quantities in qty are in the model's units (model["length_unit_in_metres"]).
- Never invent data. No imports, no file access, no printing."""


def run(text, t):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = (m.group(1) if m else text).strip()
    if re.search(r"\bimport\b|__|open\(|exec\(|eval\(|os\.|sys\.", code):
        return {"error": "rejected: code uses a forbidden construct"}, code
    env = {"__builtins__": dict(table_adapter.SAFE, repr=repr, isinstance=isinstance, dict=dict, list=list, set=set, tuple=tuple, bool=bool, any=any, all=all, sorted=sorted, round=round, abs=abs, zip=zip),
           "math": math, "re": re, "collections": collections, "model": t["model"], "elements": t["elements"]}
    try:
        exec(code, env)  # noqa: S102 - whitelisted builtins only; prototype
    except Exception as e:
        return {"error": "error: %r" % (e,)}, code
    return {k: env.get(k) for k in ("answer", "unit", "evidence", "steps", "inspect") if k in env}, code


def answer(full, question, model):
    """model: an object with generate(batch_of_messages, max_new_tokens) and a running .usd"""
    try:
        t = tables_for(full)
    except Exception as e:
        return {"answer": None, "evidence": [], "llm_tokens": 0, "cost_usd": 0.0, "note": "The IFC file could not be parsed: %r" % (e,)}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": describe(t) + "\n\nQuestion: " + question}]
    before, tok, out, code, looks = model.usd, 0, {}, "", []
    for i in range(ROUNDS + 1):
        texts, a, b = model.generate([msgs], max_new_tokens=1500)
        tok += a + b
        out, code = run(texts[0], t)
        if out.get("error"):
            fb = "Running that code failed with: %s\nReturn corrected code only." % out["error"]
        elif "answer" not in out and "inspect" in out and i < ROUNDS - 1:
            try:
                seen = json.dumps(out["inspect"], ensure_ascii=False, default=str)[:3000]
            except Exception:
                seen = repr(out["inspect"])[:3000]
            looks.append(code)
            fb = "inspect = %s\n\nNow write the next block%s." % (seen, "" if i < ROUNDS - 2 else "; it must set answer")
        elif "answer" not in out:
            fb = "Your code did not set answer. Return the final block now."
        else:
            break
        if i == ROUNDS:
            break
        msgs += [{"role": "assistant", "content": texts[0]}, {"role": "user", "content": fb}]
    ans = out.get("answer")
    if isinstance(ans, float):
        ans = int(ans) if ans == int(ans) else round(ans, 2)
    elif isinstance(ans, (dict, list, tuple, set)):
        ans = repr(ans)[:4000]
    ids = {r["id"]: r for r in t["elements"]}
    ev = [{"kind": ids[i]["class"], "name": ids[i]["name"] or ids[i]["type"] or "", "guid": i} for i in (out.get("evidence") or [])[:200] if isinstance(i, str) and i in ids]
    res = {"answer": ans, "unit": out.get("unit"), "evidence": ev, "plan": {"op": "generated code"}, "plan_source": "code",
           "calc": ((out.get("steps") or "") + "\n\nCode written by the model and run on the model's element table"
                    + (" (after %d look%s at the data)" % (len(looks), "s" * (len(looks) > 1)) if looks else "") + ":\n" + code).strip(),
           "llm_tokens": tok, "cost_usd": round(model.usd - before, 6)}
    if out.get("error"):
        res["note"] = out["error"]
    elif ans is None:
        res["note"] = out.get("steps") or "The model does not hold this information."
    return res


def make_agent(model_id=DEFAULT_MODEL):
    import llm_plan

    model = llm_plan.Model(model_id)
    return lambda full, question: answer(full, question, model)


# ---- test on IFC-Bench -----------------------------------------------------------------------

JUDGE = """You grade an answer to a question about a building information model against a reference answer.
The reference was written from the same model, so its facts are the truth. Wording, order, language and extra detail do not matter; numbers may differ by 2%.
Reply with JSON only: {"grade": "correct" | "partial" | "wrong", "why": "<one short sentence>"}
correct = every key fact of the reference that the question asks for is in the answer and nothing contradicts it.
partial = the main fact is right but some asked items are missing or off, or the answer is right for part of a multi-part question.
wrong = the main fact is missing or contradicts the reference, or there is no answer."""


def judge(rows, model_id=JUDGE_MODEL):
    """rows: [(question, reference, answer)] -> [grade]"""
    import gemini_backend as gb

    g = gb.Gemini(model_id)
    msgs = [[{"role": "system", "content": JUDGE}, {"role": "user", "content": "Question: %s\n\nReference answer:\n%s\n\nAnswer to grade:\n%s" % (q, str(ref)[:6000], str(a)[:6000])}] for q, ref, a in rows]
    texts, _, _ = g.generate(msgs, 300)
    out = []
    for x in texts:
        m = re.search(r'"grade"\s*:\s*"(\w+)"', x or "")
        out.append(m.group(1) if m else "wrong")
    return out, g.usd


def test(n=2, model_id=DEFAULT_MODEL, out_path=None):
    import random
    from concurrent.futures import ThreadPoolExecutor

    rng = random.Random(7)
    by = collections.defaultdict(list)
    for r in evalkit.load_questions():
        if evalkit.source(r["path"]) == "ifc_bench":
            by[r["path"]].append(r)
    pick = [r for p in sorted(by) for r in rng.sample(by[p], min(n, len(by[p])))]
    ag = make_agent(model_id)
    res = []
    for p in sorted({r["path"] for r in pick}):  # one model in memory at a time
        qs = [r for r in pick if r["path"] == p]
        tables_for(os.path.join(evalkit.DATA, p))
        with ThreadPoolExecutor(8) as ex:
            res += list(zip(qs, ex.map(lambda r: ag(os.path.join(evalkit.DATA, r["path"]), r["question"]), qs)))
        print(p, len(qs), flush=True)
    grades, jusd = judge([(r["question"], r["answer"], "%s %s\n%s" % (a["answer"], a.get("unit") or "", (a.get("calc") or "").split("\n\nCode")[0])) for r, a in res])
    c = collections.Counter(grades)
    per = collections.defaultdict(collections.Counter)
    for (r, a), g in zip(res, grades):
        per[os.path.basename(r["path"])][g] += 1
    print("\n%d questions: %s  | answering $%.3f ($%.4f / question), judging $%.3f" % (len(res), dict(c), sum(a["cost_usd"] for _, a in res), sum(a["cost_usd"] for _, a in res) / len(res), jusd))
    for k, v in sorted(per.items()):
        print("  %-44s %s" % (k, dict(v)))
    if out_path:
        with open(out_path, "w") as f:
            for (r, a), g in zip(res, grades):
                f.write(json.dumps({"path": r["path"], "question": r["question"], "gold": r["answer"], "answer": a["answer"], "steps": (a.get("calc") or "").split("\n\nCode")[0], "grade": g, "cost_usd": a["cost_usd"], "note": a.get("note")}, ensure_ascii=False, default=str) + "\n")
    return res, grades


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["build"]:
        build(tuple(a[1:]) or ("ifc_bench", "ifc_models"))
    elif a[:1] == ["show"] and len(a) == 2:
        print(describe(tables_for(a[1] if os.path.isabs(a[1]) else os.path.join(evalkit.DATA, a[1]))))
    elif a[:1] == ["ask"] and len(a) >= 3:
        r = make_agent(a[3] if len(a) > 3 else DEFAULT_MODEL)(a[1] if os.path.isabs(a[1]) else os.path.join(evalkit.DATA, a[1]), a[2])
        r["evidence"] = [e["guid"] for e in r["evidence"]][:10]
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    elif a[:1] == ["test"]:
        test(int(a[1]) if len(a) > 1 else 2, a[2] if len(a) > 2 else DEFAULT_MODEL, a[3] if len(a) > 3 else None)
    else:
        print(__doc__)
