"""Free-form questions on a parsed vector plan: the model writes a few lines of code over the drawing's fact tables.

The fixed query language (count / area of one entity kind) covers the dataset's question forms. Questions written by
people go further: comparisons, conditions across rooms, ratios, "which apartment ...". For those the model gets the
fact tables as text and writes Python; the code runs here, so the arithmetic and the evidence are still the code's.

  python agent.py ask <swiss pdf, relative to the dataset> "<question>" [model]
  python agent.py test [model]         a handful of hand-written complex questions, printed for reading (no gold answers)
"""
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import qa  # noqa: E402
import table_adapter  # noqa: E402

DEFAULT_MODEL = os.environ.get("STOA_AGENT_MODEL", "gemini-4-hydrogen")
KIND = {qa.WHITE: "indoor", qa.CIRC: "circulation"}
FIXT = [("toilets", "toilet", "T"), ("washbasins", "washbasin", "B"), ("bathtubs", "bathtub", "U"), ("showers", "shower", "S")]


def view(facts):
    """Fact tables for the model, with relations worked out from the geometry (which room a fixture is in,
    which rooms a door connects, which room a window belongs to). -> (tables, geometry by id)"""
    from shapely.geometry import box

    k = facts["m_per_pt"]
    rooms, geo = [], {}
    polys = []
    for i, r in enumerate(facts["rooms"]):
        rid = "R%d" % (i + 1)
        p = r["poly"]
        b = p.bounds if p is not None else None
        rooms.append({"id": rid, "number": r["no"], "label": r["label"] or "", "area_m2": round(r["area_m2"], 2) if r["area_m2"] else None,
                      "kind": KIND.get(tuple(r["fill"]) if r["fill"] is not None else None, "outdoor"),
                      "apartment": r["unit"].split()[-1] if r.get("unit") else None,
                      "width_m": round(min(b[2] - b[0], b[3] - b[1]) * k, 2) if b else None,
                      "length_m": round(max(b[2] - b[0], b[3] - b[1]) * k, 2) if b else None,
                      "perimeter_m": round(p.length * k, 2) if p is not None else None,
                      "toilets": 0, "washbasins": 0, "bathtubs": 0, "showers": 0, "doors": [], "windows": [], "connected_rooms": []})
        polys.append(p)
        geo[rid] = {"kind": "room", "name": "%s %s" % (r["no"], r["label"] or ""), "bbox": list(b) if b else None,
                    "poly": [list(c) for c in p.exterior.coords] if p is not None and p.geom_type == "Polygon" else None}

    def touching(bb, reach_m):
        g = box(*bb).buffer(reach_m / k)
        return [j for j, p in enumerate(polys) if p is not None and p.intersects(g)]

    fixtures = []
    for key, name, pre in FIXT:
        for n, bb in enumerate(facts.get(key, [])):
            fid = "%s%d" % (pre, n + 1)
            c = box(*bb).centroid
            inside = [j for j, p in enumerate(polys) if p is not None and p.contains(c)] or touching(bb, 0.05)
            room = rooms[inside[0]]["id"] if inside else None
            if inside:
                rooms[inside[0]][key] += 1
            fixtures.append({"id": fid, "type": name, "room": room})
            geo[fid] = {"kind": name, "name": name, "bbox": list(bb)}
    doors = []
    for n, bb in enumerate(facts.get("doors", [])):
        did = "D%d" % (n + 1)
        js = touching(bb, 0.15)
        ids = [rooms[j]["id"] for j in js]
        doors.append({"id": did, "rooms": ids})
        geo[did] = {"kind": "door", "name": "door", "bbox": list(bb)}
        for j in js:
            rooms[j]["doors"].append(did)
            rooms[j]["connected_rooms"] = sorted(set(rooms[j]["connected_rooms"]) | (set(ids) - {rooms[j]["id"]}))
    windows = []
    for n, bb in enumerate(facts.get("windows", [])):
        wid = "W%d" % (n + 1)
        js = touching(bb, 0.15)
        ids = [rooms[j]["id"] for j in js]
        windows.append({"id": wid, "rooms": ids, "width_m": round(max(bb[2] - bb[0], bb[3] - bb[1]) * k, 2)})
        geo[wid] = {"kind": "window", "name": "window", "bbox": list(bb)}
        for j in js:
            rooms[j]["windows"].append(wid)
    apartments = sorted({r["apartment"] for r in rooms if r["apartment"]})
    return {"rooms": rooms, "doors": doors, "windows": windows, "fixtures": fixtures, "apartments": apartments, "scale": "1:%s" % facts["scale"]}, geo


SYSTEM = """You answer questions about one floor of an apartment building. You do not see the drawing. You get tables extracted from it, and you write Python that computes the answer from those tables. The code is executed and its result is shown to the user with the entities you used highlighted on the drawing.

Variables available to your code:
- rooms: list of dicts with id, number (as printed on the plan), label (as printed, in the plan's language), area_m2, kind ("indoor" = room inside an apartment or building, "circulation" = stairs, corridors, lifts shared by the floor, "outdoor" = balcony, loggia, terrace), apartment (letter, or None for shared spaces), width_m, length_m (bounding box), perimeter_m, toilets, washbasins, bathtubs, showers (counts of fixtures inside the room), doors and windows (ids), connected_rooms (ids of rooms reached through a door of this room)
- doors: list of dicts with id, rooms (ids of the rooms the door touches; one room means the other side is outside the numbered rooms)
- windows: list of dicts with id, rooms, width_m
- fixtures: list of dicts with id, type ("toilet", "washbasin", "bathtub", "shower"), room
- apartments: list of apartment letters
- math, re
The fields above are those of a vector plan. A plan that came as an image has the same tables with these differences: rooms carry type ("bedroom", "kitchen", "bathroom", "wc", "sauna", "balcony", "living", "hall", "storage", "utility", "outdoor", "other") and floor; fixture counts per room are toilets, sinks, showers_baths; area_m2 is only what is printed in the room (often None); there is no apartment, number, width, length or perimeter. Always use the fields that are actually present in the data you are given.

Your code must set:
- answer: a number or a short string. Use None if the tables cannot answer the question (for example heights, materials, prices, orientation, anything not listed above).
- unit: "count", "m2", "m", "%" or "" for text
- evidence: list of the ids (rooms, doors, windows, fixtures) the answer rests on
- steps: one or two plain sentences, in the language of the question, saying how the answer was computed, with the key numbers. If answer is None, say what is missing.

Rules:
- Room labels are in the plan's language (German, French, Italian or English), whatever the language of the question. Match the meaning: a question about bathrooms on a German plan means labels such as "Bad", "Dusche", "Bad/WC". Decide which of the labels that actually occur fit, and list them explicitly in the code.
- Living area of an apartment means its indoor rooms; balconies and other outdoor rooms are excluded unless the question includes them.
- Never invent data. No imports, no file access, no printing.
- Reply with one ```python block and nothing else."""


def describe(tables):
    L = ["Scale %s. Apartments: %s" % (tables.get("scale", "unknown"), ", ".join(tables.get("apartments", [])) or "none marked")]
    if tables.get("floors"):
        L.append("Floors shown on the sheet: %s" % tables["floors"])
    if "readers" in tables:
        L.append("DATA QUALITY: read from an image by detectors and a vision model. Counts and room assignments are approximate; areas are only those printed on the plan.")
    L.append("Labels that occur: " + json.dumps(sorted({r["label"] for r in tables["rooms"] if r["label"]}), ensure_ascii=False))
    L.append("rooms = [")
    for r in tables["rooms"]:
        L.append("  " + json.dumps(r, ensure_ascii=False) + ",")
    L.append("]")
    L.append("doors = " + json.dumps(tables["doors"]))
    L.append("windows = " + json.dumps(tables["windows"]))
    L.append("fixtures = " + json.dumps(tables["fixtures"]))
    return "\n".join(L)


def run(text, tables):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = (m.group(1) if m else text).strip()
    if re.search(r"\bimport\b|__|open\(|exec\(|eval\(|os\.|sys\.", code):
        return {"error": "rejected: code uses a forbidden construct"}, code
    env = {"__builtins__": table_adapter.SAFE, "math": math, "re": re}
    env.update(json.loads(json.dumps(tables)))  # the code gets its own copy
    try:
        exec(code, env)  # noqa: S102 - whitelisted builtins only; prototype
    except Exception as e:
        return {"error": "error: %r" % (e,)}, code
    return {k: env.get(k) for k in ("answer", "unit", "evidence", "steps")}, code


def answer(pdf_full, question, model):
    """model: an object with generate(batch_of_messages, max_new_tokens) and a running .usd"""
    import plan_facts

    v = plan_facts.view(pdf_full)
    if v is None:
        return {"answer": None, "evidence": [], "llm_tokens": 0, "cost_usd": 0.0, "note": "No floor-plan fact tables for this file."}
    tables, geo, exact, read_tokens, read_usd = v
    if not tables["rooms"]:  # not a floor plan, or nothing could be read: do not compute on empty tables
        return {"answer": None, "evidence": [], "llm_tokens": read_tokens, "cost_usd": read_usd, "note": "No rooms were found on this drawing."}
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": describe(tables) + "\n\nQuestion: " + question}]
    before = model.usd
    texts, n_in, n_out = model.generate([msgs], max_new_tokens=900)
    out, code = run(texts[0], tables)
    if out.get("error"):  # one informed retry
        msgs += [{"role": "assistant", "content": texts[0]}, {"role": "user", "content": "Running that code failed with: %s\nReturn corrected code only." % out["error"]}]
        t2, a, b = model.generate([msgs], max_new_tokens=900)
        n_in, n_out = n_in + a, n_out + b
        out, code = run(t2[0], tables)
    ans = out.get("answer")
    if isinstance(ans, float):
        ans = int(ans) if ans == int(ans) else round(ans, 2)
    ev = [dict(geo[i], id=i) for i in (out.get("evidence") or []) if isinstance(i, str) and i in geo]
    res = {"answer": ans, "unit": out.get("unit"), "evidence": ev, "plan": {"op": "generated code"}, "plan_source": "code",
           "calc": ((out.get("steps") or "") + "\n\nCode written by the model and run on the drawing's fact tables:\n" + code).strip(),
           "llm_tokens": n_in + n_out + read_tokens, "cost_usd": round(model.usd - before + read_usd, 6)}
    if not exact:
        res["reliability"] = "computed on facts read from an image (approximate)"
    if out.get("error"):
        res["note"] = out["error"]
    elif ans is None:
        res["note"] = out.get("steps") or "The fact tables cannot answer this question."
    return res


def make_agent(model_id=DEFAULT_MODEL):
    import llm_plan

    model = llm_plan.Model(model_id)

    def f(full, question):  # same step for every drawing kind: code on that kind's fact tables
        if full.lower().endswith(".ifc"):
            import ifc_agent

            return ifc_agent.answer(full, question, model)
        return answer(full, question, model)

    return f


TESTS = [
    "Which apartment has the largest living area, and how big is it?",
    "Wie viele Zimmer hat die grösste Wohnung, ohne Bad, Küche und Balkon?",
    "What share of this floor's total area is balcony?",
    "Kuinka monessa kylpyhuoneessa on ikkuna?",
    "How many metres of skirting board do I need for all bedrooms and living rooms in apartment A? Subtract 0.9 m for each door.",
    "Which rooms have neither a window nor a door to a balcony?",
    "If I tile every bathroom floor and add 10% waste, how many m² of tiles do I order?",
    "Is there any apartment with two toilets but only one washbasin?",
    "What is the ceiling height in the living room?",
    "Compare apartments: list each one with its number of rooms and total area, smallest first.",
]


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["ask"] and len(a) >= 3:
        r = make_agent(a[3] if len(a) > 3 else DEFAULT_MODEL)(os.path.join(evalkit.DATA, a[1]), a[2])
        r["evidence"] = [e["id"] for e in r["evidence"]]
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    elif a[:1] == ["test"]:
        ag = make_agent(a[1] if len(a) > 1 else DEFAULT_MODEL)
        path = a[2] if len(a) > 2 else next(r["path"] for r in evalkit.load_questions() if evalkit.source(r["path"]) == "swiss_dwellings")
        tot = 0
        for q in TESTS:
            r = ag(os.path.join(evalkit.DATA, path), q)
            tot += r["cost_usd"]
            print("\nQ: %s\nA: %s %s | evidence %s | %d tokens, $%.4f\n%s" % (q, r["answer"], r.get("unit") or "", [e["id"] for e in r["evidence"]][:12], r["llm_tokens"], r["cost_usd"], r["calc"].split("\n\nCode")[0]))
        print("\ntotal $%.3f" % tot)
    else:
        print(__doc__)
