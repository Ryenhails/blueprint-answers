"""Step 3 adapter: IFC model -> fact table (dict), cached in one JSON file.

  python ifc_adapter.py build      parse every IFC that has questions -> data/facts_ifc.json
  python ifc_adapter.py validate   check against the English ifc_models questions

Conventions (calibrated against questions.jsonl):
  counts            elements of a class contained in a storey (or in the whole model)
  space area        footprint area of the space geometry (models rarely carry a quantity set for it)
  wall / slab sums  the model's own base quantities (Length, NetSideArea, NetVolume)
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pylibs"))  # IfcOpenShell, if it is not installed in the environment
import evalkit  # noqa: E402

FACTS = os.path.join(evalkit.ROOT, "data", "facts_ifc.json")
CLASSES = {
    "space": ["IfcSpace"], "wall": ["IfcWall"], "door": ["IfcDoor"], "window": ["IfcWindow"], "stair": ["IfcStair"],
    "column": ["IfcColumn"], "beam": ["IfcBeam"], "slab": ["IfcSlab"],
    "sanitary": ["IfcSanitaryTerminal"], "light": ["IfcLightFixture"],
}
TYPE_FALLBACK = {"sanitary": "IfcSanitaryTerminalType", "light": "IfcLightFixtureType"}  # IFC2X3: IfcFlowTerminal typed by ...


def _storey(ue, e):
    seen = 0
    c = ue.get_container(e) or ue.get_aggregate(e)
    while c is not None and not c.is_a("IfcBuildingStorey") and seen < 8:
        c = ue.get_aggregate(c) or ue.get_container(c)
        seen += 1
    return c.Name if c is not None and c.is_a("IfcBuildingStorey") else None


def _qto(ue, e, names):
    for qs in ue.get_psets(e, qtos_only=True).values():
        for n in names:
            v = qs.get(n)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def _prop_area(ue, e):
    """Last resort: a numeric property whose name mentions 'area' (some exporters store room area as a plain property)."""
    best = None
    for name, ps in ue.get_psets(e).items():
        for k, v in ps.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"area", k, re.I) and v > 0:
                if best is None or re.search(r"net|floor", k, re.I):
                    best = float(v)
    return best


def parse(path):
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as ue
    import ifcopenshell.util.shape as us

    import ifcopenshell.util.unit as uu

    f = ifcopenshell.open(path)
    try:
        length_scale = float(uu.calculate_unit_scale(f))  # model length unit -> metres
    except Exception:
        length_scale = 1.0

    def elements(kind):
        out = []
        for c in CLASSES[kind]:
            try:
                out += f.by_type(c)
            except RuntimeError:
                pass
        if not out and kind in TYPE_FALLBACK:
            try:
                for t in f.by_type(TYPE_FALLBACK[kind]):
                    out += list(ue.get_types(t))
            except RuntimeError:
                pass
        return out

    proj = f.by_type("IfcProject")
    facts = {
        "schema": f.schema,
        "project": (proj[0].LongName or proj[0].Name) if proj else None,
        "storeys": [{"name": s.Name, "elevation": s.Elevation} for s in f.by_type("IfcBuildingStorey")],
        "counts": {}, "guids": {}, "spaces": [], "walls": {}, "slabs": {},
    }
    for kind in CLASSES:
        els = elements(kind)
        if kind == "stair":  # a stair split into parts counts once
            els = [e for e in els if not ((ue.get_aggregate(e) is not None) and ue.get_aggregate(e).is_a("IfcStair"))]
        per = collections.Counter(_storey(ue, e) or "" for e in els)
        per["_total"] = len(els)
        facts["counts"][kind] = dict(per)
        g = collections.defaultdict(list)  # element ids per storey: the evidence for count answers
        for e in els:
            g[_storey(ue, e) or ""].append(e.GlobalId)
        facts["guids"][kind] = dict(g)
        if kind == "wall":
            agg = collections.defaultdict(lambda: {"n": 0, "length": 0.0, "net_side_area": 0.0, "net_volume": 0.0, "n_qto": 0})
            for e in els:
                a = agg[_storey(ue, e) or ""]
                a["n"] += 1
                ln, sa, vol = _qto(ue, e, ["Length"]), _qto(ue, e, ["NetSideArea"]), _qto(ue, e, ["NetVolume"])
                if ln is not None or vol is not None:
                    a["n_qto"] += 1
                a["length"] += (ln or 0) * length_scale
                a["net_side_area"] += sa or 0
                a["net_volume"] += vol or 0
            facts["walls"] = {k: dict(v) for k, v in agg.items()}
        if kind == "slab":
            agg = collections.defaultdict(lambda: {"n": 0, "net_volume": 0.0, "n_qto": 0})
            for e in els:
                a = agg[_storey(ue, e) or ""]
                a["n"] += 1
                vol = _qto(ue, e, ["NetVolume"])
                if vol is not None:
                    a["n_qto"] += 1
                a["net_volume"] += vol or 0
            facts["slabs"] = {k: dict(v) for k, v in agg.items()}
    # spaces: footprint area from geometry (metres)
    spaces = f.by_type("IfcSpace")
    area = {}
    if spaces:
        st = ifcopenshell.geom.settings()
        it = ifcopenshell.geom.iterator(st, f, max(1, min(8, os.cpu_count() or 1)), include=spaces)
        if it.initialize():
            while True:
                sh = it.get()
                try:
                    area[sh.guid] = float(us.get_footprint_area(sh.geometry))
                except Exception:
                    pass
                if not it.next():
                    break
    for s in spaces:
        facts["spaces"].append({
            "guid": s.GlobalId, "name": s.Name, "long_name": s.LongName, "storey": _storey(ue, s),
            "footprint_m2": area.get(s.GlobalId), "qto_net_floor_area": _qto(ue, s, ["NetFloorArea"]),
            "prop_area": _prop_area(ue, s),
        })
    for s in facts["spaces"]:  # the model's own net floor area if it has one, else the geometry footprint
        s["area_m2"] = s["qto_net_floor_area"] or (s["footprint_m2"] if s["footprint_m2"] else None) or s["prop_area"]
        s["area_source"] = "quantity set" if s["qto_net_floor_area"] else "geometry" if s["footprint_m2"] else "property" if s["prop_area"] else None
    return facts


_all = None


def facts_for(ifc_rel_path):
    """ifc_rel_path like data/ifc_models/<model>.ifc, or an absolute path. A model that is not in the store is parsed now."""
    global _all
    if _all is None:
        _all = json.load(open(FACTS)) if os.path.exists(FACTS) else {}
    if ifc_rel_path not in _all:
        full = ifc_rel_path if os.path.isabs(ifc_rel_path) else os.path.join(evalkit.DATA, ifc_rel_path)
        if not (full.lower().endswith(".ifc") and os.path.exists(full)):
            return None
        try:
            _all[ifc_rel_path] = parse(full)
        except Exception as e:
            _all[ifc_rel_path] = {"error": repr(e)[:200]}
    return _all.get(ifc_rel_path)


def model_path(q_path):
    """A question may point at the IFC or at a plan rendered from it (<model>_plan-NN-<level>.pdf)."""
    if q_path.endswith(".ifc"):
        return q_path
    return re.sub(r"_plan(-[^/]*)?\.(pdf|svg|png)$", ".ifc", q_path)


def build(sources=("ifc_models",)):
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) in sources]
    models = sorted({model_path(r["path"]) for r in rows})
    out = json.load(open(FACTS)) if os.path.exists(FACTS) else {}
    force = os.environ.get("STOA_FORCE") == "1"  # re-parse models that are already in the cache
    for i, m in enumerate(models):
        if m in out and not force:
            continue
        p = os.path.join(evalkit.DATA, m)
        try:
            out[m] = parse(p)
            print("[%d/%d] %s: %d spaces, %d storeys" % (i + 1, len(models), m, len(out[m]["spaces"]), len(out[m]["storeys"])), flush=True)
        except Exception as e:
            out[m] = {"error": repr(e)[:200]}
            print("[%d/%d] %s: ERROR %r" % (i + 1, len(models), m, e), flush=True)
        with open(FACTS, "w") as fh:  # save after every model
            json.dump(out, fh, ensure_ascii=False)
    print("wrote", FACTS, len(out), "models")


# ---- execution of plans on IFC facts ----------------------------------------------------------


def _n(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()


def space_display(s):
    if s["long_name"] and s["name"]:
        return "%s (%s)" % (s["long_name"], s["name"])
    return str(s["long_name"] or s["name"])


def execute(plan, slots, facts):
    """plan: {"op", "entity", "scope": "storey"|"model", "measure"?}; slots: storey, room.
    -> dict(answer, unit, evidence, calc)"""
    op, ent = plan["op"], plan["entity"]
    storey = slots.get("storey")
    skey = next((s["name"] for s in facts["storeys"] if _n(s["name"]) == _n(storey)), storey) if storey else None
    if op == "count":
        c = facts["counts"].get(ent, {})
        if plan.get("scope") == "model":
            return {"answer": c.get("_total", 0), "unit": "count", "evidence": [], "calc": "all %s elements in the model" % ent}
        if ent == "space":
            sel = [s for s in facts["spaces"] if _n(s["storey"]) == _n(skey)]
            return {"answer": len(sel), "unit": "count", "calc": "spaces assigned to storey '%s'" % skey,
                    "evidence": [{"kind": "space", "name": space_display(s), "guid": s["guid"]} for s in sel]}
        ev = [{"kind": ent, "name": ent, "guid": g} for g in facts.get("guids", {}).get(ent, {}).get(skey, [])]
        return {"answer": c.get(skey, 0), "unit": "count", "evidence": ev, "calc": "%s elements contained in storey '%s'" % (ent, skey)}
    if ent == "space":
        sel = [s for s in facts["spaces"] if _n(s["storey"]) == _n(skey)]
        if plan.get("by") == "room":
            want = _n(slots.get("room"))
            sel = [s for s in sel if _n(space_display(s)) == want] or [s for s in facts["spaces"] if _n(space_display(s)) == want]
        sel = [s for s in sel if s["area_m2"] is not None]
        if not sel:
            return {"answer": None, "unit": "m2", "evidence": [], "calc": "no matching space"}
        tot = sum(s["area_m2"] for s in sel)
        return {"answer": round(tot, 3), "unit": "m2", "calc": "footprint area of %d space(s) on '%s' = %.2f m²" % (len(sel), skey, tot),
                "evidence": [{"kind": "space", "name": space_display(s), "guid": s["guid"], "value": round(s["area_m2"], 2)} for s in sel]}
    table = facts["walls"] if ent == "wall" else facts["slabs"]
    a = table.get(skey)
    if not a or not a.get("n_qto"):
        return {"answer": None, "unit": None, "evidence": [], "calc": "the model carries no quantities for %ss on '%s'" % (ent, skey)}
    measure = plan["measure"]
    unit = {"length": "m", "net_side_area": "m2", "net_volume": "m3"}[measure]
    ev = [{"kind": ent, "name": ent, "guid": g} for g in facts.get("guids", {}).get(ent, {}).get(skey, [])]
    return {"answer": round(a[measure], 3), "unit": unit, "evidence": ev,
            "calc": "sum of %s over %d %s(s) on '%s' (model base quantities)" % (measure, a["n"], ent, skey)}


EN = [  # (regex, plan, slot names in group order)
    (r"How many rooms \(spaces\) are shown on the floor plan of '(.+)'\?$", {"op": "count", "entity": "space", "scope": "storey"}, ["storey"]),
    (r"How many (doors|stairs|walls|columns|windows) are shown on the floor plan of '(.+)'\?$", None, ["ent", "storey"]),
    (r"How many rooms \(spaces\) are there in the model '.+'\?$", {"op": "count", "entity": "space", "scope": "model"}, []),
    (r"How many (walls|windows|doors|beams|stairs|columns|sanitary fixtures|light fixtures) are there in the model '.+'\?$", None, ["ent"]),
    (r"What is the net floor area of the room '(.+)' on '(.+)', in m²\?$", {"op": "area", "entity": "space", "by": "room"}, ["room", "storey"]),
    (r"What is the total net floor area of all rooms \(spaces\) on '(.+)', in m²\?$", {"op": "area", "entity": "space", "by": "storey"}, ["storey"]),
    (r"What is the total length of all walls on '(.+)', in metres\?$", {"op": "sum", "entity": "wall", "measure": "length"}, ["storey"]),
    (r"What is the total net side area of all walls on '(.+)', in m²\?$", {"op": "sum", "entity": "wall", "measure": "net_side_area"}, ["storey"]),
    (r"What is the total net volume of all walls on '(.+)', in m³\?$", {"op": "sum", "entity": "wall", "measure": "net_volume"}, ["storey"]),
    (r"What is the total net volume of all slabs on '(.+)', in m³\?$", {"op": "sum", "entity": "slab", "measure": "net_volume"}, ["storey"]),
]
ENT = {"doors": "door", "stairs": "stair", "walls": "wall", "columns": "column", "windows": "window", "beams": "beam",
       "sanitary fixtures": "sanitary", "light fixtures": "light"}


def en_plan(q):
    for rx, plan, names in EN:
        m = re.match(rx, q)
        if not m:
            continue
        slots = dict(zip(names, m.groups()))
        if plan is None:
            plan = {"op": "count", "entity": ENT[slots.pop("ent")], "scope": "storey" if "storey" in slots else "model"}
        return plan, slots
    return None, None


def validate():
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "ifc_models" and re.match(r"(How|What) ", r["question"])]
    agg = collections.defaultdict(lambda: [0, 0])
    miss = collections.defaultdict(list)
    for r in rows:
        plan, slots = en_plan(r["question"])
        facts = facts_for(model_path(r["path"]))
        if plan is None:
            kind, ok, val = "UNKNOWN_TEMPLATE", False, None
        elif not facts or "error" in facts:
            kind, ok, val = "NO_FACTS", False, None
        else:
            kind = "%s:%s:%s" % (plan["op"], plan["entity"], plan.get("measure") or plan.get("by") or plan.get("scope"))
            val = execute(plan, slots, facts)["answer"]
            ok = val is not None and evalkit.is_correct(val, r["answer"], r["type"])
        for k in ("ALL", kind):
            agg[k][0] += ok
            agg[k][1] += 1
        if not ok and len(miss[kind]) < 4:
            miss[kind].append((os.path.basename(r["path"])[:38], r["answer"], val, r["question"][:70]))
    for k, (a, b) in sorted(agg.items()):
        print("  %-34s %5d / %5d  %.1f%%" % (k, a, b, 100 * a / b))
    for k, v in miss.items():
        print(k)
        for x in v:
            print("    ", x)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        build(tuple(sys.argv[2:]) or ("ifc_models",))
    elif sys.argv[1:2] == ["validate"]:
        validate()
    else:
        print(__doc__)
