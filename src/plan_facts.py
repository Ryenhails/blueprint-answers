"""One fact-table structure for every floor plan, whatever the file is.

  rooms     id, label, type, area_m2, counts of fixtures inside, ids of its doors and windows, connected_rooms
  doors     id, rooms it touches            windows   id, rooms it belongs to
  fixtures  id, type, room

Who fills it depends on the file, the structure does not:
  vector plan (PDF with a legend)  ->  the exact parser (swiss_adapter): geometry, areas and relations computed by code
  image (PNG / JPG)                ->  the image readers: object boxes from the detector or the model read
                                       (raster_ingest), a room list read once by the model, and the same
                                       geometric join (which room contains which box)
So every question-side step (query cache, code-writing model, sheet reading with facts) works on one structure and
never needs to know which reader produced it. `exact` tells them how far to trust the numbers.

  python plan_facts.py show <file>          print the fact tables
"""
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import gemini_backend as gb  # noqa: E402
import raster_ingest as ri  # noqa: E402

ROOM_MODEL = os.environ.get("STOA_ROOMLIST_MODEL", "gemini-4-hydrogen")
STORE = os.path.join(evalkit.ROOT, "data", "raster_plan.jsonl")
TYPES = ["bedroom", "kitchen", "bathroom", "wc", "sauna", "balcony", "living", "hall", "storage", "utility", "outdoor", "other"]
PROMPT = """This is a residential floor plan (Finnish plans use labels such as MH, OH, K, KK, KH, KPH, PH, WC, S, VH, ET, PARVEKE).
List every room and labelled space drawn on it. Answer with JSON only:
{"floors": <number of floors (storeys) the sheet shows>,
 "rooms": [{"label": "<text printed in the room, or empty>", "type": "<one type>", "floor": <1, 2, ... which of the drawn floors>,
            "area_m2": <number printed in the room, or null>, "box": [ymin, xmin, ymax, xmax]}]}
box uses 0-1000 coordinates relative to the image. Types:
- bedroom: bedrooms (makuuhuone, MH)
- kitchen: kitchens, including kitchenettes (KK), open kitchens and utility kitchens
- bathroom: bathrooms or washrooms (KH, KPH, PH, PESUH, SH); a room with a toilet and a shower or bath is a bathroom
- wc: separate WC rooms (a toilet room marked WC, without shower or bath)
- sauna: sauna rooms (S, SAUNA)
- balcony: balconies (parveke), including glazed balconies
- living: living rooms (OH), dining
- hall: entrance halls and corridors (ET, eteinen)
- storage: closets and storage (VH, VAR, walk-in closets)
- utility: utility and laundry rooms (KHH), technical rooms
- outdoor: terraces, patios, porches, yards
- other: garages, stairs, anything else"""
OBJ = [("windows", "W"), ("doors", "D"), ("toilets", "T"), ("sinks", "B"), ("showers_baths", "S")]


def room_list(full):
    """The model lists every room once (label, type, box, printed area); stored. -> (record, read just now?)"""
    key = ri._key(full)
    if key in ri._load(STORE):
        return ri._stores[STORE][key], False
    from PIL import Image

    W, H = Image.open(full).size
    mime = "image/png" if full.lower().endswith(".png") else "image/jpeg"
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": mime, "data": base64.b64encode(open(full, "rb").read()).decode()}}, {"text": PROMPT}]}],
            "generationConfig": {"maxOutputTokens": 30000, "thinkingConfig": {"thinkingLevel": "low"}}}
    for a in range(5):
        try:
            req = urllib.request.Request(gb.URL % ROOM_MODEL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
            j = json.loads(urllib.request.urlopen(req, timeout=300).read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and a < 4:
                time.sleep(3 * 2 ** a)
                continue
            raise RuntimeError("Gemini API %d" % e.code)
        except (urllib.error.URLError, TimeoutError):
            if a == 4:
                raise
            time.sleep(3)
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    out = json.loads(re.search(r"\{.*\}", "".join(p.get("text", "") for p in parts if not p.get("thought")), re.S).group(0))
    rooms = []
    for r in out.get("rooms") or []:
        b = r.get("box") if isinstance(r, dict) else None
        if isinstance(b, list) and len(b) == 4:
            rooms.append({"label": str(r.get("label") or ""), "type": r.get("type") if r.get("type") in TYPES else "other", "floor": r.get("floor") or 1,
                          "area_m2": r.get("area_m2") if isinstance(r.get("area_m2"), (int, float)) else None,
                          "bbox": [round(b[1] / 1000 * W), round(b[0] / 1000 * H), round(b[3] / 1000 * W), round(b[2] / 1000 * H)]})
    u = j.get("usageMetadata", {})
    n_in, n_out = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    pin, pout = gb.PRICES.get(ROOM_MODEL, (0, 0))
    rec = {"path": key, "size": [W, H], "floors": out.get("floors"), "rooms": rooms, "tokens": n_in + n_out,
           "usd": round((n_in * pin + n_out * pout) / 1e6, 5), "model": ROOM_MODEL}
    with ri._lock:
        ri._stores[STORE][key] = rec
        with open(STORE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec, True


def image_view(full):
    """-> (tables, geometry by id, tokens, usd) in the same structure agent.view() builds for vector plans."""
    key = ri._key(full)
    rec, fresh = room_list(full)
    tokens, usd = (rec["tokens"], rec["usd"]) if fresh else (0, 0.0)
    W, H = rec["size"]
    pad = 0.012 * (W + H)
    rooms, geo = [], {}
    for i, r in enumerate(rec["rooms"]):
        rid = "R%d" % (i + 1)
        rooms.append({"id": rid, "label": r["label"], "type": r["type"], "floor": r["floor"], "area_m2": r["area_m2"],
                      "kind": "outdoor" if r["type"] in ("balcony", "outdoor") else "indoor",
                      "toilets": 0, "sinks": 0, "showers_baths": 0, "doors": [], "windows": [], "connected_rooms": []})
        geo[rid] = {"kind": "room", "name": ("%s %s" % (r["label"], r["type"])).strip(), "bbox": r["bbox"]}
    boxes = [r["bbox"] for r in rec["rooms"]]

    def inside(b):  # the smallest room box containing the centre
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        js = [j for j, rb in enumerate(boxes) if rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]]
        return min(js, key=lambda j: (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])) if js else None

    def touching(b):
        return [j for j, rb in enumerate(boxes) if b[0] - pad <= rb[2] and b[2] + pad >= rb[0] and b[1] - pad <= rb[3] and b[3] + pad >= rb[1]]

    tables = {"rooms": rooms, "doors": [], "windows": [], "fixtures": [], "floors": rec["floors"], "readers": {}}
    for kind, pre in OBJ:
        have = ri.stored(key, kind)
        if not any(v["boxes"] for s, v in have.items() if s != "seg") and "model" not in have and "detector" not in have:
            g, f2 = ri.ingest(full)  # an image no reader has seen: one full model read gives the object boxes
            if f2:
                tokens, usd = tokens + g["tokens"], usd + g["usd"]
            have = ri.stored(key, kind)
        want = ri.sources().get(kind, "model")
        use = want if want in have and (have[want]["boxes"] or have[want]["count"] == 0) else next((s for s in ("detector", "model", "seg") if s in have), None)
        if use is None:
            continue
        tables["readers"][kind] = use
        for n, b in enumerate(have[use]["boxes"]):
            oid = "%s%d" % (pre, n + 1)
            geo[oid] = {"kind": kind, "name": kind.rstrip("s").replace("_", " "), "bbox": b}
            if kind in ("doors", "windows"):
                js = touching(b)
                if kind == "windows" and len(js) > 1:  # a window belongs to the room it overlaps most along its length
                    js = [max(js, key=lambda j: max(0, min(b[2], boxes[j][2]) - max(b[0], boxes[j][0])) + max(0, min(b[3], boxes[j][3]) - max(b[1], boxes[j][1])))]
                ids = [rooms[j]["id"] for j in js]
                tables[kind].append({"id": oid, "rooms": ids})
                for j in js:
                    rooms[j][kind].append(oid)
                    if kind == "doors":
                        rooms[j]["connected_rooms"] = sorted(set(rooms[j]["connected_rooms"]) | (set(ids) - {rooms[j]["id"]}))
            else:
                j = inside(b)
                tables["fixtures"].append({"id": oid, "type": {"toilets": "toilet", "sinks": "sink", "showers_baths": "shower or bathtub"}[kind], "room": rooms[j]["id"] if j is not None else None})
                if j is not None:
                    rooms[j][kind] += 1
    return tables, geo, tokens, usd


def view(full):
    """-> (tables, geo, exact, tokens, usd) for any floor plan file, or None when the file is not a plan we can table."""
    import system

    kind = system.kind_of(full)
    if kind == "swiss":
        import agent
        import qa

        t, g = agent.view(qa.facts_for(full))
        return t, g, True, 0, 0.0
    if kind == "raster":
        t, g, tok, usd = image_view(full)
        return t, g, False, tok, usd
    return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["show"]:
        import agent

        f = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(evalkit.DATA, sys.argv[2])
        t, g, exact, tok, usd = view(f)
        print("exact:", exact, "| read cost $%.4f" % usd)
        print(agent.describe(t)[:6000])
    else:
        print(__doc__)
