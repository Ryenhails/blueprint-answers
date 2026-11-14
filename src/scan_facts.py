"""Scanned floor plans go into the same floor-plan fact table as vector plans and small images.

A scan has no vector data, so the table is filled by the tools the sheet-reading step already uses, run once per sheet:

  1  one model read of the whole sheet: what kind of drawing it is, its scale, and a list of the rooms
     (number, label, kind, apartment, printed area, and one point inside each room)
  2  code measures every room from its point (measure.py): net area, perimeter, outline
  3  the sheet is read piece by piece for doors, windows and fixtures (tile_count.py): one box per object
  4  code joins boxes to rooms by geometry

The result has the fields of the vector plan table, so the query cache, the code-writing step and the sheet-reading
step work on it unchanged. Areas printed on the sheet are kept as printed; the others are measured.

Sheets that are not floor plans (piping, structural, details, elevations) have no rooms to list. They are marked as
outliers here and stay with the per-question reading step.

  python scan_facts.py show <image>       ingest if needed and print the fact tables
"""
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import gemini_backend as gb  # noqa: E402
import raster_ingest as ri  # noqa: E402

MODEL = os.environ.get("STOA_VISION_MODEL", "gemini-4-hydrogen")
STORE = os.path.join(evalkit.ROOT, "data", "raster_scan_plan.jsonl")
TYPES = ["bedroom", "kitchen", "bathroom", "wc", "living", "hall", "storage", "utility", "balcony", "outdoor", "stairs", "office", "classroom", "other"]
PROMPT = """This image is a scanned sheet of construction drawings. Say what it is and, if it is a floor plan, list its rooms.
Answer with JSON only:
{"drawing_type": "<floor plan | elevation | section | detail | schedule | site plan | services plan | structural plan | other>",
 "title": "<sheet title as printed>",
 "floor_plan": <true if the sheet's main content is a floor plan whose rooms are enclosed by walls and can be listed; otherwise false>,
 "scale": "<scale of the plan exactly as printed, e.g. 1:100 or 1/8\\" = 1'-0\\"; null if none>",
 "paper": "<sheet size if printed, e.g. A3, ISO A1, ANSI D; else null>",
 "scale_bar": {"from": [y, x], "to": [y, x], "length": "<real length between these two tick marks, e.g. 10 m>"} or null,
 "area_unit": "<unit of the areas printed in the rooms: m2, SF, or null if none are printed>",
 "symbols": {"door": "<how a door is drawn on this sheet>", "window": "...", "toilet": "...", "washbasin": "...", "bathtub": "...", "shower": "..."},
 "rooms": [{"number": "<room number as printed, or empty>", "label": "<room name as printed, or empty>", "type": "<one of: %s>",
            "kind": "<indoor | circulation | outdoor>", "apartment": "<name of the dwelling unit the room belongs to if units are marked, else null>",
            "area": <area printed in or next to the room as a number, or null>, "point": [y, x]}]}
All coordinates are 0-1000 relative to the image. point is a spot of empty floor inside the room, away from text, furniture and fixtures.
kind: circulation = stairs, lifts and corridors shared by the whole floor; outdoor = balconies, loggias, terraces; indoor = everything else.
symbols: describe each from the sheet's legend if it has one, else from what is drawn; use null for a kind that does not occur on the sheet.
List every room, including small ones without a printed area. If floor_plan is false, rooms is an empty list.""" % ", ".join(TYPES)
THINGS = [("single_doors", "door", "single door (one leaf)", "D"), ("double_doors", "door", "double door: one opening with two leaves, boxed once", "DD"),
          ("windows", "window", "window", "W"), ("toilets", "toilet", "toilet bowl", "T"), ("washbasins", "washbasin", "washbasin", "B"),
          ("bathtubs", "bathtub", "bathtub", "U"), ("showers", "shower", "shower", "S")]


def _overview(full):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(full).convert("RGB")
    W, H = im.size
    im.thumbnail((3072, 3072), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}, {"text": PROMPT}]}],
            "generationConfig": {"maxOutputTokens": 40000, "thinkingConfig": {"thinkingLevel": "low"}}}
    usd, tok, out = 0.0, 0, None
    for a in range(4):
        try:
            req = urllib.request.Request(gb.URL % MODEL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
            j = json.loads(urllib.request.urlopen(req, timeout=400).read())
        except Exception:
            time.sleep(3 * 2 ** a)
            continue
        u = j.get("usageMetadata", {})
        n_in, n_out = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        pin, pout = gb.PRICES.get(MODEL, (0, 0))
        usd, tok = usd + (n_in * pin + n_out * pout) / 1e6, tok + n_in + n_out
        text = "".join(p.get("text", "") for p in (j.get("candidates") or [{}])[0].get("content", {}).get("parts", []) if not p.get("thought"))
        try:
            out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
            break
        except Exception:
            continue  # an empty or cut-off reply: ask again
    if out is None:
        raise RuntimeError("the sheet overview could not be read")
    return out, (W, H), tok, usd


def split(poly, points, mpp):
    """Rooms that the fill could not separate (open doorways, no door leaf): cut the shared region where it is
    closest to each room's point. -> one {area_m2, perimeter_m, bbox, poly} per point (None if a part is empty)."""
    import cv2
    import numpy as np

    P = np.array(poly, dtype=np.float64)
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    k = min(1.0, 900.0 / max(P[:, 0].max() - x0, P[:, 1].max() - y0, 1.0))
    w, h = int((P[:, 0].max() - x0) * k) + 3, int((P[:, 1].max() - y0) * k) + 3
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round((P - [x0, y0]) * k + 1).astype(np.int32)], 1)
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.stack([(xx - ((px - x0) * k + 1)) ** 2 + (yy - ((py - y0) * k + 1)) ** 2 for px, py in points])
    owner = d.argmin(axis=0)
    out = []
    for i in range(len(points)):
        part = ((owner == i) & (mask > 0)).astype(np.uint8)
        cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            out.append(None)
            continue
        c = max(cs, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
        pl = [[float((p[0] - 1) / k + x0), float((p[1] - 1) / k + y0)] for p in ap]
        if len(pl) < 3:
            out.append(None)
            continue
        xs, ys = [p[0] for p in pl], [p[1] for p in pl]
        s = mpp / k
        out.append({"area_m2": round(float(part.sum()) * s * s, 2), "perimeter_m": round(float(cv2.arcLength(ap.reshape(-1, 1, 2), True)) * s, 2),
                    "bbox": [min(xs), min(ys), max(xs), max(ys)], "poly": pl})
    return out


def ingest(full):
    """Read one scanned sheet into fact tables. -> record (not stored here)"""
    import measure
    import tile_count

    ov, (W, H), tok, usd = _overview(full)
    rec = {"path": ri._key(full), "size": [W, H], "model": MODEL, "drawing_type": ov.get("drawing_type"), "title": ov.get("title"), "scale": ov.get("scale")}
    listed = [r for r in ov.get("rooms") or [] if isinstance(r, dict) and isinstance(r.get("point"), list) and len(r["point"]) == 2]
    if not ov.get("floor_plan") or not listed:
        rec.update({"outlier": "not a floor plan (%s)" % (ov.get("drawing_type") or "unknown"), "tokens": tok, "usd": round(usd, 5)})
        return rec
    mpp, how = measure.metres_per_pixel(W, H, ov.get("scale"), ov.get("paper"), ov.get("scale_bar"))
    to_m2 = 0.092903 if str(ov.get("area_unit") or "").upper().startswith(("SF", "SQ", "FT")) else 1.0
    rooms, geo, meas = [], {}, []
    ids = [measure.label_at(full, r["point"][0], r["point"][1], mpp) if mpp else None for r in listed]
    for i, r in enumerate(listed):
        m = {}
        if mpp:
            try:
                m = measure.region(full, r["point"][0], r["point"][1], mpp, others={o for o in ids if o is not None and o != ids[i]})
            except Exception as e:
                m = {"error": repr(e)[:80]}
        meas.append(m)
    pts = [[r["point"][1] / 1000.0 * W, r["point"][0] / 1000.0 * H] for r in listed]
    groups, shared = {}, {}
    for i, m in enumerate(meas):
        if "bbox" in m:
            groups.setdefault(tuple(round(v / 4) for v in m["bbox"]), []).append(i)
    for idx in groups.values():  # several listed rooms in one filled region: no line separates them, so cut it between their points
        if len(idx) > 1:
            parts = split(meas[idx[0]]["poly"], [pts[i] for i in idx], mpp)
            for i, p in zip(idx, parts):
                if p:
                    meas[i] = dict(p, method="shared region, cut between the rooms")
                    shared[i] = [j for j in idx if j != i]
    for i, r in enumerate(listed):
        rid, m = "R%d" % (i + 1), meas[i]
        printed = r.get("area") if isinstance(r.get("area"), (int, float)) and r["area"] > 0 else None
        row = {"id": rid, "number": str(r.get("number") or ""), "label": str(r.get("label") or ""), "type": r.get("type") if r.get("type") in TYPES else "other",
               "kind": r.get("kind") if r.get("kind") in ("indoor", "circulation", "outdoor") else "indoor", "apartment": r.get("apartment") or None,
               "area_m2": None, "area_source": None, "perimeter_m": m.get("perimeter_m"),
               "toilets": 0, "washbasins": 0, "bathtubs": 0, "showers": 0, "doors": [], "windows": [], "connected_rooms": []}
        if printed is not None:
            row.update({"area_m2": round(printed * to_m2, 2), "area_source": "printed on the sheet"})
        elif "area_m2" in m:
            row.update({"area_m2": m["area_m2"], "area_source": "measured by code" + (" (open to %s: no line between them, cut halfway)" % ", ".join("R%d" % (j + 1) for j in shared[i]) if i in shared else "")})
        else:
            row["area_source"] = "not measured: %s" % (m.get("error") or how)
        if i in shared:
            row["connected_rooms"] = ["R%d" % (j + 1) for j in shared[i]]
        rooms.append(row)
        geo[rid] = {"kind": "room", "name": ("%s %s" % (row["number"], row["label"])).strip() or row["type"], "value": row["area_m2"],
                    "bbox": m.get("bbox") or [pts[i][0] - 20, pts[i][1] - 20, pts[i][0] + 20, pts[i][1] + 20], "poly": m.get("poly")}
    # objects: piece by piece, inside the area the rooms cover
    xs = [v for g in geo.values() for v in (g["bbox"][0], g["bbox"][2])]
    ys = [v for g in geo.values() for v in (g["bbox"][1], g["bbox"][3])]
    padx, pady = 0.04 * (max(xs) - min(xs)) + 30, 0.04 * (max(ys) - min(ys)) + 30
    area = (max(0, min(xs) - padx), max(0, min(ys) - pady), min(W, max(xs) + padx), min(H, max(ys) + pady))
    sym = ov.get("symbols") if isinstance(ov.get("symbols"), dict) else {}
    things = {key: "%s. On this sheet: %s" % (desc, sym[s]) for key, s, desc, _ in THINGS if sym.get(s)}
    found, c_usd, n_tiles = tile_count.count(full, things, region=area) if things else ({}, 0.0, 0)
    usd += c_usd
    from shapely.geometry import Polygon, box

    polys = []
    for r in rooms:
        p = geo[r["id"]].get("poly")
        try:
            polys.append(Polygon(p).buffer(0) if p and len(p) >= 3 else None)
        except Exception:
            polys.append(None)
    reach = (0.3 / mpp) if mpp else 0.01 * (W + H)

    def near(b, k):
        g = box(*b).buffer(reach * k)
        return [j for j, p in enumerate(polys) if p is not None and not p.is_empty and p.intersects(g)]

    tables = {"rooms": rooms, "doors": [], "windows": [], "fixtures": [], "apartments": sorted({r["apartment"] for r in rooms if r["apartment"]}),
              "scale": ov.get("scale") or "unknown", "readers": {"rooms": "model list + code measurement", "objects": "piece-by-piece reading"}}
    for key, s, desc, pre in THINGS:
        for n, b in enumerate(found.get(key, [])):
            oid = "%s%d" % (pre, n + 1)
            geo[oid] = {"kind": s, "name": desc.split(":")[0].split(" (")[0], "bbox": b}
            if s == "door":
                js = near(b, 1.0)
                ids = [rooms[j]["id"] for j in js]
                tables["doors"].append({"id": oid, "rooms": ids, "leaves": 2 if key == "double_doors" else 1})
                for j in js:
                    rooms[j]["doors"].append(oid)
                    rooms[j]["connected_rooms"] = sorted(set(rooms[j]["connected_rooms"]) | (set(ids) - {rooms[j]["id"]}))
            elif s == "window":
                js = near(b, 1.3)
                if len(js) > 1:
                    js = [max(js, key=lambda j: polys[j].intersection(box(*b).buffer(reach * 1.3)).area)]
                tables["windows"].append({"id": oid, "rooms": [rooms[j]["id"] for j in js]})
                for j in js:
                    rooms[j]["windows"].append(oid)
            else:
                c = box(*b).centroid
                js = [j for j, p in enumerate(polys) if p is not None and p.contains(c)] or near(b, 0.5)
                room = rooms[js[0]] if js else None
                tables["fixtures"].append({"id": oid, "type": s, "room": room["id"] if room else None})
                if room:
                    room[key] += 1
    rec.update({"tables": tables, "geo": geo, "how": how, "tiles": n_tiles, "tokens": tok, "usd": round(usd, 5), "mpp": mpp,
                "listed": [{"point": r["point"], "area": r.get("area")} for r in listed]})  # kept so rooms can be measured again without a model call
    return rec


def ensure(full):
    """The stored record of a sheet; read now if it is new. -> (record, read just now?)"""
    key = ri._key(full)
    if key in ri._load(STORE):
        return ri._stores[STORE][key], False
    rec = ingest(full)
    with ri._lock:
        ri._stores[STORE][key] = rec
        with open(STORE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec, True


def view(full):
    """-> (tables, geo, tokens, usd) or None for a sheet that is not a floor plan."""
    rec, fresh = ensure(full)
    if "tables" not in rec:
        return None
    return json.loads(json.dumps(rec["tables"])), rec["geo"], rec["tokens"] if fresh else 0, rec["usd"] if fresh else 0.0


if __name__ == "__main__":
    if sys.argv[1:2] == ["show"]:
        import agent

        f = sys.argv[2] if os.path.isabs(sys.argv[2]) else os.path.join(evalkit.DATA, sys.argv[2])
        rec, fresh = ensure(f)
        print("%s | %s | read cost $%.3f%s" % (rec.get("drawing_type"), rec.get("title"), rec["usd"], "" if fresh else " (stored)"))
        if "tables" in rec:
            print(rec["how"], "|", rec["tiles"], "pieces")
            print(agent.describe(rec["tables"])[:5000])
        else:
            print(rec["outlier"])
    else:
        print(__doc__)
