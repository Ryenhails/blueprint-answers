"""Step 1 adapter: swiss_dwellings vector PDF -> fact table (dict).

Everything is read from the PDF itself (vector paths + text layer); the sibling
DXF is never opened. Geometry is in PDF points; `m_per_pt` converts to metres
using the scale printed in the title block.

Drawing conventions (found empirically, validated against questions.jsonl):
  room outline   fill+stroke, stroke grey 0.61, width 0.37; fill encodes room class
  unit outline   stroke red, width 1.28
  unit label     black/white rounded box (4 curves + 4 lines) with "App. X" text
  door           grey (0.34) quarter-arc 'qu'
  window         wall-thickness white rect (black stroke 0.51) containing 2 thin black lines
  washbasin      fixture rect containing a 47-segment bowl and a drain circle
  toilet         black 0.51 single line (cistern line) next to a 47-segment oval
  bathtub        black 0.51 polyline with 68 segments
  shower         black 0.51 quarter-arc 'qu'
"""
import math
import re

import fitz
from shapely.geometry import Point, Polygon

LEGEND_RE = re.compile(r"^(Légende|Legende|Legend|Legenda)$")
ROOM_NO_RE = re.compile(r"^[A-Z]?\d+\.\d+$")


def _col(c):
    return None if c is None else tuple(round(x, 2) for x in c)


def _pts(items):
    """Vertices of a path made of line / rect items."""
    out = []
    for it in items:
        if it[0] == "l":
            if not out or out[-1] != it[1]:
                out.append(it[1])
            out.append(it[2])
        elif it[0] == "re":
            r = it[1]
            out += [fitz.Point(r.x0, r.y0), fitz.Point(r.x1, r.y0), fitz.Point(r.x1, r.y1), fitz.Point(r.x0, r.y1)]
        elif it[0] == "qu":
            q = it[1]
            out += [q.ul, q.ur, q.lr, q.ll]
    return [(p.x, p.y) for p in out]


def _poly(items):
    pts = _pts(items)
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    pg = Polygon(pts)
    if not pg.is_valid:
        pg = pg.buffer(0)
    return pg if pg.area > 0 else None


def parse(pdf_path):
    page = fitz.open(pdf_path)[0]
    words = page.get_text("words")
    leg = [w[1] for w in words if LEGEND_RE.match(w[4])]
    ytop = (min(leg) - 12) if leg else page.rect.height * 0.82
    m = re.search(r"1:(\d+)", " ".join(w[4] for w in words if w[1] > ytop))
    scale = int(m.group(1)) if m else 100
    k = scale * 0.0254 / 72.0  # metres per PDF point

    rooms, units, unit_boxes, doors, win_rects, thin, wc_lines, ovals, baths, showers, dots = ([] for _ in range(11))
    for d in page.get_drawings():
        r = d["rect"]
        if r.y0 > ytop or r.width > page.rect.width * 0.8:
            continue
        its, t = d["items"], d["type"]
        color, fill, w = _col(d.get("color")), _col(d.get("fill")), round(d.get("width") or 0, 2)
        kinds = "".join(sorted(set(i[0] for i in its)))
        n = len(its)
        bb = (r.x0, r.y0, r.x1, r.y1)
        if t == "fs" and color == (0.61, 0.61, 0.61):
            pg = _poly(its)
            if pg is not None:
                rooms.append({"poly": pg, "fill": fill})
        elif t == "s" and color and color[0] > 0.5 and color[1] < 0.4:
            pg = _poly(its)
            if pg is not None:
                units.append(pg)
        elif t == "fs" and color == (0, 0, 0) and fill == (1, 1, 1) and kinds == "cl":
            unit_boxes.append(bb)
        elif t == "s" and kinds == "qu" and color not in ((0, 0, 0), None):
            doors.append(bb)
        elif t == "fs" and color == (0, 0, 0) and fill == (1, 1, 1) and kinds in ("re", "l") and n in (1, 4):
            pg = _poly(its)
            if pg is not None:
                win_rects.append(pg)
        elif t == "s" and color == (0, 0, 0) and kinds == "l" and n == 1 and w < 0.45:
            thin.append((its[0][1], its[0][2]))
        elif t == "s" and color == (0, 0, 0) and 0.45 <= w <= 0.6:
            if kinds == "l" and n == 1:
                wc_lines.append(bb)
            elif kinds == "l" and n == 47:
                ovals.append(bb)
            elif kinds == "l" and n == 68:
                baths.append(bb)
            elif kinds == "qu":
                showers.append(bb)
            elif kinds == "c" and n == 8:
                dots.append(bb)

    def _sides(pg):
        xs, ys = pg.minimum_rotated_rectangle.exterior.coords.xy
        a = math.dist((xs[0], ys[0]), (xs[1], ys[1])) * k
        b = math.dist((xs[1], ys[1]), (xs[2], ys[2])) * k
        return min(a, b), max(a, b)

    def _inside(g, b):
        return g.contains(Point(b[0], b[1])) and g.contains(Point(b[2], b[3]))

    # windows: wall-thickness white rects holding exactly two thin lines
    # washbasins: fixture rects holding a 47-gon bowl and a drain circle (a toilet has a cistern line instead)
    windows, basins = [], []
    for pg in win_rects:
        g = pg.buffer(0.6)
        short, long_ = _sides(pg)
        # glazing lines run the full length of the window (stair treads run across the short side)
        n_thin = sum(1 for a, b in thin if g.contains(Point(a.x, a.y)) and g.contains(Point(b.x, b.y))
                     and math.dist((a.x, a.y), (b.x, b.y)) * k >= 0.9 * long_)
        # drawings sometimes repeat a window at the same spot (2 rects, 4 lines): each rect counts
        # ...and a window sits in a wall, never in the interior of a room (stairs / lifts do)
        ctr = pg.centroid
        in_room = any(r["poly"].buffer(-3).contains(ctr) for r in rooms)
        if n_thin >= 2 and n_thin % 2 == 0 and short <= 1.2 and not in_room:
            windows.append(pg.bounds)
            continue
        if any(_inside(g, o) for o in ovals) and any(_inside(g, d) for d in dots) and not any(_inside(g, l) for l in wc_lines):
            basins.append(pg.bounds)

    # text: room numbers, labels, unit names
    blocks = page.get_text("dict")["blocks"]
    lines = []
    for b in blocks:
        for ln in b.get("lines", []):
            txt = " ".join(s["text"] for s in ln["spans"]).strip()
            if txt and ln["bbox"][1] < ytop:
                x0, y0, x1, y1 = ln["bbox"]
                lines.append({"text": txt, "bbox": ln["bbox"], "c": ((x0 + x1) / 2, (y0 + y1) / 2), "h": y1 - y0})
    unit_names = []
    used = set()
    for bx in unit_boxes:
        for i, ln in enumerate(lines):
            cx, cy = ln["c"]
            if bx[0] - 1 <= cx <= bx[2] + 1 and bx[1] - 1 <= cy <= bx[3] + 1:
                unit_names.append({"name": ln["text"], "c": ln["c"], "bbox": bx})
                used.add(i)
    numbers = [(i, ln) for i, ln in enumerate(lines) if ROOM_NO_RE.match(ln["text"]) and i not in used]
    for i, _ in numbers:
        used.add(i)
    label_lines = [ln for i, ln in enumerate(lines) if i not in used]

    out_rooms = []
    for i, ln in numbers:
        cx, cy = ln["c"]
        cands = [r for r in rooms if r["poly"].contains(Point(cx, cy))]
        if not cands:
            cands = sorted(rooms, key=lambda r: r["poly"].distance(Point(cx, cy)))[:1]
            if not cands or cands[0]["poly"].distance(Point(cx, cy)) > 15:
                cands = []
        room = min(cands, key=lambda r: r["poly"].area) if cands else None
        # label = text line(s) directly above the number (same column)
        lab = [l for l in label_lines if abs(l["c"][0] - cx) < 12 + 0.5 * (l["bbox"][2] - l["bbox"][0]) * 0 and 0 < cy - l["c"][1] < 2.6 * max(ln["h"], l["h"]) and abs(l["c"][0] - cx) < 14]
        lab.sort(key=lambda l: l["c"][1])
        label = " ".join(l["text"] for l in lab) if lab else None
        out_rooms.append({
            "no": ln["text"], "label": label,
            "area_m2": room["poly"].area * k * k if room else None,
            "fill": room["fill"] if room else None,
            "poly": room["poly"] if room else None,
            "c": (cx, cy),
        })
    # unit membership
    out_units = []
    for pg in units:
        nm = next((u["name"] for u in unit_names if pg.buffer(2).contains(Point(*u["c"]))), None)
        out_units.append({"name": nm, "poly": pg})
    for r in out_rooms:
        p = r["poly"].representative_point() if r["poly"] is not None else Point(*r["c"])
        r["unit"] = next((u["name"] for u in out_units if u["poly"].buffer(2).contains(p)), None)

    return {
        "scale": scale, "m_per_pt": k, "rooms": out_rooms, "units": out_units, "unit_names": unit_names,
        "n_room_polys": len(rooms),
        "doors": doors, "windows": windows, "toilets": wc_lines, "bathtubs": baths, "showers": showers, "washbasins": basins,
    }
