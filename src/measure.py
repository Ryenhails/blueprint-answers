"""A ruler for scanned sheets: the model points, code measures.

A vision model is poor at judging sizes by eye. On an image of a drawing it is asked only to say WHERE (a spot inside a
room, the corners of a zone, two ends of a run) and to read the scale. The real-world size is then worked out here:

  scale      metres per pixel from the printed scale and the sheet size (a full sheet scanned edge to edge), checked
             against the scale bar when there is one; the bar alone when the sheet size is not known
  point      flood fill from the spot to the surrounding lines -> the enclosed room: net area, perimeter
  polygon    corners given by the model -> area, perimeter (zones that are not closed by lines)
  line       two ends -> length

  python measure.py region <image> <scale, e.g. 1:200> <paper, e.g. A2> <y,x in 0-1000> [...]
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAPER_MM = {"A0": 1189, "A1": 841, "A2": 594, "A3": 420, "A4": 297, "ANSI A": 279.4, "ANSI B": 431.8, "ANSI C": 558.8, "ANSI D": 863.6,
            "ANSI E": 1117.6, "ARCH A": 304.8, "ARCH B": 457.2, "ARCH C": 609.6, "ARCH D": 914.4, "ARCH E": 1219.2, "ARCH E1": 1066.8}  # long side
PAPER_RATIO = {"A": 2 ** 0.5, "ANSI A": 11 / 8.5, "ANSI B": 17 / 11, "ANSI C": 22 / 17, "ANSI D": 34 / 22, "ANSI E": 44 / 34,
               "ARCH A": 12 / 9, "ARCH B": 18 / 12, "ARCH C": 24 / 18, "ARCH D": 36 / 24, "ARCH E": 48 / 36, "ARCH E1": 42 / 30}
WORK = 6000  # pixels on the long side for the flood fill


def _feet_inches(s):
    """'1/4"', "1'-0\\"", "40'", '1 1/2"' -> inches"""
    s = s.replace("”", '"').replace("’", "'").replace("″", '"').replace("′", "'")
    ft = re.search(r"(\d+(?:\.\d+)?)\s*'", s)
    rest = s[ft.end():] if ft else s
    inch = 0.0
    m = re.search(r"(\d+)\s+(\d+)\s*/\s*(\d+)|(\d+)\s*/\s*(\d+)|(\d+(?:\.\d+)?)", rest)
    if m:
        if m.group(1):
            inch = int(m.group(1)) + int(m.group(2)) / int(m.group(3))
        elif m.group(4):
            inch = int(m.group(4)) / int(m.group(5))
        else:
            inch = float(m.group(6))
    return (float(ft.group(1)) * 12 if ft else 0.0) + inch


def parse_scale(s):
    """'1:200' -> 200.0 ; '1/4" = 1\\'-0"' -> 48.0 ; '1" = 40\\'-0"' -> 480.0 ; None if not a scale"""
    s = str(s or "")
    m = re.search(r"1\s*[:/]\s*(\d+(?:[.,]\d+)?)\s*$|1\s*:\s*(\d+(?:[.,]\d+)?)", s)
    if m and "=" not in s:
        return float((m.group(1) or m.group(2)).replace(",", "."))
    if "=" in s:
        a, b = s.split("=", 1)
        x, y = _feet_inches(a), _feet_inches(b)
        if x > 0 and y > 0:
            return y / x
    return None


def parse_length(s):
    """'20 m', '8\\'', '2000 mm', "80'" -> metres"""
    s = str(s or "").strip().lower()
    m = re.match(r"(-?\d+(?:[.,]\d+)?)\s*(mm|cm|m|ft|feet|in|inch|inches)?\b", s)
    if "'" in s or '"' in s:
        return _feet_inches(s) * 0.0254
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v * {"mm": 0.001, "cm": 0.01, "m": 1.0, None: 1.0, "ft": 0.3048, "feet": 0.3048, "in": 0.0254, "inch": 0.0254, "inches": 0.0254}[m.group(2)]


def metres_per_pixel(W, H, scale=None, paper=None, bar=None, sheet_mm=None):
    """bar: {"from": [y, x], "to": [y, x], "length": "20 m"} in 0-1000 image coordinates.
    -> (metres per pixel or None, plain sentence saying how it was found)"""
    den = parse_scale(scale)
    by_paper = by_bar = None
    name = re.sub(r"^(ISO|DIN)\s+", "", str(paper or "").upper().strip())
    name = {"D": "ANSI D", "E": "ANSI E", "C": "ANSI C", "B": "ANSI B"}.get(name, name)
    if den and sheet_mm:  # a PDF page: its paper size is known exactly
        by_paper, name = sheet_mm / 1000.0 / max(W, H) * den, "%d mm" % round(sheet_mm)
    elif den and name in PAPER_MM:
        ratio = PAPER_RATIO["A" if name[0] == "A" and name[1:].isdigit() else name]
        if abs(max(W, H) / min(W, H) / ratio - 1) < 0.02:  # the image is the whole sheet
            by_paper = PAPER_MM[name] / 1000.0 / max(W, H) * den
    if bar and bar.get("from") and bar.get("to"):
        L = parse_length(bar.get("length"))
        d = math.hypot((bar["to"][0] - bar["from"][0]) / 1000.0 * H, (bar["to"][1] - bar["from"][1]) / 1000.0 * W)
        if L and d > 0:
            by_bar = L / d
    if den and by_bar and not by_paper:  # sheet size not printed: a standard size that fits the image shape and the bar
        fits = [(abs(mm / 1000.0 / max(W, H) * den / by_bar - 1), n, mm) for n, mm in PAPER_MM.items()
                if abs(max(W, H) / min(W, H) / PAPER_RATIO["A" if n[0] == "A" and n[1:].isdigit() else n] - 1) < 0.01]
        if fits and min(fits)[0] < 0.08:
            _, name, mm = min(fits)
            by_paper = mm / 1000.0 / max(W, H) * den
    if by_paper and by_bar:
        diff = abs(by_paper / by_bar - 1) * 100
        if diff <= 5:
            return by_paper, "scale %s on a %s sheet, %d px wide: 1 px = %.2f mm (the scale bar agrees within %.1f%%)" % (scale, name, max(W, H), by_paper * 1000, diff)
        return by_bar, "scale bar: 1 px = %.2f mm (the printed scale %s on %s paper would give %.2f mm, %.0f%% off, so the bar is used)" % (by_bar * 1000, scale, name, by_paper * 1000, diff)
    if by_paper:
        return by_paper, "scale %s on a %s sheet, %d px wide: 1 px = %.2f mm" % (scale, name, max(W, H), by_paper * 1000)
    if by_bar:
        return by_bar, "scale bar (%s between the two marked ends): 1 px = %.2f mm" % (bar.get("length"), by_bar * 1000)
    return None, "no usable scale: the sheet shows neither a scale with a known sheet size nor a scale bar"


_img = {}


def _work(full):
    """-> (dark mask at working size, labels of the free-space regions, factor original px / working px, (W, H) original)"""
    hit = _img.get(full)
    if hit is not None:
        return hit
    import cv2
    import numpy as np
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(full)
    W, H = im.size
    f = max(1.0, max(W, H) / WORK)
    if f > 1:
        im = im.resize((round(W / f), round(H / f)), Image.LANCZOS)
    g = np.asarray(im.convert("L"))
    dark = g < 170
    if im.mode not in ("L", "1", "LA", "I", "I;16"):  # coloured lines (apartment boundaries, markups) are not walls
        rgb = np.asarray(im.convert("RGB")).astype(np.int16)
        dark &= (rgb.max(axis=2) - rgb.min(axis=2)) < 60
    dark = dark.astype(np.uint8)
    n, labels = cv2.connectedComponents(1 - dark, connectivity=4)
    out = (dark, labels, f, (W, H), g, np.bincount(labels.ravel()))
    _img.clear()  # one sheet in memory at a time
    _img[full] = out
    return out


def region(full, y, x, mpp):
    """Flood fill from (y, x) in 0-1000 coordinates. -> dict(area_m2, perimeter_m, bbox, poly) in original pixels, or {"error"}"""
    import cv2
    import numpy as np

    dark, labels, f, (W, H), gray, sizes = _work(full)
    h, w = dark.shape
    px, py = int(x / 1000.0 * (w - 1)), int(y / 1000.0 * (h - 1))
    r = max(3, int(0.25 / (mpp * f)))  # look 25 cm around the spot: it may sit on a label or a furniture line
    win = labels[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1]
    ids = win[win > 0]
    if ids.size == 0:
        return {"error": "the spot is on a solid line or wall"}
    lab = int(np.bincount(ids).argmax())
    big = max(1, int(1.0 / (mpp * f) ** 2))  # regions of 1 m² or more are rooms of their own; smaller ones are pieces of a room
    mask = (labels == lab).astype(np.uint8)
    edge = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    if edge or mask.sum() > 0.35 * h * w:
        return {"error": "the spot is not enclosed by lines (the fill runs out to the sheet); give the corners as a polygon instead"}
    # thin lines cut pieces off a room (door swings, fixtures, furniture, the spot itself in a small pocket): join the small neighbours
    thin = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, int(0.06 / (mpp * f)) | 1),) * 2)
    for _ in range(6):
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = max(0, ys.min() - 9), min(h, ys.max() + 10), max(0, xs.min() - 9), min(w, xs.max() + 10)
        near = np.unique(labels[y0:y1, x0:x1][cv2.dilate(mask[y0:y1, x0:x1], thin) > 0])
        add = []
        for o in (near if len(near) < 400 else []):
            if o <= 0 or sizes[o] >= big or sizes[o] < 4:
                continue
            piece = (labels[y0:y1, x0:x1] == o).astype(np.uint8)
            if mask[y0:y1, x0:x1][piece > 0].any():
                continue
            # a strip no wider than a wall is a window or a doorway inside the wall line, not floor
            if cv2.distanceTransform(np.pad(piece, 1), cv2.DIST_L2, 3).max() * mpp * f >= 0.2:
                add.append(o)
        if not add:
            break
        mask |= np.isin(labels, add).astype(np.uint8)
    # take back what thin lines cut off inside the room: labels, furniture against the wall, door swings
    k = max(3, int(0.6 / (mpp * f))) | 1
    t = max(3, int(0.10 / (mpp * f)))  # strokes thicker than about 10 cm are walls
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = max(0, ys.min() - k), min(h, ys.max() + k + 1), max(0, xs.min() - k), min(w, xs.max() + k + 1)
    m, d, lb = mask[y0:y1, x0:x1], dark[y0:y1, x0:x1], labels[y0:y1, x0:x1]
    wall = cv2.morphologyEx(d, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (t, t)))
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    others = np.zeros_like(m)
    for o in np.unique(lb[(closed > 0) & (m == 0) & (lb > 0)]):  # other rooms are never taken in
        if sizes[o] >= big:
            others |= (lb == o).astype(np.uint8)
    full_m = ((closed > 0) & (wall == 0) & (others == 0)).astype(np.uint8)
    n, cc = cv2.connectedComponents(full_m, connectivity=8)
    if n > 2:
        full_m = (cc == cc[m > 0][0]).astype(np.uint8)
    # a door that opens outwards lets the fill run through the doorway into its swing: cut such lobes off
    # (a lobe = a piece of 0.25-1.6 m² hanging on the room by a neck no wider than a doorway)
    s = mpp * f
    kd = max(3, int(1.0 / s))
    core = cv2.morphologyEx(full_m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kd, kd)))
    if core.sum() > 0.6 * full_m.sum():
        near = cv2.dilate(core, np.ones((3, 3), np.uint8))
        nl, parts = cv2.connectedComponents(((full_m > 0) & (core == 0)).astype(np.uint8), connectivity=8)
        for i in range(1, nl):
            part = parts == i
            a = part.sum() * s * s
            if 0.25 <= a <= 1.6 and (part & (near > 0)).sum() * s <= 1.2:
                full_m[part] = 0
    cs, _ = cv2.findContours(full_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    filled = np.zeros_like(full_m)
    cv2.drawContours(filled, [c], -1, 1, -1)  # holes (labels, columns, furniture in the middle) belong to the floor
    filled[wall > 0] = 0
    band = (cv2.dilate(filled, np.ones((3, 3), np.uint8)) > 0) & (filled == 0)  # anti-aliased wall edges: count their white share
    edge_px = float((gray[y0:y1, x0:x1][band].astype(np.float32) / 255.0).sum())
    poly = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
    P = [[float((p[0] + x0) * f), float((p[1] + y0) * f)] for p in poly]
    xs2, ys2 = [p[0] for p in P], [p[1] for p in P]
    return {"area_m2": round((float(filled.sum()) + edge_px) * s * s, 2), "perimeter_m": round(float(cv2.arcLength(poly.reshape(-1, 1, 2), True)) * s, 2),
            "bbox": [min(xs2), min(ys2), max(xs2), max(ys2)], "poly": P, "method": "enclosed region"}


def polygon(pts, W, H, mpp):
    """pts: [[y, x], ...] in 0-1000 coordinates."""
    P = [[p[1] / 1000.0 * W, p[0] / 1000.0 * H] for p in pts]
    if len(P) < 3:
        return {"error": "a polygon needs at least three corners"}
    a = abs(sum(P[i][0] * P[(i + 1) % len(P)][1] - P[(i + 1) % len(P)][0] * P[i][1] for i in range(len(P)))) / 2
    per = sum(math.hypot(P[i][0] - P[(i + 1) % len(P)][0], P[i][1] - P[(i + 1) % len(P)][1]) for i in range(len(P)))
    xs, ys = [p[0] for p in P], [p[1] for p in P]
    return {"area_m2": round(a * mpp * mpp, 2), "perimeter_m": round(per * mpp, 2), "bbox": [min(xs), min(ys), max(xs), max(ys)], "poly": P, "method": "polygon from the model's corners"}


def line(pts, W, H, mpp):
    (ya, xa), (yb, xb) = pts[0], pts[-1]
    A, B = [xa / 1000.0 * W, ya / 1000.0 * H], [xb / 1000.0 * W, yb / 1000.0 * H]
    return {"length_m": round(math.hypot(A[0] - B[0], A[1] - B[1]) * mpp, 2), "bbox": [min(A[0], B[0]), min(A[1], B[1]), max(A[0], B[0]), max(A[1], B[1])], "poly": [A, B], "method": "straight line"}


def run(full, req, sheet_mm=None):
    """req: the model's request {"scale", "paper", "scale_bar", "items": [{"label", "point" | "polygon" | "line"}]}
    -> (results [{label, area_m2?, perimeter_m?, length_m?, error?}], evidence, sentence about the scale)"""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    W, H = Image.open(full).size
    mpp, how = metres_per_pixel(W, H, req.get("scale"), req.get("paper"), req.get("scale_bar"), sheet_mm)
    if mpp is None:
        return [{"error": how}], [], how
    out, ev = [], []
    for it in (req.get("items") or [])[:60]:
        try:
            if it.get("point"):
                r = region(full, it["point"][0], it["point"][1], mpp)
            elif it.get("polygon"):
                r = polygon(it["polygon"], W, H, mpp)
            elif it.get("line"):
                r = line(it["line"], W, H, mpp)
            else:
                r = {"error": "no point, polygon or line given"}
        except Exception as e:
            r = {"error": "could not be measured: %r" % (e,)}
        label = str(it.get("label", ""))[:80]
        out.append(dict({k: v for k, v in r.items() if k in ("area_m2", "perimeter_m", "length_m", "error", "method")}, label=label))
        if "bbox" in r:
            val = "%.2f m" % r["length_m"] if "length_m" in r else "%.2f m², perimeter %.2f m" % (r["area_m2"], r["perimeter_m"])
            ev.append({"kind": "measured", "name": "%s: %s" % (label, val), "bbox": r["bbox"], "poly": r["poly"]})
    return out, ev, how


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["region"] and len(a) >= 5:
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        W, H = Image.open(a[1]).size
        mpp, how = metres_per_pixel(W, H, a[2], a[3])
        print(how)
        for p in a[4:]:
            y, x = [float(v) for v in p.split(",")]
            r = region(a[1], y, x, mpp)
            r.pop("poly", None)
            print(p, r)
    else:
        print(__doc__)
