"""Instance labels from the original CubiCasa5K annotations (model.svg), for training and checking a detector.

The competition's cubicasa5k PNGs are byte-identical to CubiCasa5K's F1_original.png, and its count answers equal the
number of annotated elements (checked on 20 dev images: windows 20/20, doors 15/15, toilets 20/20). This script reads
every model.svg straight from the downloaded zip (nothing is extracted), turns each element into a box in F1_original
pixel coordinates, and writes ONE json file. It never feeds answers to the QA system: it is training data and a check.

  python cubi_labels.py build     -> /scratch/work/liuz16/stoa_raster/cubi_labels.json
  python cubi_labels.py check     -> per-kind agreement between label counts and the dataset answers
"""
import collections
import io
import json
import os
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

ZIP = "/scratch/work/liuz16/stoa_raster/cubicasa5k.zip"
OUT = "/scratch/work/liuz16/stoa_raster/cubi_labels.json"
SPLIT = "/scratch/work/liuz16/stoa_raster/cubicasa_official_split.json"
NS = "{http://www.w3.org/2000/svg}"
NUM = re.compile(r"-?\d*\.?\d+(?:e-?\d+)?")


def _mat(t):
    """SVG transform attribute -> 2x3 affine (a, b, c, d, e, f). Only matrix() and translate() occur in CubiCasa."""
    if not t:
        return (1, 0, 0, 1, 0, 0)
    m = re.match(r"\s*matrix\(([^)]*)\)", t)
    if m:
        return tuple(float(v) for v in NUM.findall(m.group(1)))
    m = re.match(r"\s*translate\(([^)]*)\)", t)
    if m:
        v = [float(x) for x in NUM.findall(m.group(1))] + [0]
        return (1, 0, 0, 1, v[0], v[1])
    return (1, 0, 0, 1, 0, 0)


def _mul(p, q):  # p then q in SVG nesting order: parent p, child q -> p*q
    a, b, c, d, e, f = p
    A, B, C, D, E, F = q
    return (a * A + c * B, b * A + d * B, a * C + c * D, b * C + d * D, a * E + c * F + e, b * E + d * F + f)


def _apply(m, x, y):
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def _local_points(g):
    """Points of the drawable geometry inside an element group (polygons, rects, lines, path endpoints)."""
    pts = []
    for el in g.iter():
        tag = el.tag.replace(NS, "")
        if tag == "polygon":
            v = [float(x) for x in NUM.findall(el.get("points", ""))]
            pts += list(zip(v[0::2], v[1::2]))
        elif tag == "rect":
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            w, h = float(el.get("width", 0)), float(el.get("height", 0))
            pts += [(x, y), (x + w, y + h)]
        elif tag == "line":
            pts += [(float(el.get("x1", 0)), float(el.get("y1", 0))), (float(el.get("x2", 0)), float(el.get("y2", 0)))]
        elif tag == "path":  # door swings: "M x,y q c1x,c1y ex,ey l ..." (relative after M); keep M, M+c, M+e
            d = el.get("d", "")
            m = re.match(r"\s*M\s*(" + NUM.pattern + r")[ ,]*(" + NUM.pattern + r")\s*q\s*([^lLzZ]*)", d)
            if m:
                x0, y0 = float(m.group(1)), float(m.group(2))
                v = [float(x) for x in NUM.findall(m.group(3))]
                pts.append((x0, y0))
                if len(v) >= 4:
                    pts += [(x0 + v[0], y0 + v[1]), (x0 + v[2], y0 + v[3])]
    return pts


def parse_svg(data):
    """-> list of (floor_id, class string, [x0, y0, x1, y1]) in SVG (= F1_scaled) coordinates."""
    root = ET.fromstring(data)
    out = []

    def walk(el, m, floor):
        cls = el.get("class") or ""
        m = _mul(m, _mat(el.get("transform")))
        if "Floorplan" in cls:
            floor = el.get("id") or cls
        if cls.startswith(("Window", "Door")) or (cls.startswith("FixedFurniture") and not cls.startswith("FixedFurnitureSet")):
            if cls.startswith("FixedFurniture"):
                bp = [c for c in el if (c.get("class") or "") == "BoundaryPolygon"]
                pts = _local_points(bp[0]) if bp else _local_points(el)
            else:
                pts = _local_points(el)
            if pts:
                P = [_apply(m, x, y) for x, y in pts]
                xs, ys = [p[0] for p in P], [p[1] for p in P]
                out.append((floor, cls.strip(), [min(xs), min(ys), max(xs), max(ys)]))
            return  # an element's sub-groups (Glass, Panel, ...) are parts of it, not new elements
        for c in el:
            walk(c, m, floor)

    walk(root, (1, 0, 0, 1, 0, 0), None)
    return out


def _png_size(z, name):
    with z.open(name) as f:
        h = f.read(24)
    return struct.unpack(">II", h[16:24])


def build():
    split = json.load(open(SPLIT))
    where = {p.strip("/"): s for s, L in split.items() for p in L}
    comp = {}
    for q in evalkit.load_questions():
        if q["path"].startswith("data/cubicasa5k/"):
            sty, num = q["path"].split("/")[-1][:-4].rsplit("-", 1)
            comp["%s/%s" % (sty, num)] = q["path"]
    z = zipfile.ZipFile(ZIP)
    names = set(z.namelist())
    labels = {}
    for n in sorted(names):
        if not n.endswith("/model.svg"):
            continue
        key = n[len("cubicasa5k/"):-len("/model.svg")]
        so, oo = n.replace("model.svg", "F1_scaled.png"), n.replace("model.svg", "F1_original.png")
        ws, hs = _png_size(z, so)
        wo, ho = _png_size(z, oo)
        sx, sy = wo / ws, ho / hs
        objs = []
        for floor, cls, b in parse_svg(z.read(n)):
            objs.append([floor, cls, [round(b[0] * sx, 1), round(b[1] * sy, 1), round(b[2] * sx, 1), round(b[3] * sy, 1)]])
        floors = sorted({o[0] for o in objs if o[0]})
        labels[key] = {"split": where.get(key), "comp_path": comp.get(key), "size": [wo, ho], "scaled_size": [ws, hs],
                       "n_floor_images": sum(1 for k in range(1, 6) if n.replace("model.svg", "F%d_original.png" % k) in names),
                       "floors": floors, "objects": objs}
    json.dump(labels, open(OUT, "w"))
    print("%d annotated images -> %s; %d in the competition set; splits %s" % (
        len(labels), OUT, sum(1 for v in labels.values() if v["comp_path"]),
        dict(collections.Counter(v["split"] for v in labels.values()))))


# label class -> question kind; refined by `check` until the counts reproduce the dataset answers
KIND_OF = [("windows", r"^Window"), ("doors", r"^Door"), ("toilets", r"^FixedFurniture Toilet"),
           ("sinks", r"^FixedFurniture (?:Round|Double|Corner|Side)?Sink"),
           ("showers_baths", r"^FixedFurniture (?:Shower$|Bathtub)")]


def kind_of(cls):
    for k, rx in KIND_OF:
        if re.search(rx, cls):
            return k
    return None


def counts(rec, floor=None):
    c = collections.Counter()
    for fl, cls, _ in rec["objects"]:
        if floor is None or fl == floor:
            k = kind_of(cls)
            if k:
                c[k] += 1
    return c


def check():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import raster_backend
    labels = json.load(open(OUT))
    by_path = {v["comp_path"]: v for v in labels.values() if v["comp_path"]}
    agree = collections.defaultdict(collections.Counter)
    bad = collections.defaultdict(list)
    for q in evalkit.load_questions():
        rec = by_path.get(q["path"])
        if rec is None or not q["question"].startswith("How"):
            continue
        k = raster_backend.question_kind(q["question"])
        if k not in dict(KIND_OF):
            continue
        n = counts(rec)[k]
        multi = rec["n_floor_images"] > 1
        a = agree[k]
        a["n"] += 1
        a["ok"] += n == q["answer"]
        a["multi_n"] += multi
        a["multi_ok"] += multi and n == q["answer"]
        if n != q["answer"]:
            bad[k].append((q["path"], q["answer"], n))
    for k, a in agree.items():
        print("%-14s labels == answer %5d / %-5d (%.2f%%); images with >1 floor image: %d/%d" % (
            k, a["ok"], a["n"], 100 * a["ok"] / a["n"], a["multi_ok"], a["multi_n"]))
        for b in bad[k][:4]:
            print("     mismatch", b)
    cl = collections.Counter(cls for v in labels.values() for _, cls, _ in v["objects"] if cls.startswith("FixedFurniture"))
    print("furniture classes:", ", ".join("%s %d" % kv for kv in cl.most_common(40)))


if __name__ == "__main__":
    {"build": build, "check": check}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
