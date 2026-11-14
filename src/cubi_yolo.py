"""Pilot: a YOLO detector for the five object counts of cubicasa5k (windows, doors, toilets, sinks, showers+baths).

Training data are the original CubiCasa5K annotations (cubi_labels.py; their counts equal the dataset answers on every
image). Images held out from training and threshold tuning:
  E1 = the 150 cubicasa5k drawings of data/dev500.jsonl (excluded from train and val wherever they sit),
  E2 = competition drawings in CubiCasa5K's official test split.
Thresholds are tuned on the official val split (minus E1). E1/E2 are only scored.

  python cubi_yolo.py prep <dir>              (GPU job) write images + YOLO labels for train/val to <dir> (node-local disk)
  python cubi_yolo.py train <dir> <out> [epochs] [imgsz] [hours]
  python cubi_yolo.py predict <weights> <out.json> [imgsz]   boxes (conf >= 0.05) for val, E1 and E2 drawings -> ONE json
  python cubi_yolo.py eval <preds.json>       (login node) tune per-class thresholds on val, score E1 and E2
"""
import collections
import io
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubi_labels as cl  # noqa: E402
import evalkit  # noqa: E402

CLASSES = ["windows", "doors", "toilets", "sinks", "showers_baths"]
CID = {k: i for i, k in enumerate(CLASSES)}


def dev_paths():
    return {r["path"] for r in map(json.loads, open(evalkit.DEV)) if r["path"].startswith("data/cubicasa5k/")}


def sets():
    """-> labels, {key: role} with role in train / val / E1 / E2 (E2 also covers test drawings not in E1)."""
    L = json.load(open(cl.OUT))
    dev = dev_paths()
    role = {}
    for k, v in L.items():
        if v["comp_path"] in dev:
            role[k] = "E1"
        elif v["split"] == "test":
            role[k] = "E2" if v["comp_path"] else None
        else:
            role[k] = v["split"]  # train / val
    return L, role


def label_lines(v):
    """YOLO label lines (class cx cy w h, normalised) of one drawing's annotated objects."""
    W, H = v["size"]
    lines = []
    for _, c, b in v["objects"]:
        kind = cl.kind_of(c)
        if kind is None:
            continue
        x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append("%d %.6f %.6f %.6f %.6f" % (CID[kind], (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return lines


def prep(root):
    L, role = sets()
    z = zipfile.ZipFile(cl.ZIP)
    n = collections.Counter()
    for k, v in L.items():
        r = role[k]
        if r not in ("train", "val"):
            continue
        lines = label_lines(v)
        name = k.replace("/", "-")
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(root, sub, r), exist_ok=True)
        with open(os.path.join(root, "images", r, name + ".png"), "wb") as f:
            f.write(z.read("cubicasa5k/%s/F1_original.png" % k))
        with open(os.path.join(root, "labels", r, name + ".txt"), "w") as f:
            f.write("\n".join(lines))
        n[r] += 1
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % root + "".join("  %d: %s\n" % (i, c) for i, c in enumerate(CLASSES)))
    print("prep:", dict(n), "->", root)


def train(root, out, epochs=60, imgsz=1024, hours=2.4, fraction=1.0, batch=16, weights=None):
    from ultralytics import YOLO
    m = YOLO(weights or os.path.join(os.path.dirname(cl.ZIP), "yolo", "yolo11s.pt"))  # weights: continue from an earlier run
    m.train(data=os.path.join(root, "data.yaml"), epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), workers=7, device=0,
            project=out, name="run", exist_ok=True, plots=False, patience=20, time=float(hours),
            fliplr=0.5, flipud=0.5, degrees=0.0, max_det=600, seed=0, deterministic=False, fraction=float(fraction))


def predict(weights, out, imgsz=1024):
    """Image list comes from the zip-free competition copies where they exist (byte-identical to F1_original)."""
    from PIL import Image
    from ultralytics import YOLO
    L, role = sets()
    z = zipfile.ZipFile(cl.ZIP)
    m = YOLO(weights)
    res = {}
    keys = [k for k in L if role[k] in ("val", "E1", "E2")]
    for i in range(0, len(keys), 16):
        chunk = keys[i:i + 16]
        ims = [Image.open(io.BytesIO(z.read("cubicasa5k/%s/F1_original.png" % k))).convert("RGB") for k in chunk]
        for k, r in zip(chunk, m.predict(ims, imgsz=int(imgsz), conf=0.05, iou=0.6, max_det=600, verbose=False)):
            b = r.boxes
            res[k] = [[int(c), round(float(p), 3)] + [round(float(x), 1) for x in xy]
                      for c, p, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
    json.dump({"weights": weights, "imgsz": imgsz, "preds": res}, open(out, "w"))
    print("predicted", len(res), "drawings ->", out)


def _count(boxes, cid, t):
    return sum(1 for b in boxes if b[0] == cid and b[1] >= t)


def evaluate(path):
    import raster_backend
    P = json.load(open(path))["preds"]
    L, role = sets()
    # 1) per-class threshold: maximise exact count accuracy on val against the annotation counts
    grid = [round(0.05 + 0.025 * i, 3) for i in range(35)]
    thr = {}
    val = [k for k in P if role[k] == "val"]
    for kind in CLASSES:
        best = max(grid, key=lambda t: (sum(_count(P[k], CID[kind], t) == cl.counts(L[k])[kind] for k in val), -abs(t - 0.4)))
        thr[kind] = best
        acc = sum(_count(P[k], CID[kind], best) == cl.counts(L[k])[kind] for k in val) / len(val)
        print("threshold %-14s %.3f   val exact %.1f%% (n=%d)" % (kind, best, 100 * acc, len(val)))
    # 2) score E1 / E2 on the dataset questions (one English question per drawing and kind), against the other routes
    gem = {json.loads(l)["path"]: json.loads(l) for l in open(os.path.join(evalkit.ROOT, "data", "raster_gemini.jsonl"))}
    seg = {json.loads(l)["path"]: json.loads(l) for l in open(os.path.join(evalkit.ROOT, "data", "raster_facts.jsonl"))}
    by = {v["comp_path"]: k for k, v in L.items() if v["comp_path"]}
    tab = collections.defaultdict(collections.Counter)
    seen = set()
    for q in evalkit.load_questions():
        k = by.get(q["path"])
        if k is None or role.get(k) not in ("E1", "E2") or not q["question"].startswith("How"):
            continue
        kind = raster_backend.question_kind(q["question"])
        if kind not in CID or (q["path"], kind) in seen:
            continue
        seen.add((q["path"], kind))
        y = _count(P[k], CID[kind], thr[kind])
        for s in (role[k], role[k] + "+gem"):
            if s.endswith("+gem") and q["path"] not in gem:
                continue
            t = tab[(s, kind)]
            t["n"] += 1
            t["yolo"] += y == q["answer"]
            t["yolo±1"] += abs(y - q["answer"]) <= 1
            t["seg"] += seg[q["path"]]["answers"][kind] == q["answer"]
            if q["path"] in gem:
                t["gem"] += gem[q["path"]]["facts"][kind]["count"] == q["answer"]
    for s in ("E1", "E1+gem", "E2"):
        print("\n== %s %s" % (s, {"E1": "(dev500 cubicasa drawings)", "E1+gem": "(dev drawings with a Gemini read)", "E2": "(official test split)"}[s]))
        print("%-14s %5s %7s %7s %7s %7s" % ("kind", "n", "YOLO", "YOLO±1", "seg", "gemini" if s.endswith("gem") else ""))
        for kind in CLASSES:
            t = tab[(s, kind)]
            if t["n"]:
                print("%-14s %5d %6.1f%% %6.1f%% %6.1f%% %s" % (kind, t["n"], 100 * t["yolo"] / t["n"], 100 * t["yolo±1"] / t["n"], 100 * t["seg"] / t["n"],
                                                             "%6.1f%%" % (100 * t["gem"] / t["n"]) if s.endswith("gem") else ""))
    json.dump(thr, open(path.replace(".json", ".thr.json"), "w"))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = {"prep": prep, "train": train, "predict": predict, "eval": evaluate}.get(a[0] if a else "")
    if cmd is None:
        print(__doc__)
    else:
        cmd(*a[1:])
