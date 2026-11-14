"""The trained object detector (cubi_yolo.py) as a reader for the image route.

  python raster_detector.py build <weights.pt> <thr.json>   (GPU job) detect on every cubicasa5k drawing of the dataset
                                                            -> data/raster_detector.jsonl, one line per drawing

Each line: {"path", "boxes": {kind: [[x0, y0, x1, y1], ...]}, "seen": true if the detector was trained or tuned on this drawing}.
Only boxes at or above the per-class confidence threshold (tuned on the validation split) are kept, so the count of a
kind is simply the number of its boxes, and the boxes shown as evidence always match the answer.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

STORE = os.path.join(evalkit.ROOT, "data", "raster_detector.jsonl")
KINDS = ["windows", "doors", "toilets", "sinks", "showers_baths"]  # class order of cubi_yolo.CLASSES
_store = None


def load():
    global _store
    if _store is None:
        _store = {}
        if os.path.exists(STORE):
            with open(STORE) as f:
                for line in f:
                    d = json.loads(line)
                    _store[d["path"]] = d
    return _store


def build(weights, thr_path, imgsz=1024):
    from PIL import Image
    from ultralytics import YOLO
    import cubi_yolo

    thr = json.load(open(thr_path))
    L, role = cubi_yolo.sets()
    seen = {v["comp_path"]: role[k] in ("train", "val") for k, v in L.items() if v["comp_path"]}
    d = os.path.join(evalkit.DATA, "data", "cubicasa5k")
    names = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    m = YOLO(weights)
    with open(STORE + ".tmp", "w") as out:
        for i in range(0, len(names), 16):
            chunk = names[i:i + 16]
            ims = [Image.open(os.path.join(d, n)).convert("RGB") for n in chunk]
            for n, r in zip(chunk, m.predict(ims, imgsz=int(imgsz), conf=0.05, iou=0.6, max_det=600, verbose=False)):
                boxes = {k: [] for k in KINDS}
                for c, p, xy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    k = KINDS[int(c)]
                    if p >= thr[k]:
                        boxes[k].append([round(float(v)) for v in xy])
                path = "data/cubicasa5k/" + n
                out.write(json.dumps({"path": path, "boxes": boxes, "seen": bool(seen.get(path, False))}) + "\n")
    os.replace(STORE + ".tmp", STORE)
    print("detector store:", len(names), "drawings ->", STORE)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"] and len(sys.argv) >= 4:
        build(*sys.argv[2:])
    else:
        print(__doc__)
