"""Cross-fitted detectors (raster_yolo_plan.md section 13): split the train + val drawings into 5 folds; detector k trains on
the other four folds and reads fold k. Every train/val drawing thus gets an out-of-fold reading, from a detector that never
saw it, which is what the combiner learns from. E1/E2 drawings are never trained on; each is read by all five detectors
(the combiner uses the detector of the fold its key hashes to, the same rule as for every other drawing).

  python cubi_cv.py prep <dir> <fold>                    (GPU job) images + labels to node-local disk
  python cubi_cv.py train <dir> <out> [epochs]           fixed epochs, no per-epoch validation, last weights
  python cubi_cv.py predict <weights> <out.json> <fold>  raw boxes (conf >= 0.05) for this fold's drawings + E1 + E2
"""
import collections
import hashlib
import io
import json
import os
import random
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubi_labels as cl  # noqa: E402
import cubi_yolo as cy  # noqa: E402

K = 5


def fold_of(key):
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % K


def prep(root, fold):
    fold = int(fold)
    L, role = cy.sets()
    z = zipfile.ZipFile(cl.ZIP)
    train = sorted(k for k in L if role[k] in ("train", "val") and fold_of(k) != fold)
    mon = random.Random(fold).sample(train, 50)  # YOLO wants a val set; it is only scored once, after the last epoch
    n = collections.Counter()
    for split, keys in (("train", train), ("val", mon)):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(root, sub, split), exist_ok=True)
        for k in keys:
            name = k.replace("/", "-")
            with open(os.path.join(root, "images", split, name + ".png"), "wb") as f:
                f.write(z.read("cubicasa5k/%s/F1_original.png" % k))
            with open(os.path.join(root, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(cy.label_lines(L[k])))
            n[split] += 1
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\nnames:\n" % root + "".join("  %d: %s\n" % (i, c) for i, c in enumerate(cy.CLASSES)))
    print("prep fold %d:" % fold, dict(n), "->", root)


def train(root, out, epochs=80):
    from ultralytics import YOLO
    m = YOLO(os.path.join(os.path.dirname(cl.ZIP), "yolo", "yolo11s.pt"))
    m.train(data=os.path.join(root, "data.yaml"), epochs=int(epochs), imgsz=1024, batch=16, workers=7, device=0,
            project=out, name="run", exist_ok=True, plots=False, val=False, fliplr=0.5, flipud=0.5, degrees=0.0,
            max_det=600, seed=0, deterministic=False)


def predict(weights, out, fold):
    from PIL import Image
    from ultralytics import YOLO
    fold = int(fold)
    L, role = cy.sets()
    z = zipfile.ZipFile(cl.ZIP)
    m = YOLO(weights)
    keys = [k for k in L if (role[k] in ("train", "val") and fold_of(k) == fold) or role[k] in ("E1", "E2")]
    res = {}
    for i in range(0, len(keys), 16):
        chunk = keys[i:i + 16]
        ims = [Image.open(io.BytesIO(z.read("cubicasa5k/%s/F1_original.png" % k))).convert("RGB") for k in chunk]
        for k, r in zip(chunk, m.predict(ims, imgsz=1024, conf=0.05, iou=0.6, max_det=600, verbose=False)):
            b = r.boxes
            res[k] = [[int(c), round(float(p), 3)] + [round(float(x), 1) for x in xy]
                      for c, p, xy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
    json.dump({"weights": weights, "fold": fold, "preds": res}, open(out, "w"))
    print("fold %d: predicted %d drawings -> %s" % (fold, len(res), out))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = {"prep": prep, "train": train, "predict": predict}.get(a[0] if a else "")
    print(__doc__) if cmd is None else cmd(*a[1:])
