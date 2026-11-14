"""Raster probe 2: run the official CubiCasa5K pretrained model on cubicasa5k PNGs.

Writes ONE jsonl (resumable) with, per image: icon polygons from the repo's
post-processing, connected components of the icon / room segmentation, and room
polygons. Scoring against questions is done separately (raster_cubicasa_score.py).

  python raster_cubicasa_infer.py <n_images> [max_seconds]
"""
import json
import os
import random
import sys
import time
import types

REPO = "/scratch/work/liuz16/stoa_raster/CubiCasa5k"
WEIGHTS = "/scratch/work/liuz16/stoa_raster/model_best_val_loss_var.pkl"
OUT = "/scratch/work/liuz16/stoa_raster/cubicasa_pred_v2.jsonl"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the repo's loaders import svgpathtools only for SVG parsing, which we never call
sys.modules.setdefault("svgpathtools", types.SimpleNamespace(parse_path=None))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy import ndimage  # noqa: E402

import evalkit  # noqa: E402
from floortrans import post_prosessing  # noqa: E402
from floortrans.loaders.augmentations import RotateNTurns  # noqa: E402
from floortrans.models.hg_furukawa_original import hg_furukawa_original  # noqa: E402

# the repo was written for scipy 1.1, where stats.mode() always returned arrays
_mode = post_prosessing.stats.mode
post_prosessing.stats.mode = lambda a, *args, **kw: _mode(a, *args, keepdims=True, **kw)

ROOM = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room", "Bedroom", "Bath", "Hallway", "Railing", "Storage", "Garage", "Other rooms"]
ICON = ["Empty", "Window", "Door", "Closet", "Electr. Appl.", "Toilet", "Sink", "Sauna bench", "Fire Place", "Bathtub", "Chimney"]
SPLIT = [21, 12, 11]
MAX_SIDE = 1600
SCALE = float(os.environ.get("STOA_SCALE", "1"))  # test-time rescale sweep
NOPOLY = os.environ.get("STOA_NOPOLY") == "1"  # fast pass: segmentation components only
if NOPOLY:
    OUT = OUT.replace("_v2.jsonl", "_cc.jsonl")
if SCALE != 1:
    OUT = OUT.replace("_v2.jsonl", "_s%s.jsonl" % os.environ["STOA_SCALE"])
    MAX_SIDE = 3200


def load_model(dev):
    model = hg_furukawa_original(n_classes=51)  # skip init_weights(): it loads a relative path
    model.conv4_ = torch.nn.Conv2d(256, 44, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(44, 44, kernel_size=4, stride=4)
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    return model.eval().to(dev)


def components(seg, names, min_area):
    out = []
    for c in range(1, len(names)):
        lab, n = ndimage.label(seg == c)
        if not n:
            continue
        for i, sl in enumerate(ndimage.find_objects(lab)):
            area = int((lab[sl] == i + 1).sum())
            if area >= min_area:
                out.append([c, sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, area])
    return out


def predict(model, path, dev):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    s = min(SCALE, MAX_SIDE / max(h0, w0))
    h, w = max(64, int(round(h0 * s / 32)) * 32), max(64, int(round(w0 * s / 32)) * 32)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x = torch.tensor(np.moveaxis(2 * (img.astype(np.float32) / 255.0) - 1, -1, 0))[None].to(dev)
    rot = RotateNTurns()
    pred = torch.zeros([4, 44, h, w])
    with torch.no_grad():
        for i, (fwd, back) in enumerate([(0, 0), (1, -1), (2, 2), (-1, 1)]):
            p = model(rot(x, "tensor", fwd))
            p = rot(p, "tensor", back)
            p = rot(p, "points", back)
            pred[i] = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=True)[0].cpu()
    pred = pred.mean(0, keepdim=True)
    heat, rooms, icons = post_prosessing.split_prediction(pred, (h, w), SPLIT)
    rec = {"size": [w0, h0], "net_size": [w, h]}
    rec["cc_icons"] = components(np.argmax(icons, 0), ICON, 12)
    rec["cc_rooms"] = components(np.argmax(rooms, 0), ROOM, 150)
    # separate drawings on the sheet (one per storey): blobs of non-background after closing small gaps
    fg = ndimage.binary_dilation(np.argmax(rooms, 0) != 0, iterations=6)
    lab, nb = ndimage.label(fg)
    rec["blobs"] = [[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop, int((lab[sl] == i + 1).sum())]
                    for i, sl in enumerate(ndimage.find_objects(lab)) if (lab[sl] == i + 1).sum() >= 0.01 * h * w]
    if SCALE != 1 or NOPOLY:
        return rec
    try:
        polys, typs, rpolys, rtyps = post_prosessing.get_polygons((heat, rooms.copy(), icons), 0.4, [1, 2])
        rec["poly_icons"] = [[int(t["class"])] + [int(v) for v in (*np.min(p, 0), *np.max(p, 0))]
                             for p, t in zip(polys, typs) if t["type"] == "icon"]
        rec["poly_rooms"] = []
        for rp, t in zip(rpolys, rtyps):  # one (multi)polygon per class: split into separate rooms
            for g in getattr(rp, "geoms", [rp]):
                if not g.is_empty and g.area > 0:
                    rec["poly_rooms"].append([int(t["class"])] + [int(v) for v in g.bounds] + [int(g.area)])
    except Exception as e:  # post-processing is brittle; keep the segmentation counts
        rec["poly_error"] = repr(e)[:200]
    return rec


def main(n, max_s):
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "cubicasa5k"]
    dev_files = []
    for l in open(evalkit.DEV):
        r = json.loads(l)
        if r["source"] == "cubicasa5k" and r["path"] not in dev_files:
            dev_files.append(r["path"])
    rest = sorted({r["path"] for r in rows} - set(dev_files))
    random.Random(5).shuffle(rest)
    files = (dev_files + rest)[:n]
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["path"] for l in open(OUT)}
    model = load_model(dev)
    print("device", dev, "todo", len([f for f in files if f not in done]), flush=True)
    with open(OUT, "a") as out:
        for i, f in enumerate(files):
            if f in done:
                continue
            if time.time() - t0 > max_s:
                print("time budget reached at", i, flush=True)
                break
            t = time.time()
            rec = predict(model, os.path.join(evalkit.DATA, f), dev)
            rec["path"] = f
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 20 == 0:
                print(i, f.split("/")[-1], "%.1fs" % (time.time() - t), rec.get("poly_error", ""), flush=True)
    print("done in %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300, float(sys.argv[2]) if len(sys.argv) > 2 else 1500)
