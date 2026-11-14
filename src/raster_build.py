"""Build the raster fact file for cubicasa5k: data/raster_facts.jsonl (ONE file).

Input : /scratch/work/liuz16/stoa_raster/cubicasa_pred_cc.jsonl  (CubiCasa5K model, connected components)
Output: data/raster_facts.jsonl, one line per image:
          {"path", "answers": {kind: int}, "evidence": {kind: [[x0,y0,x1,y1], ...]}, "method": {kind: str}}
        and a per-kind accuracy table (printed; copy into raster_notes.md).

Every stored answer is OUT-OF-FOLD: images are split into 5 folds by path hash and
the calibration classifier that answers an image was never trained on that image.
The printed accuracy is therefore a held-out estimate. (The segmentation network
itself was trained by CubiCasa on most of these images, which this cannot correct for.)

  python raster_build.py
"""
import collections
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

PRED = "/scratch/work/liuz16/stoa_raster/cubicasa_pred_cc.jsonl"
OUT = os.path.join(evalkit.ROOT, "data", "raster_facts.jsonl")
FOLDS = 5

KIND = {
    "How many windows": "windows", "How many doors": "doors", "How many toilet bowls": "toilets",
    "How many sinks": "sinks", "How many showers and bathtubs": "showers_baths", "How many saunas": "saunas",
    "How many kitchens": "kitchens", "How many bedrooms": "bedrooms", "How many bathrooms or washrooms": "bathrooms",
    "How many separate WC rooms": "wc_rooms", "How many balconies": "balconies", "How many floors": "floors",
}
# which segmentation class is shown as evidence for each kind: ("icon"|"room"|"blob", class id)
EVID = {
    "windows": ("icon", 1), "doors": ("icon", 2), "toilets": ("icon", 5), "sinks": ("icon", 6),
    "showers_baths": ("icon", 9), "saunas": ("icon", 7), "kitchens": ("room", 3), "bedrooms": ("room", 5),
    "bathrooms": ("room", 6), "wc_rooms": ("room", 6), "balconies": ("room", 1), "floors": ("blob", 0),
}
RAW = {"windows": ("icon", 1), "doors": ("icon", 2)}  # kinds where the plain component count is also tried


def area_scale(rec):
    w, h = rec["net_size"]
    return (w * h) / (800.0 * 600.0)


def comps(rec, kind_cls, min_area):
    typ, c = kind_cls
    if typ == "blob":
        return [b[:4] for b in rec.get("blobs", [])]
    src = rec["cc_icons"] if typ == "icon" else rec["cc_rooms"]
    return [x[1:5] for x in src if x[0] == c and x[5] >= min_area * area_scale(rec)]


def features(rec):
    w, h = rec["net_size"]
    f = [w, h, len(rec.get("blobs", []))]
    for c in range(1, 11):
        xs = [x for x in rec["cc_icons"] if x[0] == c and x[5] >= 20 * area_scale(rec)]
        f += [len(xs), sum(x[5] for x in xs) / (w * h) * 1e3]
    for c in (1, 3, 4, 5, 6, 7, 9, 10, 11):
        xs = [x for x in rec["cc_rooms"] if x[0] == c and x[5] >= 400 * area_scale(rec)]
        f += [len(xs), sum(x[5] for x in xs) / (w * h)]
    # toilets / tubs / benches inside Bath-class rooms help separate WC rooms from bathrooms
    baths = [x for x in rec["cc_rooms"] if x[0] == 6 and x[5] >= 400 * area_scale(rec)]

    def has(b, cls):
        return any(i[0] in cls and b[1] - 4 <= (i[1] + i[3]) / 2 <= b[3] + 4 and b[2] - 4 <= (i[2] + i[4]) / 2 <= b[4] + 4 for i in rec["cc_icons"])
    f += [sum(1 for b in baths if has(b, (5,)) and not has(b, (9, 7))), sum(1 for b in baths if has(b, (9,))),
          sum(1 for b in baths if has(b, (7,)))]
    return f


def fold(path):
    return int(hashlib.md5(path.encode()).hexdigest(), 16) % FOLDS


def main():
    recs = {}
    for l in open(PRED):
        d = json.loads(l)
        recs[d["path"]] = d
    paths = sorted(recs)
    X = {p: features(recs[p]) for p in paths}
    gold = collections.defaultdict(dict)
    for r in evalkit.load_questions():
        if r["path"] in recs and r["question"].startswith("How many"):
            k = next((k for p, k in KIND.items() if r["question"].startswith(p)), None)
            if k:
                gold[k][r["path"]] = r["answer"]
    answers = {p: {} for p in paths}
    method = {}
    table = []
    for k in sorted(EVID):
        g = gold[k]
        oof = {}
        for fo in range(FOLDS):
            tr = [p for p in g if fold(p) != fo]
            te = [p for p in paths if fold(p) == fo]
            m = HistGradientBoostingClassifier(max_iter=150, max_depth=4, learning_rate=0.08, random_state=0)
            m.fit(np.array([X[p] for p in tr], float), np.minimum([g[p] for p in tr], 30))
            for p, v in zip(te, m.predict(np.array([X[p] for p in te], float))):
                oof[p] = int(v)
        acc_clf = np.mean([oof[p] == a for p, a in g.items()])
        vals, cnt = np.unique(list(g.values()), return_counts=True)
        maj = cnt.max() / cnt.sum()
        best, acc, name = oof, acc_clf, "segmentation features + calibrated classifier"
        if k in RAW:
            raw = {p: len(comps(recs[p], RAW[k], 20)) for p in paths}
            acc_raw = np.mean([raw[p] == a for p, a in g.items()])
            if acc_raw > acc_clf:
                best, acc, name = raw, acc_raw, "count of segmentation components"
        within1 = np.mean([abs(best[p] - a) <= 1 for p, a in g.items()])
        method[k] = name
        for p in paths:
            answers[p][k] = best[p]
        table.append((k, acc, maj, within1, len(g), name))
    with open(OUT, "w") as f:
        for p in paths:
            r = recs[p]
            sx, sy = r["size"][0] / r["net_size"][0], r["size"][1] / r["net_size"][1]
            ev = {k: [[round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)] for b in comps(r, EVID[k], 20 if EVID[k][0] == "icon" else 400)]
                  for k in EVID}
            f.write(json.dumps({"path": p, "answers": answers[p], "evidence": ev}) + "\n")
    print("wrote", OUT, len(paths), "images")
    print("| 题目 | 留出准确率 | 猜最常见答案 | 误差≤1 | 题数（英文） | 方法 |")
    print("|---|---|---|---|---|---|")
    tot = [0.0, 0.0, 0]
    for k, acc, maj, w1, n, name in sorted(table, key=lambda t: -t[1]):
        print("| %s | %.1f%% | %.1f%% | %.1f%% | %d | %s |" % (k, 100 * acc, 100 * maj, 100 * w1, n, name))
        tot[0] += acc * n
        tot[1] += maj * n
        tot[2] += n
    print("| ALL | %.1f%% | %.1f%% | | %d | |" % (100 * tot[0] / tot[2], 100 * tot[1] / tot[2], tot[2]))
    with open(OUT.replace(".jsonl", "_meta.json"), "w") as f:
        json.dump({"method": method, "accuracy": {k: round(float(a), 4) for k, a, *_ in table},
                   "majority": {k: round(float(m), 4) for k, _, m, *_ in table}}, f, indent=1)


if __name__ == "__main__":
    main()
