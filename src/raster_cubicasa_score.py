"""Score the CubiCasa5K-model features (cubicasa_pred.jsonl) against the English cubicasa5k questions.

Each question kind is answered by a simple rule on the model's connected
components. Reports, per kind: rule accuracy, majority-answer accuracy (no image),
and mean absolute error. Images are split in two halves so thresholds chosen on
one half can be checked on the other.

  python raster_cubicasa_score.py
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

PRED = "/scratch/work/liuz16/stoa_raster/cubicasa_pred.jsonl"
# icon ids: 1 Window 2 Door 3 Closet 4 Appliance 5 Toilet 6 Sink 7 Sauna bench 8 Fire place 9 Bathtub 10 Chimney
# room ids: 1 Outdoor 3 Kitchen 4 Living 5 Bedroom 6 Bath 7 Hallway 9 Storage 10 Garage 11 Other

KIND = {
    "How many windows": "windows", "How many doors": "doors", "How many toilet bowls": "toilets",
    "How many sinks": "sinks", "How many showers and bathtubs": "showers_baths", "How many saunas": "saunas",
    "How many kitchens": "kitchens", "How many bedrooms": "bedrooms", "How many bathrooms or washrooms": "bathrooms",
    "How many separate WC rooms": "wc_rooms", "How many balconies": "balconies", "How many floors": "floors",
}
MAJ = {"windows": 4, "doors": 7, "toilets": 1, "sinks": 2, "showers_baths": 1, "saunas": 1, "kitchens": 1,
       "bedrooms": 1, "bathrooms": 1, "wc_rooms": 0, "balconies": 0, "floors": 1}


def n_icon(rec, c, min_area=0):
    s = scale(rec)
    return sum(1 for x in rec["cc_icons"] if x[0] == c and x[5] >= min_area * s)


def n_room(rec, c, min_area=0):
    s = scale(rec)
    return sum(1 for x in rec["cc_rooms"] if x[0] == c and x[5] >= min_area * s)


def scale(rec):
    """Area scale factor so thresholds are relative to a 800x600 sheet."""
    w, h = rec["net_size"]
    return (w * h) / (800.0 * 600.0)


def inside(inner, outer, pad=4):
    cx, cy = (inner[1] + inner[3]) / 2, (inner[2] + inner[4]) / 2
    return outer[1] - pad <= cx <= outer[3] + pad and outer[2] - pad <= cy <= outer[4] + pad


def rules(rec, P):
    baths = [x for x in rec["cc_rooms"] if x[0] == 6 and x[5] >= P["room_min"] * scale(rec)]
    icons = rec["cc_icons"]
    wc = 0
    for b in baths:  # a Bath-class room with a toilet but no tub is a separate WC
        has_t = any(i[0] == 5 and inside(i, b) for i in icons)
        has_tub = any(i[0] in (9, 7) and inside(i, b) for i in icons)
        small = b[5] < P["wc_max"] * scale(rec)
        if has_t and not has_tub and small:
            wc += 1
    return {
        "windows": n_icon(rec, 1, P["icon_min"]),
        "doors": n_icon(rec, 2, P["icon_min"]),
        "toilets": n_icon(rec, 5, P["icon_min"]),
        "sinks": n_icon(rec, 6, P["icon_min"]),
        "showers_baths": max(n_icon(rec, 9, P["icon_min"]), 1 if baths else 0),
        "saunas": 1 if n_icon(rec, 7, P["icon_min"]) else 0,
        "kitchens": n_room(rec, 3, P["room_min"]),
        "bedrooms": n_room(rec, 5, P["room_min"]),
        "bathrooms": max(0, len(baths) - wc),
        "wc_rooms": wc,
        "balconies": n_room(rec, 1, P["out_min"]),
        "floors": max(1, len(rec.get("blobs", [])) or 1),
    }


def main():
    P = {"icon_min": 20, "room_min": 400, "wc_max": 2500, "out_min": 1500}
    recs = {}
    for l in open(PRED):
        d = json.loads(l)
        recs[d["path"]] = d
    files = sorted(recs)
    half = {f: i % 2 for i, f in enumerate(files)}
    agg = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0, 0])  # ok, maj_ok, n, abs err, ok half0, n half0
    for r in evalkit.load_questions():
        if r["path"] not in recs or not r["question"].startswith("How many"):
            continue
        kind = next((k for p, k in KIND.items() if r["question"].startswith(p)), None)
        if kind is None:
            continue
        pred = rules(recs[r["path"]], P)[kind]
        a = agg[kind]
        a[0] += pred == r["answer"]
        a[1] += MAJ[kind] == r["answer"]
        a[2] += 1
        a[3] += abs(pred - r["answer"])
        if half[r["path"]] == 0:
            a[4] += pred == r["answer"]
            a[5] += 1
    print("images", len(files), "poly errors", sum(1 for d in recs.values() if "poly_error" in d))
    print("%-14s %7s %9s %7s %6s  (half A / half B)" % ("kind", "model", "majority", "MAE", "n"))
    tot = [0, 0, 0]
    for k, (ok, mj, n, ae, ok0, n0) in sorted(agg.items()):
        print("%-14s %6.1f%% %8.1f%% %7.2f %6d  (%.0f%% / %.0f%%)" % (k, 100 * ok / n, 100 * mj / n, ae / n, n, 100 * ok0 / max(1, n0), 100 * (ok - ok0) / max(1, n - n0)))
        tot[0] += ok
        tot[1] += mj
        tot[2] += n
    print("%-14s %6.1f%% %8.1f%% %14d" % ("ALL", 100 * tot[0] / tot[2], 100 * tot[1] / tot[2], tot[2]))


if __name__ == "__main__":
    main()
