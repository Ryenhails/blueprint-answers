"""Experiment: count windows and doors on overlapping tiles instead of the whole image (raster_ingest.py reads the whole image).

Why tiles: gemini-4-hydrogen bills every image as 1120 tokens whatever its pixel size (checked with the free countTokens
endpoint on 200-3000 px versions of one plan), so it resamples each image to a fixed budget. A tile therefore gets several
times more model resolution per wall metre than the whole sheet. No client-side upscaling: the model resizes anyway.

Grid: 2x2 for every drawing, neighbouring tiles overlapping by 20% of the tile size. Planned: 3 tiles along an axis of
1400 px or more (whole-image window accuracy falls from 54% below 600 px to 26% at 1300 px and more); switched off (BIG)
to spend the $4 budget on the convention variant instead. The run used 2x2 only.

De-duplication ("core+merge", the counted rule):
  core   = tile minus half the overlap on each interior side; the cores partition the image, so every location has one owner.
  merge  = detections of the same kind from two different tiles whose boxes overlap (intersection / smaller box >= 0.3,
           boxes padded by 3 px) are matched one-to-one (greedy, best overlap first) and count once. This handles a window
           cut by a seam: both tiles see a clipped piece whose centre may fall in its own core, so core alone counts it twice.
  An unmatched detection counts only if its centre lies in its own tile's core (otherwise the owning tile did not see it).
Other rules are recomputed from the stored boxes for comparison (no API calls): sum, core, merge_all.

Variants (one line per drawing and variant in data/raster_tiles_exp.jsonl):
  tiles        tiles, item definitions as in raster_ingest.KINDS
  whole_conv   whole image, windows defined by the annotation convention (each framed window segment counts)
  tiles_conv   tiles + the convention definition

  python raster_tiles.py sample                      the 40 evaluation drawings
  python raster_tiles.py run <variant> [n] [budget]  read the first n drawings of the sample (skips stored ones)
  python raster_tiles.py eval                        comparison table
  python raster_tiles.py overlay <path> <variant>    debug image into the scratchpad
"""
import base64
import concurrent.futures
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import gemini_backend as gb  # noqa: E402
import raster_backend  # noqa: E402
import raster_ingest  # noqa: E402

MODEL = "gemini-4-hydrogen"
STORE = os.path.join(evalkit.ROOT, "data", "raster_tiles_exp.jsonl")
DEBUG = "/tmp/claude-3298278/-scratch-elec-t412-aiwo-stoa/8564a16b-36ce-4cff-8924-0e60727fd007/scratchpad/tiles"
SEED, N_SAMPLE = 2026, 40
OVERLAP = 0.20          # of the tile size, between neighbouring tiles
BIG = 10 ** 9           # 3 tiles per axis switched off: budget went to the convention variant (was 1400 px)
MATCH = 0.3             # intersection / smaller box to match detections across tiles
PAD = 3
HARD_BUDGET = 4.0       # USD for the whole experiment, all variants
EST_CALL = 0.03         # reserved per in-flight call before it reports its real cost
THREADS = 6

DEFS = {k: raster_ingest.KINDS[k] for k in ("windows", "doors")}
CONV = dict(DEFS, windows="every separately framed window segment in the walls; a window split by a frame/mullion line "
                          "into a small opening part and a larger pane counts as two")
HEAD = "This is a residential floor plan (Finnish plans use labels such as MH, OH, K, KH, WC, S, PARVEKE). "
TILE_HEAD = ("This image is one tile cut from a larger residential floor plan (Finnish plans use labels such as MH, OH, K, KH, WC, S, "
             "PARVEKE), shown enlarged. Count every item whose symbol is at least partly visible in this tile, including items cut "
             "by the tile border; box only the part visible in the tile. ")
TAIL = ('Go through the image item by item and report what is drawn. Answer with JSON only. For every key give '
        '{"count": n, "boxes": [[ymin, xmin, ymax, xmax], ...]} with one box per counted item, in 0-1000 coordinates '
        'relative to the image. Keys:\n')
VARIANTS = {"tiles": (True, DEFS), "whole_conv": (False, CONV), "tiles_conv": (True, CONV)}

_lock = threading.Lock()
_spent = [0.0, 0.0]  # [spent, reserved for in-flight calls]


def prompt(tiled, defs):
    return (TILE_HEAD if tiled else HEAD) + TAIL + "\n".join('- "%s": %s' % kv for kv in defs.items())


def sample():
    rows = [json.loads(l) for l in open(evalkit.DEV)]
    paths = sorted({r["path"] for r in rows if evalkit.source(r["path"]) == "cubicasa5k"})
    return random.Random(SEED).sample(paths, N_SAMPLE)


def load():
    recs = {}
    if os.path.exists(STORE):
        for l in open(STORE):
            d = json.loads(l)
            if "error" not in d:
                recs[(d["path"], d["variant"])] = d
    return recs


def total_spent():
    return sum(json.loads(l)["usd"] for l in open(STORE)) if os.path.exists(STORE) else 0.0


def axis(L, n):
    """-> [(start, end, core_start, core_end)] for n tiles over length L with OVERLAP between neighbours."""
    if n == 1:
        return [(0, L, 0, L)]
    t = L / (n - (n - 1) * OVERLAP)
    step = t * (1 - OVERLAP)
    out = []
    for i in range(n):
        s, e = i * step, min(L, i * step + t)
        cs = 0 if i == 0 else s + OVERLAP * t / 2
        ce = L if i == n - 1 else e - OVERLAP * t / 2
        out.append((round(s), round(e), cs, ce))
    return out


def grid(W, H, tiled):
    if not tiled:
        return 1, 1
    return (3 if W >= BIG else 2), (3 if H >= BIG else 2)


def tiles_for(W, H, tiled):
    nx, ny = grid(W, H, tiled)
    return [{"box": [xs, ys, xe, ye], "core": [cxs, cys, cxe, cye]}
            for (ys, ye, cys, cye) in axis(H, ny) for (xs, xe, cxs, cxe) in axis(W, nx)]


def call(img, text):
    """One Gemini call on a PIL image -> (parsed json or None, usage dict). Same request/retry pattern as raster_ingest."""
    buf = io.BytesIO()
    img.save(buf, "PNG")
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()}}, {"text": text}]}],
            "generationConfig": {"maxOutputTokens": 20000, "thinkingConfig": {"thinkingLevel": "medium"}}}
    for a in range(5):
        try:
            req = urllib.request.Request(gb.URL % MODEL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
            j = json.loads(urllib.request.urlopen(req, timeout=600).read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and a < 4:
                time.sleep(3 * 2 ** a)
                continue
            raise RuntimeError("Gemini API %d" % e.code)
        except (urllib.error.URLError, TimeoutError):
            if a == 4:
                raise
            time.sleep(3)
    cand = (j.get("candidates") or [{}])[0]
    parts = cand.get("content", {}).get("parts", [])
    out_text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    u = j.get("usageMetadata", {})
    n_in, n_out = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    pin, pout = gb.PRICES[MODEL]
    usage = {"tok_in": n_in, "tok_out": n_out, "tok_thought": u.get("thoughtsTokenCount", 0), "finish": cand.get("finishReason"),
             "usd": round((n_in * pin + n_out * pout) / 1e6, 5)}
    m = re.search(r"\{.*\}", out_text, re.S)
    try:
        return json.loads(m.group(0)), usage
    except (AttributeError, ValueError):
        return None, usage


def read_tile(im, tile, text):
    x0, y0, x1, y1 = tile["box"]
    tw, th = x1 - x0, y1 - y0
    with _lock:
        if _spent[0] + _spent[1] + EST_CALL > HARD_BUDGET:
            raise RuntimeError("budget")
        _spent[1] += EST_CALL
    usd, tok, out = 0.0, {"tok_in": 0, "tok_out": 0, "tok_thought": 0}, None
    try:
        for _ in range(2):  # one retry when the reply is not parseable JSON
            out, u = call(im.crop((x0, y0, x1, y1)), text)
            usd += u["usd"]
            for k in tok:
                tok[k] += u[k]
            tok["finish"] = u["finish"]
            if out is not None:
                break
    finally:
        with _lock:
            _spent[1] -= EST_CALL
            _spent[0] += usd
    res = dict(tile, usd=round(usd, 5), **tok)
    if out is None:
        res["error"] = "unparseable reply"
        return res
    res["det"], res["reported"] = {}, {}
    for k in DEFS:
        v = out.get(k)
        v = v if isinstance(v, dict) else {"count": v, "boxes": []}
        boxes = []
        for b in v.get("boxes") or []:
            if isinstance(b, list) and len(b) == 4 and all(isinstance(c, (int, float)) for c in b):
                ya, xa, yb, xb = (min(max(c, 0), 1000) / 1000 for c in b)
                boxes.append([round(x0 + min(xa, xb) * tw), round(y0 + min(ya, yb) * th), round(x0 + max(xa, xb) * tw), round(y0 + max(ya, yb) * th)])
        res["det"][k] = boxes
        n = v.get("count")
        res["reported"][k] = int(n) if isinstance(n, (int, float)) else len(boxes)
    return res


# ---------------------------------------------------------------- de-duplication (pure functions of the stored boxes)

def _in_core(b, tile):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    c = tile["core"]
    return c[0] <= cx <= c[2] and c[1] <= cy <= c[3]


def _ios(a, b):
    a = [a[0] - PAD, a[1] - PAD, a[2] + PAD, a[3] + PAD]
    b = [b[0] - PAD, b[1] - PAD, b[2] + PAD, b[3] + PAD]
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return iw * ih / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def merge(rec, kind, rule="core+merge"):
    """-> list of counted boxes (full-image px) for one kind under a de-dup rule."""
    W, H, tiles = rec["W"], rec["H"], rec["tiles"]
    dets = [(ti, b) for ti, t in enumerate(tiles) for b in t["det"][kind]]
    if rule == "sum":
        return [b for _, b in dets]
    if rule == "core":
        return [b for ti, b in dets if _in_core(b, tiles[ti])]
    parent = list(range(len(dets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    pairs = sorted(((_ios(dets[i][1], dets[j][1]), i, j) for i in range(len(dets)) for j in range(i + 1, len(dets))
                    if dets[i][0] != dets[j][0]), reverse=True)
    used = set()  # one-to-one per tile pair: (detection, other tile)
    for s, i, j in pairs:
        if s < MATCH:
            break
        if (i, dets[j][0]) in used or (j, dets[i][0]) in used:
            continue
        used.add((i, dets[j][0]))
        used.add((j, dets[i][0]))
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(dets)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        bs = [dets[i][1] for i in g]
        u = [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]
        if len(g) > 1 or rule == "merge_all" or _in_core(bs[0], tiles[dets[g[0]][0]]):
            out.append(u)
    return out


RULES = ("sum", "core", "merge_all", "core+merge")


def summarize(rec):
    rec["counts_by_rule"] = {r: {k: len(merge(rec, k, r)) for k in DEFS} for r in RULES}
    rec["counts"] = rec["counts_by_rule"]["core+merge"]
    rec["boxes"] = {k: merge(rec, k) for k in DEFS}
    return rec


def run_image(path, variant):
    from PIL import Image
    tiled, defs = VARIANTS[variant]
    im = Image.open(os.path.join(evalkit.DATA, path)).convert("RGB")
    W, H = im.size
    text = prompt(tiled, defs)
    tiles = tiles_for(W, H, tiled)
    with concurrent.futures.ThreadPoolExecutor(min(4, len(tiles))) as ex:
        futs = [ex.submit(read_tile, im, t, text) for t in tiles]
        res = []
        for f in futs:
            try:
                res.append(f.result())
            except Exception as e:  # budget stop or API failure: keep the tiles that finished, so their cost is recorded
                res.append({"error": str(e)[:100], "usd": 0.0})
    rec = {"path": path, "variant": variant, "W": W, "H": H, "grid": list(grid(W, H, tiled)), "overlap": OVERLAP if tiled else 0,
           "tiles": res, "model": MODEL, "thinking": "medium",
           "tokens": sum(t.get("tok_in", 0) + t.get("tok_out", 0) for t in res), "usd": round(sum(t["usd"] for t in res), 5)}
    errs = [t["error"] for t in res if "error" in t]
    if errs:
        rec["error"] = errs[0]
    else:
        summarize(rec)
    with _lock:
        with open(STORE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def run(variant, n, budget):
    global HARD_BUDGET
    HARD_BUDGET = min(budget, 4.0)
    _spent[0] = total_spent()
    done = load()
    todo = [p for p in sample()[:n] if (p, variant) not in done]
    print("variant %s: %d drawings to read, spent so far $%.3f, stop at $%.2f" % (variant, len(todo), _spent[0], HARD_BUDGET), flush=True)
    t0 = time.time()
    # images in parallel, tiles of an image in parallel; THREADS caps the number of concurrent calls
    n_img = 2 if VARIANTS[variant][0] else THREADS  # tiled: 2 drawings x up to 4 tiles = 8 calls at most
    with concurrent.futures.ThreadPoolExecutor(n_img) as ex:
        for rec in ex.map(lambda p: run_image(p, variant), todo):
            print("  %-55s grid %s  %s  $%.4f  total $%.3f  %.0fs" % (rec["path"], rec["grid"], rec.get("counts", rec.get("error")), rec["usd"], _spent[0], time.time() - t0), flush=True)


# ---------------------------------------------------------------- evaluation

def gt_counts(paths):
    want, gt = set(paths), {}
    for l in open(os.path.join(evalkit.DATA, "questions.jsonl")):
        r = json.loads(l)
        if r["path"] in want and r["question"].startswith("How many"):  # English only; the Finnish copies have the same answers
            k = raster_backend.question_kind(r["question"])
            if k in DEFS:
                gt[(r["path"], k)] = r["answer"]
    return gt


def stats(pairs):
    n = len(pairs)
    if not n:
        return "n=0"
    d = [p - g for p, g in pairs]
    return "%2d  %5.1f%%  %5.1f%%  %2d/%2d/%2d  %+5.2f" % (n, 100 * sum(x == 0 for x in d) / n, 100 * sum(abs(x) <= 1 for x in d) / n,
                                                          sum(x < 0 for x in d), sum(x == 0 for x in d), sum(x > 0 for x in d), sum(d) / n)


def evaluate():
    paths = sample()
    gt = gt_counts(paths)
    gem = {}
    for l in open(raster_ingest.STORE):
        d = json.loads(l)
        gem[d["path"]] = d
    seg = {}
    for l in open(os.path.join(evalkit.ROOT, "data", "raster_facts.jsonl")):
        d = json.loads(l)
        if d["path"] in paths:
            seg[d["path"]] = d["answers"]
    recs = load()
    for r in recs.values():
        summarize(r)  # recompute so rule changes need no new calls
    methods = {"whole (raster_gemini)": lambda p, k: gem[p]["facts"][k]["count"] if p in gem else None,
               "segmentation": lambda p, k: seg.get(p, {}).get(k)}
    for v in VARIANTS:
        if any(key[1] == v for key in recs):
            methods[v] = (lambda v: lambda p, k: recs[(p, v)]["counts"][k] if (p, v) in recs else None)(v)
    common = [p for p in paths if all(m(p, "windows") is not None for m in methods.values())]
    print("drawings with every method available: %d of %d" % (len(common), len(paths)))
    print("%-24s %-8s  n  exact  +-1    under/ex/over  mean err" % ("method", "kind"))
    for k in DEFS:
        for name, m in methods.items():
            pairs = [(m(p, k), gt[(p, k)]) for p in common if (p, k) in gt]
            print("%-24s %-8s %s" % (name, k, stats(pairs)))
    base = methods["whole (raster_gemini)"]
    print("paired vs whole (raster_gemini): |error| smaller / same / larger, two-sided sign test on smaller vs larger")
    for k in DEFS:
        for name, m in list(methods.items())[1:]:
            ps = [p for p in common if (p, k) in gt]
            d = [abs(m(p, k) - gt[(p, k)]) - abs(base(p, k) - gt[(p, k)]) for p in ps]
            w, l = sum(x < 0 for x in d), sum(x > 0 for x in d)
            n = w + l
            pv = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
            print("  %-14s %-8s %2d / %2d / %2d   p=%.2f" % (name, k, w, len(d) - n, l, pv))
    print("windows by GT bucket (exact acc, mean err):")
    for lo, hi, lab in ((0, 5, "<=5"), (6, 10, "6-10"), (11, 999, ">10")):
        ps = [p for p in common if lo <= gt[(p, "windows")] <= hi]
        print("  GT %-5s n=%2d  " % (lab, len(ps)) + " | ".join("%s %s" % (name.split(" ")[0], stats([(m(p, "windows"), gt[(p, "windows")]) for p in ps])[4:])
                                                              for name, m in methods.items()))
    print("de-dup rules on stored boxes (exact acc / mean err):")
    for v in VARIANTS:
        if VARIANTS[v][0] and any(key[1] == v for key in recs):
            for k in DEFS:
                line = []
                for rule in RULES + ("reported_sum",):
                    def f(p):
                        r = recs[(p, v)]
                        return sum(t["reported"][k] for t in r["tiles"]) if rule == "reported_sum" else r["counts_by_rule"][rule][k]
                    pairs = [(f(p), gt[(p, k)]) for p in common if (p, k) in gt]
                    d = [a - b for a, b in pairs]
                    line.append("%s %.1f%% %+.2f" % (rule, 100 * sum(x == 0 for x in d) / len(d), sum(d) / len(d)))
                print("  %-10s %-8s %s" % (v, k, " | ".join(line)))
    print("cost:")
    for v in VARIANTS:
        rs = [r for key, r in recs.items() if key[1] == v]
        if rs:
            print("  %-10s %d drawings, %d calls, $%.3f, $%.4f per drawing, %.0f tokens per drawing" % (
                v, len(rs), sum(len(r["tiles"]) for r in rs), sum(r["usd"] for r in rs), sum(r["usd"] for r in rs) / len(rs), sum(r["tokens"] for r in rs) / len(rs)))
    print("  whole (raster_gemini, 12 kinds) $%.4f per drawing on these" % (sum(gem[p]["usd"] for p in common) / max(1, len(common))))
    print("  all API spend recorded in %s: $%.3f" % (os.path.basename(STORE), total_spent()))


def overlay(path, variant):
    from PIL import Image, ImageDraw
    rec = summarize(load()[(path, variant)])
    im = Image.open(os.path.join(evalkit.DATA, path)).convert("RGB")
    s = max(1.0, 1600 / max(im.size))
    im = im.resize((round(im.size[0] * s), round(im.size[1] * s)))
    d = ImageDraw.Draw(im, "RGBA")
    S = lambda b: [v * s for v in b]  # noqa: E731
    cols = [(230, 120, 0), (0, 150, 220), (170, 0, 200), (0, 170, 80), (200, 0, 0), (120, 120, 0), (0, 0, 200), (100, 60, 0), (0, 120, 120)]
    for ti, t in enumerate(rec["tiles"]):
        d.rectangle(S(t["box"]), outline=cols[ti % 9] + (120,), width=2)
        c = t["core"]
        d.rectangle(S(c), outline=(255, 0, 0, 90), width=1)
        for k in DEFS:
            for b in t["det"][k]:  # every raw detection, coloured by tile; dashed look via thin outline
                d.rectangle(S(b), outline=cols[ti % 9] + (255,), width=1)
    for k, col in (("windows", (0, 200, 0, 70)), ("doors", (0, 0, 255, 60))):
        for b in rec["boxes"][k]:  # counted items, filled
            d.rectangle(S(b), fill=col, outline=col[:3] + (255,), width=3)
    gt = gt_counts([path])
    title = "%s %s  windows %d (GT %s)  doors %d (GT %s)" % (os.path.basename(path), variant, rec["counts"]["windows"], gt.get((path, "windows")),
                                                            rec["counts"]["doors"], gt.get((path, "doors")))
    d.rectangle([0, 0, im.size[0], 22], fill=(255, 255, 255, 230))
    d.text((4, 4), title, fill=(0, 0, 0))
    os.makedirs(DEBUG, exist_ok=True)
    out = os.path.join(DEBUG, "%s_%s.png" % (os.path.basename(path)[:-4], variant))
    im.save(out)
    print(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["sample"]:
        for p in sample():
            print(p)
    elif a[:1] == ["run"] and len(a) >= 2 and a[1] in VARIANTS:
        run(a[1], int(a[2]) if len(a) > 2 else N_SAMPLE, float(a[3]) if len(a) > 3 else HARD_BUDGET)
    elif a[:1] == ["eval"]:
        evaluate()
    elif a[:1] == ["overlay"] and len(a) == 3:
        overlay(a[1], a[2])
    else:
        print(__doc__)
