"""Image route: every floor-plan image gets a small fact table (twelve counts, each with the boxes of the items counted),
and every question is answered from that table.

Three readers can fill the table. Each kind of count takes the reader that measured best on held-out drawings
(data/raster_sources.json, written by `select`); when two are within two points, the cheaper one wins.

  detector   trained object detector (raster_detector.py): windows, doors, toilets, sinks, showers+baths.
             Free, boxes match the count exactly. Only for the dataset's cubicasa5k drawings (it learned their drawing
             style and their counting convention), precomputed in data/raster_detector.jsonl.
  model      Gemini reads the image once and the reading is stored:
               rooms read  (gemini-3.8-flash, ~$0.004): the seven room-level counts      -> data/raster_rooms.jsonl
               full read   (gemini-4-hydrogen, ~$0.036): all twelve, for images the detector does not cover -> data/raster_gemini.jsonl
  seg        older stored predictions of the segmentation network (raster_backend); the fallback without network.

A drawing is only read when a question needs it, and never twice. When another reader already has a number for the same
count, the answer says whether it confirms or contradicts the chosen one. On held-out drawings a confirmed detector count is
right far more often than a contradicted one (windows 70% vs 32%, toilets 97% vs 58%), so this is a real reliability signal.

  python raster_ingest.py select           measure the readers on held-out drawings, choose one per kind
  python raster_ingest.py dev              answer the dev set's image questions through this route and score them
  python raster_ingest.py ask <image> "<question>"
"""
import base64
import collections
import json
import os
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
import raster_detector  # noqa: E402

MODEL = os.environ.get("STOA_VISION_MODEL", "gemini-4-hydrogen")  # full read
ROOM_MODEL = os.environ.get("STOA_ROOM_MODEL", "gemini-3.8-flash")  # rooms read
STORE = os.path.join(evalkit.ROOT, "data", "raster_gemini.jsonl")
ROOM_STORE = os.path.join(evalkit.ROOT, "data", "raster_rooms.jsonl")
SOURCES = os.path.join(evalkit.ROOT, "data", "raster_sources.json")
KINDS = {  # same names as raster_backend.question_kind
    "windows": "window symbols in the walls",
    "doors": "doorways fitted with a door leaf (swing arc or sliding leaf); open doorways without a leaf do not count",
    "toilets": "toilet bowls (WC seats) drawn",
    "sinks": "sinks and wash basins, kitchen and bathroom combined; a double-bowl sink counts as one",
    "showers_baths": "showers and bathtubs drawn, in total",
    "saunas": "sauna rooms",
    "kitchens": "kitchens, including kitchenettes, open kitchens and utility kitchens",
    "bedrooms": "bedrooms (makuuhuone, usually MH)",
    "bathrooms": "bathrooms or washrooms (KH, KPH, PH, PESUH, SH), not counting separate WC rooms",
    "wc_rooms": "separate WC rooms (toilet rooms marked WC)",
    "balconies": "balconies (parveke, including glazed balconies)",
    "floors": "floors (storeys) the sheet shows",
}
OBJECTS = raster_detector.KINDS
ROOMS = [k for k in KINDS if k not in OBJECTS]
HEAD = ("This is a residential floor plan (Finnish plans use labels such as MH, OH, K, KH, WC, S, PARVEKE). "
        "Go through the plan item by item and report what is drawn. Answer with JSON only. For every key give "
        '{"count": n, "boxes": [[ymin, xmin, ymax, xmax], ...]} with one box per counted item, in 0-1000 coordinates '
        'relative to the image ("floors" needs no boxes). Keys:\n')
NAMES = {"detector": "the object detector", "model": "the model's reading of the image", "seg": "the segmentation network"}
COST_ORDER = ["detector", "seg", "model"]  # cheapest first; used to break near-ties

_lock = threading.Lock()
_stores = {}
_sources = None


def _load(path=STORE):
    if path not in _stores:
        _stores[path] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    _stores[path][d["path"]] = d
    return _stores[path]


def _key(full):
    p = os.path.abspath(full)
    root = os.path.abspath(evalkit.DATA) + os.sep
    return p[len(root):] if p.startswith(root) else p


def sources():
    """-> {kind: reader name} as chosen by `select`; before any measurement, the model reads everything."""
    global _sources
    if _sources is None:
        _sources = {k: v["source"] for k, v in json.load(open(SOURCES))["kinds"].items()} if os.path.exists(SOURCES) else {}
    return _sources


def read(full, kinds, model, thinking, store):
    """One Gemini read of the image for the given kinds; stored, so a drawing is read at most once per store.
    -> (record {"path", "facts": {kind: {"count", "boxes" in pixels}}, "tokens", "usd", "model"}, True if read just now)"""
    key = _key(full)
    if key in _load(store):
        return _stores[store][key], False
    from PIL import Image

    W, H = Image.open(full).size
    mime = "image/png" if full.lower().endswith(".png") else "image/jpeg"
    prompt = HEAD + "\n".join('- "%s": %s' % (k, KINDS[k]) for k in kinds)
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": mime, "data": base64.b64encode(open(full, "rb").read()).decode()}}, {"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 60000, "thinkingConfig": {"thinkingLevel": thinking}}}
    for a in range(5):
        try:
            req = urllib.request.Request(gb.URL % model, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
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
    parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    facts = {}
    for k in kinds:
        v = out.get(k)
        v = v if isinstance(v, dict) else {"count": v, "boxes": []}
        boxes = [[round(b[1] / 1000 * W), round(b[0] / 1000 * H), round(b[3] / 1000 * W), round(b[2] / 1000 * H)]
                 for b in (v.get("boxes") or []) if isinstance(b, list) and len(b) == 4]
        n = v.get("count")
        facts[k] = {"count": int(n) if isinstance(n, (int, float)) else len(boxes), "boxes": boxes}
    u = j.get("usageMetadata", {})
    n_in, n_out = u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    pin, pout = gb.PRICES.get(model, (0, 0))
    rec = {"path": key, "facts": facts, "tokens": n_in + n_out, "usd": round((n_in * pin + n_out * pout) / 1e6, 5), "model": model}
    with _lock:
        _stores[store][key] = rec
        with open(store, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec, True


def ingest(full):
    """Full read (all twelve kinds). Kept under this name for callers that pre-read drawings."""
    return read(full, list(KINDS), MODEL, "medium", STORE)


MEDIAN = ("windows", "toilets", "sinks", "showers_baths")  # doors: the detector alone is best


def stored(key, kind):
    """Readings already on disk for this drawing and kind, at no cost. -> {reader: {"count", "boxes"}}"""
    r = {}
    d = raster_detector.load().get(key)
    if d and kind in OBJECTS:
        r["detector"] = {"count": len(d["boxes"][kind]), "boxes": d["boxes"][kind]}
    for store in (STORE, ROOM_STORE):
        g = _load(store).get(key)
        if g and kind in g["facts"] and "model" not in r:
            r["model"] = dict(g["facts"][kind], by=g["model"])
    facts, _ = raster_backend._load()
    s = facts.get(key)
    if s:
        r["seg"] = {"count": s["answers"][kind], "boxes": s["evidence"].get(kind, [])}
    return r


_known = None
_decider = None


def known_kind(question):
    """The dataset's own question forms (24 strings) map straight to a kind; anything else needs the decision step."""
    global _known
    if _known is None:
        _known = {}
        for r in (evalkit.load_questions() if os.path.exists(os.path.join(evalkit.DATA, "questions.jsonl")) else []):
            if r["path"].startswith("data/cubicasa5k/"):
                _known.setdefault(" ".join(r["question"].lower().split()), raster_backend.question_kind(r["question"]))
    return _known.get(" ".join(question.lower().split()))


def decide(question):
    """A new phrasing: the small model says whether it is one of the twelve plain counts, a complex question, or something
    the fact tables do not hold. -> (kind | "complex" | "other", tokens, usd). Without the API: keyword match."""
    global _decider
    try:
        if _decider is None:
            _decider = gb.Gemini(gb.DEFAULT_PLAN)
        before = _decider.usd
        sys_msg = ("A question is asked about a residential floor plan, in any language. Reply with exactly one word.\n"
                   "If it asks for exactly one plain count over the whole floor plan and nothing else, reply the name of that count:\n"
                   + "\n".join("%s = %s" % kv for kv in KINDS.items())
                   + "\nReply complex if it is about rooms, doors, windows or fixtures but needs more than one such count: a condition or a "
                     "link between things (rooms with a window, doors of the bedroom), which / where / list, a comparison, or arithmetic.\n"
                     "Reply other for anything else: text on the sheet, dimensions, areas, materials, who drew it, what the drawing shows, a question with "
                     "several parts, or a question about an elevation, section, detail, schedule or map rather than a floor plan.")
        texts, a, b = _decider.generate([[{"role": "system", "content": sys_msg}, {"role": "user", "content": question}]], 12)
        w = re.sub(r"[^a-z_]", "", texts[0].strip().lower().split()[0]) if texts[0].strip() else ""
        return (w if w in KINDS or w in ("complex", "other") else "complex"), a + b, _decider.usd - before
    except Exception:
        return raster_backend.question_kind(question) or "other", 0, 0.0


def answer(full, question):
    kind, d_tok, d_usd = known_kind(question), 0, 0.0
    if kind is None:
        kind, d_tok, d_usd = decide(question)
    if kind not in KINDS:  # complex -> the caller writes code on the plan's fact tables; other -> the caller reads the sheet
        return {"answer": None, "unit": None, "evidence": [], "plan": {"op": kind}, "plan_source": "llm", "llm_tokens": d_tok, "cost_usd": d_usd}
    key = _key(full)
    have = stored(key, kind)
    want = sources().get(kind, "model")
    tokens, usd, note = d_tok, d_usd, None
    three = kind in MEDIAN and "detector" in have and "seg" in have  # a dataset floor plan: three independent readers exist
    if (want not in have or three) and "model" not in have:  # nothing usable on disk (or the third reading is missing): one model read, the cheapest that covers this kind
        try:
            rec, _ = read(full, ROOMS, ROOM_MODEL, "low", ROOM_STORE) if kind in ROOMS else ingest(full)
            tokens, usd = tokens + rec["tokens"], usd + rec["usd"]
            have["model"] = dict(rec["facts"][kind], by=rec["model"])
        except Exception as e:  # no key, no network, unreadable reply
            note = "Model read failed (%s)." % (str(e)[:80],)
    use = want if want in have else next((s for s in ("model", "detector", "seg") if s in have), None)
    if three and "model" in have:  # the middle one of three counts beats every single reader (handoff_raster_counts.md)
        mid = sorted(have[s]["count"] for s in ("detector", "model", "seg"))[1]
        use = next(s for s in ("detector", "model", "seg") if have[s]["count"] == mid)
    if use is None:
        return {"answer": None, "unit": "count", "evidence": [], "plan": {"kind": kind}, "plan_source": "keyword", "llm_tokens": tokens, "cost_usd": usd,
                "note": (note or "") + " No reading is available for this image."}
    f = have[use]
    others = {s: v["count"] for s, v in have.items() if s != use}  # any other reading already on disk is a free cross-check
    same = [s for s, c in others.items() if c == f["count"]]
    if three and "model" in have:
        rel = "middle of three readings (%s)" % ", ".join("%s: %d" % (NAMES[s], have[s]["count"]) for s in ("detector", "model", "seg")) + \
              ("; all three agree" if len(same) == 2 else "; two agree" if same else "; all three differ")
    elif not others:
        rel = "single reading"
    elif same:
        rel = "confirmed by a second reading (%s)" % NAMES[same[0]]
    else:
        rel = "readings disagree (%s)" % ", ".join("%s: %d" % (NAMES[s], c) for s, c in others.items())
    res = {"answer": f["count"], "unit": "count", "plan": {"kind": kind, "reader": use, "other_readings": others}, "plan_source": "ingested" if not d_tok else "llm",
           "evidence": [{"kind": kind, "name": kind.rstrip("s").replace("_", " "), "bbox": b} for b in f["boxes"]],
           "calc": "%d: %s. From %s%s, stored when the drawing was first read%s." % (
               f["count"], KINDS[kind], NAMES[use], " (%s)" % f["by"] if f.get("by") else "",
               "" if tokens else "; no model call for this question"),
           "kind_from": "known question form" if not d_tok else "decision model",
           "reliability": rel, "llm_tokens": tokens, "cost_usd": usd}
    if note:
        res["note"] = note
    return res


def _gold(paths):
    g = collections.defaultdict(dict)
    for r in evalkit.load_questions():
        if r["path"] in paths and r["question"].startswith("How"):
            k = raster_backend.question_kind(r["question"])
            if k:
                g[r["path"]][k] = r["answer"]
    return g


def select():
    """Measure every reader per kind on drawings none of them was trained or tuned on, and choose."""
    det = raster_detector.load()
    gem = _load(STORE)
    paths = {p for p in gem if p.startswith("data/cubicasa5k/") and (not det or (p in det and not det[p]["seen"]))}
    gold = _gold(paths)
    acc = {k: collections.defaultdict(lambda: [0, 0]) for k in KINDS}
    agree = collections.defaultdict(lambda: [0, 0])
    for p in sorted(paths):
        for k, y in gold[p].items():
            have = stored(p, k)
            for s, v in have.items():
                acc[k][s][0] += v["count"] == y
                acc[k][s][1] += 1
            if "detector" in have and "model" in have:
                a = agree[(k, have["detector"]["count"] == have["model"]["count"])]
                a[0] += have["detector"]["count"] == y
                a[1] += 1
    out = {"measured_on": "%d dev-set cubicasa5k drawings with a stored model read%s; English question of each kind" % (
        len(paths), ", none used to train or tune the detector" if det else ""), "kinds": {}}
    print(out["measured_on"])
    print("%-14s %s  -> chosen" % ("kind", "  ".join("%-9s" % s for s in COST_ORDER)))
    for k in KINDS:
        rate = {s: c / n for s, (c, n) in acc[k].items() if n}
        best = max(rate.values())
        pick = next(s for s in COST_ORDER if s in rate and rate[s] >= best - 0.02)  # within two points: the cheaper reader
        out["kinds"][k] = {"source": pick, "correct": {s: acc[k][s] for s in rate}}
        print("%-14s %s  -> %s" % (k, "  ".join("%3d/%-3d %2.0f%%" % (acc[k][s][0], acc[k][s][1], 100 * rate[s]) if s in rate else " " * 11 for s in COST_ORDER), pick))
    if agree:
        print("\nDetector accuracy when the model's reading agrees with it / disagrees:")
        for k in OBJECTS:
            a, d = agree[(k, True)], agree[(k, False)]
            print("  %-14s agree %3d/%-3d %3.0f%%   disagree %3d/%-3d %3.0f%%" % (k, a[0], a[1], 100 * a[0] / max(1, a[1]), d[0], d[1], 100 * d[0] / max(1, d[1])))
            out["kinds"][k]["detector_correct_when_model_agrees"] = a
            out["kinds"][k]["detector_correct_when_model_disagrees"] = d
    with open(SOURCES, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", SOURCES)


def dev():
    rows = [r for r in map(json.loads, open(evalkit.DEV)) if evalkit.source(r["path"]) == "cubicasa5k"]
    agg, usd = collections.defaultdict(lambda: [0, 0, 0]), 0.0
    for r in rows:
        full = os.path.join(evalkit.DATA, r["path"])
        res = answer(full, r["question"])
        k = raster_backend.question_kind(r["question"])
        usd += res["cost_usd"]
        a = agg[(k, res["plan"].get("reader"))]
        a[0] += res["answer"] == r["answer"]
        a[1] += raster_backend.answer(full, r["question"])["answer"] == r["answer"]
        a[2] += 1
    for (k, s), (g, c, n) in sorted(agg.items(), key=str):
        print("  %-14s %-9s %2d/%-2d | segmentation alone %2d/%-2d" % (k, s, g, n, c, n))
    tot = [sum(a[i] for a in agg.values()) for i in range(3)]
    print("  ALL %d/%d (%.1f%%) | segmentation alone %d/%d (%.1f%%) | new model reads $%.3f" % (tot[0], tot[2], 100 * tot[0] / tot[2], tot[1], tot[2], 100 * tot[1] / tot[2], usd))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["select"]:
        select()
    elif a[:1] == ["dev"]:
        dev()
    elif a[:1] == ["ask"] and len(a) == 3:
        r = answer(a[1] if os.path.isabs(a[1]) else os.path.join(evalkit.DATA, a[1]), a[2])
        r["evidence"] = len(r["evidence"])
        print(json.dumps(r, ensure_ascii=False, indent=1))
    else:
        print(__doc__)
