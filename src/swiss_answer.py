"""Validation harness for the swiss_dwellings adapter (no LLM).

Maps the ENGLISH question templates to queries on the fact table, so the adapter
can be checked against every English swiss_dwellings question. Step 2 replaces
this regex mapping with a model that turns any-language questions into queries.

  python swiss_answer.py [n_files|all]
"""
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import swiss_adapter  # noqa: E402

WHITE = (1.0, 1.0, 1.0)
CIRC = (0.96, 0.96, 0.96)      # stairs / lifts / common circulation: numbered, indoor, no label
OUTDOOR = (0.91, 0.92, 0.91)   # balconies, loggias, winter gardens


def norm(s):
    return re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()


def answer(q, f):
    """Return (kind, value) or (None, None) if the template is unknown."""
    rooms = f["rooms"]
    m = re.match(r"How many rooms labelled '(.+)' are there on this floor\?$", q)
    if m:
        return "rooms_labelled", sum(1 for r in rooms if r["label"] and norm(r["label"]) == norm(m.group(1)))
    simple = {
        "How many apartments": ("apartments", lambda: len(f["unit_names"])),
        "How many doors": ("doors", lambda: len(f["doors"])),
        "How many windows": ("windows", lambda: len(f["windows"])),
        "How many toilets": ("toilets", lambda: len(f["toilets"])),
        "How many washbasins": ("washbasins", lambda: len(f["washbasins"])),
        "How many bathtubs": ("bathtubs", lambda: len(f["bathtubs"])),
        "How many showers": ("showers", lambda: len(f["showers"])),
    }
    for p, (kind, fn) in simple.items():
        if q.startswith(p):
            return kind, fn()
    m = re.match(r"What is the net floor area of room (\S+) \(", q)
    if m:
        r = [r for r in rooms if r["no"] == m.group(1)]
        return "room_area", (r[0]["area_m2"] if r and r[0]["area_m2"] else None)
    m = re.match(r"What is the total net area in m² of all rooms labelled '(.+)' on this floor\?$", q)
    if m:
        return "label_area", sum(r["area_m2"] or 0 for r in rooms if r["label"] and norm(r["label"]) == norm(m.group(1)))
    m = re.match(r"What is the net floor area of apartment (\S+) in m²", q)
    if m:
        return "apartment_area", sum(r["area_m2"] or 0 for r in rooms if r["unit"] and r["unit"].split()[-1] == m.group(1) and r["fill"] == WHITE)
    if q.startswith("What is the total net floor area in m² of all numbered indoor rooms"):
        return "floor_area", sum(r["area_m2"] or 0 for r in rooms if r["fill"] in (WHITE, CIRC))
    return None, None


def main(n):
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "swiss_dwellings"]
    files = sorted({r["path"] for r in rows})
    if n != "all":
        random.Random(7).shuffle(files)
        files = set(files[: int(n)])
    agg = collections.defaultdict(lambda: [0, 0])
    unknown = collections.Counter()
    misses = collections.defaultdict(list)
    cache = {}
    for r in rows:
        if r["path"] not in files or not re.match(r"^(How|What) ", r["question"]):
            continue
        if r["path"] not in cache:
            try:
                cache[r["path"]] = swiss_adapter.parse(evalkit.DATA + "/" + r["path"])
            except Exception as e:  # keep going, count as failure
                cache[r["path"]] = e
        f = cache[r["path"]]
        if isinstance(f, Exception):
            agg["PARSE_ERROR"][1] += 1
            continue
        kind, val = answer(r["question"], f)
        if kind is None:
            unknown[re.sub(r"'.*?'|\d+", "#", r["question"])[:70]] += 1
            continue
        ok = val is not None and evalkit.is_correct(val, r["answer"], r["type"])
        agg[kind][0] += ok
        agg[kind][1] += 1
        agg["ALL"][0] += ok
        agg["ALL"][1] += 1
        if not ok and len(misses[kind]) < 6:
            misses[kind].append((r["path"].split("/")[-1], r["answer"], None if val is None else round(val, 2), r["question"][:60]))
    print("files", len(cache))
    for k, (a, b) in sorted(agg.items()):
        print("  %-16s %6d / %6d  %.1f%%" % (k, a, b, 100 * a / max(1, b)))
    print("unknown templates:", unknown.most_common(8))
    for k, v in misses.items():
        print(k)
        for x in v:
            print("    ", x)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "120")
