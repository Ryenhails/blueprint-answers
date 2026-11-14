"""Step 0: fixed dev set + scorer.

  python evalkit.py make-dev            -> writes data/dev500.jsonl (one file)
  python evalkit.py score preds.jsonl   -> accuracy tables; preds lines: {"id", "answer", "cost_usd"?, "tokens"?}

Scoring follows the dataset README: counts exact; area/length/volume within 2%;
text is free-form (reported separately as normalised exact match = lower bound).
"""
import collections
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repository
DATA = os.environ.get("STOA_DATA", os.path.join(ROOT, "data", "junction-dataset"))  # the challenge dataset (not in the repository)
DEV = os.path.join(ROOT, "data", "dev500.jsonl")
SEED = 20261113
# (source, type) -> number of questions; one question per drawing where possible
ALLOC = {
    ("cubicasa5k", "count"): 150,
    ("swiss_dwellings", "count"): 80, ("swiss_dwellings", "area"): 70,
    ("ifc_models", "area"): 40, ("ifc_models", "count"): 30, ("ifc_models", "length"): 15, ("ifc_models", "volume"): 15,
    ("ifc_bench", "text"): 50,
    ("opening_design", "count"): 25, ("opening_design", "area"): 9, ("opening_design", "text"): 16,
}


def source(path):
    return path.split("/")[1]


def load_questions():
    with open(os.path.join(DATA, "questions.jsonl")) as f:
        return [json.loads(l) for l in f]


def to_number(x):
    if isinstance(x, (int, float)):
        return float(x)
    m = re.findall(r"-?\d+(?:\.\d+)?", str(x).replace(",", "").replace("’", ""))
    return float(m[0]) if m else None


def is_correct(pred, gold, qtype):
    """True/False for numeric types, None for text (needs a judge)."""
    if qtype == "text":
        return None
    p, g = to_number(pred), float(gold)
    if p is None:
        return False
    if qtype == "count":
        return p == g
    return abs(p - g) <= 0.02 * abs(g)


def norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def make_dev():
    rows = load_questions()
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by[(source(r["path"]), r["type"])].append((i, r))
    out = []
    for key, n in ALLOC.items():
        pool = by[key][:]
        rng.shuffle(pool)
        seen, pick, rest = set(), [], []
        for i, r in pool:  # prefer distinct drawings
            (pick if r["path"] not in seen else rest).append((i, r))
            seen.add(r["path"])
        pick = (pick + rest)[:n]
        for i, r in pick:
            out.append({"id": "d%06d" % i, "source": key[0], **r})
    out.sort(key=lambda r: r["id"])
    with open(DEV, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter((r["source"], r["type"]) for r in out)
    print("wrote", DEV, len(out), "questions,", len({r["path"] for r in out}), "drawings")
    for k, v in sorted(c.items()):
        print("  ", k, v)


def score(pred_path, dev_path=DEV):
    gold = {}
    with open(dev_path) as f:
        for l in f:
            r = json.loads(l)
            gold[r["id"]] = r
    preds = {}
    with open(pred_path) as f:
        for l in f:
            r = json.loads(l)
            preds[r["id"]] = r
    agg = collections.defaultdict(lambda: [0, 0])
    text = [0, 0]
    cost = tokens = 0.0
    for qid, g in gold.items():
        p = preds.get(qid)
        ans = p.get("answer") if p else None
        if p:
            cost += p.get("cost_usd") or 0
            tokens += p.get("tokens") or 0
        ok = is_correct(ans, g["answer"], g["type"]) if ans is not None else (None if g["type"] == "text" else False)
        if g["type"] == "text":
            text[1] += 1
            text[0] += bool(ans is not None and norm_text(ans) == norm_text(g["answer"]))
            continue
        for k in (("ALL", ""), ("source", g["source"]), ("type", g["type"]), ("source/type", g["source"] + "/" + g["type"])):
            agg[k][0] += ok
            agg[k][1] += 1
    print("answered %d / %d" % (len(set(preds) & set(gold)), len(gold)))
    for k, (a, n) in sorted(agg.items()):
        print("  %-12s %-28s %4d / %4d  %.1f%%" % (k[0], k[1], a, n, 100 * a / n))
    print("  text (exact-match lower bound)       %4d / %4d" % tuple(text))
    if cost or tokens:
        print("  cost: $%.4f total, $%.5f per question; %d tokens" % (cost, cost / max(1, len(preds)), tokens))
    return agg


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "make-dev":
        make_dev()
    elif len(sys.argv) > 2 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        print(__doc__)
