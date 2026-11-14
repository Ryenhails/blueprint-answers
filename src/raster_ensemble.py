"""Combining stored count readings on held-out cubicasa5k drawings (pre-registered in raster_yolo_plan.md section 12).

Readers: P = pilot detector, H = high-res detector (fine-tuned from P), S = segmentation network, G = Gemini full read.
E1 = the 150 dev drawings (where the vote rule was found), E2 = the 380 official-test drawings (the confirmation set).

  python raster_ensemble.py read [budget_usd] [workers]   Gemini full read of every E2 drawing not read yet, through
                                                         raster_ingest.ingest (same model, prompt and store as E1)
  python raster_ensemble.py eval                          score the rules on E1 and E2
"""
import collections
import json
import math
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubi_yolo as cy  # noqa: E402
import evalkit  # noqa: E402
import raster_backend  # noqa: E402
import raster_ingest as ri  # noqa: E402

Y = "/scratch/work/liuz16/stoa_raster/yolo/"
LETTER = {"detector": "P", "seg": "S", "model": "G"}


def e2_paths():
    L, role = cy.sets()
    return sorted(v["comp_path"] for k, v in L.items() if role[k] == "E2")


def read(budget=16.0, workers=6):
    budget, workers = float(budget), int(workers)
    todo = [p for p in e2_paths() if p not in ri._load(ri.STORE)]
    spent, done, failed = [0.0], [0], []
    lock = threading.Lock()
    print("E2 drawings to read: %d (budget $%.2f, %d workers)" % (len(todo), budget, workers), flush=True)

    def one(p):
        with lock:
            if spent[0] >= budget:
                return
        try:
            rec, new = ri.ingest(os.path.join(evalkit.DATA, p))
            with lock:
                spent[0] += rec["usd"] if new else 0
                done[0] += 1
                if done[0] % 20 == 0:
                    print("  %d read, $%.2f" % (done[0], spent[0]), flush=True)
        except Exception as e:  # one bad reply must not stop the run
            with lock:
                failed.append((p, str(e)[:100]))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print("read %d, failed %d, spent $%.2f" % (done[0], len(failed), spent[0]))
    for f in failed[:10]:
        print("  failed", f)


def _det(run):
    P = json.load(open(Y + run + "/preds.json"))["preds"]
    t = json.load(open(Y + run + "/preds.thr.json"))
    return lambda k, kind: cy._count(P[k], cy.CID[kind], t[kind])


def vote(first, others):
    """Keep `first` unless at least two of the others agree on a different number that then outnumbers it."""
    allv = collections.Counter([first] + others)
    c = collections.Counter(others).most_common()
    return c[0][0] if c and c[0][1] >= 2 and c[0][0] != first and allv[c[0][0]] > allv[first] else first


def _med(xs):
    return int(round(statistics.median(xs)))


def _p2(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def _others(r, cur, letters):
    return [r[s] for s in letters if s != cur]


RULES = {  # name -> (needs Gemini, function of readings r and the chosen letter c)
    "current": (False, lambda r, c: r[c]),
    "gemini alone": (True, lambda r, c: r["G"]),
    "free median P,H,S": (False, lambda r, c: _med([r["P"], r["H"], r["S"]])),
    "PRIMARY vote P,S,G": (True, lambda r, c: vote(r[c], _others(r, c, "PSG"))),
    "vote P,H,S,G": (True, lambda r, c: vote(r[c], _others(r, c, "PHSG"))),
    "median P,S,G": (True, lambda r, c: _med([r["P"], r["S"], r["G"]])),
}
PAID = ["PRIMARY vote P,S,G", "vote P,H,S,G", "median P,S,G"]


def evaluate():
    P, H = _det("pilot"), _det("hr1536")
    L, role = cy.sets()
    gem = ri._load(ri.STORE)
    seg = {json.loads(l)["path"]: json.loads(l) for l in open(os.path.join(evalkit.ROOT, "data", "raster_facts.jsonl"))}
    chosen = {k: LETTER[v] for k, v in ri.sources().items()}
    by = {v["comp_path"]: k for k, v in L.items() if v["comp_path"]}
    acc = collections.defaultdict(collections.Counter)
    flip = collections.defaultdict(collections.Counter)
    seen = set()
    for q in evalkit.load_questions():
        k = by.get(q["path"])
        if k is None or role.get(k) not in ("E1", "E2") or not q["question"].startswith("How"):
            continue
        kind = raster_backend.question_kind(q["question"])
        if kind not in cy.CID or (q["path"], kind) in seen:
            continue
        seen.add((q["path"], kind))
        r = {"P": P(k, kind), "H": H(k, kind), "S": seg[q["path"]]["answers"][kind]}
        if q["path"] in gem:
            r["G"] = gem[q["path"]]["facts"][kind]["count"]
        c = chosen[kind]
        y = q["answer"]
        for name, (paid, f) in RULES.items():
            if paid and "G" not in r:
                continue
            a = f(r, c)
            for E in (role[k], role[k] + " with G" if "G" in r else None):
                if E is None:
                    continue
                acc[(E, name)][kind] += a == y
                acc[(E, name)]["n_" + kind] += 1
                flip[(E, name)]["fixed"] += a == y and r[c] != y
                flip[(E, name)]["broke"] += a != y and r[c] == y
    sets = [("E1", "150 dev drawings (rule found here)"), ("E2", "380 official-test drawings, all"),
            ("E2 with G", "official-test drawings with a Gemini read (the confirmation test)")]
    holm = {}
    for E, desc in sets:
        if not acc[(E, "current")]:
            continue
        print("\n== %s: %s" % (E, desc))
        print("%-20s %s   %-10s  fixed/broke  p" % ("rule", " ".join("%8s" % k[:8] for k in cy.CLASSES), "total"))
        for name in RULES:
            t = acc[(E, name)]
            if not t:
                continue
            tot, n = sum(t[k] for k in cy.CLASSES), sum(t["n_" + k] for k in cy.CLASSES)
            fl = flip[(E, name)]
            p = _p2(fl["fixed"], fl["broke"])
            if E == "E2 with G" and name in PAID:
                holm[name] = (p, fl["fixed"], fl["broke"])
            print("%-20s %s   %4d/%-5d %s" % (name, " ".join("%7.1f%%" % (100 * t[k] / max(1, t["n_" + k])) for k in cy.CLASSES), tot, n,
                                             "" if name in ("current", "gemini alone") else "%4d/%-4d  %.4f" % (fl["fixed"], fl["broke"], p)))
    if holm:
        print("\nPre-registered decision (section 12): PRIMARY passes if, on E2 with G, fixed > broke and Holm-adjusted p < 0.05.")
        order = sorted(holm, key=lambda n: holm[n][0])
        prev = 0.0
        for i, name in enumerate(order):
            p, fx, br = holm[name]
            adj = max(prev, min(1.0, p * (len(order) - i)))
            prev = adj
            print("  %-20s fixed %3d broke %3d  p %.4f  Holm %.4f  -> %s" % (name, fx, br, p, adj, "PASS" if fx > br and adj < 0.05 else "no"))


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = {"read": read, "eval": evaluate}.get(a[0] if a else "")
    print(__doc__) if cmd is None else cmd(*a[1:])
