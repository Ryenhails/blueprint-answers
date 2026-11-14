"""Learned combiner of count readings (raster_yolo_plan.md section 13).

For every (drawing, kind) it predicts the count from all readings at once, so it can learn each reader's systematic bias
(e.g. "Gemini merges multi-sash windows", "the detector double-counts doors at low confidence").
  readers   P  detector, out-of-fold (cubi_cv.py): this kind's box count at 7 confidence levels + summed confidence,
               and the five object counts at 0.4 as context
            S  segmentation network: all twelve counts
            G  Gemini full read (data/raster_gemini.jsonl): all twelve counts + this kind's box count  (variant "G" only)
            drawing style and image size
  target    offset of the true count from an anchor (P at 0.4 for objects; G, or S without Gemini, for rooms)
  model     one HistGradientBoostingClassifier per kind and variant; answer = anchor + most likely offset
Trained on the train/val drawings (P out-of-fold), scored on E1 (150 dev) and E2 (380 official test), which it never sees.

  python raster_combiner.py read [budget_usd] [workers]   Gemini full reads of train/val drawings (val first, then train
                                                         in a fixed random order) until the budget is spent
  python raster_combiner.py fit [cv|pilot] [train_roles]   train, cross-validate, score E1/E2 (pilot/E1 = dry run only)
"""
import collections
import hashlib
import json
import math
import os
import pickle
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cubi_cv as cv  # noqa: E402
import cubi_yolo as cy  # noqa: E402
import evalkit  # noqa: E402
import raster_backend  # noqa: E402
import raster_ensemble as re_  # noqa: E402
import raster_ingest as ri  # noqa: E402

Y = "/scratch/work/liuz16/stoa_raster/yolo/"
OUT = "/scratch/work/liuz16/stoa_raster/combiner/"
KINDS = list(ri.KINDS)
OBJ = cy.CLASSES
T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
STYLE = {"colorful": 0, "high_quality": 1, "high_quality_architectural": 2}
CLIP = 5


# ---------------------------------------------------------------- Gemini reads of the training pool
def read(budget=86.0, workers=8):
    budget, workers = float(budget), int(workers)
    L, role = cy.sets()
    val = sorted(v["comp_path"] for k, v in L.items() if role[k] == "val" and v["comp_path"])
    tr = sorted(v["comp_path"] for k, v in L.items() if role[k] == "train" and v["comp_path"])
    random.Random(0).shuffle(tr)
    have = ri._load(ri.STORE)
    todo = [p for p in val + tr if p not in have]
    spent, done, failed = [0.0], [0], []
    lock = threading.Lock()
    print("pool drawings to read: %d (budget $%.2f, %d workers)" % (len(todo), budget, workers), flush=True)

    def one(p):
        with lock:
            if spent[0] >= budget:
                return
        try:
            rec, new = ri.ingest(os.path.join(evalkit.DATA, p))
            with lock:
                spent[0] += rec["usd"] if new else 0
                done[0] += 1
                if done[0] % 100 == 0:
                    print("  %d read, $%.2f" % (done[0], spent[0]), flush=True)
        except Exception as e:
            with lock:
                failed.append((p, str(e)[:100]))

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print("read %d, failed %d, spent $%.2f" % (done[0], len(failed), spent[0]))
    for f in failed[:10]:
        print("  failed", f)


# ---------------------------------------------------------------- features
def load_P(source):
    """-> {key: raw boxes} from the detector each drawing is read by."""
    if source == "pilot":  # dry runs only: in-sample on train drawings, thresholds tuned on val
        return json.load(open(Y + "pilot/preds.json"))["preds"]
    avail = [f for f in range(cv.K) if os.path.exists(Y + "%s/fold%d/preds.json" % (source, f))]
    P = {}
    for f in avail:  # train/val drawings: only those of a trained fold; E1/E2: their fold's detector, else a trained one
        for k, b in json.load(open(Y + "%s/fold%d/preds.json" % (source, f)))["preds"].items():
            fk = cv.fold_of(k)
            if fk == f or (fk not in avail and avail[fk % len(avail)] == f):
                P[k] = b
    print("detector folds used:", avail)
    return P


def cfold(key):
    """Combiner cross-validation folds (by drawing; independent of the detector folds)."""
    return int(hashlib.md5((key + "/combiner").encode()).hexdigest(), 16) % 5


def rule_menu(kind, variant):
    """The simple rules a kind may pick from: detector at each threshold, the other single readers, and (with Gemini)
    median and vote of the three readings at each detector threshold."""
    m = {}
    if kind in OBJ:
        for i, t in enumerate(T):
            m["P@%.1f" % t] = lambda r, i=i: r["Pt"][i]
            if variant == "G":
                m["median(P@%.1f,S,G)" % t] = lambda r, i=i: sorted([r["Pt"][i], r["S"], r["G"]])[1]
                m["vote(P@%.1f;S,G)" % t] = lambda r, i=i: re_.vote(r["Pt"][i], [r["S"], r["G"]])
    m["S"] = lambda r: r["S"]
    if variant == "G":
        m["G"] = lambda r: r["G"]
    return m


def pick_rule(rs, kind, variant):
    menu = rule_menu(kind, variant)
    score = {n: sum(f(r) == r["y"] for r in rs) for n, f in menu.items()}
    best = max(score, key=lambda n: (score[n], -list(menu).index(n)))
    return best, menu[best]


def _det(boxes, kind):
    c = [b[1] for b in boxes if b[0] == cy.CID[kind]]
    return [sum(x >= t for x in c) for t in T] + [round(sum(c), 2)]


def rows(P, gold, L, role, seg, gem, roles):
    """-> list of dicts: one per (drawing, kind) with a gold answer and a detector reading."""
    out = []
    for k, v in L.items():
        p = v["comp_path"]
        if role.get(k) not in roles or p not in gold or k not in P or p not in seg:
            continue
        g = gem.get(p)
        ctx = [sum(1 for b in P[k] if b[0] == i and b[1] >= 0.4) for i in range(len(OBJ))]
        S = [seg[p]["answers"][kd] for kd in KINDS]
        base = [STYLE.get(k.split("/")[0], -1)] + list(v["size"]) + ctx + S
        for kind, y in gold[p].items():
            if kind == "floors":
                continue
            r = {"key": k, "path": p, "role": role[k], "kind": kind, "y": y,
                 "S": seg[p]["answers"][kind], "G": g["facts"][kind]["count"] if g else None}
            det = _det(P[k], kind) if kind in OBJ else []
            r["P"] = det[3] if det else None  # count at 0.4
            r["Pt"] = det[:len(T)]
            r["x_free"] = det + base
            r["x_G"] = det + base + [g["facts"][kd]["count"] for kd in KINDS] + [len(g["facts"][kind]["boxes"])] if g else None
            out.append(r)
    return out


def anchor(r, variant):
    """The answer the model starts from (offset 0): with Gemini, the median of the three readings for objects, which is
    the best simple rule on held-out drawings (section 12); so the model only has to learn when to depart from it."""
    if r["kind"] in OBJ:
        return sorted([r["P"], r["S"], r["G"]])[1] if variant == "G" else r["P"]
    return r["G"] if variant == "G" else r["S"]


def _fit(rs, variant):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = [r["x_" + variant] for r in rs]
    y = [max(-CLIP, min(CLIP, r["y"] - anchor(r, variant))) for r in rs]
    n = collections.Counter(y)  # offsets seen fewer than 5 times are merged into the nearest common one
    keep = sorted(c for c, k in n.items() if k >= 5) or [n.most_common(1)[0][0]]
    y = [min(keep, key=lambda c: (abs(c - v), abs(c))) for v in y]
    if len(set(y)) < 2:
        return ("const", y[0])
    m = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, max_depth=4,
                                       l2_regularization=1.0, early_stopping=False, random_state=0)
    m.fit(X, y)
    return ("hgb", m)


def _predict(model, rs, variant):
    """-> list of (answer, probability of that answer)."""
    if model[0] == "const":
        return [(anchor(r, variant) + model[1], 1.0) for r in rs]
    m = model[1]
    pr = m.predict_proba([r["x_" + variant] for r in rs])
    out = []
    for r, p in zip(rs, pr):
        i = max(range(len(p)), key=lambda j: p[j])
        out.append((anchor(r, variant) + int(m.classes_[i]), float(p[i])))
    return out


def _p2(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def fit(source="cv", train_roles="train,val"):
    train_roles = set(train_roles.split(","))
    L, role = cy.sets()
    P = load_P(source)
    seg = {json.loads(l)["path"]: json.loads(l) for l in open(os.path.join(evalkit.ROOT, "data", "raster_facts.jsonl"))}
    gem = ri._load(ri.STORE)
    gold = collections.defaultdict(dict)
    for q in evalkit.load_questions():
        if q["path"].startswith("data/cubicasa5k/") and q["question"].startswith("How"):
            kd = raster_backend.question_kind(q["question"])
            if kd and kd not in gold[q["path"]]:
                gold[q["path"]][kd] = q["answer"]
    pool = rows(P, gold, L, role, seg, gem, train_roles)
    test = rows(P, gold, L, role, seg, gem, {"E1", "E2"} - train_roles)
    print("P source: %s | pool rows %d (with Gemini %d) from %d drawings | test rows %d" % (
        source, len(pool), sum(r["G"] is not None for r in pool), len({r["key"] for r in pool}), len(test)))

    # 1) cross-validation inside the pool (5 folds by drawing): anchor, best single reader, per-kind rule pick,
    #    and the boosted model trained on 25 / 50 / 100 % of the training part (learning curve)
    FR = (0.25, 0.5, 1.0)
    print("\n== cross-validation inside the pool (exact answers; rule pick and model never see the scored fold)")
    print("%-14s %-5s %5s %7s %10s %9s %s" % ("kind", "var", "n", "anchor", "best one", "rule pick", "  ".join("model@%d%%" % (100 * f) for f in FR)))
    for variant in ("free", "G"):
        tot = collections.Counter()
        for kind in KINDS:
            rs = [r for r in pool if r["kind"] == kind and r["x_" + variant] is not None]
            if len(rs) < 100:
                continue
            ok = collections.Counter()
            for f in range(5):
                tr = [r for r in rs if cfold(r["key"]) != f]
                te = [r for r in rs if cfold(r["key"]) == f]
                _, rule = pick_rule(tr, kind, variant)
                ok["rule"] += sum(rule(r) == r["y"] for r in te)
                for fr in (FR if kind in OBJ else (1.0,)):  # learning curve for the object kinds only
                    sub = tr if fr == 1.0 else random.Random(f).sample(tr, int(len(tr) * fr))
                    ok[fr] += sum(a == r["y"] for (a, _), r in zip(_predict(_fit(sub, variant), te, variant), te))
            single = {s_: sum(r[s_] == r["y"] for r in rs) for s_ in ("P", "S", "G") if all(r[s_] is not None for r in rs)}
            best = max(single.items(), key=lambda kv: kv[1])
            anc = sum(anchor(r, variant) == r["y"] for r in rs)
            n = len(rs)
            print("%-14s %-5s %5d %6.1f%% %3s %5.1f%% %8.1f%%  %s" % (kind, variant, n, 100 * anc / n, best[0], 100 * best[1] / n, 100 * ok["rule"] / n,
                                                                  "  ".join("%8.1f%%" % (100 * ok[fr] / n) if fr in ok else "        -" for fr in FR)))
            if kind in OBJ:
                tot["n"] += n
                tot["anchor"] += anc
                tot["rule"] += ok["rule"]
                for fr in FR:
                    tot[fr] += ok[fr]
        if tot["n"]:
            print("%-14s %-5s %5d %6.1f%% %10s %8.1f%%  %s" % ("OBJECTS", variant, tot["n"], 100 * tot["anchor"] / tot["n"], "", 100 * tot["rule"] / tot["n"],
                                                            "  ".join("%8.1f%%" % (100 * tot[fr] / tot["n"]) for fr in FR)))

    # 2) final models on the whole pool
    models = {}
    for variant in ("free", "G"):
        for kind in KINDS:
            rs = [r for r in pool if r["kind"] == kind and r["x_" + variant] is not None]
            if len(rs) >= 100:
                models[(variant, kind)] = _fit(rs, variant)
    picks = {}
    print("\n== rule picked per kind on the whole pool")
    for variant in ("free", "G"):
        for kind in KINDS:
            rs = [r for r in pool if r["kind"] == kind and r["x_" + variant] is not None]
            if len(rs) >= 30:
                picks[(variant, kind)] = pick_rule(rs, kind, variant)
                print("  %-5s %-14s %s" % (variant, kind, picks[(variant, kind)][0]))
    os.makedirs(OUT, exist_ok=True)
    tag = "%s_%s" % (source, "-".join(sorted(train_roles)))
    with open(OUT + "models_%s.pkl" % tag, "wb") as f:
        pickle.dump({"models": models, "rules": {k: v[0] for k, v in picks.items()}, "features": "see raster_combiner.rows", "clip": CLIP, "T": T}, f)

    # 3) held-out drawings: combiner vs the current system and the simple rules
    Pp = json.load(open(Y + "pilot/preds.json"))["preds"]
    thr = json.load(open(Y + "pilot/preds.thr.json"))
    chosen = ri.sources()
    res = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    conf = collections.defaultdict(lambda: [0, 0])
    by_kind = collections.defaultdict(list)
    for r in test:
        by_kind[(r["kind"], r["G"] is not None)].append(r)
    for (kind, hasG), rs in by_kind.items():
        free = _predict(models[("free", kind)], rs, "free") if ("free", kind) in models else [(None, 0)] * len(rs)
        withG = _predict(models[("G", kind)], rs, "G") if hasG and ("G", kind) in models else [(None, 0)] * len(rs)
        for r, (af, _), (ag, pg) in zip(rs, free, withG):
            pil = cy._count(Pp[r["key"]], cy.CID[kind], thr[kind]) if kind in OBJ else None
            reading = {"detector": pil, "seg": r["S"], "model": r["G"]}
            cur = reading[chosen.get(kind, "model")]
            if cur is None:
                continue
            ans = {"current": cur, "combiner free": cur if af is None else af}  # kinds without a model keep the current answer
            ans["rule pick free"] = picks[("free", kind)][1](r) if ("free", kind) in picks else cur
            if hasG:
                ans["gemini alone"] = r["G"]
                ans["combiner G"] = cur if ag is None else ag
                ans["rule pick G"] = picks[("G", kind)][1](r) if ("G", kind) in picks else cur
                if kind in OBJ:
                    ans["median P,S,G"] = sorted([pil, r["S"], r["G"]])[1]
                if kind in OBJ:
                    c = {"detector": "P", "seg": "S", "model": "G"}[chosen[kind]]
                    rd = {"P": pil, "S": r["S"], "G": r["G"]}
                    ans["vote P,S,G"] = re_.vote(rd[c], [rd[s] for s in "PSG" if s != c])
                b = conf[(r["role"], min(3, int(pg * 10) // 3) if pg < 0.9 else 3)]
                b[0] += ag == r["y"]
                b[1] += 1
            grp = "objects" if kind in OBJ else "rooms"
            for E in (r["role"], r["role"] + (" with G" if hasG else " no G")):
                for name, a in ans.items():
                    t = res[(E, grp)][name]
                    t["n"] += 1
                    t["ok"] += a == r["y"]
                    t["k_" + kind] += a == r["y"]
                    t["n_" + kind] += 1
                    t["fixed"] += a == r["y"] and cur != r["y"]
                    t["broke"] += a != r["y"] and cur == r["y"]
    for E in ("E1 with G", "E2 with G", "E1", "E2"):
        for grp, kinds in (("objects", OBJ), ("rooms", [k for k in KINDS if k not in OBJ and k != "floors"])):
            R = res.get((E, grp))
            if not R:
                continue
            print("\n== %s, %s" % (E, grp))
            print("%-14s %s  %-11s fixed/broke   p" % ("rule", " ".join("%8s" % k[:8] for k in kinds), "total"))
            for name in ("current", "gemini alone", "vote P,S,G", "median P,S,G", "rule pick free", "rule pick G", "combiner free", "combiner G"):
                t = R.get(name)
                if not t or not t["n"]:
                    continue
                print("%-14s %s  %4d/%-5d  %s" % (name, " ".join("%7.1f%%" % (100 * t["k_" + k] / t["n_" + k]) if t["n_" + k] else "       -" for k in kinds),
                                                 t["ok"], t["n"], "" if name == "current" else "%4d/%-4d %.4f" % (t["fixed"], t["broke"], _p2(t["fixed"], t["broke"]))))
    print("\n== combiner G: accuracy by its own probability for the answer (reliability signal)")
    for E in ("E1", "E2"):
        print("  %s  " % E + "  ".join("p%s %d/%d" % (lab, *conf[(E, i)]) for i, lab in enumerate(["<.3", ".3-.6", ".6-.9", ">=.9"]) if conf[(E, i)][1]))
    print("\nmodels ->", OUT + "models_%s.pkl" % tag)


if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = {"read": read, "fit": fit}.get(a[0] if a else "")
    print(__doc__) if cmd is None else cmd(*a[1:])
