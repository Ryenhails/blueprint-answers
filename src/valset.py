"""Validation set: questions with gold answers that were not used while the system was built.

  python valset.py make                 -> data/val1200.jsonl   (fixed seed; nothing from data/dev500.jsonl)
  python valset.py run [workers]        answer every question with the full system (resumes) -> data/val_results.jsonl
  python valset.py report               the score tables (markdown)

What is in it
  vector plans      swiss_dwellings PDFs, counts and areas, four languages
  BIM               ifc_models, counts, areas, lengths, volumes, six languages
  floor-plan images cubicasa5k drawings of the official CubiCasa5K test split (the detector never saw them), English and Finnish
  scans             the PNG twin of vector plans: same questions and gold answers, but nothing can be read from vector data
  BIM, open         IFC-Bench free-text questions; graded by a model against the reference answer
  schedules         opening_design: only 64 questions exist and the prompts were tuned on them, so they are listed apart
"""
import collections
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

VAL = os.path.join(evalkit.ROOT, "data", "val1200.jsonl")
OUT = os.path.join(evalkit.ROOT, "data", "val_results.jsonl")
SEED = 20261115
GROUPS = [("vector plans", "Vector plan PDFs"), ("bim", "BIM models (fixed forms)"), ("images", "Floor-plan images"), ("scans", "Scanned sheets (PNG twins)"),
          ("bim open", "BIM models (open questions)"), ("schedules", "Schedule sheets (tuned on)")]


def lang(q):
    w = q.split()[0].lower().strip("¿")
    return {"how": "en", "what": "en", "wie": "de", "quelle": "fr", "combien": "fr", "quel": "fr", "quanti": "it", "quante": "it", "qual": "it", "quanto": "it",
            "kuinka": "fi", "wat": "nl", "hoeveel": "nl", "hvad": "da", "hvor": "da", "kui": "et", "mitu": "et", "cuál": "es", "cuántos": "es", "cuántas": "es"}.get(w, "other")


def make():
    import raster_backend
    import raster_detector
    import raster_ingest
    import scan_eval

    rng = random.Random(SEED)
    dev = [json.loads(l) for l in open(evalkit.DEV)]
    dev_q = {(r["path"], r["question"]) for r in dev}
    dev_paths = {r["path"] for r in dev}
    tuned_scans = {json.loads(l)["image"][:-4] + ".pdf" for l in open(scan_eval.OUT)} if os.path.exists(scan_eval.OUT) else set()
    rows = evalkit.load_questions()
    by = collections.defaultdict(list)
    for r in rows:
        if (r["path"], r["question"]) not in dev_q:
            by[evalkit.source(r["path"])].append(r)
    out = []

    def take(pool, n, key, group, spread=None, **extra):
        """n questions, balanced over key(r), one per drawing while drawings last."""
        pool = pool[:]
        rng.shuffle(pool)
        cells = collections.defaultdict(list)
        for r in pool:
            cells[key(r)].append(r)
        used, got = set(), []
        names = sorted(cells)
        while len(got) < n and any(cells.values()):
            for k in names:
                c = cells[k]
                i = next((i for i, r in enumerate(c) if r["path"] not in used), 0 if c else None)
                if i is None or len(got) >= n:
                    continue
                r = c.pop(i)
                used.add(r["path"])
                got.append(dict(r, group=group, **extra))
        out.extend(got)

    sw = [r for r in by["swiss_dwellings"] if r["path"] not in dev_paths]
    take(sw, 300, lambda r: (r["type"], lang(r["question"])), "vector plans")
    take([r for r in by["ifc_models"]], 150, lambda r: (r["type"], lang(r["question"]) == "en"), "bim")
    det, gem = raster_detector.load(), raster_ingest._load(raster_ingest.STORE)
    e2 = [r for r in by["cubicasa5k"] if r["path"] in det and not det[r["path"]]["seen"] and r["path"] in gem and r["path"] not in dev_paths]
    take(e2, 300, lambda r: (raster_backend.question_kind(r["question"]), lang(r["question"])), "images")
    for name, pat in scan_eval.FORMS:  # scans: English forms, 20 each, drawings not used when the measuring step was tuned
        pool = [r for r in sw if r["path"] not in tuned_scans and re.search(pat, r["question"])]
        take(pool, 20, lambda r: 0, "scans", form=name)
    for r in out:
        if r["group"] == "scans":
            r["ask_path"] = r["path"][:-4] + ".png"
    take(by["ifc_bench"], 200, lambda r: r["path"], "bim open")
    out.extend(dict(r, group="schedules") for r in rows if evalkit.source(r["path"]) == "opening_design")
    with open(VAL, "w") as f:
        for i, r in enumerate(out):
            f.write(json.dumps(dict(r, id="v%04d" % i), ensure_ascii=False) + "\n")
    c = collections.Counter(r["group"] for r in out)
    print("wrote", VAL, len(out), dict(c), "| drawings:", len({r["path"] for r in out}))


def run(workers=6):
    import demo_server
    import ifc_agent
    import system

    demo_server.gemini_backends()
    b = {k: demo_server.BACKENDS[k] for k in ("llm", "agent", "table_llm", "raster", "vision")}
    rows = [json.loads(l) for l in open(VAL)]
    if os.environ.get("STOA_VAL_GROUP"):  # re-run one group only (its old rows must be removed from the results file first)
        rows = [r for r in rows if r["group"] == os.environ["STOA_VAL_GROUP"]]
    done = {json.loads(l)["id"] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [r for r in rows if r["id"] not in done]
    lock = threading.Lock()
    fh = open(OUT, "a")

    def one(r):
        try:
            res = system.ask(r.get("ask_path", r["path"]), r["question"], **b)
        except Exception as e:
            res = {"answer": None, "note": "crash %r" % (e,)}
        steps = str(res.get("note") or (res.get("calc") or "").split("\n\nCode")[0])[:1500]
        o = {"id": r["id"], "group": r["group"], "source": evalkit.source(r["path"]), "path": r.get("ask_path", r["path"]), "question": r["question"], "type": r["type"],
             "lang": lang(r["question"]), "form": r.get("form"), "gold": r["answer"], "answer": res.get("answer"), "unit": res.get("unit"), "steps": steps,
             "via": res.get("plan_source"), "kind": res.get("kind"), "reliability": res.get("reliability"), "cost_usd": res.get("cost_usd", 0), "seconds": res.get("seconds"),
             "ok": None if r["type"] == "text" else bool(res.get("answer") is not None and evalkit.is_correct(res["answer"], r["answer"], r["type"]))}
        with lock:
            fh.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
            fh.flush()

    big = [r for r in todo if r["group"] == "bim open"]  # one building model in memory at a time
    rest = [r for r in todo if r["group"] != "bim open"]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, rest))
    for p in sorted({r["path"] for r in big}):
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, [r for r in big if r["path"] == p]))
        print("done", p, flush=True)
    fh.close()
    grade()
    report()


def grade():
    """Text answers: a model compares each with the reference answer (no human check)."""
    import ifc_agent

    rows = [json.loads(l) for l in open(OUT)]
    todo = [r for r in rows if r["type"] == "text" and not r.get("grade")]
    if todo:
        grades, usd = ifc_agent.judge([(r["question"], r["gold"], "%s %s\n%s" % (r["answer"], r.get("unit") or "", r["steps"])) for r in todo])
        for r, g in zip(todo, grades):
            r["grade"] = g if r["answer"] is not None else "wrong"
        with open(OUT, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print("graded %d text answers, $%.3f" % (len(todo), usd))


def _cell(rs):
    num = [r for r in rs if r["type"] != "text"]
    txt = [r for r in rs if r["type"] == "text"]
    parts = []
    if num:
        parts.append("%d / %d (%.1f%%)" % (sum(r["ok"] for r in num), len(num), 100 * sum(r["ok"] for r in num) / len(num)))
    if txt:
        c = collections.Counter(r.get("grade", "?") for r in txt)
        parts.append("%d correct, %d partial, %d wrong of %d" % (c["correct"], c["partial"], c["wrong"], len(txt)))
    return "; ".join(parts), sum(r["cost_usd"] or 0 for r in rs) / max(1, len(rs)), sorted(r["seconds"] or 0 for r in rs)[len(rs) // 2]


def report():
    rows = [json.loads(l) for l in open(OUT)]
    print("%d of %d questions answered so far\n" % (len(rows), sum(1 for _ in open(VAL))))
    print("| Group | Questions | Result | Cost per question | Median time |\n|---|---|---|---|---|")
    for g, title in GROUPS:
        rs = [r for r in rows if r["group"] == g]
        if rs:
            res, usd, sec = _cell(rs)
            print("| %s | %d | %s | $%.4f | %.1f s |" % (title, len(rs), res, usd, sec))
    new = [r for r in rows if r["group"] != "schedules" and r["type"] != "text"]
    if new:
        print("| **All unseen numeric questions** | %d | **%s** | $%.4f | |" % (len(new), _cell(new)[0], _cell(new)[1]))
    for title, key in (("By question type", lambda r: (r["group"], r["type"])), ("By language", lambda r: (r["group"], r["lang"])),
                       ("Scans by question form", lambda r: (r["form"],) if r["group"] == "scans" else None),
                       ("By answer route", lambda r: (r["group"], str(r["via"])))):
        cells = collections.defaultdict(list)
        for r in rows:
            k = key(r)
            if k is not None and r["group"] != "schedules":
                cells[k].append(r)
        print("\n%s\n\n| | Questions | Result | Cost per question |\n|---|---|---|---|" % title)
        for k in sorted(cells, key=str):
            res, usd, _ = _cell(cells[k])
            print("| %s | %d | %s | $%.4f |" % (" · ".join(str(x) for x in k), len(cells[k]), res, usd))
    print("\ntotal API cost of the run: $%.2f" % sum(r["cost_usd"] or 0 for r in rows))


def summary():
    """The numbers of report() as data, for the demo page."""
    if not os.path.exists(OUT):
        return None
    rows = [json.loads(l) for l in open(OUT)]

    def cell(rs, label):
        num = [r for r in rs if r["type"] != "text"]
        txt = collections.Counter(r.get("grade", "ungraded") for r in rs if r["type"] == "text")
        return {"label": label, "n": len(rs), "numeric": len(num), "ok": sum(bool(r["ok"]) for r in num), "text": dict(txt),
                "usd": round(sum(r["cost_usd"] or 0 for r in rs) / max(1, len(rs)), 5), "sec": sorted(r["seconds"] or 0 for r in rs)[len(rs) // 2] if rs else 0}

    out = {"total": sum(1 for _ in open(VAL)), "answered": len(rows), "groups": [], "types": [], "langs": [], "scans": [],
           "usd": round(sum(r["cost_usd"] or 0 for r in rows), 2)}
    for g, title in GROUPS:
        rs = [r for r in rows if r["group"] == g]
        if rs:
            out["groups"].append(dict(cell(rs, title), key=g))
    unseen = [r for r in rows if r["group"] != "schedules"]
    out["unseen"] = len(unseen)
    out["all"] = cell([r for r in unseen if r["type"] != "text"], "All unseen numeric questions")
    for name, key, where in (("types", lambda r: r["type"], unseen), ("langs", lambda r: r["lang"], [r for r in unseen if r["type"] != "text"]),
                             ("scans", lambda r: r["form"], [r for r in rows if r["group"] == "scans"])):
        cells = collections.defaultdict(list)
        for r in where:
            cells[key(r)].append(r)
        out[name] = [cell(v, str(k)) for k, v in sorted(cells.items(), key=lambda kv: -len(kv[1]))]
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["make"]:
        make()
    elif a[:1] == ["run"]:
        run(int(a[1]) if len(a) > 1 else 6)
    elif a[:1] == ["grade"]:
        grade()
    elif a[:1] == ["report"]:
        report()
    else:
        print(__doc__)
