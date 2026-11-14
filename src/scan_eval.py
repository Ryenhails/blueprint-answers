"""Scanned-drawing check with real gold answers: the vector plans of the dataset also come as PNG images.

The same questions are asked on the PNG, so nothing can be read from vector data: counts come from the image readers,
areas from the measuring step (the model points, code measures). Gold answers are those of questions.jsonl.

  python scan_eval.py run [n per question form] [out.jsonl]     default 6 -> data/scan_eval.jsonl
  python scan_eval.py report [out.jsonl]
"""
import collections
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

OUT = os.path.join(evalkit.ROOT, "data", "scan_eval.jsonl")
FORMS = [("area of one room", r"net floor area of room \d"), ("total floor area", r"total net floor area in m² of all numbered"), ("area of rooms with a label", r"total net area in m² of all rooms labelled"),
         ("area of an apartment", r"net floor area of apartment"), ("rooms with a label", r"How many rooms labelled"), ("apartments", r"How many apartments"),
         ("doors", r"How many doors"), ("windows", r"How many windows"), ("toilets", r"How many toilets"), ("washbasins", r"How many washbasins"), ("bathtubs", r"How many bathtubs")]


def sample(n):
    rng = random.Random(11)
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "swiss_dwellings"]
    rng.shuffle(rows)
    out, used = [], set()
    for name, pat in FORMS:
        k = 0
        for r in rows:
            if k < n and r["path"] not in used and re.search(pat, r["question"]):
                used.add(r["path"])
                out.append(dict(r, form=name, image=r["path"][:-4] + ".png"))
                k += 1
    return out


def run(n=6, out_path=OUT):
    import demo_server
    import system

    demo_server.gemini_backends()
    b = {k: demo_server.BACKENDS[k] for k in ("llm", "agent", "table_llm", "raster", "vision")}
    rows = sample(n)

    def one(r):
        try:
            res = system.ask(r["image"], r["question"], **b)
        except Exception as e:
            res = {"answer": None, "note": "crash %r" % (e,)}
        return {"form": r["form"], "image": r["image"], "question": r["question"], "gold": r["answer"], "type": r["type"], "answer": res.get("answer"),
                "ok": bool(res.get("answer") is not None and evalkit.is_correct(res["answer"], r["answer"], r["type"])),
                "via": res.get("plan_source"), "reliability": res.get("reliability"), "cost_usd": res.get("cost_usd", 0), "seconds": res.get("seconds"),
                "steps": str(res.get("note") or (res.get("calc") or "").split("\n\nCode")[0])[:600]}

    with ThreadPoolExecutor(6) as ex, open(out_path, "w") as f:
        for o in ex.map(one, rows):
            f.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
            f.flush()
    report(out_path)


def report(out_path=OUT):
    rows = [json.loads(l) for l in open(out_path)]
    agg = collections.defaultdict(lambda: [0, 0, 0.0, []])
    for r in rows:
        for k in (r["form"], "ALL " + r["type"], "ALL"):
            a = agg[k]
            a[0] += r["ok"]
            a[1] += 1
            a[2] += r["cost_usd"] or 0
            p = evalkit.to_number(r["answer"])
            if p is not None and r["gold"]:
                a[3].append(abs(p - float(r["gold"])) / abs(float(r["gold"])))
    for k, (ok, n, usd, err) in agg.items():
        err.sort()
        print("  %-28s %3d / %3d   $%.3f per question   median error %s" % (k, ok, n, usd / n, "%.1f%%" % (100 * err[len(err) // 2]) if err else "-"))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["run"]:
        run(int(a[1]) if len(a) > 1 else 6, a[2] if len(a) > 2 else OUT)
    elif a[:1] == ["report"]:
        report(a[1] if len(a) > 1 else OUT)
    else:
        print(__doc__)
