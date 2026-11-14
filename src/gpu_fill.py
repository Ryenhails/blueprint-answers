"""Run the model-backed routes once on a GPU node and store the results in data/answer_cache.json,
so the demo (and the dev-set score) can show them without a GPU.

  python gpu_fill.py <plan_model_id> <table_model_id>

Fills:
  - every opening_design schedule question (table route: the model writes code, the code runs on the tables)
  - a few re-phrased / other-language questions on the demo's swiss drawings (plan route: the model
    translates the question into a query, the query runs on the geometry)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_server  # noqa: E402
import evalkit  # noqa: E402
import llm_plan  # noqa: E402
import qa  # noqa: E402
import table_adapter  # noqa: E402

PARAPHRASES = [  # (question, dataset question it should agree with: startswith match on the English form)
    ("How many WCs do I need to order for this floor?", "How many toilets"),
    ("Montako ovea tässä kerroksessa on?", "How many doors"),
    ("¿Cuántas ventanas hay en esta planta?", "How many windows"),
    ("Hoeveel appartementen telt deze verdieping?", "How many apartments"),
    ("Wie viele Badewannen muss ich für dieses Geschoss bestellen?", "How many bathtubs"),
    ("What is the total indoor floor area of this storey?", "What is the total net floor area in m² of all numbered indoor rooms"),
]


def main(plan_model, table_model):
    cache = json.load(open(demo_server.ANSWER_CACHE)) if os.path.exists(demo_server.ANSWER_CACHE) else {}
    rows = evalkit.load_questions()

    # --- plan route on the demo's swiss drawings
    m = llm_plan.Model(plan_model)
    for kind, _, path in demo_server.EXAMPLES:
        if kind != "swiss":
            continue
        facts = qa.facts_for(os.path.join(evalkit.DATA, path))
        gold = {r["question"]: r for r in rows if r["path"] == path}
        for q, twin in PARAPHRASES:
            t0 = time.time()
            before = m.usd
            texts, n_in, n_out = m.generate([llm_plan.messages(facts, q)])
            plan, slots = llm_plan.parse_plan(texts[0])
            g = next((r for qq, r in gold.items() if qq.startswith(twin)), None)
            res = {"answer": None, "kind": "swiss", "evidence": [], "plan": plan, "slots": slots, "plan_source": "llm",
                   "llm_tokens": n_in + n_out, "cost_usd": round(m.usd - before, 6), "model": plan_model}
            if plan and plan.get("op") != "unsupported":
                res.update(qa.execute(plan, slots, facts))
            elif plan:
                res["note"] = "the model judged this question to be outside what the drawing data can answer"
            res["seconds"] = round(time.time() - t0, 2)
            if g:
                res["gold"], res["type"], res["unit"] = g["answer"], g["type"], res.get("unit") or g["unit"]
            for e in res["evidence"]:
                e["bbox"] = [float(v) for v in e["bbox"]]
            cache[path + "\n" + q] = res
            print("plan | %s | %s -> %s (gold %s)" % (os.path.basename(path)[:24], q[:50], res["answer"], g["answer"] if g else "?"), flush=True)
    plan_usd = m.usd
    import agent
    import vision
    ag = agent.make_agent()
    path = next(p for k, _, p in demo_server.EXAMPLES if k == "swiss")
    for q in agent.TESTS:
        res = ag(os.path.join(evalkit.DATA, path), q)
        res.update({"kind": "swiss", "model": agent.DEFAULT_MODEL})
        for e in res["evidence"]:
            e["bbox"] = [float(v) for v in e["bbox"]] if e.get("bbox") else None
        cache[path + "\n" + q] = res
        print("code | %s -> %s" % (q[:60], str(res["answer"])[:70]), flush=True)
    if m.api is None:
        del m
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

    # --- table route on every opening_design question
    m = llm_plan.Model(table_model)
    todo = [r for r in rows if evalkit.source(r["path"]) == "opening_design"]
    facts = {}
    ok = n = 0
    for r in todo:
        if r["path"] not in facts:
            facts[r["path"]] = table_adapter.parse(os.path.join(evalkit.DATA, r["path"]))
        f = facts[r["path"]]
        t0 = time.time()
        before = m.usd
        ans, used, err, code, tk = table_adapter.answer_with(m, f, r["question"])
        n_in, n_out = tk, 0
        code = code.strip()
        res = {"answer": ans, "unit": r["unit"], "kind": "table", "evidence": table_adapter.evidence(f, used),
               "calc": "The model wrote this code and it was run on the extracted tables:\n" + code[:900],
               "plan": {"op": "generated code"}, "plan_source": "code", "llm_tokens": n_in + n_out, "cost_usd": round(m.usd - before, 6),
               "seconds": round(time.time() - t0, 2), "gold": r["answer"], "type": r["type"], "model": table_model}
        if err:
            res["note"] = err
        if ans is None or (isinstance(ans, str) and ans.strip().lower() in ("none", "", "not found in tables")):
            v = vision.answer(os.path.join(evalkit.DATA, r["path"]), r["question"])  # the tables do not hold it: read the sheet
            v.update({"kind": "table", "gold": r["answer"], "type": r["type"], "unit": v.get("unit") or r["unit"],
                      "llm_tokens": v["llm_tokens"] + tk, "cost_usd": round(v["cost_usd"] + res["cost_usd"], 6)})
            res, ans = v, v["answer"]
        cache[r["path"] + "\n" + r["question"]] = res
        if r["type"] != "text":
            n += 1
            good = bool(ans is not None and evalkit.is_correct(ans, r["answer"], r["type"]))
            ok += good
            if not good:
                print("MISS gold=%s got=%s %s | %s\n%s\n" % (r["answer"], ans, err or "", r["question"].split("drawings")[-1][:110], code[:700]), flush=True)
        else:
            print("TEXT gold=%r got=%r | %s" % (str(r["answer"])[:70], str(ans)[:70], r["question"].split("drawings")[-1][:80]), flush=True)
    print("table numeric: %d / %d | API cost: plan route $%.4f, table route $%.4f" % (ok, n, plan_usd, m.usd))
    with open(demo_server.ANSWER_CACHE, "w") as fh:
        json.dump(cache, fh, ensure_ascii=False, default=str)
    print("wrote", demo_server.ANSWER_CACHE, len(cache), "answers")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
