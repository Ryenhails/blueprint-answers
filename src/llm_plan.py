"""LLM backend for question -> query plan (used only on plan-cache misses).

  python llm_plan.py test <model_id> [<model_id> ...]     run on a GPU node (see run_llm_plan.sh)

The model sees the question plus the room labels / apartment names found on that drawing,
and must answer with one JSON object. The plan is then executed by qa.execute, so the
test measures end-to-end answer accuracy on question forms the plan cache has never seen.
"""
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import qa  # noqa: E402

SYSTEM = """You translate a question about a residential floor plan into a JSON query. The question can be in any language.

Reply with exactly one JSON object and nothing else:
{"op": ..., "entity": ..., "by": ..., "label": ..., "room": ..., "apartment": ...}

op: "count" (how many), "area" (floor area in m2), "complex", or "unsupported".
"complex" = the question is about rooms, apartments, doors, windows or sanitary fixtures, but one count or one area sum of this schema cannot answer it: it links things (bathrooms with a window, doors of the bedrooms), compares or ranks (which apartment is largest), asks for a list, a ratio, a length or perimeter, a condition, or further arithmetic (waste factor, price per m2). A stronger model will handle it.
These are NOT complex, they are one area sum: the area of one apartment, with or without the words "excluding balconies / loggias / terraces" (by "apartment"); the indoor or net area of the whole floor (by "floor"); the total area of all rooms with one label (by "label"). Counting one kind of thing on the whole floor is never complex, however the question is worded (for example "how many showers do I need to plan for").
"unsupported" = the answer is not in a floor plan's rooms, openings and fixtures at all (heights, materials, costs, people, dates).
entity: "room", "apartment", "door", "window", "toilet", "washbasin", "bathtub", "shower".
by: only for rooms. "label" = rooms with a given label; "number" = one room by its number; "apartment" = all rooms of one apartment; "floor" = all indoor rooms of the whole floor. Otherwise null.
label: must be copied exactly from the list of room labels on the drawing (pick the label that means what the question asks for), else null.
room: the room number such as "3.14" when by is "number", else null.
apartment: the apartment letter such as "B" when by is "apartment", else null.

Counting apartments is {"op":"count","entity":"apartment"}. Area questions always use entity "room".
Fixtures are counted as symbols, never as rooms: shower -> "shower", bathtub -> "bathtub", toilet / WC -> "toilet", washbasin / sink -> "washbasin". For these, by and label are null.
If the question asks for an area, a surface, square metres or how much flooring, op is "area" even when it names a room label."""

SHOTS = [
    ("Room labels on the drawing: Küche, Bad, Zimmer, Wohnen/Essen, Korridor\nApartments: A, B\nQuestion: How many kitchens does this floor have?",
     {"op": "count", "entity": "room", "by": "label", "label": "Küche", "room": None, "apartment": None}),
    ("Room labels on the drawing: Cuisine, Bain, Pièce, Séjour/repas, Couloir\nApartments: A, B, C\nQuestion: Quelle est la superficie de l'appartement C ?",
     {"op": "area", "entity": "room", "by": "apartment", "label": None, "room": None, "apartment": "C"}),
    ("Room labels on the drawing: Cucina, Bagno, Camera, Soggiorno, Corridoio\nApartments: A\nQuestion: Quante finestre ci sono?",
     {"op": "count", "entity": "window", "by": None, "label": None, "room": None, "apartment": None}),
    ("Room labels on the drawing: Küche, Bad, Zimmer\nApartments: A, B\nQuestion: Wie hoch ist die Decke in Raum 2.03?",
     {"op": "unsupported", "entity": None, "by": None, "label": None, "room": None, "apartment": None}),
    ("Room labels on the drawing: Küche, Bad, Zimmer, Balkon\nApartments: A, B\nQuestion: Which apartment has the larger balcony, and by how much?",
     {"op": "complex", "entity": None, "by": None, "label": None, "room": None, "apartment": None}),
    ("Room labels on the drawing: Küche, Bad, Zimmer, Balkon\nApartments: A, B\nQuestion: What is the net floor area of apartment B, summing all its rooms except balconies and loggias?",
     {"op": "area", "entity": "room", "by": "apartment", "label": None, "room": None, "apartment": "B"}),
    ("Room labels on the drawing: Cuisine, Bain, Pièce, Séjour/repas\nApartments: A, B\nQuestion: Combien de pièces de l'appartement A ont au moins deux fenêtres ?",
     {"op": "complex", "entity": None, "by": None, "label": None, "room": None, "apartment": None}),
    ("Room labels on the drawing: Küche, Bad, Dusche, Zimmer, Korridor\nApartments: A, B\nQuestion: Wie viele Duschen sind eingezeichnet?",
     {"op": "count", "entity": "shower", "by": None, "label": None, "room": None, "apartment": None}),
    ("Room labels on the drawing: Cuisine, Bain, Pièce, Séjour/repas\nApartments: A, B\nQuestion: Surface totale des pièces désignées 'Pièce' ?",
     {"op": "area", "entity": "room", "by": "label", "label": "Pièce", "room": None, "apartment": None}),
    ("Room labels on the drawing: Cucina, Bagno, Camera, Soggiorno\nApartments: A, B\nQuestion: Quante vasche da bagno ci sono?",
     {"op": "count", "entity": "bathtub", "by": None, "label": None, "room": None, "apartment": None}),
    ("Room labels on the drawing: Küche, Bad, Zimmer, Wohnen/Essen\nApartments: A, B\nQuestion: Size of room 2.07 in square metres?",
     {"op": "area", "entity": "room", "by": "number", "label": None, "room": "2.07", "apartment": None}),
]


def context(facts, question):
    labels = sorted({r["label"] for r in facts["rooms"] if r["label"]})
    apts = sorted({u["name"].split()[-1] for u in facts["unit_names"] if u["name"]})
    return "Room labels on the drawing: %s\nApartments: %s\nQuestion: %s" % (", ".join(labels), ", ".join(apts) or "-", question)


def messages(facts, question):
    m = [{"role": "system", "content": SYSTEM}]
    for u, a in SHOTS:
        m += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    return m + [{"role": "user", "content": context(facts, question)}]


def parse_plan(text):
    """-> (plan, slots) in qa.execute's format, or (None, None)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None, None
    op, ent, by = j.get("op"), j.get("entity"), j.get("by")
    if op in ("unsupported", "complex"):
        return {"op": op}, {}
    ents = ("room", "apartment", "door", "window", "toilet", "washbasin", "bathtub", "shower")
    if op not in ("count", "area") or ent not in ents:
        return None, None
    if op == "area":
        ent = "room"
        if by not in ("label", "number", "apartment", "floor"):
            return None, None
    plan = {"op": op, "entity": ent}
    if ent == "room":
        if by not in ("label", "number", "apartment", "floor"):
            return None, None
        if op == "count" and by != "label":
            return None, None
        plan["by"] = by
    slots = {}
    if j.get("label"):
        slots["Q"] = str(j["label"])
    if j.get("room"):
        slots["R"] = str(j["room"])
    if j.get("apartment"):
        slots["L"] = str(j["apartment"]).split()[-1]
    return plan, slots


class Model:
    usd = 0.0  # running API spend; stays 0 for local models

    def __init__(self, model_id):
        self.api = None
        if model_id.lower().startswith("gemini"):  # Google API instead of a local model: no GPU, no torch
            import gemini_backend

            self.api = gemini_backend.Gemini(model_id)
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        # resolve to the snapshot folder in the cluster's shared cache (a personal HF cache setting would hide it)
        import glob
        snaps = sorted(glob.glob("/scratch/shareddata/dldata/huggingface-hub-cache/hub/models--%s/snapshots/*/" % model_id.replace("/", "--")))
        if snaps:
            model_id = snaps[-1]
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.eval()
        self.qwen3 = "qwen3" in model_id.lower().replace("--", "/")

    def generate(self, batch_msgs, max_new_tokens=80):
        if self.api is not None:
            out = self.api.generate(batch_msgs, max_new_tokens)
            self.usd = self.api.usd
            return out
        kw = {"enable_thinking": False} if self.qwen3 else {}
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw) for m in batch_msgs]
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
        n_in = int(enc["attention_mask"].sum())
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gen, skip_special_tokens=True)
        n_out = int((gen != self.tok.pad_token_id).sum())
        return texts, n_in, n_out


def make_llm(model_id):
    """-> callable(facts, question) -> (plan, slots, tokens) for qa.ask(..., llm=...)"""
    model = Model(model_id)

    def f(facts, question):
        before = model.usd
        texts, n_in, n_out = model.generate([messages(facts, question)])
        f.last_usd = model.usd - before
        plan, slots = parse_plan(texts[0])
        return plan, slots or {}, n_in + n_out

    f.last_usd = 0.0
    return f


# ---- test set: question forms that are NOT in the plan cache ---------------------------------
# {label} {room} {apt} are filled from real questions, whose gold answers are reused.
PARA = {
    "count:door": ["Count the doors on this plan.", "Montako ovea tässä pohjapiirroksessa on?", "¿Cuántas puertas hay en este plano?", "Hoeveel deuren zijn er op deze plattegrond?", "Wie viele Türen gibt es insgesamt?"],
    "count:window": ["Number of windows?", "Kuinka monta ikkunaa kerroksessa on?", "Hvor mange vinduer er der på denne plantegning?", "Combien y a-t-il de fenêtres en tout ?", "Mitu akent on sellel plaanil?"],
    "count:toilet": ["How many WCs do I need to order for this floor?", "Montako WC-istuinta tarvitaan tähän kerrokseen?", "¿Cuántos inodoros hay en esta planta?", "Quanti water ci sono in totale?"],
    "count:washbasin": ["How many sinks for washing hands are there?", "Kuinka monta pesuallasta piirustuksessa on?", "Wie viele Waschbecken brauche ich für dieses Geschoss?", "Hoeveel wastafels staan er op de tekening?"],
    "count:bathtub": ["Total number of bath tubs on the floor?", "Montako kylpyammetta kerroksessa on?", "¿Cuántas bañeras hay?", "Wie viele Wannen muss ich bestellen?"],
    "count:shower": ["How many shower units are on this level?", "Kuinka monta suihkua piirustuksessa on?", "Combien de douches faut-il prévoir pour cet étage ?", "Hvor mange brusere er der?"],
    "count:apartment": ["How many flats are on this floor?", "Montako asuntoa tässä kerroksessa on?", "¿Cuántas viviendas hay en esta planta?", "Hoeveel appartementen telt deze verdieping?"],
    "count:label": ["Count the rooms that carry the label {label}.", "Montako huonetta on merkitty nimellä {label}?", "¿Cuántas habitaciones llevan la etiqueta {label}?", "Wie oft kommt die Raumbezeichnung {label} in diesem Geschoss vor?"],
    "area:number": ["How big is room {room}?", "Mikä on huoneen {room} pinta-ala neliömetreinä?", "¿Cuál es la superficie de la habitación {room}?", "Fläche von Raum {room}?", "Wat is de oppervlakte van ruimte {room} in m²?"],
    "area:label": ["Add up the floor area of every room labelled {label}.", "Mikä on kaikkien {label}-nimisten huoneiden yhteispinta-ala?", "Somme des surfaces des pièces nommées {label} ?", "Wie viel Bodenbelag in m² brauche ich für alle Räume mit der Bezeichnung {label}?"],
    "area:apartment": ["How many square metres does apartment {apt} have, not counting balconies?", "Mikä on asunnon {apt} huoneistoala ilman parvekkeita?", "¿Cuántos metros cuadrados tiene el apartamento {apt} sin balcones?", "Wohnfläche der Wohnung {apt}?"],
    "area:floor": ["What is the total indoor floor area of this storey?", "Mikä on koko kerroksen sisätilojen yhteenlaskettu pinta-ala?", "¿Cuál es la superficie interior total de esta planta?", "Wie viele m² Innenfläche hat das gesamte Geschoss?"],
}
UNSUPPORTED = ["What is the ceiling height of the living room?", "Welches Material haben die Wände?", "Kuka on suunnitellut tämän rakennuksen?", "Quel est le prix de cet appartement ?"]


def build_cases(per_template=2, seed=3):
    rng = random.Random(seed)
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "swiss_dwellings"]
    by_kind, by_tpl = {}, {}
    for r in rows:
        t, slots = qa.normalise(r["question"])
        plan = qa.SEED.get(t)
        by_tpl.setdefault(t, []).append(r)
        if plan:
            by_kind.setdefault(plan["op"] + ":" + plan.get("by", plan["entity"]), []).append((r, slots))
    cases = []
    # A. every real template (all 4 languages) with the cache switched off
    for t, rs in sorted(by_tpl.items()):
        for r in rng.sample(rs, min(3, len(rs))):
            cases.append({"set": "A_dataset_templates", "path": r["path"], "question": r["question"], "gold": r["answer"], "type": r["type"]})
    # B. paraphrases and languages the dataset never uses for these drawings
    for kind, paras in PARA.items():
        for p in paras:
            for r, slots in rng.sample(by_kind[kind], per_template):
                q = p.format(label="'%s'" % slots.get("Q", ""), room=slots.get("R", ""), apt=slots.get("L", ""))
                cases.append({"set": "B_paraphrases", "path": r["path"], "question": q, "gold": r["answer"], "type": r["type"], "kind": kind})
    # C. out of scope -> must say unsupported
    for q in UNSUPPORTED:
        r, _ = rng.choice(by_kind["count:door"])
        cases.append({"set": "C_unsupported", "path": r["path"], "question": q, "gold": None, "type": "unsupported"})
    return cases


def test(model_ids, deadline_s=1500):
    t_start = time.time()
    cases = build_cases()
    import agent
    cases += [{"set": "D_complex", "path": cases[i * 7]["path"], "question": q, "gold": None, "type": "complex"} for i, q in enumerate(agent.TESTS) if "ceiling" not in q]
    facts = {c["path"]: qa.facts_for(evalkit.DATA + "/" + c["path"]) for c in cases}
    print("cases:", len(cases), "| drawings:", len(facts), "| setup %.0fs" % (time.time() - t_start), flush=True)
    for mid in model_ids:
        if time.time() - t_start > deadline_s - 420:
            print("SKIPPED %s: not enough time left in the debug slot" % mid)
            continue
        t0 = time.time()
        model = Model(mid)
        t_load = time.time() - t0
        agg, fails, n_in, n_out, bad_json = {}, [], 0, 0, 0
        t0 = time.time()
        B = 16
        for i in range(0, len(cases), B):
            chunk = cases[i:i + B]
            texts, a, b = model.generate([messages(facts[c["path"]], c["question"]) for c in chunk])
            n_in += a
            n_out += b
            for c, txt in zip(chunk, texts):
                plan, slots = parse_plan(txt)
                if plan is None:
                    bad_json += 1
                    ok = False
                elif c["type"] in ("unsupported", "complex"):
                    ok = plan["op"] == c["type"]
                elif plan["op"] in ("unsupported", "complex"):
                    ok = False
                else:
                    try:
                        ans = qa.execute(plan, slots, facts[c["path"]])["answer"]
                    except Exception:
                        ans = None
                    ok = ans is not None and evalkit.is_correct(ans, c["gold"], c["type"])
                k = agg.setdefault(c["set"], [0, 0])
                k[0] += ok
                k[1] += 1
                if not ok and len(fails) < 25:
                    fails.append((c["set"], c["question"][:90], txt.strip().replace("\n", " ")[:150], c["gold"]))
        dt = time.time() - t0
        print("\n=== %s | load %.0fs | %d cases in %.0fs (%.2fs/question) | tokens in %d out %d | unparseable %d" % (mid, t_load, len(cases), dt, dt / len(cases), n_in, n_out, bad_json))
        for k, (a, b) in sorted(agg.items()):
            print("  %-22s %4d / %4d  %.1f%%" % (k, a, b, 100 * a / b))
        for f in fails:
            print("   FAIL", f)
        sys.stdout.flush()
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "test":
        test(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "cases":
        cs = build_cases()
        import collections
        print(collections.Counter(c["set"] for c in cs))
        for c in cs[::37]:
            print(c)
    else:
        print(__doc__)
