"""Raster backend for system.ask(): answers cubicasa5k counting questions from precomputed facts.

    answer(full_path, question) -> {
        "answer": int | None, "unit": "count",
        "evidence": [{"kind", "name", "bbox": [x0, y0, x1, y1]}],   # image pixels
        "calc": str, "plan": dict, "plan_source": str, "llm_tokens": 0,
        "reliability": "ok" | "low" | "unknown",
    }

No model runs here. Facts come from data/raster_facts.jsonl (built by raster_build.py from
the CubiCasa5K segmentation model). Questions are mapped to one of 12 kinds by keyword, in
English or Finnish. Images without precomputed facts, or questions outside the 12 kinds,
return answer None so the caller can fall back to something else.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(ROOT, "data", "raster_facts.jsonl")
META = os.path.join(ROOT, "data", "raster_facts_meta.json")

# order matters: first match wins
KEYWORDS = [
    ("toilets", r"toilet bowl|wc seat|wc-istuin"),
    ("bathrooms", r"bathrooms? or washrooms?|kylpy- tai pesuhuone"),
    ("wc_rooms", r"separate wc|wc rooms?|erillist\S* wc"),
    ("sinks", r"\bsinks?\b|wash ?basins?|tiskiallas|pesuallas|pesu- ja tiskiallas"),
    ("showers_baths", r"showers?|bathtubs?|suihku|kylpyamme"),
    ("saunas", r"sauna"),
    ("kitchens", r"kitchen|keitti"),
    ("bedrooms", r"bedroom|makuuhuone"),
    ("balconies", r"balcon|parveke|parvekke"),
    ("floors", r"floors? \(storeys?\)|storeys?|how many floors|kerros"),
    ("windows", r"window|ikkuna"),
    ("doors", r"\bdoors?\b|\bove[at]\b|\bovia\b"),
]
NAME = {
    "windows": "window", "doors": "door", "toilets": "toilet", "sinks": "sink", "showers_baths": "bathtub",
    "saunas": "sauna bench", "kitchens": "kitchen", "bedrooms": "bedroom", "bathrooms": "bath-type room",
    "wc_rooms": "bath-type room", "balconies": "outdoor area", "floors": "separate plan on the sheet",
}
LOW = 0.70  # kinds whose held-out accuracy is below this are flagged "low"

_facts = None
_meta = None


def _load():
    global _facts, _meta
    if _facts is None:
        _facts = {}
        if os.path.exists(FACTS):
            with open(FACTS) as f:
                for line in f:
                    d = json.loads(line)
                    _facts[d["path"]] = d
        _meta = json.load(open(META)) if os.path.exists(META) else {"method": {}, "accuracy": {}, "majority": {}}
    return _facts, _meta


def question_kind(question):
    q = question.lower()
    for kind, rx in KEYWORDS:
        if re.search(rx, q):
            return kind
    return None


def _key(full_path):
    p = full_path.replace("\\", "/")
    i = p.find("data/cubicasa5k/")
    return p[i:] if i >= 0 else None


def answer(full_path, question):
    facts, meta = _load()
    kind = question_kind(question)
    res = {"answer": None, "unit": "count", "evidence": [], "calc": "", "plan": {"kind": kind},
           "plan_source": "keyword", "llm_tokens": 0, "reliability": "unknown"}
    rec = facts.get(_key(full_path) or "")
    if kind is None:
        res["calc"] = "Question is not one of the 12 supported floor-plan counting kinds."
        return res
    if rec is None:
        res["calc"] = "No precomputed raster facts for this image."
        return res
    n = rec["answers"][kind]
    boxes = rec["evidence"].get(kind, [])
    acc = meta["accuracy"].get(kind)
    res["answer"] = n
    res["evidence"] = [{"kind": kind, "name": NAME[kind], "bbox": b} for b in boxes]
    res["reliability"] = "unknown" if acc is None else ("ok" if acc >= LOW else "low")
    how = meta["method"].get(kind, "segmentation model")
    res["calc"] = "%d by %s; the segmentation model marked %d %s region(s), shown as boxes%s." % (
        n, how, len(boxes), NAME[kind],
        "" if acc is None else " (held-out accuracy for this question kind: %.0f%%)" % (100 * acc))
    res["plan"] = {"kind": kind, "n_boxes": len(boxes), "held_out_accuracy": acc, "majority_baseline": meta["majority"].get(kind)}
    return res


if __name__ == "__main__":
    import sys
    print(json.dumps(answer(sys.argv[1], sys.argv[2]), indent=1, ensure_ascii=False))
