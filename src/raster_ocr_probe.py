"""Raster probe 1: can plain OCR of room labels answer the label-type cubicasa5k questions?

Runs PaddleOCR on N cubicasa5k images, counts Finnish room-label tokens, and
scores the English label-type questions (bedrooms, saunas, kitchens, bathrooms,
WC rooms, balconies). OCR output is cached in ONE jsonl file.

  python raster_ocr_probe.py [n_images]
"""
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402

CACHE = "/scratch/work/liuz16/tmp/stoa_raster_ocr_probe.jsonl"

# question prefix -> (kind, token regex applied to each upper-cased OCR token)
KINDS = {
    "How many bedrooms": ("bedrooms", r"^(MH|MAKUUH\w*|MAK\.?H\w*)$"),
    "How many saunas": ("saunas", r"^(S|SA|SAUNA)$"),
    "How many kitchens": ("kitchens", r"^(K|KT|KK|KEITTI\w*|KEITTO\w*|AVOK\w*|TUPAK\w*|APUK\w*)$"),
    "How many bathrooms or washrooms": ("bathrooms", r"^(KH|KPH|PH|PESUH\w*|PESU|SH|KYLPYH\w*|PE)$"),
    "How many separate WC rooms": ("wc_rooms", r"^WC$"),
    "How many balconies": ("balconies", r"^(PARVEKE|PARV|LAS\.?PARV\w*|LASITETTU|PARVI)$"),
}


def tokens(texts):
    out = []
    for t in texts:
        for w in re.split(r"[\s/+,;:()]+", t.upper()):
            w = re.sub(r"[\d.,]+(M2|M²)?$", "", w).strip(".-")  # "MH12,5" -> "MH"
            if w:
                out.append(w)
    return out


def main(n):
    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "cubicasa5k" and r["question"].startswith("How many")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["path"]].append(r)
    files = sorted(by)
    random.Random(11).shuffle(files)
    files = files[:n]
    cache = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            d = json.loads(l)
            cache[d["path"]] = d["texts"]
    todo = [f for f in files if f not in cache]
    if todo:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        with open(CACHE, "a") as out:
            for i, f in enumerate(todo):
                res = ocr.ocr(os.path.join(evalkit.DATA, f), cls=True)
                texts = [[x[1][0], round(x[1][1], 3), [[round(v) for v in p] for p in x[0]]] for x in (res[0] or [])]
                cache[f] = texts
                out.write(json.dumps({"path": f, "texts": texts}, ensure_ascii=False) + "\n")
                out.flush()
                print("ocr %d/%d %s boxes=%d" % (i + 1, len(todo), f.split("/")[-1], len(texts)), flush=True)
    agg = collections.defaultdict(lambda: [0, 0, 0])  # ocr-correct, majority-correct, n
    maj = {"bedrooms": 1, "saunas": 1, "kitchens": 1, "bathrooms": 1, "wc_rooms": 0, "balconies": 0}
    miss = collections.defaultdict(list)
    for f in files:
        toks = tokens([t[0] for t in cache[f]])
        for r in by[f]:
            for p, (kind, rx) in KINDS.items():
                if r["question"].startswith(p):
                    pred = sum(1 for t in toks if re.match(rx, t))
                    agg[kind][0] += pred == r["answer"]
                    agg[kind][1] += maj[kind] == r["answer"]
                    agg[kind][2] += 1
                    if pred != r["answer"] and len(miss[kind]) < 4:
                        miss[kind].append((f.split("/")[-1], "gold", r["answer"], "pred", pred, [t for t in toks if len(t) < 9][:25]))
    print("images", len(files), "avg text boxes %.1f" % (sum(len(cache[f]) for f in files) / len(files)))
    print("%-10s %8s %10s %4s" % ("kind", "ocr", "majority", "n"))
    for k, (a, b, c) in sorted(agg.items()):
        print("%-10s %7.0f%% %9.0f%% %4d" % (k, 100 * a / c, 100 * b / c, c))
    for k, v in miss.items():
        print(k)
        for x in v:
            print("   ", x)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
