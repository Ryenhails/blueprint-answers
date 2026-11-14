"""Collects what the film shows that is not in the exported demo page: fact-table rows of the example plan,
one wrong answer of the image-only baseline, the fact table read from a scan, the size of the validation sample.

  python video/prepare_data.py <out.json> [scan image]     (run in the environment that runs the system)
"""
import base64
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import agent  # noqa: E402
import demo_server  # noqa: E402
import evalkit  # noqa: E402
import qa  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def jpg(path_or_im, side):
    im = Image.open(path_or_im) if isinstance(path_or_im, str) else path_or_im
    im = im.convert("RGB")
    im.thumbnail((side, side), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def main(out, scan=None):
    D = {}
    plan = demo_server.EXAMPLES[0][2]  # the vector plan the film follows
    t, geo = agent.view(qa.facts_for(os.path.join(evalkit.DATA, plan)))
    D["plan"] = {"path": plan, "rooms": [{k: r[k] for k in ("id", "number", "label", "area_m2", "kind", "apartment", "perimeter_m", "toilets", "washbasins", "bathtubs", "showers")}
                                         | {"doors": len(r["doors"]), "windows": len(r["windows"]), "door_ids": r["doors"], "poly": geo[r["id"]]["poly"], "bbox": geo[r["id"]]["bbox"]} for r in t["rooms"]],
                 "doors": {d["id"]: geo[d["id"]]["bbox"] for d in t["doors"]}, "windows": {w["id"]: geo[w["id"]]["bbox"] for w in t["windows"]},
                 "fixtures": {f["id"]: geo[f["id"]]["bbox"] for f in t["fixtures"]}, "counts": {k: len(t[k]) for k in ("rooms", "doors", "windows", "fixtures")}}
    # one wrong answer of the image-only baseline (trial01_opus_image_only.md, question q34)
    p = "data/cubicasa5k/high_quality_architectural-13185.png"
    D["baseline"] = {"image": jpg(os.path.join(evalkit.DATA, p), 1400), "question": "How many windows are shown on this floor plan?", "gold": 10, "model": 6,
                     "score": [36, 50], "name": "Opus 5.5, image + question only"}
    # thumbnails of the five kinds of drawings
    thumbs = [("Vector plan PDF", plan), ("BIM model (IFC)", demo_server.EXAMPLES[3][2]), ("Schedule sheet", demo_server.EXAMPLES[7][2]), ("Floor-plan image", demo_server.EXAMPLES[5][2])]
    D["kinds"] = []
    for name, pth in thumbs:
        d = demo_server.drawing(pth, width=900)
        if d["mode"] == "image":
            D["kinds"].append({"name": name, "image": d["image"]})
        else:  # the BIM plan is an SVG in the demo: use its rendered PNG twin, or the PDF
            import fitz

            pg = fitz.open(os.path.join(evalkit.DATA, pth))[0]
            pix = pg.get_pixmap(matrix=fitz.Matrix(900 / pg.rect.width, 900 / pg.rect.width))
            D["kinds"].append({"name": name, "image": jpg(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), 900)})
    if scan and os.path.exists(scan):
        import scan_facts

        rec, _ = scan_facts.ensure(scan)
        D["kinds"].append({"name": "Scanned sheet", "image": jpg(scan, 900)})
        if "tables" in rec:
            D["scan"] = {"image": jpg(scan, 5200), "size": rec["size"], "title": rec.get("title"), "how": rec.get("how"), "usd": rec["usd"],
                         "rooms": [{"label": r["label"] or r["type"], "area_m2": r["area_m2"], "perimeter_m": r["perimeter_m"], "doors": len(r["doors"]), "windows": len(r["windows"]),
                                    "poly": rec["geo"][r["id"]].get("poly"), "bbox": rec["geo"][r["id"]]["bbox"]} for r in rec["tables"]["rooms"]],
                         "objects": [{"kind": g["kind"], "bbox": g["bbox"]} for k, g in rec["geo"].items() if g["kind"] != "room"],
                         "counts": {k: len(rec["tables"][k]) for k in ("rooms", "doors", "windows", "fixtures")}}
    D["sample"] = {"pool": sum(1 for _ in open(os.path.join(evalkit.DATA, "questions.jsonl"))), "drawings": len({json.loads(l)["path"] for l in open(os.path.join(evalkit.ROOT, "data", "val1200.jsonl"))})}
    json.dump(D, open(out, "w"))
    print("wrote", out, "%.1f MB" % (os.path.getsize(out) / 1e6), {k: (len(v) if hasattr(v, "__len__") else v) for k, v in D.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
