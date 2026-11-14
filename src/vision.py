"""Last-resort route: a vision model reads the sheet itself.

Used when no parser covers the drawing (scans, photos, unknown drawing styles) or when the parsed facts cannot answer
(for example "how many toilets are drawn in enlarged plan 6A" on a sheet whose tables say nothing about it).
The answer is the model's reading, not a computation, so every result is marked reliability "model reading",
and the model must point at what it counted: the boxes it returns are drawn on the sheet.

  python vision.py ask <file relative to the dataset, or absolute> "<question>" [model]
"""
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402
import gemini_backend as gb  # noqa: E402

DEFAULT_MODEL = os.environ.get("STOA_VISION_MODEL", "gemini-4-hydrogen")
LONG_SIDE = 3072  # pixels sent to the model

PROMPT = """You are reading a construction drawing to answer a quantity or lookup question. Work like a careful estimator:
find the part of the sheet the question is about, read labels, schedules, legends and dimensions, and count symbols one by one.

Question: %s

Answer with JSON only:
{"answer": <number or short string; null if the sheet does not show it>,
 "unit": "<count, m2, m, SF, ... or empty>",
 "steps": "<two or three plain sentences in the language of the question: where you looked and how you got the answer>",
 "evidence": [{"label": "<what this is>", "box_2d": [ymin, xmin, ymax, xmax]}],
 "confidence": "<high, medium or low>"}
box_2d uses 0-1000 coordinates relative to the image. Give one box per counted item (at most 40), or one box for the
table row, note or dimension you read. Do not guess: if the drawing is unreadable at this resolution, say so and answer null.

Sizes: use areas and lengths that are printed on the sheet (room tags, schedules, dimension strings) and show the arithmetic.
When the question needs an area or a length that is NOT printed, never estimate it by eye. Reply instead with this JSON only,
and code will measure it on the image and give you the numbers for your final answer:
{"measure": {"scale": "<scale of that view exactly as printed, e.g. 1:200 or 1/4\" = 1'-0\">",
  "paper": "<sheet size if printed in the title block, e.g. A2, ISO A1, ANSI D; else null>",
  "scale_bar": {"from": [y, x], "to": [y, x], "length": "<real length between these two tick marks, e.g. 20 m>"} or null,
  "items": [{"label": "<room or zone>", "point": [y, x]}, {"label": "...", "polygon": [[y, x], ...]}, {"label": "...", "line": [[y, x], [y, x]]}]}}
point = a spot of empty floor inside a room that is closed by walls: code fills out to the walls and returns net area and perimeter.
polygon = the corners of a zone that is not closed by lines. line = the two ends of a straight run. At most 60 items; all coordinates 0-1000.

Counting: you see the sheet shrunk, so small symbols are easy to miss. When the question asks how many of some drawn symbol there are
(doors, windows, fixtures, valves, columns ...) and there may be more than about eight, do not count on this view. Reply instead with
{"count": {"things": {"<short key>": "<what it looks like on this drawing, and what does not count>"}, "region": [ymin, xmin, ymax, xmax] or null}}
and code will go over the sheet piece by piece at full resolution and give you the count of each key with a box per instance.
region limits the count to one part of the sheet (a wing, one view, one floor). For doors use two keys, single doors (one leaf) and double doors
(one opening with two leaves, counted as one door), unless the question defines doors differently, and state the convention in steps. You may ask for "measure" and "count" in the same reply."""


def page_image(full, page=0):
    """-> (png bytes, mime, (width, height) of the frame the demo draws in)"""
    low = full.lower()
    if low.endswith((".pdf", ".svg")):
        import fitz

        pg = fitz.open(full)[page]
        z = LONG_SIDE / max(pg.rect.width, pg.rect.height)
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
        return pix.tobytes("png"), "image/png", (pg.rect.width, pg.rect.height)
    from PIL import Image
    import io

    Image.MAX_IMAGE_PIXELS = None  # scanned sheets are large
    im = Image.open(full)
    size = im.size
    if max(size) > LONG_SIDE:
        im.thumbnail((LONG_SIDE, LONG_SIDE))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG")
    return buf.getvalue(), "image/png", size


_router = None


def effort(question):
    """One cheap call decides how hard the reading is. -> ("low" | "medium", tokens, usd)"""
    global _router
    try:
        if _router is None:
            _router = gb.Gemini(gb.DEFAULT_PLAN)
        before = _router.usd
        texts, a, b = _router.generate([[{"role": "system", "content": "A question is asked about a construction drawing. Reply with one word. "
            "LOOKUP if answering means reading one piece of text off the sheet (a title, scale, date, a name, one value from a schedule or note). "
            "COUNT if it means counting symbols, measuring, comparing several places, or adding things up."},
            {"role": "user", "content": question}]], 10)
        return ("low" if "LOOKUP" in texts[0].upper() else "medium"), a + b, _router.usd - before
    except Exception:
        return "medium", 0, 0.0


FACTS = """

Facts already extracted from this drawing are given below. %s
Use them together with the image: take numbers and names from the facts where they cover the question, and use the image
for what they do not contain or to check them. If the image clearly contradicts a fact, trust the image and say so in steps.

%s"""
TRUST = {"exact": "They were computed by code from the drawing's vector data or text layer and are exact.",
         "approximate": "They come from detectors and earlier model reads of the image; counts can be off by one or two."}


def answer(full, question, model_id=DEFAULT_MODEL, thinking=None, facts=None, trust="exact"):
    """facts: optional text of the parsed fact tables, given to the model next to the image (trust: exact | approximate)."""
    t0 = time.time()
    r_tok, r_usd = 0, 0.0
    if thinking is None:
        thinking, r_tok, r_usd = effort(question)
    data, mime, (W, H) = page_image(full)
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}}, {"text": PROMPT % question + (FACTS % (TRUST[trust], facts[:60000]) if facts else "")}]}],
            "generationConfig": {"maxOutputTokens": 30000, "thinkingConfig": {"thinkingLevel": thinking}}}
    res = {"answer": None, "unit": None, "evidence": [], "plan": {"op": "model reads the sheet" + (" with the extracted facts" if facts else "")}, "plan_source": "vision", "reliability": "model reading",
           "llm_tokens": 0, "cost_usd": 0.0, "model": model_id}
    def call(contents, again=True):
        nonlocal n_in, n_out
        body["contents"] = contents
        for a in range(5):
            try:
                req = urllib.request.Request(gb.URL % model_id, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
                j = json.loads(urllib.request.urlopen(req, timeout=300).read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 503) and a < 4:
                    time.sleep(3 * 2 ** a)
                    continue
                raise RuntimeError("Gemini API %d: %s" % (e.code, e.read().decode("utf-8", "replace")[:200]))
            except Exception as e:
                if a < 4:
                    time.sleep(3)
                    continue
                raise RuntimeError("Gemini API: %r" % (e,))
        parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        u = j.get("usageMetadata", {})
        n_in, n_out = n_in + u.get("promptTokenCount", 0), n_out + u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)), text
        except Exception:
            if not again:
                return None, text
            return call(contents, again=False)  # an empty or cut-off reply: ask once more

    n_in = n_out = 0
    measured, how = [], None
    contents = body["contents"]
    try:
        out, text = call(contents)
        if out and (isinstance(out.get("measure"), dict) or isinstance(out.get("count"), dict)):  # the model asks; code measures and counts; then it answers
            import measure

            src, sheet_mm, k = full, None, 1.0
            if full.lower().endswith((".pdf", ".svg")):  # work on the rendered page; its paper size is exact
                import tempfile

                import fitz

                fd, src = tempfile.mkstemp(prefix="stoa_measure_", suffix=".png")
                os.close(fd)
                pg = fitz.open(full)[0]
                z = 9000.0 / max(W, H)  # finer than the view the model saw
                pg.get_pixmap(matrix=fitz.Matrix(z, z)).save(src)
                sheet_mm = max(W, H) / 72.0 * 25.4
                k = 1 / z  # rendered pixels -> page coordinates
            back = []
            if isinstance(out.get("measure"), dict):
                results, measured, how = measure.run(src, out["measure"], sheet_mm)
                back.append("Measured by code on the image.\nScale: %s\n%s" % (how, json.dumps(results, ensure_ascii=False)))
            if isinstance(out.get("count"), dict) and isinstance(out["count"].get("things"), dict):
                import tile_count
                from PIL import Image

                things = {str(a)[:40]: str(b)[:300] for a, b in list(out["count"]["things"].items())[:6]}
                iw, ih = Image.open(src).size
                rg = out["count"].get("region")
                rg = (rg[1] / 1000 * iw, rg[0] / 1000 * ih, rg[3] / 1000 * iw, rg[2] / 1000 * ih) if isinstance(rg, list) and len(rg) == 4 else None
                found, c_usd, n_tiles = tile_count.count(src, things, region=rg)
                r_usd += c_usd
                for key, boxes in found.items():
                    measured += [{"kind": "counted", "name": key, "bbox": b, "poly": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]} for b in boxes]
                how = (how + "; " if how else "") + "counted on %d pieces of the sheet at full resolution" % n_tiles
                back.append("Counted by code, piece by piece at full resolution (%d pieces, overlaps merged): %s" % (n_tiles, json.dumps({a: len(b) for a, b in found.items()}, ensure_ascii=False)))
            for e in measured:
                e["bbox"] = [c * k for c in e["bbox"]]
                e["poly"] = [[p[0] * k, p[1] * k] for p in e["poly"]]
            if src != full:
                os.remove(src)
            contents = contents + [{"role": "model", "parts": [{"text": text}]},
                                   {"role": "user", "parts": [{"text": "\n\n".join(back) + "\n\nNow give the final JSON answer (answer, unit, steps, evidence, confidence). Use these numbers, do the arithmetic explicitly in steps, convert units if the question asks for others, and say if something failed. Leave evidence empty for what code measured or counted: those places are shown already."}]}]
            out, text = call(contents)
    except RuntimeError as e:
        res["note"] = str(e)
        return res
    pin, pout = gb.PRICES.get(model_id, (0, 0))
    res.update({"llm_tokens": n_in + n_out + r_tok, "cost_usd": round((n_in * pin + n_out * pout) / 1e6 + r_usd, 6), "seconds": round(time.time() - t0, 2), "thinking": thinking})
    if out is None or "measure" in out or "count" in out:
        res["note"] = "the model's reply could not be parsed"
        return res
    ans = out.get("answer")
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    ev = []
    for e in out.get("evidence") or []:
        b = e.get("box_2d") if isinstance(e, dict) else None
        if isinstance(b, list) and len(b) == 4:
            y0, x0, y1, x1 = [float(v) / 1000 for v in b]
            ev.append({"kind": "model box", "name": str(e.get("label", ""))[:80], "bbox": [x0 * W, y0 * H, x1 * W, y1 * H]})
    res.update({"answer": ans, "unit": out.get("unit") or None, "evidence": measured + ev,
                "calc": "%s\n\nRead from the sheet by %s (confidence: %s). This is the model's reading, not a computation; check the boxes." % (out.get("steps", ""), model_id, out.get("confidence", "?"))})
    if measured:
        res.update({"plan": {"op": "model points, code measures and counts"}, "reliability": "measured or counted by code on the image; what to look for was chosen by the model",
                    "calc": "%s\n\nDone by code on the image: %s. The model chose what to measure or count and did the final arithmetic (%s, confidence: %s)." % (out.get("steps", ""), how, model_id, out.get("confidence", "?"))})
    if ans is None:
        res["note"] = out.get("steps") or "The sheet does not show this."
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["ask"] and len(a) >= 3:
        full = a[1] if os.path.isabs(a[1]) else os.path.join(evalkit.DATA, a[1])
        r = answer(full, a[2], *(a[3:4]))
        r["evidence"] = ["%s %s" % (e["name"], [round(v) for v in e["bbox"]]) for e in r["evidence"]][:12]
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        print(__doc__)
