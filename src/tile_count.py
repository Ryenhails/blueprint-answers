"""Counting on large sheets: the model looks at the sheet piece by piece, code joins the pieces.

A whole sheet sent as one image is shrunk until door swings and window lines are a few pixels wide, and the model
misses many of them. Here the sheet is cut into overlapping tiles, the model boxes the asked objects in every tile,
and the boxes are mapped back to the sheet and merged where tiles overlap. Every counted object keeps its box.

  python tile_count.py count <image> "<what to count>" [more ...]
"""
import base64
import io
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemini_backend as gb  # noqa: E402

MODEL = os.environ.get("STOA_VISION_MODEL", "gemini-4-hydrogen")
TILE = 1600  # pixels sent per tile (long side)
PROMPT = """This image is one piece of a larger construction drawing sheet. Find every instance of each of these, including ones cut by the edge of the piece:
%s
Count only what is drawn in the drawing itself. Ignore legends, symbol keys, schedules, title blocks and key plans.
Answer with JSON only: {%s} where each value is a list of boxes [ymin, xmin, ymax, xmax] in 0-1000 coordinates of this piece, one tight box per instance. Empty list if there is none."""


def content_box(im):
    """Bounding box of the ink, so empty margins are not tiled."""
    import numpy as np

    g = np.asarray(im.convert("L").resize((max(1, im.width // 8), max(1, im.height // 8))))
    ys, xs = np.where(g < 200)
    if len(xs) == 0:
        return 0, 0, im.width, im.height
    return int(xs.min()) * 8, int(ys.min()) * 8, min(im.width, int(xs.max() + 1) * 8), min(im.height, int(ys.max() + 1) * 8)


def tiles(box, target):
    """Overlapping tiles of about target x target original pixels covering box."""
    x0, y0, x1, y1 = box
    out = []
    nx, ny = max(1, round((x1 - x0) / (target * 0.8))), max(1, round((y1 - y0) / (target * 0.8)))
    w, h = (x1 - x0) / nx, (y1 - y0) / ny
    for j in range(ny):
        for i in range(nx):
            out.append((max(x0, int(x0 + i * w - 0.12 * w)), max(y0, int(y0 + j * h - 0.12 * h)), min(x1, int(x0 + (i + 1) * w + 0.12 * w)), min(y1, int(y0 + (j + 1) * h + 0.12 * h))))
    return out


def _ask(png, prompt, thinking):
    body = {"contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode()}}, {"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 20000, "thinkingConfig": {"thinkingLevel": thinking}}}
    for a in range(5):
        try:
            req = urllib.request.Request(gb.URL % MODEL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
            j = json.loads(urllib.request.urlopen(req, timeout=300).read())
            break
        except Exception:
            if a == 4:
                return {}, 0.0
            time.sleep(3 * 2 ** a)
    text = "".join(p.get("text", "") for p in (j.get("candidates") or [{}])[0].get("content", {}).get("parts", []) if not p.get("thought"))
    u = j.get("usageMetadata", {})
    pin, pout = gb.PRICES.get(MODEL, (0, 0))
    usd = (u.get("promptTokenCount", 0) * pin + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * pout) / 1e6
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)), usd
    except Exception:
        return {}, usd


def _merge(boxes):
    """Boxes of the same object seen in two tiles: keep one. Same object = centres closer than half the smaller box."""
    keep = []
    for b in sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1])):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        dup = False
        for k in keep:
            kx, ky = (k[0] + k[2]) / 2, (k[1] + k[3]) / 2
            lim = 0.5 * max(min(b[2] - b[0], k[2] - k[0]), min(b[3] - b[1], k[3] - k[1]))
            ix = max(0, min(b[2], k[2]) - max(b[0], k[0])) * max(0, min(b[3], k[3]) - max(b[1], k[1]))
            if (abs(cx - kx) <= lim and abs(cy - ky) <= lim) or ix > 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


def count(full, things, tile_px=None, thinking="low", region=None):
    """things: {"key": "description"}; region: optional (x0, y0, x1, y1) in original pixels.
    -> ({key: [box in original pixels]}, usd, number of tiles)"""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    if full.lower().endswith(".pdf"):
        import fitz

        pg = fitz.open(full)[0]
        z = 3072.0 / max(pg.rect.width, pg.rect.height)  # same frame as the sheet-reading step
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        back = 1 / z
    else:
        im, back = Image.open(full).convert("RGB"), 1.0
    box = region or content_box(im)
    if tile_px is None:  # about a dozen tiles for a full sheet, never less detail than the original
        tile_px = int(max(900, min(2600, max(box[2] - box[0], box[3] - box[1]) / 3.2)))
    ts = tiles(box, tile_px)
    prompt = PROMPT % ("\n".join("- %s: %s" % kv for kv in things.items()), ", ".join('"%s": [...]' % k for k in things))

    def one(t):
        c = im.crop(t)
        k = TILE / max(c.size)
        c = c.resize((max(1, round(c.width * k)), max(1, round(c.height * k))), Image.LANCZOS)
        buf = io.BytesIO()
        c.save(buf, "PNG")
        out, usd = _ask(buf.getvalue(), prompt, thinking)
        res = {}
        for key in things:
            res[key] = []
            for b in out.get(key) or []:
                if isinstance(b, list) and len(b) == 4:
                    y0, x0, y1, x1 = [float(v) / 1000 for v in b]
                    res[key].append([t[0] + x0 * (t[2] - t[0]), t[1] + y0 * (t[3] - t[1]), t[0] + x1 * (t[2] - t[0]), t[1] + y1 * (t[3] - t[1])])
        return res, usd

    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(one, ts))
    usd = sum(p[1] for p in parts)
    return {key: [[v * back for v in b] for b in _merge([b for p in parts for b in p[0][key]])] for key in things}, usd, len(ts)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["count"] and len(a) >= 3:
        res, usd, n = count(a[1], {"t%d" % i: d for i, d in enumerate(a[2:])})
        print({k: len(v) for k, v in res.items()}, "| %d tiles, $%.3f" % (n, usd))
    else:
        print(__doc__)
