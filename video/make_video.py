"""Builds the two-minute pitch video from the system's own outputs.

Pictures: the sheets, evidence shapes, answers and validation numbers are read from an exported demo page
(python src/demo_server.py export ...), so everything shown is what the system produced. Voice: Gemini text-to-speech.

  python video/make_video.py <exported demo .html> <out folder> [--no-voice]
Needs pillow, numpy, imageio-ffmpeg. Writes <out>/blueprint_answers.mp4 and keeps the narration clips in <out>/voice/.
"""
import base64
import io
import json
import math
import os
import subprocess
import sys
import urllib.request
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
Image.MAX_IMAGE_PIXELS = None
W, H, FPS = 1920, 1080, 24
BG, PANEL, INK, MUTED, LINE = (238, 241, 244), (255, 255, 255), (20, 32, 46), (91, 104, 120), (207, 214, 222)
BLUE, GREEN, ORANGE, RED = (20, 99, 200), (11, 122, 83), (217, 72, 15), (179, 38, 30)
ROUTES = [("Vector plan PDF", (29, 98, 184), "code reads the PDF's lines"), ("BIM model (IFC)", (31, 122, 77), "one row per element"),
          ("Schedule sheet", (168, 93, 0), "tables with named columns"), ("Floor-plan image", (122, 63, 176), "three readers, middle count"),
          ("Scanned sheet", (179, 55, 47), "measured and counted by code")]
FONT_DIR = [os.path.expanduser("~/.local/share/fonts/mpl"), "/usr/share/fonts/dejavu-sans-fonts", "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]
VOICE, TTS_MODEL = "Charon", "gemini-3.8-flash-tts"

SCRIPT = [
    ("hook", "Before anything gets built, someone counts the doors and measures the floors, by hand. "
             "We gave a frontier model the image and the question: thirty-six of fifty. It reads the drawing fine. It just can't count or measure."),
    ("idea", "Blueprint Answers splits the job. Models decide and point. Code counts, sums and measures. "
             "Every drawing is read once into facts, by a parser chosen from what the file is."),
    ("vector", "Ask how large an apartment is. The question form is known, so no model is called. The rooms light up, the sum is shown, the cost is zero."),
    ("ladder", "A new kind of question climbs a ladder. A small model routes it. If it is complex, a larger model writes code over the facts, and the code runs. "
               "The answer slip shows every step and its cost."),
    ("measure", "Scans are the hard part. The model only points at a room. Code reads the scale, checks it against the scale bar, "
                "fills the room to its walls, and returns area and perimeter."),
    ("count", "To count doors, the sheet is cut into pieces. The model boxes each symbol, code merges the boxes, and every counted door is drawn back on the sheet."),
    ("valid", None),  # written from the validation numbers
    ("close", "Blueprint Answers. Drawings that answer, and show their work."),
]


def font(size, bold=False, mono=False):
    name = ("DejaVuSansMono" if mono else "DejaVuSans") + ("-Bold" if bold else "") + ".ttf"
    for d in FONT_DIR:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(a, b, k)) for a, b in zip(c1, c2))


def wrap(d, text, f, width):
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) <= width or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def para(d, xy, text, f, width, fill=INK, gap=1.32):
    x, y = xy
    for ln in wrap(d, text, f, width):
        d.text((x, y), ln, font=f, fill=fill)
        y += int(f.size * gap)
    return y


def panel(d, box, fill=PANEL, outline=LINE, r=10, width=2):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


def load_static(path):
    h = open(path, encoding="utf-8").read()
    i, j = h.index("const STATIC = ") + 15, h.index(";\nconst LIVE" if ";\nconst LIVE" in h else ";\nconst $ =")
    return json.loads(h[i:j].replace("<\\/", "</"))


def sheet_image(st, path, max_side=4200):
    d = st["drawings"][path]
    im = Image.open(io.BytesIO(base64.b64decode(d["image"].split(",", 1)[1]))).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im, d["clip"]


class Sheet:
    """A drawing with shapes in its own coordinates, shown through a moving window."""

    def __init__(self, im, clip):
        self.im, self.clip = im, clip
        self.sx, self.sy = im.width / (clip[2] - clip[0]), im.height / (clip[3] - clip[1])

    def px(self, x, y):
        return (x - self.clip[0]) * self.sx, (y - self.clip[1]) * self.sy

    def view(self, box, size, shapes=()):
        """box: window in image pixels (x0, y0, x1, y1). shapes: [(kind, pts in drawing coords, colour, alpha, width)]"""
        x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), min(self.im.width, box[2]), min(self.im.height, box[3])
        bw, bh = x1 - x0, y1 - y0
        k = min(size[0] / bw, size[1] / bh)
        out = Image.new("RGB", size, (255, 255, 255))
        crop = self.im.crop((int(x0), int(y0), int(x1), int(y1))).resize((max(1, int(bw * k)), max(1, int(bh * k))), Image.BILINEAR)
        ox, oy = (size[0] - crop.width) // 2, (size[1] - crop.height) // 2
        out.paste(crop, (ox, oy))
        if shapes:
            ov = Image.new("RGBA", size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for kind, pts, col, alpha, width in shapes:
                P = [((self.px(x, y)[0] - x0) * k + ox, (self.px(x, y)[1] - y0) * k + oy) for x, y in pts]
                if kind == "poly":
                    d.polygon(P, fill=col + (int(255 * alpha),), outline=col + (255,))
                    d.line(P + [P[0]], fill=col + (255,), width=width)
                elif kind == "box":
                    d.rectangle([P[0], P[1]], outline=col + (255,), width=width, fill=col + (int(255 * alpha),))
                elif kind == "line":
                    d.line(P, fill=col + (255,), width=width)
            out = Image.alpha_composite(out.convert("RGBA"), ov).convert("RGB")
        return out

    def full(self):
        return (0, 0, self.im.width, self.im.height)

    def around(self, bbox, pad=0.25, aspect=1.0):
        (x0, y0), (x1, y1) = self.px(bbox[0], bbox[1]), self.px(bbox[2], bbox[3])
        w, h = (x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad)
        if w / h < aspect:
            w = h * aspect
        else:
            h = w / aspect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def win(a, b, k):
    return tuple(lerp(p, q, k) for p, q in zip(a, b))


def frame():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def header(d, kicker, title):
    d.text((90, 60), kicker.upper(), font=font(24, True), fill=MUTED)
    d.text((90, 96), title, font=font(54, True), fill=INK)


def caption(d, text, t, T):
    """Narration as a caption, a line or two at a time."""
    f = font(30)
    lines = wrap(d, text, f, W - 360)
    per = 2
    groups = [" ".join(lines[i:i + per]) for i in range(0, len(lines), per)]
    words = [len(g.split()) for g in groups]
    pos, acc = t / max(T, 0.01) * sum(words), 0
    g = groups[-1]
    for gi, n in zip(groups, words):
        acc += n
        if pos <= acc:
            g = gi
            break
    ls = wrap(d, g, f, W - 360)
    y = H - 46 - 42 * len(ls)
    d.rectangle([0, y - 18, W, H], fill=(20, 32, 46))
    for ln in ls:
        d.text(((W - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill=(238, 241, 244))
        y += 42


def answer_card(d, box, question, answer, unit, badges, steps, t, cost="$0"):
    """The answer slip of the interface. steps: [(label, detail, state 'done'|'pass'|'skip')]"""
    x0, y0, x1, y1 = box
    panel(d, box)
    y = y0 + 26
    d.text((x0 + 28, y), "QUESTION", font=font(18, True), fill=MUTED)
    y = para(d, (x0 + 28, y + 30), question, font(26), x1 - x0 - 56) + 14
    if t > 0.25:
        d.line([x0 + 28, y, x1 - 28, y], fill=LINE, width=2)
        d.text((x0 + 28, y + 16), "ANSWER", font=font(18, True), fill=MUTED)
        fa = font(64 if len(str(answer)) < 14 else 34, mono=True)
        ya = para(d, (x0 + 28, y + 46), "%s %s" % (answer, unit), fa, x1 - x0 - 56)
        bx = x0 + 28
        for txt, col in badges:
            fb = font(20)
            w = d.textlength(txt, font=fb) + 24
            d.rounded_rectangle([bx, ya + 6, bx + w, ya + 42], 6, fill=mix(col, (255, 255, 255), 0.86))
            d.text((bx + 12, ya + 12), txt, font=fb, fill=col)
            bx += w + 10
        y = ya + 64
    if t > 0.45:
        d.line([x0 + 28, y, x1 - 28, y], fill=LINE, width=2)
        d.text((x0 + 28, y + 16), "ANSWER PATH", font=font(18, True), fill=MUTED)
        y += 52
        n = int(1 + (t - 0.45) / 0.4 * len(steps))
        for i, (lab, det, state) in enumerate(steps[:n]):
            col = BLUE if state == "done" else MUTED
            if state == "done":
                d.ellipse([x0 + 30, y + 4, x0 + 50, y + 24], fill=BLUE)
            else:
                d.ellipse([x0 + 30, y + 4, x0 + 50, y + 24], outline=MUTED, width=3)
            d.text((x0 + 64, y), "%d · %s" % (i, lab), font=font(23, True), fill=INK if state != "skip" else MUTED)
            y = para(d, (x0 + 64, y + 30), det, font(20), x1 - x0 - 100, fill=INK if state == "done" else MUTED) + 12
        if t > 0.85:
            d.text((x0 + 28, y1 - 60), "API cost of this answer", font=font(20), fill=MUTED)
            fc = font(30, True, mono=True)
            d.text((x1 - 28 - d.textlength(cost, font=fc), y1 - 66), cost, font=fc, fill=GREEN if cost == "$0" else INK)


# ---- scenes: each returns a function (t seconds, T duration) -> image -------------------------


def scene_hook(st):
    path = next(e["path"] for e in st["examples"] if "cubicasa" in e["path"])
    sh = Sheet(*sheet_image(st, path))

    def f(t, T):
        im, d = frame()
        v = sh.view(sh.full(), (820, 820))
        im.paste(v, (90, 110))
        d.rectangle([90, 110, 910, 930], outline=LINE, width=2)
        x = 1000
        for i, (txt, at) in enumerate([("Someone counts the doors", 0.4), ("and measures the floors.", 2.2), ("By hand.", 4.4)]):
            if t > at:
                d.text((x, 150 + i * 80), txt, font=font(58, True), fill=mix(BG, INK, ease((t - at) * 2)))
        if t > T * 0.45:
            k = ease((t - T * 0.45) * 1.5)
            panel(d, [x, 440, 1830, 760])
            d.text((x + 36, 464), "FRONTIER MODEL · IMAGE + QUESTION ONLY", font=font(22, True), fill=MUTED)
            d.text((x + 36, 510), "36 / 50", font=font(150, True, mono=True), fill=mix(PANEL, RED, k))
            d.text((x + 36, 690), "counts must be exact, sizes within 2%", font=font(24), fill=MUTED)
        if t > T * 0.72:
            d.text((x, 800), "It reads the drawing fine.", font=font(40), fill=INK)
            d.text((x, 856), "It can't count or measure.", font=font(40, True), fill=RED)
        return im

    return f


def scene_idea(st):
    def f(t, T):
        im, d = frame()
        header(d, "Blueprint Answers", "Models decide and point.")
        if t > 1.6:
            d.text((90, 166), "Code counts, sums and measures.", font=font(54, True), fill=BLUE)
        y0 = 330
        d.text((90, y0 - 50), "EVERY DRAWING IS READ ONCE, INTO FACTS", font=font(22, True), fill=MUTED)
        cw = (W - 180 - 4 * 20) // 5
        for i, (name, col, to) in enumerate(ROUTES):
            at = T * 0.38 + i * 0.75
            if t < at:
                continue
            k = ease((t - at) * 2.5)
            x = 90 + i * (cw + 20)
            y = y0 + int((1 - k) * 40)
            panel(d, [x, y, x + cw, y + 300])
            d.rectangle([x, y, x + cw, y + 12], fill=col)
            para(d, (x + 22, y + 40), name, font(30, True), cw - 44)
            d.line([x + 22, y + 190, x + cw - 22, y + 190], fill=LINE, width=2)
            d.text((x + 22, y + 152), "↓", font=font(28, True), fill=col)
            para(d, (x + 22, y + 206), to, font(22, mono=True), cw - 44, fill=INK)
        if t > T * 0.38 + 4.2:
            panel(d, [90, 690, W - 90, 800], fill=(20, 32, 46), outline=(20, 32, 46))
            d.text((126, 722), "fact tables", font=font(34, True, mono=True), fill=(238, 241, 244))
            d.text((400, 728), "rooms · doors · windows · fixtures · elements · schedule rows — each with its place on the sheet", font=font(26), fill=(207, 214, 222))
        return im

    return f


def scene_vector(st):
    ex = next(e for e in st["examples"] if e["kind"] == "swiss")
    cands = [x["q"] for x in ex["questions"] if x.get("type") == "area" and st["answers"][ex["path"] + "\n" + x["q"]].get("correct")
             and len(st["answers"][ex["path"] + "\n" + x["q"]].get("evidence") or []) >= 3]
    q = next((c for c in cands if "apartment" in c.lower() or "appartement" in c.lower()), cands[0])
    a = st["answers"][ex["path"] + "\n" + q]
    sh = Sheet(*sheet_image(st, ex["path"]))
    polys = [e for e in a["evidence"] if e.get("poly")]

    def f(t, T):
        im, d = frame()
        header(d, "Vector plan · known question form", "No model call at all")
        n = int(ease((t - 2.0) / 3.5) * len(polys)) if t > 2.0 else 0
        shapes = [("poly", e["poly"], BLUE, 0.28, 4) for e in polys[:n]]
        v = sh.view(sh.full(), (1120, 760), shapes)
        im.paste(v, (90, 190))
        d.rectangle([90, 190, 1210, 950], outline=LINE, width=2)
        steps = [("Known question form", "query from the plan cache, computed by code on the vector geometry", "done"),
                 ("Decision model", "not needed", "skip"), ("Model writes code on the facts", "not needed", "skip"), ("Model reads the sheet", "not needed", "skip")]
        ans = ("%.2f" % a["answer"]) if isinstance(a["answer"], float) else str(a["answer"])
        answer_card(d, [1250, 190, W - 90, 950], q[:110] + ("…" if len(q) > 110 else ""), ans, "m²",
                    [("matches the dataset answer", GREEN), ("computed by code", GREEN)], steps, ease(t / (T * 0.8)), "$0")
        return im

    return f


def scene_ladder(st):
    ex = next(e for e in st["examples"] if e["kind"] == "swiss")
    q = next(x["q"] for x in ex["questions"] if "skirting" in x["q"])
    a = st["answers"][ex["path"] + "\n" + q]
    code = (a.get("calc") or "").split("fact tables:\n")[-1].strip().split("\n")
    rungs = [("Known question form", "177 forms, a dozen languages", "$0"), ("Decision model", "routes a new phrasing", "≈ $0.0004"),
             ("Model writes code", "Python over the fact tables, run in a sandbox", "≈ $0.01–0.06"), ("Model reads the sheet", "may ask code to measure or count", "≈ $0.01–0.20")]

    def f(t, T):
        im, d = frame()
        header(d, "A question climbs only as far as it has to", "Cheapest step that can answer wins")
        cw = (W - 180) // 4
        active = min(3, int(t / (T * 0.16))) if t < T * 0.5 else 2
        for i, (name, det, cost) in enumerate(rungs):
            x = 90 + i * cw
            on = i == active
            d.rectangle([x, 200, x + cw - 8, 360], fill=(20, 32, 46) if on else PANEL, outline=LINE, width=2)
            d.text((x + 20, 216), "%d  %s" % (i, name), font=font(27, True), fill=(238, 241, 244) if on else INK)
            para(d, (x + 20, 262), det, font(21), cw - 48, fill=(207, 214, 222) if on else MUTED)
            d.text((x + 20, 322), cost, font=font(21, mono=True), fill=(120, 200, 160) if on else GREEN)
        if t > T * 0.42:
            k = (t - T * 0.42) / (T * 0.5)
            panel(d, [90, 400, 1000, 950])
            d.text((118, 420), "QUESTION", font=font(18, True), fill=MUTED)
            y = para(d, (118, 452), q, font(27), 850)
            d.text((118, y + 18), "ANSWER", font=font(18, True), fill=MUTED)
            if k > 0.75:
                d.text((118, y + 48), "%s m" % a["answer"], font=font(64, mono=True), fill=INK)
                para(d, (118, y + 140), (a.get("calc") or "").split("\n\n")[0], font(21), 850, fill=MUTED)
            panel(d, [1030, 400, W - 90, 950], fill=(247, 248, 250))
            d.text((1058, 420), "CODE THE MODEL WROTE · RUN ON THE FACTS", font=font(18, True), fill=MUTED)
            n = int(ease(k * 1.4) * min(len(code), 19))
            fy = 456
            for ln in code[:n]:
                d.text((1058, fy), ln[:66], font=font(19, mono=True), fill=INK)
                fy += 25
        return im

    return f


def _school(st):
    ex = next((e for e in st["examples"] if "school" in e["title"].lower()), None)
    if ex is None:
        ex = next(e for e in st["examples"] if "scanned image" in e["title"])
    return ex, Sheet(*sheet_image(st, ex["path"], 5200))


def scene_measure(st):
    ex, sh = _school(st)
    q = next((x["q"] for x in ex["questions"] if "baseboard" in x["q"]), ex["questions"][0]["q"])
    a = st["answers"][ex["path"] + "\n" + q]
    regs = [e for e in a["evidence"] if e.get("kind") == "measured" and len(e.get("poly") or []) > 2]
    lines = [e for e in a["evidence"] if e.get("kind") == "measured" and len(e.get("poly") or []) == 2]
    how = (a.get("calc") or "").split("Done by code on the image: ")[-1].split(". The model chose")[0]
    target = sh.around(regs[0]["bbox"], 0.9, 1120 / 760) if regs else sh.full()

    def f(t, T):
        im, d = frame()
        header(d, "Scanned sheet · the model points, code measures", "Sizes are never guessed by eye")
        k = ease((t - 1.5) / 3.0)
        box = win(sh.full(), target, k) if t < T * 0.62 else win(target, sh.full(), ease((t - T * 0.62) / 2.0))
        shapes = []
        if t > 5.0:
            fillk = ease((t - 5.0) / 1.5)
            for e in regs:
                shapes.append(("poly", e["poly"], GREEN, 0.34 * fillk, 5))
            for e in lines:
                shapes.append(("line", e["poly"], ORANGE, 1, 7))
        v = sh.view(box, (1120, 760), shapes)
        im.paste(v, (90, 190))
        d.rectangle([90, 190, 1210, 950], outline=LINE, width=2)
        panel(d, [1250, 190, W - 90, 950])
        x, y = 1278, 214
        d.text((x, y), "QUESTION", font=font(18, True), fill=MUTED)
        y = para(d, (x, y + 30), q, font(24), 520) + 16
        items = [(2.5, "1  the model reads the scale and points", "a spot inside each room, the ends of each door opening"),
                 (5.0, "2  code calibrates", how[:150]),
                 (7.5, "3  code fills each room to its walls", "\n".join(e["name"] for e in regs[:4])),
                 (T * 0.66, "4  the model does the arithmetic", (a.get("calc") or "").split("\n\n")[0][:260])]
        for at, head, body in items:
            if t > at:
                d.text((x, y), head, font=font(23, True), fill=GREEN)
                y = para(d, (x, y + 32), body, font(19), 520, fill=INK) + 14
        if t > T * 0.8:
            d.text((x, 860), "%s %s" % (a["answer"], a.get("unit") or ""), font=font(58, mono=True), fill=INK)
        return im

    return f


def scene_count(st):
    ex, sh = _school(st)
    q = next((x["q"] for x in ex["questions"] if "doors" in x["q"].lower()), ex["questions"][0]["q"])
    a = st["answers"][ex["path"] + "\n" + q]
    boxes = [e for e in a["evidence"] if e.get("bbox") and e.get("kind") in ("counted", "model box")]
    boxes.sort(key=lambda e: (round(e["bbox"][1] / ((sh.clip[3] - sh.clip[1]) / 4)), e["bbox"][0]))
    xs = [b["bbox"][0] for b in boxes] + [b["bbox"][2] for b in boxes]
    ys = [b["bbox"][1] for b in boxes] + [b["bbox"][3] for b in boxes]
    area = sh.around([min(xs), min(ys), max(xs), max(ys)], 0.04, 1120 / 760) if boxes else sh.full()
    names = sorted({e["name"] for e in boxes})

    def f(t, T):
        im, d = frame()
        header(d, "Scanned sheet · counting piece by piece", "Every counted object is drawn back on the sheet")
        n = int(ease((t - 3.0) / (T - 5.5)) * len(boxes)) if t > 3.0 else 0
        shapes = [("box", [(e["bbox"][0], e["bbox"][1]), (e["bbox"][2], e["bbox"][3])], BLUE if "window" in e["name"].lower() else ORANGE, 0.12, 4) for e in boxes[:n]]
        v = sh.view(area, (1120, 760), shapes)
        if 1.0 < t < T - 2.0:  # the tile grid
            dv = ImageDraw.Draw(v)
            for i in range(1, 4):
                dv.line([i * 280, 0, i * 280, 760], fill=(20, 32, 46), width=2)
            for j in range(1, 3):
                dv.line([0, j * 253, 1120, j * 253], fill=(20, 32, 46), width=2)
        im.paste(v, (90, 190))
        d.rectangle([90, 190, 1210, 950], outline=LINE, width=2)
        panel(d, [1250, 190, W - 90, 950])
        x, y = 1278, 214
        d.text((x, y), "QUESTION", font=font(18, True), fill=MUTED)
        y = para(d, (x, y + 30), q, font(26), 520) + 24
        for nm in names:
            c = sum(1 for e in boxes[:n] if e["name"] == nm)
            col = BLUE if "window" in nm.lower() else ORANGE
            d.rectangle([x, y + 8, x + 26, y + 34], outline=col, width=5)
            d.text((x + 44, y), nm.replace("_", " "), font=font(26), fill=INK)
            fc = font(44, True, mono=True)
            d.text((W - 118 - d.textlength(str(c), font=fc), y - 8), str(c), font=fc, fill=col)
            y += 64
        if t > T - 2.2:
            d.line([x, y + 6, W - 118, y + 6], fill=LINE, width=2)
            para(d, (x, y + 22), str(a["answer"]), font(34, True), 520)
        return im

    return f


def scene_valid(st):
    v = st.get("validation") or {}
    groups = [g for g in v.get("groups", []) if g["key"] != "schedules"]

    def f(t, T):
        im, d = frame()
        header(d, "Validation", "%d questions the system was not built on" % sum(g["n"] for g in groups))
        al = v.get("all", {"ok": 0, "numeric": 1, "usd": 0})
        tiles = [("%.1f%%" % (100 * al["ok"] / max(1, al["numeric"])), "of %d numeric answers correct" % al["numeric"]),
                 ("$%.3f" % al["usd"], "average API cost per question"), ("%d" % len(groups), "kinds of drawings, one interface")]
        for i, (big, small) in enumerate(tiles):
            x = 90 + i * 590
            panel(d, [x, 190, x + 560, 360])
            d.text((x + 28, 206), big, font=font(76, True, mono=True), fill=INK)
            d.text((x + 28, 308), small, font=font(23), fill=MUTED)
        y = 410
        for i, g in enumerate(groups):
            if t < 1.5 + i * 0.9:
                break
            k = ease((t - 1.5 - i * 0.9) * 1.6)
            d.text((90, y + 6), g["label"], font=font(27, True), fill=INK)
            x0, x1 = 640, 1330
            d.rectangle([x0, y + 4, x1, y + 44], fill=PANEL, outline=LINE, width=2)
            tx = g.get("text") or {}
            nt = sum(tx.get(a, 0) for a in ("correct", "partial", "wrong"))
            if g["numeric"]:
                share = g["ok"] / g["numeric"]
                d.rectangle([x0, y + 4, x0 + (x1 - x0) * share * k, y + 44], fill=GREEN)
                lab = "%d / %d  (%.1f%%)" % (g["ok"], g["numeric"], 100 * share)
            elif nt:
                c, p = tx.get("correct", 0) / nt, tx.get("partial", 0) / nt
                d.rectangle([x0, y + 4, x0 + (x1 - x0) * c * k, y + 44], fill=GREEN)
                d.rectangle([x0 + (x1 - x0) * c * k, y + 4, x0 + (x1 - x0) * (c + p) * k, y + 44], fill=(200, 140, 20))
                lab = "%d right, %d partly, of %d" % (tx.get("correct", 0), tx.get("partial", 0), nt)
            else:
                lab = ""
            d.text((x1 + 24, y + 8), lab, font=font(24, mono=True), fill=INK)
            y += 84
        return im

    return f


def scene_close(st):
    def f(t, T):
        im = Image.new("RGB", (W, H), (20, 32, 46))
        d = ImageDraw.Draw(im)
        d.text((90, 360), "Blueprint Answers", font=font(120, True), fill=(238, 241, 244))
        d.text((94, 520), "Drawings that answer, and show their work.", font=font(46), fill=(160, 190, 235))
        d.text((94, 640), os.environ.get("STOA_REPO_URL", ""), font=font(34, mono=True), fill=(207, 214, 222))
        d.text((94, 700), "Junction 2026 · Stoa — Make Blueprints Talk", font=font(28), fill=(147, 161, 177))
        return im

    return f


SCENES = {"hook": scene_hook, "idea": scene_idea, "vector": scene_vector, "ladder": scene_ladder, "measure": scene_measure, "count": scene_count,
          "valid": scene_valid, "close": scene_close}


def validation_line(st):
    v = st.get("validation") or {}
    al = v.get("all", {"ok": 0, "numeric": 1, "usd": 0})
    g = {x["key"]: x for x in v.get("groups", [])}

    def pc(k):
        return "%d percent" % round(100 * g[k]["ok"] / max(1, g[k]["numeric"])) if k in g and g[k]["numeric"] else "unknown"

    cents = al["usd"] * 100
    return ("On %d questions it was not built on, %d percent of numeric answers are correct, at about %s a question. "
            "Vector plans and building models are near perfect. Images reach %s, scans %s. We show the hard numbers too."
            % (sum(x["n"] for x in v.get("groups", []) if x["key"] != "schedules"), round(100 * al["ok"] / max(1, al["numeric"])), ("%.1f cents" % cents) if cents >= 0.1 else "a tenth of a cent", pc("images"), pc("scans")))


def speak(text, out_wav):
    import gemini_backend as gb

    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}}}}
    req = urllib.request.Request(gb.URL % TTS_MODEL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": gb._key()})
    j = json.loads(urllib.request.urlopen(req, timeout=300).read())
    p = j["candidates"][0]["content"]["parts"][0]["inlineData"]
    data = base64.b64decode(p["data"])
    if data[:4] == b"RIFF":
        open(out_wav, "wb").write(data)
    else:  # raw 16-bit PCM, 24 kHz mono
        with wave.open(out_wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(data)


def read_wav(path):
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            a = a.reshape(-1, 2).mean(axis=1).astype(np.int16)
        return a, w.getframerate()


def main(static_path, out_dir, voice=True, only=None):
    import imageio_ffmpeg

    st = load_static(static_path)
    os.makedirs(os.path.join(out_dir, "voice"), exist_ok=True)
    script = [(k, t if t is not None else validation_line(st)) for k, t in SCRIPT if only is None or k in only]
    clips, rate = [], 24000
    for k, text in script:
        wav = os.path.join(out_dir, "voice", k + ".wav")
        txt = wav[:-4] + ".txt"
        if voice and not (os.path.exists(wav) and os.path.exists(txt) and open(txt).read() == text):
            speak(text, wav)
            open(txt, "w").write(text)
        if voice:
            a, rate = read_wav(wav)
            loud = np.where(np.abs(a.astype(np.int32)) > 500)[0]  # cut the silence the voice model leaves at both ends
            if len(loud):
                a = a[max(0, loud[0] - rate // 20):loud[-1] + rate // 10]
            with wave.open(wav[:-4] + ".trim.wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(a.tobytes())
        else:
            a = np.zeros(int(rate * len(text.split()) / 2.6), dtype=np.int16)
        clips.append(a)
    LIMIT, PAD = 116.0, 1.1  # the film must stay under two minutes
    speech = sum(len(c) for c in clips) / rate
    f = max(1.0, speech / (LIMIT - PAD * len(clips)))
    if f > 1.0 and voice:  # speak a little faster rather than cut words
        print("narration is %.1f s: sped up by %.2f" % (speech, f))
        fast = []
        for (k, _), c in zip(script, clips):
            src, dst = os.path.join(out_dir, "voice", k + ".trim.wav"), os.path.join(out_dir, "voice", k + ".fast.wav")
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, "-filter:a", "atempo=%.3f" % min(f, 1.6), "-ar", str(rate), "-ac", "1", dst], check=True)
            fast.append(read_wav(dst)[0])
        clips = fast
    clips = [np.concatenate([np.zeros(int(rate * 0.2), np.int16), c, np.zeros(int(rate * (PAD - 0.2)), np.int16)]) for c in clips]
    total = sum(len(c) for c in clips) / rate
    print("narration: %.1f s in %d scenes: %s" % (total, len(clips), ", ".join("%s %.1f" % (k, len(c) / rate) for (k, _), c in zip(script, clips))))
    silent = os.path.join(out_dir, "silent.mp4")
    wr = imageio_ffmpeg.write_frames(silent, (W, H), fps=FPS, quality=7, macro_block_size=8, pix_fmt_out="yuv420p")
    wr.send(None)
    for (k, text), c in zip(script, clips):
        T = len(c) / rate
        draw = SCENES[k](st)
        for i in range(int(round(T * FPS))):
            t = i / FPS
            im = draw(t, T)
            if k != "close":
                caption(ImageDraw.Draw(im), text, max(0.0, t - 0.25), T - 0.7)
            wr.send(np.asarray(im).tobytes())
        print("scene", k, "%.1f s" % T, flush=True)
    wr.close()
    audio = os.path.join(out_dir, "narration.wav")
    with wave.open(audio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.concatenate(clips).tobytes())
    out = os.path.join(out_dir, "blueprint_answers.mp4")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", silent, "-i", audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out], check=True)
    os.remove(silent)
    print("wrote %s: %.1f s, %.1f MB" % (out, total, os.path.getsize(out) / 1e6))


def scene_limits(st):
    v = st.get("validation") or {}
    rows = [("Cost", "Known question forms cost nothing. A new phrasing costs about $0.0004, generated code $0.01–0.06, reading a scanned sheet $0.01–0.20. Reading a drawing in is a few cents at most."),
            ("What is exact", "Vector plans and building models: computed from the file's own geometry and quantities."),
            ("What is approximate", "Images and scans. Answers say so, show every counted box and measured region, and flag when two independent readings disagree."),
            ("Known weak spots", "Doors on scans are undercounted by 10–35% on every drawing. Sums over many rooms on a scan are unstable. The vector parser is tied to this dataset's drawing style."),
            ("Graded by a model", "Open BIM answers were graded by a model against the reference answers.")]

    def f(t, T):
        im, d = frame()
        header(d, "Cost and limits", "What it costs, and where it is still wrong")
        y = 210
        for head, body in rows:
            panel(d, [90, y, W - 90, y + 138])
            d.text((118, y + 20), head, font=font(28, True), fill=BLUE)
            para(d, (480, y + 20), body, font(25), W - 90 - 480 - 30)
            y += 152
        return im

    return f


def deck(static_path, out_pdf):
    """The pitch deck: the film's key frames as slides, plus cost and limits."""
    st = load_static(static_path)
    slides = []
    for k in ("hook", "idea", "vector", "ladder", "measure", "count", "valid"):
        slides.append(SCENES[k](st)(1000.0, 1000.0).convert("RGB"))
    slides.append(scene_limits(st)(0, 1))
    slides.append(scene_close(st)(0, 1))
    slides[0].save(out_pdf, save_all=True, append_images=slides[1:], resolution=150.0)
    print("wrote %s: %d slides" % (out_pdf, len(slides)))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["deck"] and len(a) == 3:
        deck(a[1], a[2])
    elif len(a) >= 2:
        only = a[a.index("--only") + 1].split(",") if "--only" in a else None
        main(a[0], a[1], voice="--no-voice" not in a, only=only)
    else:
        print(__doc__)
