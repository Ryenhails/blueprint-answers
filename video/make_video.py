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
    ("task", "A construction drawing, and a question. The answer has to come with proof. Drawings come in five kinds: vector plans, building models, schedules, "
             "floor-plan images, and scans. Questions in four: count, measure, look up, and combine."),
    ("motive", "We first gave Opus five point five just the image and the question. Thirty-six of fifty. It understood every drawing. It miscounted and mismeasured."),
    ("facts", "So our idea starts with a fact table. Every room, door, window and fixture on the drawing becomes one row, with its size and its place on the sheet."),
    ("lanes", "Each drawing is read into this table once, by a reader chosen for its kind. Vector files are parsed exactly. Images and scans are read by models and measured by code."),
    ("ladder", "Then a question takes the cheapest step that can answer it. A known form costs nothing. A small model routes new wording. A larger model writes code on the table. "
               "Only if the table falls short does a model read the sheet."),
    ("case", "Take one question: how much skirting board for the bedrooms and living room of apartment A, minus the doors? It is new, so the router marks it complex. "
             "The model writes a few lines of code on the fact table. The code runs, and every room and door it used is drawn on the plan."),
    ("scan", "A scan gets the same table. A model lists the rooms and points at each one. Code measures them."),
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
    size = 54
    while size > 30 and d.textlength(title, font=font(size, True)) > W - 180:
        size -= 2
    d.text((90, 96), title, font=font(size, True), fill=INK)


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
D = {}  # video_data.json (prepare_data.py)


def _img(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


def fit(im, box):
    """Paste size and offset to fit im inside box (x0, y0, x1, y1), centred."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    k = min(bw / im.width, bh / im.height)
    w, h = max(1, int(im.width * k)), max(1, int(im.height * k))
    return im.resize((w, h), Image.BILINEAR), (box[0] + (bw - w) // 2, box[1] + (bh - h) // 2)


def scene_task(st):
    thumbs = [(_img(k["image"]), k["name"]) for k in D["kinds"]]
    cols = {n: c for n, c, _ in ROUTES}
    qs = [("Count", "How many doors are on this floor?"), ("Measure", "How large is apartment A?"), ("Look up", "What type is door 121?"),
          ("Combine", "Skirting board for the bedrooms, minus the doors?")]

    def f(t, T):
        im, d = frame()
        header(d, "The task", "A drawing and a question in. An answer with proof out.")
        d.text((90, 196), "FIVE KINDS OF DRAWINGS", font=font(22, True), fill=MUTED)
        cw = (W - 180 - 4 * 24) // 5
        for i, (th, name) in enumerate(thumbs):
            at = 1.2 + i * 0.9
            if t < at:
                continue
            x = 90 + i * (cw + 24)
            panel(d, [x, 236, x + cw, 236 + 330])
            p, off = fit(th, (x + 10, 246, x + cw - 10, 500))
            im.paste(p, off)
            col = cols.get(name, cols.get("Scanned sheet"))
            d.rectangle([x, 512, x + cw, 566], fill=col)
            d.text((x + 16, 524), name, font=font(25, True), fill=(255, 255, 255))
        if t > T * 0.55:
            d.text((90, 610), "FOUR KINDS OF QUESTIONS", font=font(22, True), fill=MUTED)
            qw = (W - 180 - 3 * 24) // 4
            for i, (kind, q) in enumerate(qs):
                at = T * 0.55 + 0.4 + i * 0.8
                if t < at:
                    continue
                x = 90 + i * (qw + 24)
                panel(d, [x, 650, x + qw, 840])
                d.text((x + 22, 668), kind, font=font(34, True), fill=BLUE)
                para(d, (x + 22, 722), "“%s”" % q, font(25), qw - 44)
        return im

    return f


def scene_motive(st):
    b = D["baseline"]
    pic = _img(b["image"])

    def f(t, T):
        im, d = frame()
        header(d, "Why not just ask a frontier model?", "Opus 5.5, given only the image and the question")
        panel(d, [90, 200, 980, 950])
        p, off = fit(pic, (110, 220, 960, 930))
        im.paste(p, off)
        x = 1040
        if t > 1.0:
            k = ease((t - 1.0) * 1.5)
            panel(d, [x, 200, W - 90, 480])
            d.text((x + 36, 222), "50 DATASET QUESTIONS", font=font(22, True), fill=MUTED)
            d.text((x + 36, 262), "%d / %d" % tuple(b["score"]), font=font(150, True, mono=True), fill=mix(PANEL, RED, k))
        if t > T * 0.4:
            panel(d, [x, 510, W - 90, 800])
            para(d, (x + 36, 530), "“%s”" % b["question"], font(28), W - 90 - x - 72)
            d.text((x + 36, 640), "Opus 5.5", font=font(26), fill=MUTED)
            d.text((x + 36, 676), str(b["model"]), font=font(84, True, mono=True), fill=RED)
            d.text((x + 400, 640), "on the drawing", font=font(26), fill=MUTED)
            d.text((x + 400, 676), str(b["gold"]), font=font(84, True, mono=True), fill=GREEN)
        if t > T * 0.7:
            d.text((x, 840), "It understood the drawing.", font=font(38), fill=INK)
            d.text((x, 892), "It miscounted and mismeasured.", font=font(38, True), fill=RED)
        return im

    return f


def _plan(st):
    sh = Sheet(*sheet_image(st, D["plan"]["path"]))
    return sh, D["plan"]


def scene_facts(st):
    sh, P = _plan(st)
    rows = [r for r in P["rooms"] if r["apartment"] and r["poly"]][:9]
    xs = [p[0] for r in rows for p in r["poly"]]
    ys = [p[1] for r in rows for p in r["poly"]]
    focus = sh.around([min(xs), min(ys), max(xs), max(ys)], 0.12, 1000 / 740)

    def f(t, T):
        im, d = frame()
        header(d, "The core idea", "A fact table: every room is one row, with its place on the sheet")
        n = min(len(rows), int(max(0, t - 1.2) / (T * 0.62) * len(rows)) + (1 if t > 1.2 else 0))
        shapes = [("poly", r["poly"], BLUE, 0.42 if i == n - 1 else 0.14, 5 if i == n - 1 else 3) for i, r in enumerate(rows[:n])]
        objs = t > T * 0.74
        if objs:
            for b in P["doors"].values():
                shapes.append(("box", [(b[0], b[1]), (b[2], b[3])], ORANGE, 0.25, 3))
            for b in P["windows"].values():
                shapes.append(("box", [(b[0], b[1]), (b[2], b[3])], GREEN, 0.25, 3))
        box = win(sh.full(), focus, ease(t / 1.2))
        im.paste(sh.view(box, (1000, 740), shapes), (90, 200))
        d.rectangle([90, 200, 1090, 940], outline=LINE, width=2)
        x0, x1 = 1130, W - 90
        panel(d, [x0, 200, x1, 940])
        cols = [("room", 0), ("label", 110), ("m²", 330), ("apt", 440), ("doors", 520), ("win.", 620)]
        for name, dx in cols:
            d.text((x0 + 24 + dx, 218), name.upper(), font=font(18, True), fill=MUTED)
        d.line([x0 + 16, 250, x1 - 16, 250], fill=LINE, width=2)
        y = 262
        for i, r in enumerate(rows[:n]):
            if i == n - 1:
                d.rectangle([x0 + 8, y - 4, x1 - 8, y + 44], fill=(219, 232, 250))
            vals = [r["number"], (r["label"] or "")[:13], "%.2f" % r["area_m2"], r["apartment"], str(r["doors"]), str(r["windows"])]
            for (name, dx), v in zip(cols, vals):
                d.text((x0 + 24 + dx, y + 6), v, font=font(23, mono=name != "label"), fill=INK)
            y += 52
        if objs:
            c = P["counts"]
            d.line([x0 + 16, 800, x1 - 16, 800], fill=LINE, width=2)
            para(d, (x0 + 24, 816), "The same for every object:", font(23), x1 - x0 - 48, fill=MUTED)
            d.text((x0 + 24, 852), "%d rooms · %d doors · %d windows · %d fixtures" % (c["rooms"], c["doors"], c["windows"], c["fixtures"]), font=font(24, True), fill=INK)
        return im

    return f


READ = ["code parses the PDF's own lines", "code lists every element of the model", "code lifts the tables from the text layer", "three readers count, the middle one wins",
        "a model lists the rooms, code measures them"]
RUNGS = [("Known question form", "$0"), ("Small model routes it", "$0.0004"), ("Model writes code on the table", "≈ 1¢"), ("Model reads the sheet", "1–20¢")]


def architecture(d, lanes, rungs, done):
    """The whole system, drawn as far as the film has got. lanes: 0-5 readers shown; rungs: 0-4; done: output shown."""
    d.text((90, 196), "READ ONCE", font=font(22, True), fill=MUTED)
    lh = 62
    for i, (name, col, _) in enumerate(ROUTES[:lanes]):
        y = 232 + i * (lh + 10)
        d.rounded_rectangle([90, y, 520, y + lh], 8, fill=mix(col, (255, 255, 255), 0.86), outline=col, width=3)
        d.text((108, y + 6), name, font=font(24, True), fill=col)
        d.text((108, y + 34), READ[i], font=font(18), fill=INK)
        d.line([520, y + lh // 2, 640, y + lh // 2, 640, 412, 700, 412], fill=col, width=4)
    if lanes:
        d.polygon([(700, 400), (724, 412), (700, 424)], fill=INK)
        d.rounded_rectangle([724, 300, 1130, 524], 10, fill=(20, 32, 46))
        d.text((752, 320), "fact table", font=font(40, True, mono=True), fill=(238, 241, 244))
        para(d, (752, 380), "rooms · doors · windows · fixtures · elements · schedule rows", font(22), 350, fill=(207, 214, 222))
        d.text((752, 470), "each with size and place on the sheet", font=font(19), fill=(160, 190, 235))
    if rungs:
        d.text((90, 624), "EVERY QUESTION: CHEAPEST STEP THAT CAN ANSWER", font=font(22, True), fill=MUTED)
        rw = 296
        for i, (name, cost) in enumerate(RUNGS[:rungs]):
            x = 90 + i * (rw + 14)
            on = i == rungs - 1 and not done
            d.rounded_rectangle([x, 660, x + rw, 800], 8, fill=(20, 32, 46) if on else PANEL, outline=LINE, width=2)
            d.text((x + 16, 672), str(i), font=font(30, True, mono=True), fill=(160, 190, 235) if on else BLUE)
            para(d, (x + 50, 674), name, font(23, True), rw - 64, fill=(238, 241, 244) if on else INK)
            d.text((x + 16, 760), cost, font=font(22, mono=True), fill=(120, 200, 160) if on else GREEN)
        d.line([927, 524, 927, 620], fill=INK, width=4)  # the table feeds the steps
        d.polygon([(915, 620), (939, 620), (927, 644)], fill=INK)
    if done:
        d.line([90 + 4 * 310 - 14, 730, 1420, 730], fill=INK, width=4)
        d.polygon([(1420, 718), (1444, 730), (1420, 742)], fill=INK)
        d.rounded_rectangle([1444, 620, W - 90, 840], 10, fill=PANEL, outline=BLUE, width=4)
        d.text((1470, 636), "ANSWER", font=font(22, True), fill=MUTED)
        for i, s in enumerate(["computed by code", "evidence on the sheet", "every step and its cost"]):
            d.text((1470, 676 + i * 46), "· " + s, font=font(25), fill=INK)
        d.rounded_rectangle([1180, 300, W - 90, 524], 10, fill=PANEL, outline=LINE, width=2)
        d.text((1206, 318), "THE SPLIT", font=font(22, True), fill=MUTED)
        d.text((1206, 356), "Models decide and point.", font=font(30, True), fill=INK)
        para(d, (1206, 402), "Code counts, sums and measures.", font(30, True), 600, fill=BLUE)


def scene_lanes(st):
    def f(t, T):
        im, d = frame()
        header(d, "The core idea", "Each drawing is read into the table once")
        architecture(d, min(5, int(t / (T * 0.8) * 5) + 1), 0, False)
        return im

    return f


def scene_ladder(st):
    def f(t, T):
        im, d = frame()
        header(d, "The core idea", "Then every question takes the cheapest step that can answer it")
        r = min(4, int(t / (T * 0.72) * 4) + 1)
        architecture(d, 5, r, t > T * 0.78)
        return im

    return f


def scene_case(st):
    ex = next(e for e in st["examples"] if e["path"] == D["plan"]["path"])
    q = next(x["q"] for x in ex["questions"] if "skirting" in x["q"])
    a = st["answers"][ex["path"] + "\n" + q]
    sh, P = _plan(st)
    by = {r["id"]: r for r in P["rooms"]}
    used = [by[e["id"]] for e in a["evidence"] if e.get("id") in by]
    doors = [e for e in a["evidence"] if e.get("kind") == "door"]
    code = [l for l in (a.get("calc") or "").split("fact tables:\n")[-1].strip().split("\n") if l.strip()]
    xs = [p[0] for r in used for p in r["poly"]]
    ys = [p[1] for r in used for p in r["poly"]]
    focus = sh.around([min(xs), min(ys), max(xs), max(ys)], 0.3, 900 / 740)
    per, nd = sum(r["perimeter_m"] for r in used), len(doors)

    def f(t, T):
        im, d = frame()
        header(d, "One question, step by step", "“%s”" % (q if len(q) < 74 else q[:72] + "…"))
        s = [T * 0.14, T * 0.34, T * 0.6, T * 0.8]  # when steps 2, 3, 4 and the answer start
        shapes = []
        if t > s[0]:
            shapes += [("poly", r["poly"], BLUE, 0.3, 4) for r in used]
        if t > s[2]:
            shapes += [("box", [(e["bbox"][0], e["bbox"][1]), (e["bbox"][2], e["bbox"][3])], ORANGE, 0.35, 4) for e in doors]
        im.paste(sh.view(win(sh.full(), focus, ease(t / s[0])), (900, 740), shapes), (90, 200))
        d.rectangle([90, 200, 990, 940], outline=LINE, width=2)
        x0, x1 = 1030, W - 90
        panel(d, [x0, 200, x1, 940])
        y = 218
        d.text((x0 + 24, y), "1  ROUTE", font=font(20, True), fill=BLUE)
        d.text((x0 + 24, y + 30), "not a known form  →  small model: “complex”  →  write code", font=font(22), fill=INK)
        y += 78
        if t > s[0]:
            d.text((x0 + 24, y), "2  ROWS OF THE FACT TABLE THE CODE WILL USE", font=font(20, True), fill=BLUE)
            y += 32
            for r in used[:5]:
                d.text((x0 + 24, y), "%-5s %-14s perimeter %6.2f m   doors %d" % (r["number"], (r["label"] or "")[:14], r["perimeter_m"], r["doors"]), font=font(21, mono=True), fill=INK)
                y += 30
            y += 16
        if t > s[1]:
            d.text((x0 + 24, y), "3  CODE WRITTEN BY THE MODEL, RUN IN A SANDBOX", font=font(20, True), fill=BLUE)
            y += 32
            n = int(ease((t - s[1]) / (s[2] - s[1])) * min(len(code), 9))
            for ln in code[:n]:
                d.text((x0 + 24, y), ln[:62], font=font(19, mono=True), fill=INK)
                y += 26
            y = max(y, 218 + 78 + 32 + 30 * min(5, len(used)) + 16 + 32 + 26 * 9) + 14
        if t > s[2]:
            d.text((x0 + 24, y), "4  THE CODE'S ARITHMETIC", font=font(20, True), fill=BLUE)
            d.text((x0 + 24, y + 32), "%.2f m of wall  −  %d doors × 0.9 m" % (per, nd), font=font(26, mono=True), fill=INK)
        if t > s[3]:
            d.text((x0 + 24, 822), "%s m" % a["answer"], font=font(68, True, mono=True), fill=INK)
            d.text((x0 + 24, 902), "every room and door used is on the plan · API cost $%.3f" % a.get("cost_usd", 0), font=font(22), fill=MUTED)
        return im

    return f


def scene_scan(st):
    S = D["scan"]
    pic = _img(S["image"])
    sh = Sheet(pic, [0, 0, S["size"][0], S["size"][1]])
    rooms = [r for r in S["rooms"] if r["poly"] and len(r["poly"]) >= 3]
    xs = [v for r in rooms for v in (r["bbox"][0], r["bbox"][2])]
    ys = [v for r in rooms for v in (r["bbox"][1], r["bbox"][3])]
    focus = sh.around([min(xs), min(ys), max(xs), max(ys)], 0.03, 1120 / 740)
    pal = [(20, 99, 200), (11, 122, 83), (217, 72, 15), (122, 63, 176), (168, 93, 0), (31, 122, 77)]
    order = sorted(range(len(rooms)), key=lambda i: (rooms[i]["bbox"][0] // 800, rooms[i]["bbox"][1]))

    def f(t, T):
        im, d = frame()
        header(d, "The same table from a scan", "A model lists the rooms and points; code measures each one")
        n = int(ease((t - 0.8) / (T * 0.7)) * len(order)) if t > 0.8 else 0
        shapes = [("poly", rooms[i]["poly"], pal[k % len(pal)], 0.38, 3) for k, i in enumerate(order[:n])]
        if t > T * 0.75:
            shapes += [("box", [(o["bbox"][0], o["bbox"][1]), (o["bbox"][2], o["bbox"][3])], (20, 32, 46), 0.0, 2) for o in S["objects"] if o["kind"] in ("door", "window")]
        im.paste(sh.view(focus, (1120, 740), shapes), (90, 200))
        d.rectangle([90, 200, 1210, 940], outline=LINE, width=2)
        x0, x1 = 1250, W - 90
        panel(d, [x0, 200, x1, 940])
        d.text((x0 + 24, 218), "ROOM", font=font(18, True), fill=MUTED)
        d.text((x1 - 250, 218), "M²", font=font(18, True), fill=MUTED)
        d.text((x1 - 130, 218), "PERIM. M", font=font(18, True), fill=MUTED)
        d.line([x0 + 16, 250, x1 - 16, 250], fill=LINE, width=2)
        y = 262
        for k, i in list(enumerate(order[:n]))[-12:]:
            r = rooms[i]
            d.rectangle([x0 + 24, y + 6, x0 + 44, y + 26], fill=pal[k % len(pal)])
            d.text((x0 + 56, y + 2), str(r["label"]).replace("\n", " ")[:20], font=font(21), fill=INK)
            d.text((x1 - 250, y + 2), "%.1f" % (r["area_m2"] or 0), font=font(21, mono=True), fill=INK)
            d.text((x1 - 130, y + 2), "%.1f" % (r["perimeter_m"] or 0), font=font(21, mono=True), fill=INK)
            y += 40
        if t > T * 0.75:
            c = S["counts"]
            d.line([x0 + 16, 800, x1 - 16, 800], fill=LINE, width=2)
            d.text((x0 + 24, 816), "%d rooms · %d doors · %d windows" % (c["rooms"], c["doors"], c["windows"]), font=font(26, True), fill=INK)
            para(d, (x0 + 24, 858), "read once for $%.2f; the scale was checked against the scale bar" % S["usd"], font(21), x1 - x0 - 48, fill=MUTED)
        return im

    return f


def scene_valid(st):
    v = st.get("validation") or {}
    groups = [g for g in v.get("groups", []) if g["key"] != "schedules"]
    unseen = sum(g["n"] for g in groups)

    def f(t, T):
        im, d = frame()
        header(d, "Evaluation", "%d questions the system was not built on" % unseen)
        steps = ["{:,} questions in the dataset".format(D["sample"]["pool"]), "random sample, fixed seed, by drawing kind, question type, language",
                 "{:,} never used while building".format(unseen)]
        x = 90
        for i, s in enumerate(steps):
            if t < i * 1.2:
                break
            fw = font(21)
            w = d.textlength(s, font=fw) + 36
            d.rounded_rectangle([x, 196, x + w, 244], 8, fill=PANEL, outline=LINE, width=2)
            d.text((x + 18, 206), s, font=fw, fill=INK)
            x += w + 14
            if i < 2:
                d.text((x, 204), "→", font=font(26, True), fill=MUTED)
                x += 44
        y = 290
        for i, g in enumerate(groups):
            if t < 3.0 + i * 1.0:
                break
            k = ease((t - 3.0 - i * 1.0) * 1.6)
            d.text((90, y + 8), g["label"], font=font(27, True), fill=INK)
            x0, x1 = 600, 1250
            d.rectangle([x0, y + 4, x1, y + 48], fill=PANEL, outline=LINE, width=2)
            tx = g.get("text") or {}
            nt = sum(tx.get(a, 0) for a in ("correct", "partial", "wrong"))
            if g["numeric"]:
                share = g["ok"] / g["numeric"]
                d.rectangle([x0, y + 4, x0 + (x1 - x0) * share * k, y + 48], fill=GREEN)
                lab = "%d / %d  (%.1f%%)" % (g["ok"], g["numeric"], 100 * share)
            else:
                c, p = tx.get("correct", 0) / max(1, nt), tx.get("partial", 0) / max(1, nt)
                d.rectangle([x0, y + 4, x0 + (x1 - x0) * c * k, y + 48], fill=GREEN)
                d.rectangle([x0 + (x1 - x0) * c * k, y + 4, x0 + (x1 - x0) * (c + p) * k, y + 48], fill=(200, 140, 20))
                lab = "%d right, %d partly / %d" % (tx.get("correct", 0), tx.get("partial", 0), nt)
            d.text((x1 + 20, y + 12), lab, font=font(22, mono=True), fill=INK)
            d.text((W - 90 - 130, y + 12), "$%.3f / q" % g["usd"], font=font(22, mono=True), fill=MUTED)
            y += 78
        al = v.get("all", {"ok": 0, "numeric": 1, "usd": 0})
        if t > 3.0 + len(groups) * 1.0:
            tiles = [("%.1f%%" % (100 * al["ok"] / max(1, al["numeric"])), "of %d numeric answers correct" % al["numeric"]),
                     ("%.1f¢" % (al["usd"] * 100), "average API cost per question"), ("$%.0f" % v.get("usd", 0), "for the whole evaluation run"),
                     ("72%", "Opus 5.5 on image + question (50 other questions)")]
            for i, (big, small) in enumerate(tiles):
                x = 90 + i * 440
                panel(d, [x, 720, x + 416, 900])
                d.text((x + 24, 736), big, font=font(72, True, mono=True), fill=INK if i < 3 else MUTED)
                para(d, (x + 24, 838), small, font(20), 370, fill=MUTED)
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


SCENES = {"task": scene_task, "motive": scene_motive, "facts": scene_facts, "lanes": scene_lanes, "ladder": scene_ladder, "case": scene_case, "scan": scene_scan,
          "valid": scene_valid, "close": scene_close}


def validation_line(st):
    v = st.get("validation") or {}
    al = v.get("all", {"ok": 0, "numeric": 1, "usd": 0})
    g = {x["key"]: x for x in v.get("groups", [])}

    def pc(k):
        return "%d percent" % round(100 * g[k]["ok"] / max(1, g[k]["numeric"])) if k in g and g[k]["numeric"] else "unknown"

    cents = al["usd"] * 100
    return ("We test on %d questions the system was not built on, drawn at random across every kind of drawing and eight languages. "
            "%d percent of the numeric answers are correct, at about %s a question. Vector plans and building models are near perfect. Scans are the hardest, at %s."
            % (sum(x["n"] for x in v.get("groups", []) if x["key"] != "schedules"), round(100 * al["ok"] / max(1, al["numeric"])),
               ("%.1f cents" % cents) if cents >= 0.1 else "a tenth of a cent", pc("scans")))


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
    D.update(json.load(open(os.path.join(out_dir, "video_data.json"))))
    os.makedirs(os.path.join(out_dir, "voice"), exist_ok=True)
    script = [(k, t if t is not None else validation_line(st)) for k, t in SCRIPT if (only is None or k in only) and (k != "scan" or "scan" in D)]
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
    """The pitch deck: the film's scenes at their last frame, plus cost and limits."""
    st = load_static(static_path)
    D.update(json.load(open(os.path.join(os.path.dirname(out_pdf), "video_data.json"))))
    slides = [SCENES[k](st)(1000.0, 1000.0).convert("RGB") for k, _ in SCRIPT if k != "close" and (k != "scan" or "scan" in D)]
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
