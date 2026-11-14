"""Step 4 adapter: schedule sheets (door / window / finish schedules) -> tables with cell coordinates,
answered by a model that writes a few lines of Python over the tables; the code is executed, so
counting and summing are exact and the rows it used become the evidence.

  python table_adapter.py show <pdf>              print the extracted tables
  python table_adapter.py test <model_id>         answer the opening_design questions (GPU node)
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evalkit  # noqa: E402


def parse(pdf_path, page_no=0):
    """-> {"tables": [{"id", "bbox", "rows": [[str]], "row_bboxes": [bbox]}]} ; split header tables are merged."""
    import fitz

    page = fitz.open(pdf_path)[page_no]
    raw = []
    for t in page.find_tables().tables:
        rows, rb = [], []
        for cells, trow in zip(t.extract(), t.rows):
            cells = [(c or "").strip() for c in cells]
            # the table detector sometimes fuses two or more sheet rows into one (every cell holds k lines): split them
            counts = [len(c.split("\n")) for c in cells if c]
            k = max(set(counts), key=counts.count) if counts else 1
            if k >= 2 and len(counts) >= 3 and counts.count(k) / len(counts) >= 0.6:
                x0, y0, x1, y1 = trow.bbox
                for i in range(k):
                    rows.append([(c.split("\n")[i] if len(c.split("\n")) == k else c.replace("\n", " ")).strip() for c in cells])
                    rb.append([x0, y0 + (y1 - y0) * i / k, x1, y0 + (y1 - y0) * (i + 1) / k])
            else:
                rows.append([c.replace("\n", " ") for c in cells])
                rb.append(list(trow.bbox))
        keep = [i for i, r in enumerate(rows) if any(r)]
        if len(keep) == 0:
            continue
        raw.append({"bbox": list(t.bbox), "rows": [rows[i] for i in keep], "row_bboxes": [rb[i] for i in keep], "ncol": t.col_count})
    raw.sort(key=lambda t: (round(t["bbox"][0] / 50), t["bbox"][1]))
    merged = []
    for t in raw:
        p = merged[-1] if merged else None
        # a short table sitting right above one with the same columns is that table's header
        if p and p["ncol"] == t["ncol"] and len(p["rows"]) <= 3 and abs(p["bbox"][0] - t["bbox"][0]) < 5 and 0 <= t["bbox"][1] - p["bbox"][3] < 40:
            p["rows"] += t["rows"]
            p["row_bboxes"] += t["row_bboxes"]
            p["bbox"] = [p["bbox"][0], p["bbox"][1], max(p["bbox"][2], t["bbox"][2]), t["bbox"][3]]
        else:
            merged.append(t)
    out = []
    for t in merged:
        cells = sum(1 for r in t["rows"] for c in r if c)
        if cells < 2:
            continue
        t["id"] = len(out)
        out.append(t)
    return {"tables": out, "page_rect": list(page.rect)}


def structure(t):
    """Raw rows -> (title, column names, data rows as dicts, original row index of each data row).
    Handles a title row and one- or two-level headers (a parent header cell spans the empty cells after it)."""
    rows = t["rows"]
    i, title = 0, ""
    if sum(1 for c in rows[0] if c) == 1 and len(rows) > 1:
        title, i = next(c for c in rows[0] if c), 1
    if i >= len(rows):
        return title, [], [], []
    h1 = rows[i]
    h2 = rows[i + 1] if i + 1 < len(rows) else None
    # a second header row: first cell empty, some labels, and no digits (data rows carry numbers)
    two = bool(h2) and not h2[0] and bool(h1[0]) and any(h2) and not any(re.search(r"\d", c) for c in h2)
    cols, parent = [], ""
    for k, a in enumerate(h1):
        sub = h2[k] if two else ""
        if a:
            parent = a
        name = ("%s %s" % (parent, sub)).strip() if (two and sub) else (a or ("%s (%d)" % (parent, k) if parent and not two else "col%d" % k))
        while name in cols:
            name += "_"
        cols.append(name)
    start = i + (2 if two else 1)
    data, src = [], []
    for j in range(start, len(rows)):
        if any(rows[j]):
            data.append(dict(zip(cols, rows[j])))
            src.append(j)
    return title, cols, data, src


def structured(facts):
    out = []
    for t in facts["tables"]:
        title, cols, data, src = structure(t)
        out.append({"title": title, "columns": cols, "rows": data, "src": src})
    return out


def render_tables(facts, max_chars=26000):
    """Tables as text for the model: column names once, then one line per data row."""
    st = structured(facts)
    parts, used = [], 0
    for i in sorted(range(len(st)), key=lambda i: -len(st[i]["rows"])):
        t = st[i]
        if not t["rows"]:
            continue
        lines = ["tables[%d]  title: %r  (%d rows)" % (i, t["title"], len(t["rows"])), "columns: " + json.dumps(t["columns"], ensure_ascii=False)]
        for j, r in enumerate(t["rows"]):
            lines.append("%d: %s" % (j, " | ".join(r[c] for c in t["columns"])))
        s = "\n".join(lines)
        if used + len(s) > max_chars:
            continue
        parts.append((i, s))
        used += len(s)
    return "\n\n".join(s for _, s in sorted(parts))


SYSTEM = """You answer questions about a construction drawing sheet whose schedules (door, window, finish schedules, keys and legends) have been extracted as tables.

tables[i]["columns"] is the list of column names; tables[i]["rows"] is a list of dicts, one per data row, mapping column name -> cell string. Header and title rows are already removed. Cells can be empty strings.

Write Python code that computes the answer from `tables`.
Rules:
- Output only one Python code block. Short comments are fine. No other text.
- Set `answer`: a number for "how many" and area questions, otherwise a short string taken from the table cells (add the meaning from a key / legend table when the question asks what a code means).
- Set `rows` to a list of (table_index, row_index) pairs for every data row the answer is based on.
- Helpers available: num(s) -> first number in a string as float (0.0 if none), e.g. num("741 SF") == 741.0;
  Cell text compares without regard to case and spaces, so r["Type Mark"] == "u01" matches "U01" and "sliding" in r["Type"] works.
  norm(s) -> lower-case with all spaces removed, e.g. norm('3\\' - 0"') == norm('3\\'-0"'). Write feet-and-inch literals in single quotes with the apostrophe escaped, as in that example. Always compare cell text with norm(...) or use `in` on norm(...) values, because spacing in the sheet differs from the question.
- A "Count" column means the row stands for that many items: add up num(r["Count"]) instead of counting rows.
- A sheet can hold several similar schedules (for example one row per door, and another one per door TYPE with a Count column). Do not assume a table index: loop over all tables and use those whose "columns" contain the columns you need.
- Words in parentheses in the question only explain a code (for example "type B2 (double aluminium door)"): filter on the code itself, not on the explanation.
- The tables hold only what is written in schedules, keys and notes. If the question is about things drawn on the sheet (symbols on a plan, an elevation, a detail) or no table covers it, set answer = None. A count of 0 is only right when a table that covers the question has no matching row.
- `re` is available. Do not import anything, do not read files, do not print."""

SHOT_Q = """tables[0]  title: 'Lighting Schedule - by room'  (2 rows)
columns: ["Room", "Mark", "Switch"]
0: OFFICE | L1 | S1
1: LOBBY | L2 | S2

tables[1]  title: 'Lighting Schedule - by type'  (3 rows)
columns: ["Mark", "Room", "Count", "Fixture Type", "Fixture Size"]
0: L1 | OFFICE | 4 | RECESSED | 2' - 0"
1: L2 | LOBBY | 6 | PENDANT | 1' - 6"
2: L1 | STORAGE | 2 | RECESSED | 2' - 0"

Question: How many 2'-0" recessed fixtures are scheduled in total?"""
SHOT_A = """```python
rows = []
total = 0
for ti, t in enumerate(tables):
    if "Count" in t["columns"] and "Fixture Type" in t["columns"]:
        for j, r in enumerate(t["rows"]):
            if r["Fixture Size"] == '2\\'-0"' and "recessed" in r["Fixture Type"]:
                total += num(r["Count"])
                rows.append((ti, j))
answer = total
```"""


def messages(facts, question):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": SHOT_Q},
        {"role": "assistant", "content": SHOT_A},
        {"role": "user", "content": render_tables(facts) + "\n\nQuestion: " + question},
    ]


SAFE = {"len": len, "sum": sum, "min": min, "max": max, "int": int, "float": float, "str": str, "range": range, "enumerate": enumerate,
        "sorted": sorted, "set": set, "list": list, "dict": dict, "tuple": tuple, "any": any, "all": all, "abs": abs, "round": round,
        "zip": zip, "bool": bool, "KeyError": KeyError, "IndexError": IndexError, "TypeError": TypeError, "isinstance": isinstance, "ValueError": ValueError, "Exception": Exception, "map": map, "filter": filter}


def _num(x):
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group(0)) if m else 0.0


def _n(x):
    return re.sub(r"\s+", "", str(x)).lower()


class Cell(str):
    """A cell string that compares without regard to case and spacing, so that code written by the model
    as r["Width"] == "3'-0\"" or "sliding" in r["Type"] matches the sheet's "3' - 0\"" / "SLIDING"."""

    def __eq__(self, o):
        return _n(self) == _n(o) if isinstance(o, str) else False

    def __ne__(self, o):
        return not self.__eq__(o)

    def __hash__(self):
        return hash(_n(self))

    def __contains__(self, item):
        return _n(item) in _n(self)

    def startswith(self, p, *a):
        return _n(self).startswith(tuple(_n(x) for x in p) if isinstance(p, tuple) else _n(p))

    def endswith(self, p, *a):
        return _n(self).endswith(tuple(_n(x) for x in p) if isinstance(p, tuple) else _n(p))


def _norm(x):
    return Cell(_n(x))


def run_code(text, facts):
    """Execute the model's code with a small whitelist of builtins. -> (answer, rows, error)
    Returned rows are (table index, ORIGINAL row index) so they map to cell coordinates."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    if re.search(r"\bimport\b|__|open\(|exec\(|eval\(|os\.|sys\.", code):
        return None, [], "rejected: code uses a forbidden construct"
    st = structured(facts)
    env = {"__builtins__": SAFE, "re": re, "num": _num, "norm": _norm,
           "tables": [{"title": t["title"], "columns": t["columns"], "rows": [{k: Cell(v) for k, v in r.items()} for r in t["rows"]]} for t in st]}
    try:
        exec(code, env)  # noqa: S102 - whitelisted builtins only; prototype
    except Exception as e:
        return None, [], "error: %r" % (e,)
    rows = []
    for x in env.get("rows", []) or []:
        try:
            ti, ri = int(x[0]), int(x[1])
            rows.append((ti, st[ti]["src"][ri]))
        except Exception:
            pass
    ans = env.get("answer")
    if isinstance(ans, float) and ans == int(ans):
        ans = int(ans)
    return ans, rows, None


def _feedback(code, facts, err, ans):
    """What the model gets back after a failed or empty first attempt: the error, or the real values of the
    columns it filtered on, so it can see why nothing matched."""
    if err:
        return "Running that code failed with: %s\nReturn corrected code only." % err
    st = structured(facts)
    cols = list(dict.fromkeys(re.findall(r"""r\[["']([^"']+)["']\]""", code)))
    lines = []
    for i, t in enumerate(st):
        for c in cols:
            if c in t["columns"]:
                vals = list(dict.fromkeys(r[c] for r in t["rows"] if r[c]))
                lines.append("tables[%d] (%r) column %r has values: %s" % (i, t["title"], c, json.dumps(vals[:40], ensure_ascii=False)))
    return ("That code matched no rows (answer = %r). Look at the real values and at which table is the right one:\n%s\n"
            "Filter only on what the question requires: a code such as a type mark is enough on its own, and words in parentheses "
            "are explanations, not extra conditions. Return corrected code only." % (ans, "\n".join(lines[:14])))


def answer_with(model, facts, question, max_new_tokens=450):
    """First attempt; if the code crashed or matched nothing, one informed second attempt.
    -> (answer, rows, error, code, tokens)"""
    msgs = messages(facts, question)
    texts, n_in, n_out = model.generate([msgs], max_new_tokens=max_new_tokens)
    ans, rows, err = run_code(texts[0], facts)
    tokens = n_in + n_out
    empty = ans in (None, 0, "", "0") or (isinstance(ans, (list, tuple)) and not any(ans))
    if err or empty:
        msgs = msgs + [{"role": "assistant", "content": texts[0]}, {"role": "user", "content": _feedback(texts[0], facts, err, ans)}]
        t2, a, b = model.generate([msgs], max_new_tokens=max_new_tokens)
        tokens += a + b
        ans2, rows2, err2 = run_code(t2[0], facts)
        if not err2 and (err or ans2 not in (None, 0, "", "0")):
            return ans2, rows2, None, t2[0], tokens
    return ans, rows, err, texts[0], tokens


def evidence(facts, rows):
    ev = []
    for ti, ri in rows:
        try:
            t = facts["tables"][ti]
            ev.append({"kind": "table_row", "name": " | ".join(c for c in t["rows"][ri] if c)[:80], "bbox": t["row_bboxes"][ri]})
        except Exception:
            pass
    return ev


_PARSE = __import__("threading").Lock()


def make_table_llm(model_id):
    """-> callable(full_pdf_path, question) -> result dict for system.ask(..., table_llm=...)"""
    import llm_plan

    model = llm_plan.Model(model_id)
    cache = {}

    def f(full, question):
        with _PARSE:  # PyMuPDF is not safe to use from several threads at once
            if full not in cache:
                cache[full] = parse(full)
        if not cache[full].get("tables"):  # a sheet without schedules: nothing to compute on, the caller reads the sheet
            return {"answer": None, "unit": None, "evidence": [], "plan": {"op": "no tables on this sheet"}, "plan_source": None, "llm_tokens": 0, "cost_usd": 0.0}
        before = model.usd
        ans, rows, err, code, tokens = answer_with(model, cache[full], question)
        res = {"cost_usd": round(model.usd - before, 6), "answer": ans, "unit": None, "evidence": evidence(cache[full], rows), "plan": {"op": "generated code"}, "plan_source": "code",
               "llm_tokens": tokens, "calc": "The model wrote this code and it was run on the extracted tables:\n" + code.strip()[:900]}
        if err:
            res["note"] = err
        return res

    return f


def test(model_ids):
    import llm_plan

    rows = [r for r in evalkit.load_questions() if evalkit.source(r["path"]) == "opening_design"]
    facts = {}
    for r in rows:
        if r["path"] not in facts:
            facts[r["path"]] = parse(os.path.join(evalkit.DATA, r["path"]))
            print(os.path.basename(r["path"]), "tables:", [(t["id"], len(t["rows"]), t["ncol"]) for t in facts[r["path"]]["tables"]][:12])
    for mid in model_ids:
        t0 = time.time()
        model = llm_plan.Model(mid)
        print("\n=== %s | load %.0fs" % (mid, time.time() - t0), flush=True)
        t0 = time.time()
        n_in = n_out = 0
        num = [0, 0]
        text_rows = []
        for r in rows:
            ans, used, err, code, tk = answer_with(model, facts[r["path"]], r["question"])
            n_in += tk
            q = r["question"].split("drawings")[-1][:95]
            if r["type"] == "text":
                text_rows.append((q, r["answer"], ans, err))
                continue
            ok = ans is not None and evalkit.is_correct(ans, r["answer"], r["type"])
            num[0] += ok
            num[1] += 1
            if not ok:
                print("   MISS gold=%s got=%s %s | %s" % (r["answer"], ans, err or "", q), flush=True)
        dt = time.time() - t0
        print("numeric: %d / %d | %.1fs per question | tokens %d | API cost $%.4f" % (num[0], num[1], dt / len(rows), n_in, model.usd))
        print("text questions (judge by eye):")
        for q, g, a, e in text_rows:
            print("   gold=%r | got=%r %s | %s" % (str(g)[:60], str(a)[:60], e or "", q[:70]))
        sys.stdout.flush()
        del model
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    if sys.argv[1:2] == ["show"]:
        f = parse(sys.argv[2])
        print(render_tables(f)[:6000])
    elif sys.argv[1:2] == ["test"]:
        test(sys.argv[2:])
    else:
        print(__doc__)
