"""Google Gemini API as the model behind the two model-backed steps (question -> query, table question -> code).

Same interface as the local llm_plan.Model: generate(batch_of_chat_messages, max_new_tokens) -> (texts, n_in, n_out),
so every caller works unchanged. Standard library only (no extra package), no GPU.

  export GEMINI_API_KEY=...          # the AI Studio key of the hackathon project; never put it in code or on a command line
  python gemini_backend.py ping [model]

The key is read from the environment and sent in a request header; it is never printed or written to disk.
Prices are USD per 1M tokens, Standard paid tier, read from the AI Studio pricing page on 2026-11-14
(thinking tokens are billed as output). Check the page again before quoting costs.
"""
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent"
PRICES = {  # model id: (input, output)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-4-hydrogen": (1.50, 7.50),
    "gemini-4-argon": (2.00, 10.00),
    "gemini-4-carbon": (2.00, 10.00),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}
KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api.txt")
DEFAULT_PLAN = os.environ.get("STOA_PLAN_MODEL", "gemini-3.1-flash-lite")
DEFAULT_TABLE = os.environ.get("STOA_TABLE_MODEL", "gemini-3.8-flash")


def is_gemini(model_id):
    return model_id.lower().startswith("gemini")


def _key():
    k = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not k and os.path.exists(KEY_FILE):  # one line, owner-readable only; keep it out of git
        k = open(KEY_FILE).read().strip()
    if not k:
        raise RuntimeError("set GEMINI_API_KEY in the environment, or put the key in %s (chmod 600)" % KEY_FILE)
    return k


def _body(msgs, max_out, thinking):
    system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in msgs if m["role"] != "system"]
    cfg = {"maxOutputTokens": max_out}
    if thinking:
        cfg["thinkingConfig"] = {"thinkingLevel": thinking}
    body = {"contents": contents, "generationConfig": cfg}
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    return body


class Gemini:
    def __init__(self, model_id, thinking="low", workers=8):
        self.model_id = model_id
        # the 2.5 family has thinking off by default and takes a different switch; leave it alone
        if "flash-lite" in model_id and thinking == "low":
            thinking = "minimal"  # the lite models accept it; one-line answers do not need more
        self.thinking = None if model_id.startswith("gemini-2") else thinking
        self.workers = workers
        self.price = PRICES.get(model_id, (0.0, 0.0))
        self.usd = 0.0  # running total for this object
        self.calls = 0

    def _one(self, msgs, max_new_tokens):
        # thinking tokens count against the output limit, so leave room for them
        thinking = self.thinking
        for attempt in range(6):
            body = _body(msgs, max_new_tokens + (2048 if thinking else 0), thinking)
            req = urllib.request.Request(URL % self.model_id, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": _key()})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 400 and thinking:  # this model does not take the thinking switch
                    thinking = self.thinking = None
                    continue
                if e.code in (429, 500, 503) and attempt < 5:  # rate limit / overloaded: back off and retry
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Gemini API %d for %s: %s" % (e.code, self.model_id, detail))
            except (urllib.error.URLError, TimeoutError):
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
            parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            u = j.get("usageMetadata", {})
            n_in = u.get("promptTokenCount", 0)
            n_out = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
            return text, n_in, n_out
        raise RuntimeError("Gemini API: no answer after retries")

    def generate(self, batch_msgs, max_new_tokens=80):
        if len(batch_msgs) == 1:
            outs = [self._one(batch_msgs[0], max_new_tokens)]
        else:
            with concurrent.futures.ThreadPoolExecutor(self.workers) as ex:
                outs = list(ex.map(lambda m: self._one(m, max_new_tokens), batch_msgs))
        n_in = sum(o[1] for o in outs)
        n_out = sum(o[2] for o in outs)
        self.usd += (n_in * self.price[0] + n_out * self.price[1]) / 1e6
        self.calls += len(outs)
        return [o[0] for o in outs], n_in, n_out


if __name__ == "__main__":
    if sys.argv[1:2] == ["ping"]:
        mid = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PLAN
        g = Gemini(mid)
        t0 = time.time()
        texts, a, b = g.generate([[{"role": "system", "content": "Answer with one word."}, {"role": "user", "content": "Which language is 'Montako ovea?' written in?"}]], 20)
        print("%s -> %r  (%d in, %d out, $%.6f, %.2fs)" % (mid, texts[0].strip(), a, b, g.usd, time.time() - t0))
    else:
        print(__doc__)
