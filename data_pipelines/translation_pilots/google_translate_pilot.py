"""Pilot only: Google Translate (free web endpoint, throttled) for the same 200 pairs; back-translated with Gemini for a like-for-like check.
usage: python google_translate_pilot.py --lang twi"""
import argparse, json, random, re, time, requests
from collections import defaultdict
p = argparse.ArgumentParser(); p.add_argument("--lang", choices=["twi", "ewe"], required=True); p.add_argument("--n", type=int, default=200)
p.add_argument("--back_model", default="gemini-3.8-flash"); a = p.parse_args()
GCODE = {"twi": "ak", "ewe": "ee"}[a.lang]
KEY = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
LANG = {"twi": "Asante Twi (Akan)", "ewe": "Ewe (Eʋegbe)"}[a.lang]
src = open("gemini_translate.py").read()
ns = {}; exec(src[src.index("def names_of"):src.index("FWD = ")], {"re": re}, ns)      # reuse names_of / mask / unmask
names_of, mask, unmask = ns["names_of"], ns["mask"], ns["unmask"]


def gtrans(text, tl, sl="en"):
    """same call as the nsanku MT benchmark (GoogleFreeMT): clients5 endpoint, one text per request, throttled"""
    for t in range(6):
        try:
            r = requests.post("https://clients5.google.com/translate_a/t", params={"client": "dict-chrome-ex", "sl": sl, "tl": tl},
                              data={"q": text}, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if r.status_code != 200: raise RuntimeError(f"google {r.status_code}")
            j = r.json(); out = j[0] if isinstance(j, list) else j
            out = out[0] if isinstance(out, list) else out
            time.sleep(1.5)
            if out.strip() == text.strip() and len(text) > 20: raise RuntimeError("unchanged output (soft block)")
            return out
        except Exception as e:
            err = e; time.sleep(min(90, 2 ** (t + 2)))
    print("fail", err); return ""


BACK = """Translate these {lang} texts into English as literally and accurately as possible. Keep every placeholder like [P1] exactly.
Return a JSON array, same order: {{"id": ..., "input": ..., "output": ...}}.
ITEMS:
{items}"""


def gem(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0, "thinkingConfig": {"thinkingBudget": 0}}}
    for t in range(5):
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{a.back_model}:generateContent?key={KEY}", json=body, timeout=180)
            if r.status_code in (429, 503): time.sleep(3 * (t + 1)); continue
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception: pass
    return []


rows = [json.loads(l) for l in open("data_reason/train.jsonl")]; random.Random(7).shuffle(rows); bt = defaultdict(list)
for r in rows:
    if len(bt[r["task"]]) < a.n // 8 + 1: bt[r["task"]].append(r)
pick = [r for v in bt.values() for r in v][:a.n]
items = []; t0 = time.time()
for k, r in enumerate(pick):
    m = {}; nm = names_of(r["skeleton"]); mi = mask(r["user"], nm, m); mo = mask(r["answer"], nm, m)
    items.append(dict(id=r["id"], row=r, m=m, t_in=gtrans(mi, GCODE), t_out=gtrans(mo, GCODE)))
    if k % 50 == 0: print(k, "translated", f"{time.time()-t0:.0f}s", flush=True)
for i in range(0, len(items), 8):
    b = items[i:i + 8]
    res = gem(BACK.format(lang=LANG, items=json.dumps([{"id": x["id"], "input": x["t_in"], "output": x["t_out"]} for x in b], ensure_ascii=False)))
    got = {x.get("id"): x for x in res if isinstance(x, dict)}
    for x in b: x["b_in"] = (got.get(x["id"]) or {}).get("input", ""); x["b_out"] = (got.get(x["id"]) or {}).get("output", "")
with open(f"results/gt_pilot_{a.lang}.jsonl", "w") as f:
    for x in items:
        r, m = x["row"], x["m"]; lost = [c for c in m if c not in x["t_in"] + " " + x["t_out"]] if x["t_out"] else list(m)
        f.write(json.dumps(dict(id=r["id"], task=r["task"], skeleton=r["skeleton"], en_input=r["user"], en_output=r["answer"],
                                input=unmask(x["t_in"], m), output=unmask(x["t_out"], m), back_input=unmask(x["b_in"], m), back_output=unmask(x["b_out"], m),
                                checks=dict(translated=bool(x["t_out"]), names_kept=not lost)), ensure_ascii=False) + "\n")
print("done", a.lang, f"{time.time()-t0:.0f}s")
