"""Translate (request, answer) pairs to Twi / Ewe with Gemini, place names protected by placeholders; back-translate and check.
usage: python gemini_translate.py --lang twi --model gemini-3.5-flash --n 200 --out results/gem_pilot"""
import argparse, json, random, re, time, concurrent.futures as cf, requests
from collections import Counter, defaultdict
p = argparse.ArgumentParser(); p.add_argument("--lang", choices=["twi", "ewe"], required=True); p.add_argument("--model", default="gemini-3.5-flash")
p.add_argument("--n", type=int, default=200); p.add_argument("--data", default="data_reason/train.jsonl"); p.add_argument("--out", default="results/gem_pilot")
p.add_argument("--temp", type=float, default=0.3); p.add_argument("--back_model", default=""); p.add_argument("--batch", type=int, default=8); p.add_argument("--workers", type=int, default=16)
a = p.parse_args()
KEY = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
LANG = {"twi": "Asante Twi (Akan), written with the standard Twi alphabet (ɛ, ɔ)", "ewe": "Ewe (Eʋegbe), written with the standard Ewe alphabet (ɖ, ɛ, ƒ, ɣ, ŋ, ɔ, ʋ)"}[a.lang]


def names_of(sk):
    n = set()
    for l in sk.splitlines():
        m = re.match(r"(Start|Destination): (.+?) \(", l); n.update([m.group(2)] if m else [])
        n.update(re.findall(r"area ([^,;\n]+)", l))
        m = re.search(r"(?:on|onto) (.+?)(?: \(unnamed\))?(?:,|;|$)", l)
        if m and re.match(r"^\d+\.", l.strip()) and "(unnamed)" not in l.split(";")[0]: n.add(m.group(1).strip())
        if "sees " in l: n.update(x.strip() for x in re.findall(r"([^,;()]+?) \((?:on your (?:left|right)|straight ahead)\)", l.split("sees ")[-1]))
        m = re.search(r"Asked about: (.+?) ->", l); n.update([m.group(1)] if m else [])
        m = re.search(r"Avoid \w+: (.+?) \(", l); n.update([m.group(1)] if m else [])
        if l.startswith("Nearby"): n.update(re.findall(r"(?:: |; )([^;(]+?) \(", l))
    return sorted((x.strip() for x in n if len(x.strip()) > 2), key=len, reverse=True)


def mask(text, names, m):
    for nm in names:
        if re.search(re.escape(nm), text, re.I):
            code = next((c for c, v in m.items() if v == nm), None) or f"[P{len(m)+1}]"
            m[code] = nm; text = re.sub(re.escape(nm), code, text, flags=re.I)
    return text


def unmask(t, m):
    for c, nm in m.items(): t = t.replace(c, nm)
    return t


FWD = """Translate these navigation messages from English into natural, everyday {lang} as people in Ghana actually speak and text it.
Each item has "input" (what a user asks) and "output" (the directions given back).
Rules:
- Keep every placeholder like [P1], [P2] EXACTLY as written (they stand for place and road names). Do not translate, drop or add them.
- Translate directions exactly: left stays left, right stays right; "first/second/third left/right" keeps the same number and side; "keep straight", "turn", "on your left/right" must keep their meaning.
- Do not add or remove information. Keep it simple and clear. No English words except where Ghanaians would normally use them.
Return a JSON array with one object per item, in the same order: {{"id": ..., "input": ..., "output": ...}}.
ITEMS:
{items}"""
BACK = """Translate these {lang} texts into English as literally and accurately as possible. Keep every placeholder like [P1] exactly.
Return a JSON array, same order: {{"id": ..., "input": ..., "output": ...}}.
ITEMS:
{items}"""


THINK = [{"thinkingBudget": 0}]      # as in the nsanku MT benchmark; falls back to minimal thinking if a model rejects it


def gem(prompt, model=None):
    model = model or a.model
    for t in range(6):
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": a.temp,
                "thinkingConfig": THINK[0]}}
        try:
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={KEY}", json=body, timeout=180)
            if r.status_code in (429, 503): time.sleep(3 * (t + 1)); continue
            if r.status_code == 400 and "think" in r.text.lower(): THINK[0] = {"thinkingLevel": "minimal"}; continue
            if not r.ok: raise RuntimeError(r.text[:200])
            parts = r.json()["candidates"][0]["content"]["parts"]
            return json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
        except Exception as e: err = e
    print("fail", err); return []


SIDE = re.compile(r"\b(?:on your|to your|turn|take a|make a|bear|keep|veer|slight|sharp|the (?:first|second|third|next))\s+(?:slight\s+|sharp\s+)?(left|right)\b", re.I)
ORD = re.compile(r"\b(first|second|third)\s+(?:slight\s+|sharp\s+)?(left|right)\b", re.I)
rows = [json.loads(l) for l in open(a.data)]; random.Random(7).shuffle(rows); bt = defaultdict(list)
for r in rows:
    if len(bt[r["task"]]) < a.n // 8 + 1: bt[r["task"]].append(r)
pick = [r for v in bt.values() for r in v][:a.n]
items = []
for r in pick:
    nm = names_of(r["skeleton"]); m = {}
    items.append(dict(id=r["id"], row=r, m=m, input=mask(r["user"], nm, m), output=mask(r["answer"], nm, m)))
batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]
fmt = lambda b, k1, k2: json.dumps([{"id": x["id"], "input": x[k1], "output": x[k2]} for x in b], ensure_ascii=False, indent=0)
t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex: fwd = list(ex.map(lambda b: gem(FWD.format(lang=LANG, items=fmt(b, "input", "output"))), batches))
for b, res in zip(batches, fwd):
    got = {x.get("id"): x for x in res if isinstance(x, dict)}
    for x in b: x["t_in"] = (got.get(x["id"]) or {}).get("input", ""); x["t_out"] = (got.get(x["id"]) or {}).get("output", "")
with cf.ThreadPoolExecutor(a.workers) as ex: back = list(ex.map(lambda b: gem(BACK.format(lang=LANG, items=fmt(b, "t_in", "t_out")), a.back_model or a.model), batches))
for b, res in zip(batches, back):
    got = {x.get("id"): x for x in res if isinstance(x, dict)}
    for x in b: x["b_in"] = (got.get(x["id"]) or {}).get("input", ""); x["b_out"] = (got.get(x["id"]) or {}).get("output", "")
stats = Counter(); out = []
for x in items:
    r, m = x["row"], x["m"]
    lost = [c for c in m if c not in x["t_in"] + " " + x["t_out"]] if x["t_out"] else list(m)
    bout = unmask(x["b_out"], m)
    o_en = Counter((o.lower(), s.lower()) for o, s in ORD.findall(r["answer"])); o_bt = Counter((o.lower(), s.lower()) for o, s in ORD.findall(bout))
    s_en = Counter(w.lower() for w in SIDE.findall(r["answer"])); s_bt = Counter(w.lower() for w in SIDE.findall(bout))
    tin, tout = unmask(x["t_in"], m), unmask(x["t_out"], m)
    chk = dict(translated=bool(x["t_out"]), names_kept=not lost, ordinals_kept=(o_en == o_bt) or not o_en, sides_kept=s_en == s_bt, leftover_codes=bool(re.search(r"\[P\d+\]", tin + tout)))
    chk["all_ok"] = chk["translated"] and chk["names_kept"] and chk["ordinals_kept"] and chk["sides_kept"] and not chk["leftover_codes"]
    for k, v in chk.items(): stats[k] += v
    out.append(dict(id=r["id"], task=r["task"], skeleton=r["skeleton"], en_input=r["user"], en_output=r["answer"], input=tin, output=tout,
                    back_input=unmask(x["b_in"], m), back_output=bout, checks=chk))
n = len(out)
print(a.lang, a.model, {k: f"{100*v/n:.0f}%" for k, v in stats.items()}, f"| {time.time()-t0:.0f}s")
with open(f"{a.out}_{a.lang}_{a.model}.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
