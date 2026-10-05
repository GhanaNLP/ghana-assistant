"""Pilot: translate (user request, answer) pairs to Twi / Ewe with NLLB-3.3B, keeping place names intact, then back-translate and check.
usage: python nllb_pilot.py --n 200 --langs twi_Latn ewe_Latn --mask codes"""
import argparse, json, random, re, torch
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=200); p.add_argument("--langs", nargs="+", default=["twi_Latn", "ewe_Latn"])
p.add_argument("--mask", choices=["codes", "none"], default="codes"); p.add_argument("--beams", type=int, default=4); p.add_argument("--out", default="results/nllb_pilot")
a = p.parse_args()
M = "facebook/nllb-200-3.3B"
tok = AutoTokenizer.from_pretrained(M); model = AutoModelForSeq2SeqLM.from_pretrained(M, torch_dtype=torch.bfloat16).to("cuda").eval()
for l in a.langs: assert tok.convert_tokens_to_ids(l) != tok.unk_token_id, l


def names_of(sk):
    n = set()
    for l in sk.splitlines():
        m = re.match(r"(Start|Destination): (.+?) \(", l);  n.update([m.group(2)] if m else [])
        n.update(re.findall(r"area ([^,;\n]+)", l))
        m = re.search(r"(?:on|onto) (.+?)(?: \(unnamed\))?(?:,|;|$)", l)
        if m and re.match(r"^\d+\.", l.strip()) and "(unnamed)" not in l.split(";")[0]: n.add(m.group(1).strip())
        n.update(x.strip() for x in re.findall(r"([^,;()]+?) \((?:on your (?:left|right)|straight ahead)\)", l.split("sees ")[-1]) if "sees " in l)
        m = re.search(r"Asked about: (.+?) ->", l);  n.update([m.group(1)] if m else [])
        m = re.search(r"Avoid \w+: (.+?) \(", l);  n.update([m.group(1)] if m else [])
        if l.startswith("Nearby"): n.update(re.findall(r"(?:: |; )([^;(]+?) \(", l))
    return sorted((x.strip() for x in n if len(x.strip()) > 2), key=len, reverse=True)


def mask(text, names):
    m = {}
    for i, nm in enumerate(names):
        code = f"ZX{i+1}Q"
        if re.search(re.escape(nm), text, re.I): text = re.sub(re.escape(nm), code, text, flags=re.I); m[code] = nm
    return text, m


def unmask(text, m):
    for c, nm in m.items(): text = text.replace(c, nm)
    return text


SPLIT = re.compile(r"(?<=[.!?])\s+")


def translate(texts, src, tgt, bs=48):
    """sentence-split, translate, re-join"""
    sents, owner = [], []
    for i, t in enumerate(texts):
        for s in SPLIT.split(t.strip()):
            if s: sents.append(s); owner.append(i)
    out = [None] * len(sents); tok.src_lang = src
    order = sorted(range(len(sents)), key=lambda k: len(sents[k]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([sents[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids(tgt), num_beams=a.beams, max_new_tokens=256)
        for j, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)): out[j] = t
    res = [[] for _ in texts]
    for o, t in zip(owner, out): res[o].append(t)
    return [" ".join(r) for r in res]


SIDE = re.compile(r"\b(?:on your|to your|turn|take a|make a|bear|keep|veer|slight|sharp|the (?:first|second|third|next))\s+(?:slight\s+|sharp\s+)?(left|right)\b", re.I)
ORD = re.compile(r"\b(first|second|third)\s+(?:slight\s+|sharp\s+)?(left|right)\b", re.I)
rows = [json.loads(l) for l in open("data_reason/train.jsonl")]
random.Random(7).shuffle(rows)
bytask = defaultdict(list)
for r in rows:
    if len(bytask[r["task"]]) < a.n // 8 + 1: bytask[r["task"]].append(r)
pick = [r for v in bytask.values() for r in v][:a.n]
print("pilot rows", len(pick), dict(Counter(r["task"] for r in pick)))
for lang in a.langs:
    items = []
    for r in pick:
        nm = names_of(r["skeleton"]) if a.mask == "codes" else []
        ui, um = mask(r["user"], nm); ao, am = mask(r["answer"], nm)
        items.append((r, ui, um, ao, am))
    tu = translate([x[1] for x in items], "eng_Latn", lang); ta = translate([x[3] for x in items], "eng_Latn", lang)
    bu = translate(tu, lang, "eng_Latn"); ba = translate(ta, lang, "eng_Latn")
    stats = Counter(); out = []
    for (r, ui, um, ao, am), t1, t2, b1, b2 in zip(items, tu, ta, bu, ba):
        lost = [c for c in list(um) + list(am) if c not in (t1 + " " + t2)]
        tin, tout = unmask(t1, um), unmask(t2, am); bin_, bout = unmask(b1, um), unmask(b2, am)
        o_en = Counter((o.lower(), s.lower()) for o, s in ORD.findall(r["answer"])); o_bt = Counter((o.lower(), s.lower()) for o, s in ORD.findall(bout))
        side_en = Counter(w.lower() for w in SIDE.findall(r["answer"])); side_bt = Counter(w.lower() for w in SIDE.findall(bout))
        chk = dict(names_kept=not lost, ordinals_kept=o_en == o_bt or not o_en, sides_kept=side_en == side_bt, leftover_codes=bool(re.search(r"ZX\d+Q", tin + tout)))
        for k, v in chk.items(): stats[k] += v
        out.append(dict(id=r["id"], task=r["task"], skeleton=r["skeleton"], en_input=r["user"], en_output=r["answer"], input=tin, output=tout,
                        back_input=bin_, back_output=bout, lost_names=[um.get(c) or am.get(c) for c in lost], checks=chk))
    n = len(out); print(lang, {k: f"{100*v/n:.0f}%" for k, v in stats.items()},
                         "| all checks pass:", f"{100*sum(o['checks']['names_kept'] and o['checks']['ordinals_kept'] and o['checks']['sides_kept'] and not o['checks']['leftover_codes'] for o in out)/n:.0f}%", flush=True)
    with open(f"{a.out}_{lang}_{a.mask}.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
