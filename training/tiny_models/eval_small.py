"""Router-mode eval for tiny models on the same 200 held-out examples as the Qwen runs; saves predictions.
usage: python eval_small.py --run runs/s_smol135 --out results/s_smol135.json"""
import argparse, json, random, sys, time, torch
from collections import defaultdict
sys.path.insert(0, "."); from factcheck import check
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--out", required=True)
p.add_argument("--split", default="test_area"); p.add_argument("--n", type=int, default=200); p.add_argument("--bs", type=int, default=32)
a = p.parse_args()
cfg = json.load(open(f"{a.run}/small_args.json")); S2S = cfg["s2s"]
tok = AutoTokenizer.from_pretrained(a.run, padding_side="left" if not S2S else "right")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = (AutoModelForSeq2SeqLM if S2S else AutoModelForCausalLM).from_pretrained(a.run, torch_dtype=torch.float32 if S2S else torch.bfloat16).to("cuda").eval()
SC = {}
for c in ("kumasi", "accra"):
    for l in open(f"scenarios/{c}_scenarios.jsonl"): d = json.loads(l); SC[d["id"]] = d
rows = [json.loads(l) for l in open(f"data_reason/{a.split}.jsonl")]; random.Random(1).shuffle(rows); rows = rows[:a.n]
src = lambda r: f"Request: {r['user']}\nRoute plan:\n{r['skeleton']}"
def prompt(r):
    if S2S: return "directions: " + src(r)
    return tok.apply_chat_template([{"role": "system", "content": cfg["system"]}, {"role": "user", "content": src(r)}], tokenize=False, add_generation_prompt=True)
stat = defaultdict(float); preds = []; t0 = time.time()
for i in range(0, len(rows), a.bs):
    b = rows[i:i + a.bs]
    enc = tok([prompt(r) for r in b], return_tensors="pt", padding=True, truncation=True, max_length=768, add_special_tokens=S2S).to("cuda")
    with torch.no_grad(): out = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
    for r, o in zip(b, out):
        txt = tok.decode(o if S2S else o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(dict(id=r["id"], task=r["task"], user=r["user"], pred=txt))
        why = check(SC[r["id"]], dict(input=r["user"], output=txt)) if txt else "no_answer"
        stat["pass"] += why is None
        if why: stat["fail_" + why] += 1
n = len(rows); res = {k: round(v / n, 3) for k, v in stat.items()}; res["n"] = n; print(json.dumps(res))
json.dump(dict(run=a.run, results=res), open(a.out, "w"), indent=1)
with open(a.out.replace(".json", "_preds.jsonl"), "w") as f:
    for x in preds: f.write(json.dumps(x, ensure_ascii=False) + "\n")
