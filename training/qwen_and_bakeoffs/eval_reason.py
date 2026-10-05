"""Evaluate a Qwen3.8 reasoning-format model (base or LoRA) on held-out data.
 --mode model : model writes the skeleton (<think>) itself, then the answer  -> trace accuracy + answer pass rate
 --mode router: the TRUE skeleton is pre-filled as the thinking, model writes only the answer -> answer pass rate (router + verbaliser)
usage: python eval_reason.py --mode model --adapter runs/q38_nav_10k --split test_area --n 200 --out results/x.json"""
import argparse, json, re, random, sys, time, torch
from collections import defaultdict
sys.path.insert(0, "."); from factcheck import check
from unsloth import FastLanguageModel
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["model", "router"], required=True); p.add_argument("--adapter"); p.add_argument("--base", default="unsloth/Qwen3.8-27B")
p.add_argument("--split", default="test_area"); p.add_argument("--n", type=int, default=200); p.add_argument("--bs", type=int, default=16); p.add_argument("--out", required=True)
a = p.parse_args()
model, tok = FastLanguageModel.from_pretrained(a.adapter or a.base, max_seq_length=2048, load_in_4bit=False, dtype=torch.bfloat16)
FastLanguageModel.for_inference(model); tk = getattr(tok, "tokenizer", tok); tk.padding_side = "left"
if tk.pad_token is None: tk.pad_token = tk.eos_token
SC = {}
for c in ("kumasi", "accra"):
    for l in open(f"scenarios/{c}_scenarios.jsonl"): d = json.loads(l); SC[d["id"]] = d
rows = [json.loads(l) for l in open(f"data_reason/{a.split}.jsonl")]; random.Random(1).shuffle(rows); rows = rows[:a.n]


def prompt(r):
    m = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
    s = tk.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True, reasoning_effort="low")
    return s + (r["skeleton"] + "\n</think>\n\n" if a.mode == "router" else "")


def parse(trace):
    L = [l.strip() for l in trace.strip().splitlines()]
    start = next((l for l in L if l.startswith(("Start:", "Position:"))), ""); dest = next((l for l in L if l.startswith("Destination:")), "")
    steps = []
    for l in L:
        m = re.match(r"^\d+\.\s+(.*)$", l)
        if not m: continue
        t = m.group(1); kind = re.match(r"(Head \S+|TURN (?:LEFT|RIGHT)(?: \(the \w+ (?:left|right)\))?|TURN (?:LEFT|RIGHT)|KEEP STRAIGHT)", t)
        road = re.search(r"(?:on|onto) (.+?)(?:,|;|$)", t)
        lms = re.findall(r"sees (.+)$", t)
        names = re.findall(r"([^,;()]+?) \((?:on your (?:left|right)|straight ahead)\)", lms[0]) if lms else []
        steps.append((kind.group(1).strip() if kind else "", road.group(1).strip() if road else "", frozenset(n.strip() for n in names)))
    return start, dest, steps


def f1(p, t):
    if not p and not t: return 1.0
    i = len(p & t); return 0 if not i else 2 * i / (len(p) + len(t))


stat = defaultdict(float); bytask = defaultdict(lambda: [0, 0]); samples = []; preds = []; t0 = time.time()
for i in range(0, len(rows), a.bs):
    b = rows[i:i + a.bs]
    enc = tk([prompt(r) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad(): out = model.generate(**enc, max_new_tokens=700 if a.mode == "model" else 260, do_sample=False, pad_token_id=tk.pad_token_id)
    for r, o in zip(b, out):
        txt = tk.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if a.mode == "model":
            trace, _, ans = txt.partition("</think>"); ans = ans.strip(); ps, pd, pst = parse(trace); ts, td, tst = parse(r["skeleton"])
            stat["trace_exact"] += trace.strip() == r["skeleton"].strip(); stat["start_ok"] += ps == ts; stat["dest_ok"] += pd == td
            stat["steps_n_ok"] += len(pst) == len(tst); stat["road_seq_ok"] += [x[1] for x in pst] == [x[1] for x in tst]
            stat["turn_seq_ok"] += [x[0] for x in pst] == [x[0] for x in tst]
            stat["road_f1"] += f1({x[1] for x in pst}, {x[1] for x in tst}); stat["landmark_f1"] += f1(set().union(*[x[2] for x in pst]) if pst else set(), set().union(*[x[2] for x in tst]) if tst else set())
        else: ans = txt.strip()
        preds.append(dict(id=r["id"], task=r["task"], user=r["user"], pred=ans))
        sc = SC[r["id"]]; why = check(sc, dict(input=r["user"], output=ans)) if ans else "no_answer"
        stat["pass"] += why is None; bytask[r["task"]][0] += why is None; bytask[r["task"]][1] += 1
        if why: stat["fail_" + why] += 1
        if len(samples) < 12 and i == 0: samples.append(dict(task=r["task"], user=r["user"], true_skeleton=r["skeleton"], ref=r["answer"], pred=txt, verdict=why))
    print(f"{min(i + a.bs, len(rows))}/{len(rows)} done, {time.time()-t0:.0f}s", flush=True)
n = len(rows); res = {k: round(v / n, 3) for k, v in stat.items()}; res["n"] = n; res["by_task_pass"] = {t: round(x[0] / x[1], 2) for t, x in bytask.items()}
print(json.dumps(res, indent=1))
import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(dict(args=vars(a), results=res, samples=samples), open(a.out, "w"), indent=1, ensure_ascii=False)
with open(a.out.replace('.json', '_preds.jsonl'), 'w') as f:
    for x in preds: f.write(json.dumps(x, ensure_ascii=False) + '\n')
