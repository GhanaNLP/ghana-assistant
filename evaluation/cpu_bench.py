"""CPU latency: time to generate one answer (router mode) on this machine, no GPU.
usage: python cpu_bench.py runs_local/s_falcon100m [n] [threads]"""
import sys, json, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
run = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 15; th = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.set_num_threads(th)
cfg = json.load(open(f"{run}/small_args.json")); S2S = cfg["s2s"]
tok = AutoTokenizer.from_pretrained(run)
model = (AutoModelForSeq2SeqLM if S2S else AutoModelForCausalLM).from_pretrained(run, torch_dtype=torch.float32).eval()
rows = [json.loads(l) for l in open("data_reason/test_area.jsonl")]; random.Random(5).shuffle(rows); rows = rows[:N]
src = lambda r: f"Request: {r['user']}\nRoute plan:\n{r['skeleton']}"
def prompt(r):
    if S2S: return "directions: " + src(r)
    return tok.apply_chat_template([{"role": "system", "content": cfg["system"]}, {"role": "user", "content": src(r)}], tokenize=False, add_generation_prompt=True)
times, toks = [], []
for k, r in enumerate(rows):
    enc = tok(prompt(r), return_tensors="pt", add_special_tokens=S2S)
    t = time.time()
    with torch.no_grad(): o = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    dt = time.time() - t; n = o.shape[1] - (0 if S2S else enc["input_ids"].shape[1])
    if k: times.append(dt); toks.append(n)          # skip the first (warm-up)
times.sort()
print(f"{run.split('/')[-1]:16} threads={th} | median {times[len(times)//2]:.2f}s per answer | mean {sum(times)/len(times):.2f}s | {sum(toks)/sum(times):.1f} tokens/s | ~{sum(toks)/len(toks):.0f} tokens per answer")
