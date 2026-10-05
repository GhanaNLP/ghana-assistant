"""GPU inference speed (router mode) for the tiny models and the Qwen3.5-0.8B LoRA reference.
Latency: batch 1 (one user). Throughput: batch 32 (serving many users). Same prompts for every model.
usage: python gpu_bench.py runs/s_falcon100m runs/s_smol135 ... [--qwen runs/rt_Qwen3.5-0.8B]"""
import sys, json, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
args = sys.argv[1:]; qwen = None
if "--qwen" in args: i = args.index("--qwen"); qwen = args[i + 1]; args = args[:i] + args[i + 2:]
rows = [json.loads(l) for l in open("data_reason/test_area.jsonl")]; random.Random(9).shuffle(rows)
src = lambda r: f"Request: {r['user']}\nRoute plan:\n{r['skeleton']}"


def load(run):
    if run == qwen:                                    # LoRA on Qwen3.5-0.8B, merged for a fair speed test
        from peft import PeftModel
        base = json.load(open(f"{run}/adapter_config.json"))["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(run); m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda()
        m = PeftModel.from_pretrained(m, run).merge_and_unload().eval(); tk = getattr(tok, "tokenizer", tok)
        def prompt(r):
            s = tk.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            return s + r["skeleton"] + "\n</think>\n\n"
        return tk, m, False, prompt
    cfg = json.load(open(f"{run}/small_args.json")); s2s = cfg["s2s"]
    tok = AutoTokenizer.from_pretrained(run)
    m = (AutoModelForSeq2SeqLM if s2s else AutoModelForCausalLM).from_pretrained(run, dtype=torch.float32 if s2s else torch.bfloat16).cuda().eval()
    def prompt(r):
        if s2s: return "directions: " + src(r)
        return tok.apply_chat_template([{"role": "system", "content": cfg["system"]}, {"role": "user", "content": src(r)}], tokenize=False, add_generation_prompt=True)
    return tok, m, s2s, prompt


def gen(tok, m, s2s, prompts):
    tok.padding_side = "right" if s2s else "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=s2s).to("cuda")
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad(): o = m.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(); dt = time.time() - t
    new = o if s2s else o[:, enc["input_ids"].shape[1]:]
    return dt, int((new != tok.pad_token_id).sum())


for run in args + ([qwen] if qwen else []):
    tok, m, s2s, prompt = load(run)
    P = [prompt(r) for r in rows[:160]]
    for p in P[:3]: gen(tok, m, s2s, [p])                                      # warm-up
    lat = []; ntok = 0
    for p in P[3:33]: dt, n = gen(tok, m, s2s, [p]); lat.append(dt); ntok += n
    lat.sort(); t0 = time.time(); nb = 0
    for i in range(32, 160, 32): dt, n = gen(tok, m, s2s, P[i:i + 32]); nb += 32
    thr = nb / (time.time() - t0)
    params = sum(x.numel() for x in m.parameters()) / 1e6
    print(f"{run.split('/')[-1]:22} {params:6.0f}M | batch 1: median {lat[len(lat)//2]:.2f}s/answer, {ntok/sum(lat):.0f} tok/s | batch 32: {thr:.1f} answers/s", flush=True)
    del m; torch.cuda.empty_cache()
