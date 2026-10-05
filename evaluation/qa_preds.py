"""Generate answers for held-out GhanaQA questions with two decoding settings (for grading).
usage: python qa_preds.py runs/t5s_multi results/qa_preds.jsonl"""
import sys, json, random, re, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
src = open("t5_multi.py").read(); g = {"re": re, "json": json, "random": random}
exec(src[src.index("def _norm(x)"):src.index('if a.mode == "train":')], g)
run, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(run); model = AutoModelForSeq2SeqLM.from_pretrained(run).cuda().eval()
rows = g["load"]("ghanaqa/rag_sel_test_b1.jsonl", 150, seed=11) + g["load"]("ghanaqa/rag_sel_test_b2.jsonl", 150, seed=11)
def gen(srcs, **kw):
    res = []
    for i in range(0, len(srcs), 32):
        enc = tok(srcs[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
        with torch.no_grad(): o = model.generate(**enc, max_new_tokens=120, **kw)
        res += tok.batch_decode(o, skip_special_tokens=True)
    return res
S = [g["qa_src"](r) for r in rows]
greedy = gen(S, do_sample=False)
fixed = gen(S, do_sample=False, num_beams=4, no_repeat_ngram_size=3, early_stopping=True)
with open(out, "w") as f:
    for r, a, b in zip(rows, greedy, fixed):
        f.write(json.dumps(dict(qid=r["qid"], batch=r["batch"], question=r["question"], contexts=r["contexts"], reference=r["answer"],
                                source_hit=r.get("source_hit"), pred_greedy=a, pred_fixed=b), ensure_ascii=False) + "\n")
print("PREDS_DONE", len(rows))
