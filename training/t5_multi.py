"""Flan-T5-small with two skills, selected by prefix:
  directions: Request ... Route plan ...      -> spoken navigation directions
  answer: Question ... Context: 1..8 sentences -> short answer (GhanaQA, retrieval-augmented)
usage: python t5_multi.py train --out runs/t5s_multi [--nav 0] [--qa 0]      (0 = all)
       python t5_multi.py eval  --run runs/t5s_multi"""
import sys, json, random, time, re, argparse, torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
p = argparse.ArgumentParser(); p.add_argument("mode"); p.add_argument("--out"); p.add_argument("--run"); p.add_argument("--model", default="google/flan-t5-small")
p.add_argument("--nav", type=int, default=0); p.add_argument("--qa", type=int, default=0); p.add_argument("--intent", type=int, default=50000, help="examples per class for intent detection"); p.add_argument("--parse", type=int, default=0, help="parse examples (0 = one per navigation example)"); p.add_argument("--epochs", type=float, default=1)
p.add_argument("--bs", type=int, default=64); p.add_argument("--out_json"); p.add_argument("--tok", default="google/flan-t5-small"); p.add_argument("--lr", type=float, default=5e-4); p.add_argument("--n_eval", type=int, default=500)
a = p.parse_args()


def _norm(x): return re.sub(r"[^a-z0-9 ]", " ", str(x).lower()).split()


def mentioned(name, text):
    """name appears in the message (all its words present, order-free), so the model is never taught to invent it"""
    if not name: return False
    w = [t for t in _norm(name) if len(t) > 1]; tw = set(_norm(text))
    return bool(w) and sum(t in tw for t in w) >= max(1, round(0.8 * len(w)))


def parse_target(user, sc):
    TASK = {"route": "route", "reverse": "route", "short_hop": "route", "gps_start": "route", "avoid_road": "avoid",
            "avoid_landmark": "avoid", "nearby": "nearby", "connects": "connects"}          # what the router needs, not how the data was sampled
    f = [("task", TASK[sc["task"]])]; s, e = sc["start"], sc.get("end")
    if s.get("name") and mentioned(s["name"], user): f.append(("start", s["name"]))
    elif s.get("near") and mentioned(s["near"]["name"], user): f.append(("start", "near " + s["near"]["name"]))
    elif mentioned(s.get("area"), user): f.append(("start_area", s["area"]))
    if e:
        if mentioned(e["name"], user): f.append(("end", e["name"]))
        elif mentioned(e.get("area"), user): f.append(("end_area", e["area"]))
    if sc.get("avoid") and mentioned(sc["avoid"]["name"], user): f.append(("avoid", sc["avoid"]["name"]))
    if sc.get("question_landmark") and mentioned(sc["question_landmark"], user): f.append(("asked", sc["question_landmark"]))
    return " | ".join(f"{k}: {v}" for k, v in f)


def load_scenarios():
    SC = {}
    for c in ("kumasi", "accra"):
        for l in open(f"scenarios/{c}_scenarios.jsonl"): d = json.loads(l); SC[d["id"]] = d
    return SC


def nav_src(r): return f"directions: Request: {r['user']}\nRoute plan:\n{r['skeleton']}"
def qa_src(r): return "answer: Question: " + r["question"] + "\nContext:\n" + "\n".join(f"{i+1}. {c}" for i, c in enumerate(r["contexts"]))


CHARS = re.compile(r"\s*\(\s*~?\d+\s*(?:chars?|characters?)\s*\)", re.I)       # "(160 chars)" left over from the generation prompt
BROKEN = re.compile(r"(^\s*\*?A\d*[:.])|\bQ\d+[:.]")                                # parse leftovers like "A1:" / "Q2:"


def clean_qa(rows):
    """GhanaQA cleaning (training copy only): strip "(N chars)" notes and ** markers, drop broken parses."""
    out = []
    for r in rows:
        a = CHARS.sub("", r["answer"]).replace("**", "").strip()
        if BROKEN.search(a) or BROKEN.search(r["question"]) or len(a) < 10: continue
        r["answer"] = a; out.append(r)
    return out


def load(path, n, seed=0):
    rows = [json.loads(l) for l in open(path)]
    if "ghanaqa" in path: rows = clean_qa(rows)
    random.Random(seed).shuffle(rows); return rows[:n] if n else rows


if a.mode == "train":
    tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForSeq2SeqLM.from_pretrained(a.model).cuda()
    nav = load("data_reason/train.jsonl", a.nav); qa = load("ghanaqa/rag_sel_train.jsonl", a.qa)
    pairs = [(nav_src(r), r["answer"]) for r in nav] + [(qa_src(r), r["answer"]) for r in qa]
    # intent detection: navigation requests vs knowledge questions (labels come from which dataset a message is from)
    ni = [("intent: " + r["user"], "navigation") for r in random.Random(3).sample(nav, min(a.intent, len(nav)))]
    qi = [("intent: " + r["question"], "knowledge") for r in random.Random(4).sample(qa, min(a.intent, len(qa)))]
    SC = load_scenarios(); src_rows = nav if not a.parse else random.Random(6).sample(nav, min(a.parse, len(nav)))
    pp = [("parse: " + r["user"], parse_target(r["user"], SC[r["id"]])) for r in src_rows]
    pairs += ni + qi + pp; random.Random(1).shuffle(pairs)
    print(f"navigation {len(nav):,} + ghanaqa {len(qa):,} + intent {len(ni)+len(qi):,} + parse {len(pp):,} = {len(pairs):,} examples", flush=True)
    print("parse examples:", pp[:3], flush=True)
    t0 = time.time(); ds = []; cache = f"{a.out}_tokenised.pkl"                 # restarts skip the ~8 min tokenisation
    import pickle, os
    if os.path.exists(cache): ds = pickle.load(open(cache, "rb")); print("loaded tokenised cache", flush=True)
    else:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        for i in range(0, len(pairs), 100000):
            ch = pairs[i:i + 100000]
            X = tok([x for x, _ in ch], truncation=True, max_length=768)["input_ids"]; Y = tok([y for _, y in ch], truncation=True, max_length=200)["input_ids"]
            ds += [dict(input_ids=x, labels=y) for x, y in zip(X, Y)]
        pickle.dump(ds, open(cache, "wb"), protocol=4)
    print(f"tokenised in {time.time()-t0:.0f}s | truncated inputs: {sum(len(d['input_ids']) >= 768 for d in ds):,}", flush=True)

    def collate(b):
        L = max(len(x["input_ids"]) for x in b); T = max(len(x["labels"]) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long); lab = torch.full((len(b), T), -100)
        for i, x in enumerate(b):
            ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"]); att[i, :len(x["input_ids"])] = 1; lab[i, :len(x["labels"])] = torch.tensor(x["labels"])
        return dict(input_ids=ids, attention_mask=att, labels=lab)
    torch.backends.cuda.matmul.allow_tf32 = True
    args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
                             warmup_steps=500, tf32=True, logging_steps=200, save_strategy="steps", save_steps=5000, save_total_limit=2,
                             report_to=[], remove_unused_columns=False, dataloader_num_workers=4)
    Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
    model.save_pretrained(a.out); tok.save_pretrained(a.out); print("TRAIN_DONE", f"{time.time()-t0:.0f}s")

else:
    sys.path.insert(0, "."); from factcheck import check
    import os
    tok = AutoTokenizer.from_pretrained(a.run if os.path.exists(f"{a.run}/tokenizer_config.json") else a.tok)   # checkpoints have no tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(a.run).cuda().eval()
    def gen(srcs, bs=64):
        out = []
        for i in range(0, len(srcs), bs):
            enc = tok(srcs[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=768).to("cuda")
            with torch.no_grad(): o = model.generate(**enc, max_new_tokens=200, do_sample=False)
            out += tok.batch_decode(o, skip_special_tokens=True)
        return out
    res = {}
    # navigation: same 200 held-out examples as all earlier runs
    SC = load_scenarios()
    nav = [json.loads(l) for l in open("data_reason/test_area.jsonl")]; random.Random(1).shuffle(nav); nav = nav[:200]
    P = gen([nav_src(r) for r in nav]); ok = sum(check(SC[r["id"]], dict(input=r["user"], output=t)) is None for r, t in zip(nav, P))
    res["navigation_pass"] = round(ok / len(nav), 3)
    with open(f"{a.run}/nav_preds.jsonl", "w") as f:
        for r, t in zip(nav, P): f.write(json.dumps(dict(id=r["id"], task=r["task"], user=r["user"], pred=t), ensure_ascii=False) + "\n")
    # GhanaQA: token-F1 / ROUGE-L vs reference; split by whether retrieval found the source article (batch 2)
    W = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    def f1(p, r):
        p, r = W(p), W(r); c = sum(min(p.count(w), r.count(w)) for w in set(p))
        return 0 if not c else 2 * c / (len(p) + len(r))
    def rougeL(p, r):
        p, r = W(p), W(r)
        if not p or not r: return 0
        dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]
        for i in range(len(p)):
            for j in range(len(r)): dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
        l = dp[-1][-1]; return 0 if not l else 2 * l / (len(p) + len(r))
    def support(p, ctx):                                   # share of answer words that appear in the retrieved sentences
        cw = set(W(" ".join(ctx))); pw = [w for w in W(p) if len(w) > 3]; return sum(w in cw for w in pw) / max(1, len(pw))
    samples = []
    for split in ("test_b1", "test_b2"):
        rows = load(f"ghanaqa/rag_sel_{split}.jsonl", a.n_eval, seed=2); P = gen([qa_src(r) for r in rows]); g = defaultdict(list)
        for r, t in zip(rows, P):
            key = "all"; m = dict(f1=f1(t, r["answer"]), rougeL=rougeL(t, r["answer"]), support=support(t, r["contexts"]))
            for k in ([key] + ([("hit" if r.get("source_hit") else "miss")] if "source_hit" in r else [])):
                for mk, v in m.items(): g[f"{k}_{mk}"].append(v)
            if len(samples) < 10: samples.append(dict(split=split, q=r["question"], ref=r["answer"], pred=t, hit=r.get("source_hit")))
        res[split] = {k: round(sum(v) / len(v), 3) for k, v in g.items()}; res[split]["n"] = len(rows)
        if "source_hit" in rows[0]: res[split]["source_hit_rate"] = round(sum(r["source_hit"] for r in rows) / len(rows), 3)
    # parse: structured fields from held-out navigation messages
    pr = load("data_reason/test_area.jsonl", 500, seed=7); gold = [parse_target(r["user"], SC[r["id"]]) for r in pr]
    pp = gen(["parse: " + r["user"] for r in pr], bs=128)
    fields = lambda t: dict(x.split(": ", 1) for x in t.split(" | ") if ": " in x)
    res["parse_exact"] = round(sum(p.strip() == g for p, g in zip(pp, gold)) / len(pr), 4)
    fa = defaultdict(list)
    for p, g in zip(pp, gold):
        P, G = fields(p), fields(g)
        for k in G: fa[k].append(P.get(k) == G[k])
    res["parse_field_accuracy"] = {k: round(sum(v) / len(v), 3) for k, v in fa.items()}
    res["parse_errors"] = [(r["user"], g, p) for r, g, p in zip(pr, gold, pp) if p.strip() != g][:10]
    # intent detection on held-out messages from both skills
    im = [(r["user"], "navigation") for r in load("data_reason/test_area.jsonl", 500, seed=5)] + [(r["user"], "navigation") for r in load("data_reason/test_pair.jsonl", 500, seed=5)]
    im += [(r["question"], "knowledge") for r in load("ghanaqa/rag_sel_test_b1.jsonl", 500, seed=5)] + [(r["question"], "knowledge") for r in load("ghanaqa/rag_sel_test_b2.jsonl", 500, seed=5)]
    pi = gen(["intent: " + m for m, _ in im], bs=128)
    res["intent_accuracy"] = round(sum(p.strip().lower() == y for p, (_, y) in zip(pi, im)) / len(im), 4)
    res["intent_errors"] = [(m, y, p) for p, (m, y) in zip(pi, im) if p.strip().lower() != y][:15]
    print(json.dumps(res, indent=1)); json.dump(dict(results=res, samples=samples), open(a.out_json or f"{a.run}/eval.json", "w"), indent=1, ensure_ascii=False)
