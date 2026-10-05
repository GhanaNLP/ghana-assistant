"""Full fine-tune of tiny models (<=250M) for router mode: (user request + route plan) -> spoken directions.
Decoder models use their chat template; T5 models get a plain text input.
usage: python train_small.py --model HuggingFaceTB/SmolLM2-135M-Instruct --out runs/s_smol135 --n 50000"""
import argparse, json, random, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=50000)
p.add_argument("--epochs", type=float, default=1); p.add_argument("--lr", type=float); p.add_argument("--bs", type=int, default=32)
p.add_argument("--max_src", type=int, default=768); p.add_argument("--max_tgt", type=int, default=200)
a = p.parse_args()
S2S = "t5" in a.model.lower()
SYSTEM = "You are a navigation assistant for Ghana. Using the route plan, give short spoken directions with counted turns and landmarks, no distances, in simple English."


def src_text(user, skel):
    return f"Request: {user}\nRoute plan:\n{skel}"


tok = AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = (AutoModelForSeq2SeqLM if S2S else AutoModelForCausalLM).from_pretrained(a.model, torch_dtype=torch.bfloat16 if not S2S else torch.float32).to("cuda")


def encode(r):
    if S2S:
        x = tok("directions: " + src_text(r["user"], r["skeleton"]), truncation=True, max_length=a.max_src)["input_ids"]
        y = tok(r["answer"], truncation=True, max_length=a.max_tgt)["input_ids"]
        return dict(input_ids=x, labels=y)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": src_text(r["user"], r["skeleton"])}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pi = tok(prompt, add_special_tokens=False)["input_ids"]; ai = tok(r["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (pi + ai)[:a.max_src + a.max_tgt]
    return dict(input_ids=ids, labels=([-100] * len(pi) + ai)[:len(ids)])


rows = [json.loads(l) for l in open("data_reason/train.jsonl")]; random.Random(0).shuffle(rows)
rows = rows[:a.n] if a.n else rows
ds = [encode(r) for r in rows]
print("examples", len(ds), "| avg input tokens", sum(len(d["input_ids"]) for d in ds) / len(ds), "| truncated?", sum(len(d["input_ids"]) >= a.max_src for d in ds))


def collate(b):
    if S2S:
        L = max(len(x["input_ids"]) for x in b); T = max(len(x["labels"]) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long); lab = torch.full((len(b), T), -100)
        for i, x in enumerate(b):
            ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"]); att[i, :len(x["input_ids"])] = 1; lab[i, :len(x["labels"])] = torch.tensor(x["labels"])
        return dict(input_ids=ids, attention_mask=att, labels=lab)
    L = max(len(x["input_ids"]) for x in b)
    ids = torch.full((len(b), L), tok.pad_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        n = len(x["input_ids"]); ids[i, :n] = torch.tensor(x["input_ids"]); lab[i, :n] = torch.tensor(x["labels"]); att[i, :n] = 1
    return dict(input_ids=ids, labels=lab, attention_mask=att)


torch.backends.cuda.matmul.allow_tf32 = True
lr = a.lr or (5e-4 if S2S else 2e-4)
args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, num_train_epochs=a.epochs, learning_rate=lr, lr_scheduler_type="cosine",
                         warmup_steps=max(1, int(0.03 * len(ds) / a.bs * a.epochs)), bf16=not S2S, tf32=True, logging_steps=50,
                         save_strategy="no", report_to=[], remove_unused_columns=False, dataloader_num_workers=2)
t0 = time.time(); Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
print(f"TRAIN_DONE {time.time()-t0:.0f}s"); model.save_pretrained(a.out); tok.save_pretrained(a.out)
json.dump(dict(vars(a), s2s=S2S, system=SYSTEM), open(f"{a.out}/small_args.json", "w"))
