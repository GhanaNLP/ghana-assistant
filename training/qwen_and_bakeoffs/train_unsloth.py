"""LoRA SFT of Qwen3.8 with Unsloth. Target = skeleton (as <think>) + spoken directions.
usage: python train_unsloth.py --out runs/q38_smoke --n 96 --epochs 1"""
import argparse, json, random, time, torch
from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments
p = argparse.ArgumentParser()
p.add_argument("--model", default="unsloth/Qwen3.8-27B"); p.add_argument("--out", required=True)
p.add_argument("--n", type=int, default=0, help="train on a random subset of n examples (0 = all)")
p.add_argument("--epochs", type=float, default=1); p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--bs", type=int, default=4); p.add_argument("--accum", type=int, default=4)
p.add_argument("--r", type=int, default=16); p.add_argument("--max_len", type=int, default=1024)
p.add_argument("--save_steps", type=int, default=0); p.add_argument("--answer_only", action="store_true", help="router mode: skeleton is context, loss only on the answer"); p.add_argument("--load_4bit", action="store_true"); p.add_argument("--data", default="data_reason/train.jsonl")
a = p.parse_args()

model, tok = FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_len, load_in_4bit=a.load_4bit, dtype=torch.bfloat16)
tk = getattr(tok, "tokenizer", tok)
if tk.pad_token is None: tk.pad_token = tk.eos_token
model = FastLanguageModel.get_peft_model(
    model, r=a.r, lora_alpha=a.r, lora_dropout=0, bias="none", random_state=3407, use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkv", "in_proj_z", "out_proj"])
model.print_trainable_parameters()

KW = dict(enable_thinking=True, reasoning_effort="low")      # short structured thinking, not the default "xhigh"


def encode(r):
    msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
    prompt = tk.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **KW)      # ends with "<think>\n"
    if a.answer_only:       # router mode: prompt + skeleton + </think> is context (exactly as at inference); learn only the answer
        ctx = tk(prompt + r["skeleton"] + "\n</think>\n\n", add_special_tokens=False)["input_ids"]
        ans = tk(r["answer"] + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (ctx + ans)[:a.max_len]; return dict(input_ids=ids, labels=([-100] * len(ctx) + ans)[:a.max_len])
    target = r["skeleton"] + "\n</think>\n\n" + r["answer"] + "<|im_end|>\n"
    pi = tk(prompt, add_special_tokens=False)["input_ids"]; ti = tk(target, add_special_tokens=False)["input_ids"]
    ids = (pi + ti)[:a.max_len]
    return dict(input_ids=ids, labels=([-100] * len(pi) + ti)[:a.max_len])


rows = [json.loads(l) for l in open(a.data)]
random.Random(0).shuffle(rows)
if a.n: rows = rows[:a.n]
ds = [encode(r) for r in rows]
print("examples", len(ds), "| avg tokens", sum(len(d["input_ids"]) for d in ds) / len(ds), "| max", max(len(d["input_ids"]) for d in ds))
print("SAMPLE PROMPT+TARGET:\n", tk.decode(ds[0]["input_ids"])[:1500])


def collate(b):
    L = max(len(x["input_ids"]) for x in b)
    ids = torch.full((len(b), L), tk.pad_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
    for i, x in enumerate(b):
        n = len(x["input_ids"]); ids[i, :n] = torch.tensor(x["input_ids"]); lab[i, :n] = torch.tensor(x["labels"]); att[i, :n] = 1
    return dict(input_ids=ids, labels=lab, attention_mask=att)


steps = max(1, int(len(ds) / (a.bs * a.accum) * a.epochs))
args = TrainingArguments(output_dir=a.out, per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.accum, num_train_epochs=a.epochs,
                         learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=max(1, min(10, steps // 20)), bf16=True, logging_steps=10,
                         save_strategy=("steps" if a.save_steps else "no"), save_steps=a.save_steps or 500, save_total_limit=2, report_to=[], remove_unused_columns=False, optim="adamw_8bit", seed=3407)
t0 = time.time()
Trainer(model=model, args=args, train_dataset=ds, data_collator=collate).train()
el = time.time() - t0
print(f"TRAIN_DONE {el:.0f}s | {len(ds)*a.epochs/el:.2f} examples/s | peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
model.save_pretrained(a.out); tok.save_pretrained(a.out); print("saved", a.out)
