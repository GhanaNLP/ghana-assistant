"""Publish the exact Flan-T5 training set (same construction and seeds as t5_multi.py train) as a ready-to-train dataset.
usage (from map-nav/): python build_train_hf.py out_dir  -> parquet splits + README; then upload."""
import sys, os, json, random, re
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
src = open("t5_multi.py").read()
g = {"re": re, "json": json, "random": random}
exec(src[src.index("def _norm(x)"):src.index('if a.mode == "train":')], g)   # helpers: clean_qa, load, mentioned, parse_target, load_scenarios, nav_src, qa_src
load, nav_src, qa_src, parse_target, load_scenarios = g["load"], g["nav_src"], g["qa_src"], g["parse_target"], g["load_scenarios"]
OUT = sys.argv[1]; os.makedirs(f"{OUT}/data", exist_ok=True)

# train: identical to t5_multi.py train (defaults: all navigation, all GhanaQA, 50k intent per class, one parse per navigation example)
nav = load("data_reason/train.jsonl", 0); qa = load("ghanaqa/rag_sel_train.jsonl", 0)
rows = [(nav_src(r), r["answer"], "directions") for r in nav] + [(qa_src(r), r["answer"], "answer") for r in qa]
ni = [("intent: " + r["user"], "navigation", "intent") for r in random.Random(3).sample(nav, min(50000, len(nav)))]
qi = [("intent: " + r["question"], "knowledge", "intent") for r in random.Random(4).sample(qa, min(50000, len(qa)))]
SC = load_scenarios(); pp = [("parse: " + r["user"], parse_target(r["user"], SC[r["id"]]), "parse") for r in nav]
pairs = [(x, y) for x, y, _ in rows] + [(x, y) for x, y, _ in ni + qi + pp]
tasks = [t for _, _, t in rows] + [t for _, _, t in ni + qi + pp]
order = list(range(len(pairs))); random.Random(1).shuffle(order)
# t5_multi shuffles the pair list itself with Random(1); shuffling indices with the same seed gives the same permutation
train = pd.DataFrame({"input": [pairs[i][0] for i in order], "target": [pairs[i][1] for i in order], "task": [tasks[i] for i in order]})
pq.write_table(pa.Table.from_pandas(train, preserve_index=False), f"{OUT}/data/train.parquet", compression="zstd")

def nav_test(split):
    rs = [json.loads(l) for l in open(f"data_reason/{split}.jsonl")]
    return pd.DataFrame([(nav_src(r), r["answer"], "directions") for r in rs] + [("parse: " + r["user"], parse_target(r["user"], SC[r["id"]]), "parse") for r in rs]
                        + [("intent: " + r["user"], "navigation", "intent") for r in rs], columns=["input", "target", "task"])
def qa_test(split):
    rs = load(f"ghanaqa/rag_sel_{split}.jsonl", 0)
    return pd.DataFrame([(qa_src(r), r["answer"], "answer") for r in rs] + [("intent: " + r["question"], "knowledge", "intent") for r in rs], columns=["input", "target", "task"])
tests = {"test_area": nav_test("test_area"), "test_pair": nav_test("test_pair"), "test_qa_b1": qa_test("test_b1"), "test_qa_b2": qa_test("test_b2")}
for k, d in tests.items(): pq.write_table(pa.Table.from_pandas(d, preserve_index=False), f"{OUT}/data/{k}.parquet", compression="zstd")
counts = {"train": len(train), **{k: len(d) for k, d in tests.items()}}
by_task = train.task.value_counts().to_dict()
json.dump(dict(counts=counts, train_by_task=by_task), open(f"{OUT}/stats.json", "w"), indent=1)
print("BUILT", counts, by_task)
