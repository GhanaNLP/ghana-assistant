"""Runtime PhraseStore must pick exactly the training contexts."""
import sys, json, time, random
sys.path.insert(0, "ga_pkg")
from ghana_assistant.knowledge.phrases import PhraseStore
ps = PhraseStore("ga_out/knowledge"); rows = [json.loads(l) for l in open("ghanaqa/rag_sel_test_b2.jsonl")]; random.Random(3).shuffle(rows); rows = rows[:300]
same = sum([x["line"] for x in ps.search(r["question"])] == r["contexts"] for r in rows)
print(f"VERIFY identical {same}/{len(rows)}")
