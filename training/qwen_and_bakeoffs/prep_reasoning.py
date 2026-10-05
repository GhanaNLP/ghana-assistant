"""Build chat data where the skeleton is the model's reasoning (<think>), followed by the spoken directions.
usage (from train/): python prep_reasoning.py kumasi accra  -> data_reason/{train,test_area,test_pair}.jsonl
Rows: id, task, city, system, user, skeleton, answer."""
import sys, json, os
sys.path.insert(0, ".."); from skeleton import skeleton_text
SYSTEM = ("You are a navigation assistant for Ghana. First work out the route and the landmarks as a short structured plan, then reply "
          "with short, spoken, landmark-based directions the way a local would: counted turns, a landmark at each turn to confirm it, "
          "no distances and no coordinates, in simple English.")
cities = sys.argv[1:]; os.makedirs("data_reason", exist_ok=True)
SC = {}
for c in cities:
    for l in open(f"../{c}_scenarios.jsonl"): d = json.loads(l); SC[d["id"]] = d
cnt = {}
out = {sp: open(f"data_reason/{sp}.jsonl", "w") for sp in ("train", "test_area", "test_pair")}
for c in cities:
    for l in open(f"../{c}_pairs.jsonl"):
        r = json.loads(l); sc = SC[r["id"]]
        out[r["split"]].write(json.dumps(dict(id=r["id"], task=r["task"], city=r["city"], system=SYSTEM, user=r["input"],
                                              skeleton=skeleton_text(sc), answer=r["output"]), ensure_ascii=False) + "\n")
        cnt[r["split"]] = cnt.get(r["split"], 0) + 1
print(cnt)
