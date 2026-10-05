"""Grade GhanaQA answers with Gemini: correct / partly correct / wrong, supported by context, repetitive, misleading.
usage: python grade_qa.py results/qa_preds.jsonl"""
import sys, json, time, requests, concurrent.futures as cf
from collections import Counter
KEY = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]; MODEL = "gemini-3.8-flash"
PROMPT = """You are grading a small assistant's answer to a question about Ghana.

QUESTION: {q}

CONTEXT the assistant was given (retrieved news/research sentences):
{ctx}

REFERENCE ANSWER (written by a strong model with the source article): {ref}

ASSISTANT ANSWER: {pred}

Judge the ASSISTANT ANSWER:
- "correct": "correct" if it answers the question with the same essential meaning as the reference (wording may differ), "partial" if it is on topic but incomplete, vague or partly wrong, "wrong" if it is incorrect, off-topic or contradicts the reference.
- "supported": "yes" / "partly" / "no": are its claims backed by the CONTEXT?
- "repetitive": true if it repeats words or phrases unnaturally.
- "misleading": true if it states something confidently that is false and could mislead a user (e.g. wrong legal, health or financial advice).
Return JSON: {{"correct": "...", "supported": "...", "repetitive": true/false, "misleading": true/false, "note": "<few words>"}}"""


def grade(args):
    r, field = args
    p = PROMPT.format(q=r["question"], ctx="\n".join(r["contexts"]) or "(none)", ref=r["reference"], pred=r[field])
    body = {"contents": [{"parts": [{"text": p}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0, "thinkingConfig": {"thinkingBudget": 0}}}
    for t in range(5):
        try:
            x = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={KEY}", json=body, timeout=120)
            if x.status_code in (429, 503): time.sleep(3 * (t + 1)); continue
            parts = x.json()["candidates"][0]["content"]["parts"]
            return json.loads("".join(q.get("text", "") for q in parts if not q.get("thought")))
        except Exception: time.sleep(2)
    return None


rows = [json.loads(l) for l in open(sys.argv[1])]
for field in ("pred_greedy", "pred_fixed"):
    with cf.ThreadPoolExecutor(16) as ex: G = list(ex.map(grade, [(r, field) for r in rows]))
    ok = [(r, gr) for r, gr in zip(rows, G) if gr]; n = len(ok)
    c = Counter(gr["correct"] for _, gr in ok); s = Counter(gr["supported"] for _, gr in ok)
    print(f"\n== {field} (graded {n}/{len(rows)})")
    print(f"  correct {100*c['correct']/n:.0f}% | partial {100*c['partial']/n:.0f}% | wrong {100*c['wrong']/n:.0f}%")
    print(f"  supported by context: yes {100*s['yes']/n:.0f}% | partly {100*s['partly']/n:.0f}% | no {100*s['no']/n:.0f}%")
    print(f"  repetitive {100*sum(gr['repetitive'] for _, gr in ok)/n:.0f}% | misleading {100*sum(gr['misleading'] for _, gr in ok)/n:.0f}%")
    for key, lab in (("hit", True), ("miss", False)):
        sub = [gr for r, gr in ok if r["batch"] == "b2" and r["source_hit"] is lab]
        if sub: print(f"  b2 source article {'retrieved' if lab else 'NOT retrieved'} (n={len(sub)}): correct {100*sum(x['correct']=='correct' for x in sub)/len(sub):.0f}%")
    json.dump([dict(r, grade=gr, field=field) for r, gr in zip(rows, G)], open(sys.argv[1].replace(".jsonl", f"_graded_{field}.json"), "w"), indent=1, ensure_ascii=False)
