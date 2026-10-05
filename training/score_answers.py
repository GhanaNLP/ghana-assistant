"""Quality scoring of router-mode answers against the skeleton (no model needed), optional Gemini judge.
usage: python score_answers.py data_reason/test_area.jsonl results/x_preds.jsonl [--ref] [--judge N]
 --ref scores the reference (Gemini) answers in the data file instead of predictions."""
import sys, json, re, random, os, time, concurrent.futures as cf
ORDW = {"first": "first|1st", "second": "second|2nd", "third": "third|3rd"}
def norm(s): return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).strip()
CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][\w'’&.-]*(?:\s+(?:of|and|&|the)?\s*[A-Z][\w'’&.-]*)*)")

def parse_sk(sk):
    roads, lms, ords = [], [], []
    for l in sk.splitlines():
        m = re.match(r"^\d+\.\s+(.*)$", l.strip())
        if not m: continue
        t = m.group(1)
        r = re.search(r"(?:on|onto) (.+?)(?: \(unnamed\))?(?:,|;|$)", t)
        if r and "(unnamed)" not in t.split(";")[0] and not t.startswith("Head") or (r and t.startswith("Head") and "(unnamed)" not in t): roads.append(r.group(1).strip())
        o = re.search(r"the (first|second|third) (left|right)", t)
        if o: ords.append((o.group(1), o.group(2)))
        s = re.search(r"sees (.+)$", t)
        if s: lms += [x.strip() for x in re.findall(r"([^,;()]+?) \((?:on your (?:left|right)|straight ahead)\)", s.group(1))]
    return roads, lms, ords

def score(row, text):
    roads, lms, ords = parse_sk(row["skeleton"]); T = norm(text); out = {}
    out["words"] = len(text.split())
    out["ordinal_recall"] = (sum(bool(re.search(rf"\b({ORDW[o]})\s+(?:slight\s+|sharp\s+)?{s}\b", text, re.I)) for o, s in ords) / len(ords)) if ords else None
    out["landmark_recall"] = (sum(norm(x) in T for x in lms) / len(lms)) if lms else None
    out["road_recall"] = (sum(norm(x) in T for x in roads) / len(roads)) if roads else None
    # order: first-mention positions of skeleton roads+landmarks should increase
    seq = [T.find(norm(x)) for x in roads + lms]; pos = [p for p in seq if p >= 0]
    out["order_ok"] = (sum(a < b for a, b in zip(pos, pos[1:])) / (len(pos) - 1)) if len(pos) > 1 else None
    known = norm(row["skeleton"] + " " + row["user"]); extra = [c for c in CAP.findall(text) if len(norm(c)) > 3 and norm(c) not in known and not any(norm(c) in k or k in norm(c) for k in map(norm, lms + roads))]
    out["extra_names"] = len(extra); out["has_extra"] = bool(extra); out["extra_examples"] = extra[:3]
    return out

def agg(rows):
    keys = ["words", "ordinal_recall", "landmark_recall", "road_recall", "order_ok", "extra_names", "has_extra"]
    return {k: round(sum(r[k] for r in rows if r[k] is not None) / max(1, sum(r[k] is not None for r in rows)), 3) for k in keys}

JUDGE = """You are grading directions written for a landmark-based navigation assistant in Ghana.
USER REQUEST: {user}

GROUND-TRUTH ROUTE PLAN (facts from the map; the directions must follow it):
{sk}

DIRECTIONS TO GRADE:
{ans}

Score each from 1 (poor) to 5 (excellent):
- faithful: follows the plan, no invented roads/landmarks/turns, counted turns match
- complete: includes the key turns and the landmarks that confirm them
- clear: a local could follow it easily, plain simple English, natural spoken style
Return JSON: {{"faithful": n, "complete": n, "clear": n, "problem": "<one short phrase or none>"}}"""

def judge(args):
    row, text, key = args
    body = {"contents": [{"parts": [{"text": JUDGE.format(user=row["user"], sk=row["skeleton"], ans=text)}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}}
    for a in range(4):
        try:
            import requests
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent?key={key}", json=body, timeout=120)
            if r.status_code in (429, 503): time.sleep(3 * (a + 1)); continue
            return json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
        except Exception: pass
    return None

if __name__ == "__main__":
    data = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
    items = [(data[json.loads(l)["id"]], json.loads(l)["pred"]) for l in open(sys.argv[2])] if "--ref" not in sys.argv else None
    if items is None:
        ids = [json.loads(l)["id"] for l in open(sys.argv[2])]; items = [(data[i], data[i]["answer"]) for i in ids]
    res = [score(r, t) for r, t in items]; print(json.dumps(agg(res)))
    if "--judge" in sys.argv:
        N = int(sys.argv[sys.argv.index("--judge") + 1]); key = dict(l.strip().split("=", 1) for l in open(".env") if "=" in l)["GEMINI_API_KEY"]
        sub = items[:N]
        with cf.ThreadPoolExecutor(16) as ex: J = list(ex.map(judge, [(r, t, key) for r, t in sub]))
        J = [j for j in J if j]; print("judge n =", len(J), {k: round(sum(j[k] for j in J) / len(J), 2) for k in ("faithful", "complete", "clear")})
