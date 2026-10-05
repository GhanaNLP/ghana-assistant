"""Score translation pilot files (*.jsonl with en_output / back_output / checks) with idiom-aware left/right and ordinal checks."""
import json, re, sys, glob
from collections import Counter
IDIOM = re.compile(r"\bright\s+(beside|ahead|there|away|in front|next|at|by|now|here|behind|after|before|on|outside|opposite|across|in your)\b|\bthe right (turn|way|road|track|one|direction|place|route|path|junction|street)\b|\ball right\b", re.I)
ORD = re.compile(r"\b(first|second|third)\s+(?:(?:slight|sharp)\s+)?(?:(?:road|turn|turning|junction|street|corner|exit)\s+)?(?:(?:on|to)\s+(?:the|your)\s+)?(left|right)\b", re.I)
sides = lambda t: Counter(w.lower() for w in re.findall(r"\b(left|right)\b", IDIOM.sub(" ", t), re.I))
ords = lambda t: Counter((o.lower(), s.lower()) for o, s in ORD.findall(t))
for f in sorted(sum([glob.glob(p) for p in sys.argv[1:]], [])):
    R = [json.loads(l) for l in open(f)]; n = len(R)
    s_ok = [sides(r["en_output"]) == sides(r["back_output"]) for r in R]
    o_ok = [ords(r["en_output"]) == ords(r["back_output"]) or not ords(r["en_output"]) for r in R]
    nm = [r["checks"]["names_kept"] and r["checks"]["translated"] for r in R]
    allok = sum(a and b and c for a, b, c in zip(s_ok, o_ok, nm))
    print(f"{f.split('/')[-1][:-6]:38} n={n} | names {100*sum(nm)/n:.0f}% | left/right {100*sum(s_ok)/n:.0f}% | ordinals {100*sum(o_ok)/n:.0f}% | ALL {100*allok/n:.0f}%")
