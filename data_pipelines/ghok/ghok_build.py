"""Build the GHOK retrieval (RAG) training set: sentence corpus -> embeddings -> top-k per prompt -> (question, contexts, answer).
stages:  python ghok_build.py split | embed | retrieve | index"""
import sys, os, re, json, time, random
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
OUT = "ghok"; os.makedirs(OUT, exist_ok=True)
EMB = "BAAI/bge-small-en-v1.5"; QPFX = "Represent this sentence for searching relevant passages: "
K = 8; N_TEST = 2000
stage = sys.argv[1]

META = re.compile(r"(reference material|provided (text|material|context|document|information|reference)|user('s)? request|^here (are|is)\b|^in summary\b|^response to)", re.I)
MD = [(re.compile(r"^\s*#{1,6}\s*"), ""), (re.compile(r"\*\*|__|`"), ""), (re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+"), ""), (re.compile(r"\s+"), " ")]
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def sentences(text):
    out = []
    for line in text.split("\n"):
        for pat, rep in MD: line = pat.sub(rep, line)
        line = line.strip()
        for s in SPLIT.split(line):
            s = s.strip().strip("*").strip()
            if 25 <= len(s) <= 400 and not META.search(s) and re.search(r"[a-zA-Z]{3}", s): out.append(s)
    return out


if stage == "split":
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    f = hf_hub_download("ghananlpcommunity/ghok-parallel-en-twi", "data/train.parquet", repo_type="dataset")
    t = pq.read_table(f, columns=["source_prompt", "english"]); prompts = t.column("source_prompt").to_pylist(); answers = t.column("english").to_pylist()
    print("rows", len(prompts), f"{time.time()-t0:.0f}s", flush=True)
    ids = list(range(len(prompts))); random.Random(0).shuffle(ids); test = set(ids[:N_TEST])
    pq.write_table(pa.table({"doc": list(range(len(prompts))), "prompt": prompts, "answer": answers,
                             "split": ["test" if i in test else "train" for i in range(len(prompts))]}), f"{OUT}/docs.parquet")
    seen = {}; S_text, S_doc = [], []; dup = 0; empty = 0
    for d, ans in enumerate(answers):
        ss = sentences(ans or "")
        if not ss: empty += 1
        for s in ss:
            k = s.lower()
            if k in seen: dup += 1; continue
            seen[k] = 1; S_text.append(s); S_doc.append(d)
        if d % 100000 == 0: print(d, "docs ->", len(S_text), "sentences", flush=True)
    pq.write_table(pa.table({"sid": list(range(len(S_text))), "doc": S_doc, "text": S_text}), f"{OUT}/sentences.parquet")
    L = [len(s) for s in S_text]
    print(f"SPLIT_DONE sentences {len(S_text):,} | duplicates dropped {dup:,} | answers with no usable sentence {empty:,} | "
          f"avg {np.mean(L):.0f} chars, median {np.median(L):.0f} | {time.time()-t0:.0f}s")
    random.seed(1); print("samples:", random.sample(S_text, 8))
