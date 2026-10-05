"""GhanaQA retrieval (RAG) training set: news + KNUST sentences -> bge-small embeddings -> top-k per question -> (question, contexts, answer).
stages (run from map-nav/):  python ghanaqa_build.py corpus | qa | embed | retrieve | index"""
import sys, os, re, json, time, random, glob
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
RAW, OUT = "ghanaqa/raw", "ghanaqa"; os.makedirs(OUT, exist_ok=True)
EMB = "BAAI/bge-small-en-v1.5"; QPFX = "Represent this sentence for searching relevant passages: "
K = 8; N_TEST = 2000; stage = sys.argv[1]
GLUE = re.compile(r"([a-z0-9)\]][.!?…])([A-Z])")                      # "decade.Then" -> "decade. Then"
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(“])")
JUNK = re.compile(r"(read more|also read|click here|subscribe|follow us|advertis|copyright|all rights reserved|sign up|newsletter|share this|whatsapp channel|join our|download our)", re.I)


def sents(text):
    text = GLUE.sub(r"\1 \2", str(text).replace("\n", " ").replace("…", " ")).strip()
    return [s.strip() for s in SPLIT.split(re.sub(r"\s+", " ", text)) if 30 <= len(s.strip()) <= 400 and not JUNK.search(s) and re.search(r"[a-zA-Z]{3}", s)]


if stage == "corpus":
    t0 = time.time(); seen = set(); rows = []
    for f, src in [(f"{RAW}/data/news/citinews_scraped-data.csv", "citinews"), (f"{RAW}/data/news/myjoyonline.csv", "myjoyonline"),
                   (f"{RAW}/data/news/ghanaweb.csv", "ghanaweb"), (f"{RAW}/data/research/knust-tokenised-data.csv", "knust")]:
        n0 = len(rows); docs = 0
        for ch in pd.read_csv(f, chunksize=20000):
            for r in ch.itertuples():
                doc = f"{r.filename}#{r.page_range}" if src == "knust" else str(r.url)
                docs += 1
                for s in sents(r.content):
                    k = s.lower()
                    if k in seen: continue
                    seen.add(k); rows.append((src, doc, s))
        print(f"{src}: {docs:,} docs -> {len(rows)-n0:,} sentences ({time.time()-t0:.0f}s)", flush=True)
    pq.write_table(pa.table({"sid": list(range(len(rows))), "source": [r[0] for r in rows], "doc": [r[1] for r in rows], "text": [r[2] for r in rows]}), f"{OUT}/sentences.parquet")
    print(f"CORPUS_DONE {len(rows):,} sentences"); random.seed(1); print("samples:", [r[2] for r in random.sample(rows, 6)])

elif stage == "qa":
    qa = []
    for r in pd.read_csv(f"{RAW}/output/batch_1/cleaning/parallel_qa.csv").itertuples():
        if isinstance(r.question, str) and isinstance(r.answer, str): qa.append(("b1", r.question.strip(), r.answer.strip(), ""))
    PAIR = re.compile(r"\*{0,2}Q\d+[:.]\*{0,2}\s*(.+?)\s*\n+\s*\*{0,2}A\d+[:.]\*{0,2}\s*(.+?)(?=\n\s*\*{0,2}Q\d+[:.]|\Z)", re.S)
    for f in sorted(glob.glob(f"{RAW}/output/batch_2/*.csv")):
        for r in pd.read_csv(f).itertuples():
            for q, a in PAIR.findall(str(r.generated_queries)):
                q, a = q.strip().strip("*").strip(), re.sub(r"\s+", " ", a).strip().strip("*").strip()
                if len(q) > 10 and len(a) > 10: qa.append(("b2", q, a, str(r.url)))
    seen = set(); uniq = []
    for x in qa:
        k = x[1].lower()
        if k not in seen: seen.add(k); uniq.append(x)
    random.Random(0).shuffle(uniq)
    b1 = [x for x in uniq if x[0] == "b1"]; b2 = [x for x in uniq if x[0] == "b2"]
    split = {}
    for x in b1[:N_TEST]: split[x[1]] = "test_b1"
    for x in b2[:N_TEST]: split[x[1]] = "test_b2"
    df = pd.DataFrame(uniq, columns=["batch", "question", "answer", "source_url"]); df["split"] = df.question.map(lambda q: split.get(q, "train"))
    df.insert(0, "qid", range(len(df))); df.to_parquet(f"{OUT}/qa.parquet")
    print(f"QA_DONE raw {len(qa):,} -> unique {len(uniq):,} | b1 {len(b1):,}, b2 {len(b2):,} | splits {df.split.value_counts().to_dict()}")
    print(df.sample(3, random_state=2)[["question", "answer"]].to_dict("records"))

elif stage == "embed":
    import torch; from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB, device="cuda"); m.half()
    t0 = time.time(); S = pq.read_table(f"{OUT}/sentences.parquet", columns=["text"]).column("text").to_pylist()
    E = m.encode(S, batch_size=1024, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    np.save(f"{OUT}/sent_emb.npy", E); print(f"corpus emb {E.shape} {time.time()-t0:.0f}s", flush=True)
    Q = pd.read_parquet(f"{OUT}/qa.parquet").question.tolist()
    QE = m.encode([QPFX + q for q in Q], batch_size=1024, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    np.save(f"{OUT}/q_emb.npy", QE); print(f"EMBED_DONE questions {QE.shape} {time.time()-t0:.0f}s")

elif stage == "retrieve":
    import torch
    S = pq.read_table(f"{OUT}/sentences.parquet").to_pandas(); qa = pd.read_parquet(f"{OUT}/qa.parquet")
    E = torch.from_numpy(np.load(f"{OUT}/sent_emb.npy")).cuda(); QE = torch.from_numpy(np.load(f"{OUT}/q_emb.npy")).cuda()
    top = np.zeros((len(qa), K), dtype=np.int64); sc = np.zeros((len(qa), K), dtype=np.float32); t0 = time.time()
    for i in range(0, len(qa), 4096):
        s = QE[i:i + 4096] @ E.T; v, ix = torch.topk(s.float(), K, dim=1); top[i:i + 4096] = ix.cpu().numpy(); sc[i:i + 4096] = v.cpu().numpy()
    print(f"retrieved {len(qa):,} x top{K} in {time.time()-t0:.0f}s", flush=True)
    text, doc = S.text.values, S.doc.values
    outs = {sp: open(f"{OUT}/rag_{sp}.jsonl", "w") for sp in ("train", "test_b1", "test_b2")}; hit1 = hit8 = nb2 = 0
    for j, r in enumerate(qa.itertuples()):
        docs = [doc[k] for k in top[j]]
        rec = dict(qid=int(r.qid), batch=r.batch, question=r.question, contexts=[text[k] for k in top[j]], context_docs=docs,
                   scores=[round(float(x), 4) for x in sc[j]], answer=r.answer, source_url=r.source_url)
        if r.batch == "b2" and r.source_url:
            nb2 += 1; h = [d == r.source_url for d in docs]; hit1 += h[0]; hit8 += any(h); rec["source_hit"] = any(h)
        outs[r.split].write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"RETRIEVE_DONE | batch-2 questions with known source article: {nb2:,} | source article in top-1: {100*hit1/max(nb2,1):.1f}% | in top-{K}: {100*hit8/max(nb2,1):.1f}%")

elif stage == "index":
    import faiss
    E = np.load(f"{OUT}/sent_emb.npy").astype(np.float32); d = E.shape[1]; nlist = 8192
    q = faiss.IndexFlatIP(d); idx = faiss.IndexIVFPQ(q, d, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
    samp = E[np.random.RandomState(0).choice(len(E), min(len(E), 600000), replace=False)]
    t0 = time.time(); idx.train(samp); idx.add(E); idx.nprobe = 32
    faiss.write_index(idx, f"{OUT}/ghanaqa_ivfpq.faiss"); print(f"INDEX_DONE {idx.ntotal:,} vectors, {os.path.getsize(f'{OUT}/ghanaqa_ivfpq.faiss')/1e6:.0f} MB, {time.time()-t0:.0f}s")
