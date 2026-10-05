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

elif stage in ("embed", "embed_check"):
    # Fast path: tokenise big chunks with the Rust tokenizer on all cores, then length-sorted fp16 batches on the GPU.
    # bge-small = [CLS] token, L2-normalised (same as sentence-transformers with normalize_embeddings=True).
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    import torch; from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(EMB); model = AutoModel.from_pretrained(EMB, dtype=torch.float16).cuda().eval()

    def embed(texts, name, max_len, bs=4096, chunk=500_000):
        n = len(texts); out = np.zeros((n, model.config.hidden_size), np.float16); t0 = time.time()
        for c0 in range(0, n, chunk):
            enc = tok(texts[c0:c0 + chunk], truncation=True, max_length=max_len, padding=False)["input_ids"]   # parallel over all cores
            lens = np.fromiter((len(x) for x in enc), dtype=np.int32, count=len(enc)); order = np.argsort(lens, kind="stable")
            for b0 in range(0, len(order), bs):
                idx = order[b0:b0 + bs]; L = int(lens[idx].max())
                ids = np.zeros((len(idx), L), np.int64); att = np.zeros((len(idx), L), np.int64)
                for j, i in enumerate(idx): ids[j, :lens[i]] = enc[i]; att[j, :lens[i]] = 1
                with torch.inference_mode():
                    h = model(input_ids=torch.from_numpy(ids).cuda(non_blocking=True), attention_mask=torch.from_numpy(att).cuda(non_blocking=True)).last_hidden_state[:, 0]
                    out[c0 + idx] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu().numpy()
            done = min(c0 + chunk, n); el = time.time() - t0
            print(f"{name}: {done:,}/{n:,} | {el:.0f}s | {done/el:,.0f}/s | ETA {el/done*(n-done):.0f}s", flush=True)
        return out

    S = pq.read_table(f"{OUT}/sentences.parquet", columns=["text"]).column("text").to_pylist()
    if stage == "embed_check":                                  # must match sentence-transformers on a sample
        from sentence_transformers import SentenceTransformer
        samp = S[:: max(1, len(S) // 2000)][:2000]
        ref = SentenceTransformer(EMB, device="cuda").encode(samp, normalize_embeddings=True, batch_size=256)
        mine = embed(samp, "check", 128, bs=512).astype(np.float32)
        cos = (ref * mine).sum(1); print(f"EMBED_CHECK cosine vs sentence-transformers: mean {cos.mean():.5f}, min {cos.min():.5f}")
    else:
        E = embed(S, "corpus", 256); np.save(f"{OUT}/sent_emb.npy", E)
        Q = pd.read_parquet(f"{OUT}/qa.parquet").question.tolist()
        QE = embed([QPFX + q for q in Q], "questions", 96); np.save(f"{OUT}/q_emb.npy", QE)
        print(f"EMBED_DONE corpus {E.shape} questions {QE.shape}")

elif stage == "retrieve":
    import torch
    S = pq.read_table(f"{OUT}/sentences.parquet").to_pandas(); qa = pd.read_parquet(f"{OUT}/qa.parquet")
    E = torch.from_numpy(np.load(f"{OUT}/sent_emb.npy")).cuda(); QE = torch.from_numpy(np.load(f"{OUT}/q_emb.npy")).cuda()
    top = np.zeros((len(qa), K), dtype=np.int64); sc = np.zeros((len(qa), K), dtype=np.float32); t0 = time.time()
    QB = 512                                                    # 512 x 6.3M fp16 scores = ~6.4 GB per step (the GPU is shared)
    for i in range(0, len(qa), QB):
        v, ix = torch.topk(QE[i:i + QB] @ E.T, K, dim=1); top[i:i + QB] = ix.cpu().numpy(); sc[i:i + QB] = v.float().cpu().numpy()
        if i % (QB * 400) == 0: print(f"  {i:,}/{len(qa):,} questions ({time.time()-t0:.0f}s)", flush=True)
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

elif stage == "nouns":
    # Noun index: spaCy POS tags on all cores -> nouns (lemmatised) + proper-noun phrases ("john mahama") per sentence -> inverted index.
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])        # the parser is needed for noun phrases
    t0 = time.time()

    HYPH = re.compile(r"\b([A-Z][a-zA-Z]+(?:-[A-Z][a-zA-Z]+)+)\b")      # Ga-Mashie, Sekondi-Takoradi, Asante-Akim

    EDGE_POS = ("DET", "PRON", "PUNCT", "PART", "CCONJ", "ADP", "SCONJ", "AUX", "SPACE")

    def phrase_of(span):
        """noun phrase -> cleaned key: no articles/pronouns/possessives/stop-words at the edges; words kept exactly as written
        (singular and plural are separate terms). "the Ghana's poultry industries" -> "ghana poultry industries"."""
        toks = [t for t in span if t.pos_ not in ("DET", "PRON", "PUNCT", "SPACE") and t.text.lower() not in ("'s", "’s")]
        while toks and (toks[0].is_stop or toks[0].pos_ in EDGE_POS or not (toks[0].is_alpha or toks[0].like_num)): toks.pop(0)
        while toks and (toks[-1].is_stop or toks[-1].pos_ in EDGE_POS or not toks[-1].is_alpha): toks.pop()
        if not toks: return None
        words = [t.text.lower() for t in toks]
        p = " ".join(words).replace(" - ", "-"); return p if len(p) >= 2 else None

    def terms_of(doc):
        """noun phrases (main signal) + names (>=2 letters, hyphenated, mis-tagged capitalised words) + single nouns (backup)"""
        out, run = set(), []
        for ch in doc.noun_chunks:
            p = phrase_of(ch)
            if p:
                out.add(p); w = p.split()
                if len(w) >= 3: out.add(" ".join(w[-2:]))          # "ghana poultry industry" also as "poultry industry"
        for t in doc:
            name = t.is_alpha and (t.pos_ == "PROPN" or (t.is_title and not t.is_sent_start and t.i > 0 and t.pos_ not in ("PRON", "DET", "AUX", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT")))
            if name:
                run.append(t.text.lower()); out.add(t.text.lower()); continue
            if t.text == "-" and run: continue
            if len(run) > 1: out.add(" ".join(run))
            run = []
            if t.pos_ == "NOUN" and t.is_alpha and len(t.text) > 2: out.add(t.text.lower())
        if len(run) > 1: out.add(" ".join(run))
        for h in HYPH.findall(doc.text):
            out.add(h.lower()); out.update(p.lower() for p in h.split("-"))
        return {x for x in out if len(x) >= 2}

    S = pq.read_table(f"{OUT}/sentences.parquet", columns=["text"]).column("text").to_pylist()
    vocab, post = {}, []
    for sid, doc in enumerate(nlp.pipe(S, batch_size=2000, n_process=18)):
        for w in terms_of(doc):
            j = vocab.get(w)
            if j is None: j = vocab[w] = len(post); post.append([])
            post[j].append(sid)
        if sid % 1_000_000 == 0: print(f"  corpus {sid:,}/{len(S):,} ({time.time()-t0:.0f}s)", flush=True)
    terms = list(vocab); offs = np.zeros(len(post) + 1, np.int64); offs[1:] = np.cumsum([len(p) for p in post])
    ids = np.fromiter((x for p in post for x in p), dtype=np.int32, count=int(offs[-1]))
    np.savez(f"{OUT}/noun_index.npz", offs=offs, ids=ids, N=len(S)); json.dump(terms, open(f"{OUT}/noun_terms.json", "w"))
    print(f"index: {len(terms):,} terms, {len(ids):,} postings ({time.time()-t0:.0f}s)", flush=True)
    Q = pd.read_parquet(f"{OUT}/qa.parquet").question.tolist()
    with open(f"{OUT}/q_nouns.jsonl", "w") as f:
        for d in nlp.pipe(Q, batch_size=2000, n_process=18): f.write(json.dumps(sorted(terms_of(d))) + "\n")
    print(f"NOUNS_DONE {time.time()-t0:.0f}s")

elif stage == "retrieve_nouns":
    import multiprocessing as mp
    Z = np.load(f"{OUT}/noun_index.npz"); OFFS, IDS, N = Z["offs"], Z["ids"], int(Z["N"])
    VOC = {w: i for i, w in enumerate(json.load(open(f"{OUT}/noun_terms.json")))}
    DF = np.diff(OFFS); IDF = np.log((N + 1) / (DF + 1)); COMMON = 50_000          # nouns in >50k sentences only re-rank

    def search(q_terms, k=K):
        tj = [VOC[w] for w in q_terms if w in VOC]
        if not tj: return [], []
        rare = [j for j in tj if DF[j] <= COMMON]; common = [j for j in tj if DF[j] > COMMON]
        base = rare or sorted(common, key=lambda j: DF[j])[:1]
        cand = np.concatenate([IDS[OFFS[j]:OFFS[j + 1]] for j in base]); w = np.concatenate([np.full(DF[j], IDF[j], np.float32) for j in base])
        u, inv = np.unique(cand, return_inverse=True); sc = np.bincount(inv, weights=w).astype(np.float32)
        for j in (common if rare else [x for x in common if x not in base]):
            p = IDS[OFFS[j]:OFFS[j + 1]]; pos = np.searchsorted(p, u); pos[pos >= len(p)] = 0; sc += np.where(p[pos] == u, IDF[j], 0).astype(np.float32)
        top = np.argsort(-sc)[:k]; return u[top].tolist(), sc[top].round(3).tolist()

    QN = [json.loads(l) for l in open(f"{OUT}/q_nouns.jsonl")]
    t0 = time.time()
    with mp.Pool(18) as pool: R = pool.map(search, QN, chunksize=2000)
    print(f"searched {len(QN):,} questions in {time.time()-t0:.0f}s", flush=True)
    S = pq.read_table(f"{OUT}/sentences.parquet").to_pandas(); qa = pd.read_parquet(f"{OUT}/qa.parquet")
    text, doc = S.text.values, S.doc.values; outs = {sp: open(f"{OUT}/rag_noun_{sp}.jsonl", "w") for sp in ("train", "test_b1", "test_b2")}
    hit1 = hit8 = nb2 = empty = 0
    for r, qn, (top, sc) in zip(qa.itertuples(), QN, R):
        empty += not top; docs = [doc[i] for i in top]
        rec = dict(qid=int(r.qid), batch=r.batch, question=r.question, nouns=qn, contexts=[text[i] for i in top], context_docs=docs, scores=sc, answer=r.answer, source_url=r.source_url)
        if r.batch == "b2" and r.source_url:
            nb2 += 1; h = [d == r.source_url for d in docs]; hit1 += bool(h and h[0]); hit8 += any(h); rec["source_hit"] = any(h)
        outs[r.split].write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"RETRIEVE_NOUNS_DONE | questions with no usable noun: {empty:,} | batch-2 source article in top-1: {100*hit1/nb2:.1f}% | in top-{K}: {100*hit8/nb2:.1f}%")

elif stage == "meta":
    # per sentence: article date (YYYY-MM; research has none) and exact Flan-T5 token length (for the context budget)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    S = pq.read_table(f"{OUT}/sentences.parquet", columns=["doc", "text", "source"]).to_pandas(); dates = {}
    for f in ("citinews_scraped-data.csv", "myjoyonline.csv", "ghanaweb.csv"):
        d = pd.read_csv(f"{RAW}/data/news/{f}", usecols=["url", "date"]); p = pd.to_datetime(d.date, errors="coerce", utc=True, format="mixed")
        dates.update(zip(d.url.astype(str), p.dt.strftime("%Y-%m").fillna("")))
    date = [("" if src == "knust" else dates.get(doc, "")) for doc, src in zip(S.doc.values, S.source.values)]
    tok = AutoTokenizer.from_pretrained("google/flan-t5-small"); L = np.zeros(len(S), np.int16); T = S.text.tolist()
    for i in range(0, len(T), 500_000): L[i:i + 500_000] = [len(x) for x in tok(T[i:i + 500_000], add_special_tokens=False)["input_ids"]]
    pq.write_table(pa.table({"sid": S.index.values, "date": date, "toklen": L}), f"{OUT}/sentence_meta.parquet")
    print(f"META_DONE dated sentences {sum(bool(x) for x in date):,}/{len(S):,} | median tokens {int(np.median(L))}")

elif stage == "select":
    # Inference-identical context selection: phrase/noun match (rarity-weighted) with a gentle recency boost (<= +5%),
    # filled to a 600-token budget, <= 3 sentences per article, near-duplicates skipped, lines prefixed with [YYYY-MM] / [research].
    import multiprocessing as mp
    BUDGET, PER_DOC, POOL, LINE_COST = 600, 3, 300, 6
    Z = np.load(f"{OUT}/noun_index.npz"); OFFS, IDS, N = Z["offs"], Z["ids"], int(Z["N"])
    VOC = {w: i for i, w in enumerate(json.load(open(f"{OUT}/noun_terms.json")))}
    DF = np.diff(OFFS); IDF = np.log((N + 1) / (DF + 1)); COMMON = 50_000
    S = pq.read_table(f"{OUT}/sentences.parquet", columns=["doc", "text"]).to_pandas(); M = pq.read_table(f"{OUT}/sentence_meta.parquet").to_pandas()
    TEXT = S.text.values; DOCI, DOCS = pd.factorize(S.doc.values); DATE = M.date.values; TOKL = M.toklen.values.astype(np.int32)
    yr = pd.to_numeric(pd.Series(DATE).str[:4], errors="coerce").values + (pd.to_numeric(pd.Series(DATE).str[5:7], errors="coerce").values - 1) / 12
    REC = np.where(np.isnan(yr), 0.5, np.clip((yr - 2010) / 16, 0, 1)).astype(np.float32)     # undated research = neutral

    sys.path.insert(0, "ga_pkg"); from ghana_assistant.knowledge.phrases import rank_and_fill   # the exact runtime code

    def select(q_terms):
        return rank_and_fill([VOC[w] for w in sorted(q_terms) if w in VOC], OFFS, IDS, REC, DOCI, TOKL, lambda sids: [TEXT[s] for s in sids])

    QN = [json.loads(l) for l in open(f"{OUT}/q_nouns.jsonl")]; t0 = time.time()
    with mp.Pool(18) as pool: R = pool.map(select, QN, chunksize=2000)
    print(f"selected contexts for {len(QN):,} questions in {time.time()-t0:.0f}s", flush=True)
    qa = pd.read_parquet(f"{OUT}/qa.parquet"); outs = {sp: open(f"{OUT}/rag_sel_{sp}.jsonl", "w") for sp in ("train", "test_b1", "test_b2")}
    hit = nb2 = 0; nctx = []
    for r, qn, sel in zip(qa.itertuples(), QN, R):
        lines = [f"[{DATE[s] or 'research'}] {TEXT[s]}" for s in sel]; docs = [DOCS[DOCI[s]] for s in sel]; nctx.append(len(sel))
        rec = dict(qid=int(r.qid), batch=r.batch, question=r.question, nouns=qn, contexts=lines, context_docs=docs, answer=r.answer, source_url=r.source_url)
        if r.batch == "b2" and r.source_url: nb2 += 1; h = r.source_url in docs; hit += h; rec["source_hit"] = h
        outs[r.split].write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"SELECT_DONE | sentences per question: mean {np.mean(nctx):.1f}, median {int(np.median(nctx))} | no context: {sum(n == 0 for n in nctx):,} | batch-2 source article selected: {100*hit/nb2:.1f}%")

elif stage == "index":
    import faiss
    E = np.load(f"{OUT}/sent_emb.npy").astype(np.float32); d = E.shape[1]; nlist = 8192
    q = faiss.IndexFlatIP(d); idx = faiss.IndexIVFPQ(q, d, nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
    samp = E[np.random.RandomState(0).choice(len(E), min(len(E), 600000), replace=False)]
    t0 = time.time(); idx.train(samp); idx.add(E); idx.nprobe = 32
    faiss.write_index(idx, f"{OUT}/ghanaqa_ivfpq.faiss"); print(f"INDEX_DONE {idx.ntotal:,} vectors, {os.path.getsize(f'{OUT}/ghanaqa_ivfpq.faiss')/1e6:.0f} MB, {time.time()-t0:.0f}s")
