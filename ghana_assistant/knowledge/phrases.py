"""Noun-phrase retrieval for the knowledge skill (no embeddings, no FAISS).

Index:  noun phrases + names + single nouns per sentence (spaCy), kept exactly as written (plural != singular),
        stored as an inverted index (postings) that is memory-mapped from disk.
Select: rarity-weighted phrase match, gentle recency boost (<= +5%), filled to a 600-token budget,
        at most 3 sentences per article, near-duplicates skipped; lines prefixed [YYYY-MM] or [research].
The same functions build the training contexts, so training and inference see identical input.

Store layout (a folder):
  postings_offs.npy  int64 [n_terms+1]   postings_ids.npy int32 [n_postings]   (memory-mapped)
  rec.npy float32, docidx.npy int32, toklen.npy int16  [n_sentences]          (memory-mapped)
  store.sqlite   terms(term -> tid), sentences(sid -> text, date)
"""
import os, re, sqlite3
import numpy as np

BUDGET, PER_DOC, POOL, LINE_COST, COMMON, RECENCY = 600, 3, 300, 6, 50_000, 0.05
HYPH = re.compile(r"\b([A-Z][a-zA-Z]+(?:-[A-Z][a-zA-Z]+)+)\b")              # Ga-Mashie, Sekondi-Takoradi
EDGE_POS = ("DET", "PRON", "PUNCT", "PART", "CCONJ", "ADP", "SCONJ", "AUX", "SPACE")
NOT_NAME_POS = ("PRON", "DET", "AUX", "ADP", "CCONJ", "SCONJ", "PART", "PUNCT")


def load_nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])                    # parser is needed for noun phrases


def phrase_of(span):
    """noun phrase -> key without articles/pronouns/possessives/stop-words at the edges, words kept as written."""
    toks = [t for t in span if t.pos_ not in ("DET", "PRON", "PUNCT", "SPACE") and t.text.lower() not in ("'s", "’s")]
    while toks and (toks[0].is_stop or toks[0].pos_ in EDGE_POS or not (toks[0].is_alpha or toks[0].like_num)): toks.pop(0)
    while toks and (toks[-1].is_stop or toks[-1].pos_ in EDGE_POS or not toks[-1].is_alpha): toks.pop()
    if not toks: return None
    p = " ".join(t.text.lower() for t in toks).replace(" - ", "-")
    return p if len(p) >= 2 else None


def terms_of(doc):
    """noun phrases (+ last two words of long ones) + names (>=2 letters, hyphenated, mis-tagged capitalised words) + single nouns"""
    out, run = set(), []
    for ch in doc.noun_chunks:
        p = phrase_of(ch)
        if p:
            out.add(p); w = p.split()
            if len(w) >= 3: out.add(" ".join(w[-2:]))
    for t in doc:
        name = t.is_alpha and (t.pos_ == "PROPN" or (t.is_title and not t.is_sent_start and t.i > 0 and t.pos_ not in NOT_NAME_POS))
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


def rank_and_fill(tids, OFFS, IDS, REC, DOCI, TOKL, text_of):
    """core selection shared by training-data building and inference. tids: term ids of the question.
    text_of(list of sids) -> list of texts (for near-duplicate checks). Returns selected sentence ids in order."""
    if not tids: return []
    DFq = {j: int(OFFS[j + 1] - OFFS[j]) for j in tids}; N = len(DOCI)
    IDF = {j: float(np.log((N + 1) / (DFq[j] + 1))) for j in tids}
    rare = [j for j in tids if DFq[j] <= COMMON]; common = [j for j in tids if DFq[j] > COMMON]
    base = rare or sorted(common, key=lambda j: DFq[j])[:1]
    cand = np.concatenate([np.asarray(IDS[OFFS[j]:OFFS[j + 1]]) for j in base])
    w = np.concatenate([np.full(DFq[j], IDF[j], np.float32) for j in base])
    u, inv = np.unique(cand, return_inverse=True); sc = np.bincount(inv, weights=w).astype(np.float32)
    for j in (common if rare else [x for x in common if x not in base]):
        p = np.asarray(IDS[OFFS[j]:OFFS[j + 1]]); pos = np.searchsorted(p, u); pos[pos >= len(p)] = 0
        sc += np.where(p[pos] == u, IDF[j], 0).astype(np.float32)
    sc = sc * (1 + RECENCY * np.asarray(REC[u]))
    pool = u[np.argsort(-sc)[:POOL]]                                         # same sort as the training-data builder
    texts = text_of([int(s) for s in pool])
    used, per, seen, out = 0, {}, [], []
    for s, txt in zip(pool, texts):
        d = int(DOCI[s])
        if per.get(d, 0) >= PER_DOC: continue
        toks = set(re.findall(r"[a-z0-9]+", txt.lower()))
        if any(len(toks & t) / max(1, len(toks | t)) > 0.8 for t in seen): continue
        cost = int(TOKL[s]) + LINE_COST
        if used + cost > BUDGET: continue
        used += cost; per[d] = per.get(d, 0) + 1; seen.append(toks); out.append(int(s))
        if used >= BUDGET - 20: break
    return out


class PhraseStore:
    """Memory-light runtime store: postings and per-sentence arrays are memory-mapped; terms and text live in SQLite."""

    def __init__(self, folder, nlp=None):
        mm = lambda f: np.load(os.path.join(folder, f), mmap_mode="r")
        self.OFFS, self.IDS = mm("postings_offs.npy"), mm("postings_ids.npy")
        self.REC, self.DOCI, self.TOKL = mm("rec.npy"), mm("docidx.npy"), mm("toklen.npy")
        self.db = sqlite3.connect(os.path.join(folder, "store.sqlite"), check_same_thread=False)
        self.nlp = nlp or load_nlp()

    def _tids(self, terms):
        if not terms: return []
        q = ",".join("?" * len(terms))
        found = dict(self.db.execute(f"SELECT term, tid FROM terms WHERE term IN ({q})", list(terms)))
        return [found[t] for t in sorted(found)]                                 # alphabetical, as in training (sums in the same order)

    def _texts(self, sids):
        if not sids: return []
        rows = dict(self.db.execute(f"SELECT sid, text FROM sentences WHERE sid IN ({','.join('?' * len(sids))})", sids))
        return [rows[s] for s in sids]

    def search(self, question):
        """-> list of dicts {sid, text, date, line} in the order given to the model"""
        sel = rank_and_fill(self._tids(terms_of(self.nlp(question))), self.OFFS, self.IDS, self.REC, self.DOCI, self.TOKL, self._texts)
        if not sel: return []
        rows = {s: (t, d) for s, t, d in self.db.execute(f"SELECT sid, text, date FROM sentences WHERE sid IN ({','.join('?' * len(sel))})", sel)}
        return [dict(sid=s, text=rows[s][0], date=rows[s][1], line=f"[{rows[s][1] or 'research'}] {rows[s][0]}") for s in sel]
