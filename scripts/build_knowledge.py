"""Convert the knowledge build outputs into the memory-light runtime store used by PhraseStore.
  python scripts/build_knowledge.py ghanaqa/ out/knowledge
Inputs (from data_pipelines/ghanaqa/ghanaqa_build.py): noun_index.npz, noun_terms.json, sentences.parquet, sentence_meta.parquet"""
import sys, os, json, sqlite3, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True); t0 = time.time()
Z = np.load(os.path.join(src, "noun_index.npz"))
np.save(os.path.join(dst, "postings_offs.npy"), Z["offs"].astype(np.int64)); np.save(os.path.join(dst, "postings_ids.npy"), Z["ids"].astype(np.int32))
S = pq.read_table(os.path.join(src, "sentences.parquet"), columns=["doc", "text"]).to_pandas()
M = pq.read_table(os.path.join(src, "sentence_meta.parquet")).to_pandas()
doci, docs = pd.factorize(S.doc.values)
yr = pd.to_numeric(M.date.str[:4], errors="coerce").values + (pd.to_numeric(M.date.str[5:7], errors="coerce").values - 1) / 12
rec = np.where(np.isnan(yr), 0.5, np.clip((yr - 2010) / 16, 0, 1)).astype(np.float32)        # undated research = neutral
np.save(os.path.join(dst, "rec.npy"), rec); np.save(os.path.join(dst, "docidx.npy"), doci.astype(np.int32)); np.save(os.path.join(dst, "toklen.npy"), M.toklen.values.astype(np.int16))
db_path = os.path.join(dst, "store.sqlite")
if os.path.exists(db_path): os.remove(db_path)
db = sqlite3.connect(db_path); db.execute("PRAGMA journal_mode=OFF"); db.execute("PRAGMA synchronous=OFF")
db.execute("CREATE TABLE terms(term TEXT PRIMARY KEY, tid INTEGER) WITHOUT ROWID")
db.executemany("INSERT INTO terms VALUES (?, ?)", ((t, i) for i, t in enumerate(json.load(open(os.path.join(src, "noun_terms.json"))))))
db.execute("CREATE TABLE sentences(sid INTEGER PRIMARY KEY, text TEXT, date TEXT, doc TEXT)")
db.executemany("INSERT INTO sentences VALUES (?, ?, ?, ?)", zip(range(len(S)), S.text.values, M.date.values, S.doc.values))
db.commit(); db.execute("VACUUM"); db.close()
size = {f: round(os.path.getsize(os.path.join(dst, f)) / 1e6) for f in os.listdir(dst)}
print(f"knowledge store -> {dst} in {time.time()-t0:.0f}s | sizes MB: {size}")
