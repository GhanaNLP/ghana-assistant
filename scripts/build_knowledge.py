"""Package the knowledge store for runtime: FAISS index + sentence text/source (from training/ghanaqa_build.py outputs).
  python scripts/build_knowledge.py ghanaqa/ out/knowledge"""
import sys, os, shutil, pyarrow.parquet as pq
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
shutil.copy(os.path.join(src, "ghanaqa_ivfpq.faiss"), os.path.join(dst, "index.faiss"))
t = pq.read_table(os.path.join(src, "sentences.parquet"), columns=["text", "doc", "source"])
pq.write_table(t, os.path.join(dst, "sentences.parquet"), compression="zstd")
print("knowledge store:", t.num_rows, "sentences ->", dst)
