"""Question -> bge-small embedding -> FAISS top-k -> sentences (Ghanaian news + KNUST research)."""
import numpy as np, pyarrow.parquet as pq


class Retriever:
    def __init__(self, index_path, sentences_path, embedder, query_prefix, nprobe=32):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"): self.index.nprobe = nprobe
        t = pq.read_table(sentences_path, columns=["text", "doc"])
        self.text = t.column("text").to_pylist(); self.doc = t.column("doc").to_pylist()
        self.emb = SentenceTransformer(embedder, device="cpu"); self.qp = query_prefix

    def search(self, question, k=8):
        q = self.emb.encode([self.qp + question], normalize_embeddings=True).astype(np.float32)
        scores, ids = self.index.search(q, k)
        return [dict(text=self.text[i], source=self.doc[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]
