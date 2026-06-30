import os
import json
import numpy as np
import pickle
from config import DB_DIR
from embeddings import embed_batch, embed_text

class PersistentVectorStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.session_dir = os.path.join(DB_DIR, f"persist_{session_id}")
        self.meta_path = os.path.join(self.session_dir, "meta.json")
        self.emb_path = os.path.join(self.session_dir, "embeddings.npy")
        
        self.ids = []
        self.texts = []
        self.metadatas = []
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        
        self.file_mtimes = {}
        
        self.bm25 = None
        self.bm25_enabled = False
        
        self._load()

    def _load(self):
        if os.path.exists(self.meta_path) and os.path.exists(self.emb_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ids = data["ids"]
            self.texts = data["texts"]
            self.metadatas = data["metadatas"]
            self.file_mtimes = data.get("file_mtimes", {})
            self.embeddings = np.load(self.emb_path)
            self._build_bm25()

    def _save(self):
        os.makedirs(self.session_dir, exist_ok=True)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.ids,
                "texts": self.texts,
                "metadatas": self.metadatas,
                "file_mtimes": self.file_mtimes
            }, f)
        if self.embeddings.size > 0:
            np.save(self.emb_path, self.embeddings)

    def _build_bm25(self):
        try:
            from rank_bm25 import BM25Okapi
            tokenized_corpus = [doc.lower().split(" ") for doc in self.texts]
            if tokenized_corpus:
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.bm25_enabled = True
        except ImportError:
            self.bm25_enabled = False

    def remove_source(self, source_name: str):
        """Removes chunks for a specific file (Delta Updates)."""
        indices_to_keep = [i for i, meta in enumerate(self.metadatas) if meta.get("source") != source_name]
        
        if len(indices_to_keep) < len(self.metadatas):
            self.ids = [self.ids[i] for i in indices_to_keep]
            self.texts = [self.texts[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            if self.embeddings.size > 0:
                self.embeddings = self.embeddings[indices_to_keep]
            if source_name in self.file_mtimes:
                del self.file_mtimes[source_name]

    def remove_dependent_syntheses(self, source_name: str):
        """Removes any cached Master Synthesis that relied on this modified file."""
        indices_to_keep = []
        deleted_count = 0
        for i, meta in enumerate(self.metadatas):
            if meta.get("type") == "synthesis" and source_name in meta.get("dependent_files", []):
                deleted_count += 1
            else:
                indices_to_keep.append(i)
                
        if deleted_count > 0:
            print(f"🗑️ [Cache Invalidation]: Deleted {deleted_count} stale Master Syntheses that relied on '{source_name}'.")
            self.ids = [self.ids[i] for i in indices_to_keep]
            self.texts = [self.texts[i] for i in indices_to_keep]
            self.metadatas = [self.metadatas[i] for i in indices_to_keep]
            if self.embeddings.size > 0:
                self.embeddings = self.embeddings[indices_to_keep]

    def add_chunks(self, chunks: list[dict], source_name: str, mtime: float):
        if chunks:
            if source_name != "map_reduce_cache":
                print(f"Embedding {len(chunks)} chunks... This may take a moment.")
            texts = [c["text"] for c in chunks]
            embs = embed_batch(texts)
            
            self.ids.extend([c["chunk_id"] for c in chunks])
            self.texts.extend(texts)
            self.metadatas.extend([c["metadata"] for c in chunks])
            
            new_emb = np.array(embs, dtype=np.float32)
            if self.embeddings.size == 0:
                self.embeddings = new_emb
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb])
                
        if source_name != "map_reduce_cache" and source_name != "codebase_blueprints":
            self.file_mtimes[source_name] = mtime
            
        self._build_bm25()
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.ids:
            return []
            
        query_vec = embed_text(query)
        if query_vec is None or self.embeddings.size == 0:
            return []
            
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        matrix_norm = self.embeddings / norms
        dense_sims = (matrix_norm @ query_norm).flatten()
        
        if np.max(dense_sims) != np.min(dense_sims):
            dense_sims = (dense_sims - np.min(dense_sims)) / (np.max(dense_sims) - np.min(dense_sims))
        
        sparse_sims = np.zeros(len(self.ids))
        if self.bm25_enabled:
            tokenized_query = query.lower().split(" ")
            sparse_sims = np.array(self.bm25.get_scores(tokenized_query))
            if np.max(sparse_sims) != np.min(sparse_sims):
                sparse_sims = (sparse_sims - np.min(sparse_sims)) / (np.max(sparse_sims) - np.min(sparse_sims))

        alpha = 0.5
        hybrid_scores = (alpha * dense_sims) + ((1 - alpha) * sparse_sims)
        
        k = min(top_k, len(self.ids))
        top_idx = np.argsort(hybrid_scores)[::-1][:k]
        
        results = []
        for i in top_idx:
            results.append({
                "text": self.texts[i],
                "metadata": self.metadatas[i],
                "score": float(hybrid_scores[i])
            })
        return results

    def cleanup(self):
        pass
