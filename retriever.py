import os
import json
from typing import List, Dict, Any, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_dir: str, embed_model: str):
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "chunks.jsonl")

        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index not found. Run `python ingest.py` first.")

        self.index = faiss.read_index(self.index_path)
        self.chunks = self._load_chunks(self.meta_path)
        self.model = SentenceTransformer(embed_model)

    def _load_chunks(self, jsonl_path: str) -> List[Dict[str, Any]]:
        chunks = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
        return chunks

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        qv = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(qv)

        D, I = self.index.search(qv, k=top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            c = self.chunks[int(idx)]
            results.append((float(score), c))
        return results
