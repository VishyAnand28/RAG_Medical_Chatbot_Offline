# app/rag/retriever.py
from typing import List, Dict
import json, pathlib
from langchain_huggingface import HuggingFaceEmbeddings

# Prefer modern import if you installed langchain-chroma; fallback otherwise.
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever  # weighted hybrid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

PERSIST_DIR = "data/processed/chroma"
BM25_JSONL  = "data/processed/chunks.jsonl"
EMB_MODEL   = "BAAI/bge-m3"

class HybridRetriever:
    def __init__(self, k_pool: int = 10, bm25_weight: float = 0.45, dense_weight: float = 0.55, use_reranker: bool = True):
        # 1. dense store with chroma
        self.embed = HuggingFaceEmbeddings(
            model_name=EMB_MODEL,
            model_kwargs={"device":"cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vs = Chroma(collection_name="aok", embedding_function=self.embed, persist_directory=PERSIST_DIR)
        self.dense = self.vs.as_retriever(search_kwargs={"k": k_pool})

        # 2. bm25 in-memory (from jsonl)
        p = pathlib.Path(BM25_JSONL)
        if not p.exists():
            raise FileNotFoundError(f"BM25 corpus not found: {BM25_JSONL}. Re-run ingest to generate it.")
        texts, metas = [], []
        for line in p.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            texts.append(row["text"]); metas.append(row.get("metadata", {}))
        self.bm25 = BM25Retriever.from_texts(texts=texts, metadatas=metas)
        self.bm25.k = k_pool

        # 3. ensemble fusion - sparse + dense weights
        self.hybrid = EnsembleRetriever(retrievers=[self.bm25, self.dense],
                                        weights=[bm25_weight, dense_weight])

        # 4. cross-encoder reranker (optional) - high precision
        self.use_reranker = use_reranker
        if use_reranker:
            self.tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
            self.mdl = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        # 1) fused candidates (BM25 + Dense)
        docs = self.hybrid.invoke(query)  # fusion result
        items = [{"text": d.page_content, "meta": d.metadata} for d in docs]

        # 2) optional rerank on CPU (pairwise scoring)
        if not self.use_reranker:
            return items[:top_k]

        batch = [(query, it["text"]) for it in items]
        inputs = self.tok.batch_encode_plus(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            scores = self.mdl(**inputs).logits.squeeze(-1).tolist()
        ranked = sorted([{**it, "ce": s} for it, s in zip(items, scores)], key=lambda x: x["ce"], reverse=True)
        return ranked[:top_k]