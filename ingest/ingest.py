# ingest/ingest.py
import os, argparse, yaml, json, tempfile, requests, pathlib
import trafilatura, fitz  # PyMuPDF
from typing import Dict, List
from ingest.cleaners import clean_text, chunk_text

# Vector DB + embeddings (CPU-friendly)
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def load_yaml(path:str):
    # read YAML (seed_sources.yaml or faq_de.yaml) -> Python objects (list/dicts)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def fetch_html(url:str) -> str:
    # trafilatura.fetch_url() downloads HTML
    # trafilatura.extract() strips nav/JS, returns clean text
    # then we clean whitespace with clean_text()
    html = trafilatura.fetch_url(url)
    if not html: return ""
    text = trafilatura.extract(html, include_tables=False, include_images=False) or ""
    return clean_text(text)

def read_pdf(path_or_url:str) -> str:
    # support local file (data/docs/...) or http(s) URLs
    # if URL: requests.get() -> write bytes to a temp file
    # PyMuPDF (fitz) opens the PDF; get_text() per page -> join
    # then clean_text()
    fn = path_or_url
    if path_or_url.startswith("http"):
        resp = requests.get(path_or_url, timeout=60)
        resp.raise_for_status()
        fd = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        fd.write(resp.content); fd.close()
        fn = fd.name
    doc = fitz.open(fn)
    text = "\n".join(p.get_text() for p in doc)
    doc.close()
    return clean_text(text)

def build_vs(persist_dir:str):
    # make (or open) a Chroma collection called "aok"
    # embedding model = BAAI/bge-m3 via HuggingFaceEmbeddings (CPU)
    # returns a LangChain Chroma vectorstore object
    os.makedirs(persist_dir, exist_ok=True)
    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device":"cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(collection_name="aok", embedding_function=emb, persist_directory=persist_dir)

def upsert_texts(vs, texts:List[str], base_meta:Dict):
    # add texts to the vectorstore with the same metadata for each text
    vs.add_texts(texts=texts, metadatas=[base_meta]*len(texts))

def ingest_seed(vs, manifest_path:str):
    # loop over curated URLs from seed_sources.yaml
    # pick loader: PDF vs HTML
    # chunk each document and add chunks to vectorstore with metadata
    items = load_yaml(manifest_path)
    added = 0
    for it in items:
        meta = {k: it.get(k) for k in ("id","url","title","category","language","type")}
        t = it.get("type","html")
        try:
            if t == "pdf":
                text = read_pdf(it["url"])
            else:
                text = fetch_html(it["url"])
            for ch in chunk_text(text):
                upsert_texts(vs, [ch], meta)
                added += 1
        except Exception as e:
            print(f"⚠️  skip {it.get('id')} ({it.get('url')}): {e}")
    return added

def ingest_faq(vs, faq_path:str):
    # add each FAQ 'answer' as a single chunk with metadata (id/topic/etc.)
    faqs = load_yaml(faq_path)
    added = 0
    for q in faqs:
        meta = {
            "id": q["id"], "type":"faq", "topic": q.get("topic"),
            "routing_hint": q.get("routing_hint"), "region": q.get("region","DE"),
            "sources": json.dumps(q.get("sources",[]))
        }
        text = clean_text(q["answer"])
        if text:
            upsert_texts(vs, [text], meta)
            added += 1
    return added

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/seed_sources.yaml")
    ap.add_argument("--faq", default="data/faq_de.yaml")
    ap.add_argument("--persist", default="data/processed/chroma")
    args = ap.parse_args()

    vs = build_vs(args.persist)
    n_seed = ingest_seed(vs, args.manifest)
    n_faq  = ingest_faq(vs, args.faq)
    vs.persist()
    # --- write BM25 corpus for hybrid retrieval ---
    out = pathlib.Path("data/processed/chunks.jsonl")
    all_docs = vs.get(include=["metadatas","documents"])
    with out.open("w", encoding="utf-8") as f:
        for text, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            f.write(json.dumps({"text": text, "metadata": meta}, ensure_ascii=False) + "\n")
    print(f"📝 wrote BM25 corpus → {out}")
    print(f"✅ ingestion done → {args.persist} | chunks: seed={n_seed}, faq={n_faq}, total={n_seed+n_faq}")