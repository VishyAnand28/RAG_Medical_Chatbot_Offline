# ingest/cleaners.py
import re

def clean_text(txt: str) -> str:
    # normalize all whitespace to single spaces; trim ends
    # purpose: make chunks consistent for both BM25 and embeddings
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def chunk_text(txt: str, size: int = 1000, overlap: int = 200):
    # sliding window over the text:
    #   [0:1000], [800:1800], [1600:2600], ...
    # step = size - overlap (here 800)
    # returns only non-empty slices
    # purpose: produce fixed-size passages for retrieval
    chunks, i, n = [], 0, len(txt)
    step = max(1, size - overlap)
    while i < n:
        chunks.append(txt[i:i+size])
        i += step
    return [c for c in chunks if c.strip()]