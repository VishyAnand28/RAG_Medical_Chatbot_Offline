# scripts/smoke_retrieval.py
from langchain_huggingface import HuggingFaceEmbeddings

# Prefer the new package to silence deprecation warnings:
# pip install -U langchain-chroma
try:
    from langchain_chroma import Chroma  # modern import
except ImportError:
    from langchain_community.vectorstores import Chroma  # fallback

PERSIST_DIR = "data/processed/chroma"
EMB_MODEL = "BAAI/bge-m3"

def build_retriever(k=4):
    embed = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma(collection_name="aok",
                embedding_function=embed,          # <-- IMPORTANT
                persist_directory=PERSIST_DIR)
    return vs.as_retriever(search_kwargs={"k": k})

def ask(retriever, q: str):
    # use invoke() (get_relevant_documents is deprecated)
    docs = retriever.invoke(q)
    print(f"\nQ: {q}")
    for i, d in enumerate(docs, 1):
        m = d.metadata or {}
        title = m.get("title") or m.get("id") or m.get("type")
        url = m.get("url") or m.get("sources") or ""
        snippet = (d.page_content[:180] + "…") if len(d.page_content) > 180 else d.page_content
        print(f"{i}. {title} -> {url}\n   {snippet}")

if __name__ == "__main__":
    retriever = build_retriever(k=4)
    ask(retriever, "Wie lade ich eine Mitgliedsbescheinigung herunter?")
    ask(retriever, "Was ist die elektronische Patientenakte (ePA)?")
    ask(retriever, "Hilfsmittel – wo prüfen?")
    ask(retriever, "Wofür steht die Abkürzung AOK?")
    ask(retriever, "Was ist die AOK allgemein?")