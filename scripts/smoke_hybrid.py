from rag.retriever import HybridRetriever

if __name__ == "__main__":
    r = HybridRetriever()
    for q in [
        "Wie lade ich eine Mitgliedsbescheinigung herunter?",
        "Was ist die elektronische Patientenakte (ePA)?",
        "Hilfsmittel – wo prüfen?",
        "Wofür steht die Abkürzung AOK?",
        "Was ist die AOK allgemein?"
    ]:
        hits = r.retrieve(q, top_k=4)
        print(f"\nQ: {q}")
        for i,h in enumerate(hits,1):
            m = h["meta"] or {}
            title = m.get("title") or m.get("id") or m.get("type")
            url = m.get("url") or m.get("sources") or ""
            print(f"{i}. {title} -> {url}")
            print("   " + (h["text"][:160] + "…"))