# scripts/smoke_graph.py
from rag.graph import build_graph, RAGState

if __name__ == "__main__":
    graph = build_graph()

    queries = [
        "Wie lade ich eine Mitgliedsbescheinigung herunter?",
        "Was ist die elektronische Patientenakte (ePA)?",
        "Wofür steht die Abkürzung AOK?"
    ]

    for q in queries:
        print("=" * 80)
        print(f"❓ Frage: {q}\n")

        # run graph, returns dict-like object
        result = graph.invoke(RAGState(question=q))

        # Access fields safely as dict
        answer = result.get("answer", "")
        citations = result.get("citations", [])

        print(f"💡 Antwort:\n{answer}\n")
        if citations:
            print("🔗 Quellen:")
            for c in citations:
                print(" ", c)