import json, os
from rag.graph import build_graph, RAGState

INP="tests/eval_set.jsonl"
OUT="tests/eval_set_with_preds.jsonl"

def main(top_k=4):
    graph = build_graph()
    os.makedirs("tests", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as out, open(INP, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = r["user_input"]  # was: r["question"]
            state = RAGState(question=q)  # add top_k if you wired it in state
            res = graph.invoke(state)
            ctxs = [h.get("text","") for h in res.get("docs",[])]  # was: contexts
            outrow = {
                "user_input": q,                      # was: "question": q
                "reference": r["reference"],          # was: "ground_truths": r["ground_truths"]
                "response": res.get("answer",""),     # was: "answer": ...
                "retrieved_contexts": ctxs,           # was: "contexts": ctxs
            }
            out.write(json.dumps(outrow, ensure_ascii=False) + "\n")
    print(f"✅ Predictions → {OUT}")

if __name__=="__main__":
    main()