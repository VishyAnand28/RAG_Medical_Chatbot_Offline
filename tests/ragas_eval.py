# tests/ragas_eval.py
import os
os.environ["OLLAMA_NUM_PARALLEL"] = "1"  # serialize judge calls to avoid timeouts

import json, pathlib
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  # pip install langchain-ollama

INP = "tests/eval_set_with_preds.jsonl"

def sanity_check(sanity_path):
    p = pathlib.Path(sanity_path)
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    print("rows:", len(rows))
    print("empty reference:", sum(not bool(r.get("reference")) for r in rows))
    print("empty retrieved_contexts:", sum(not isinstance(r.get("retrieved_contexts"), list) or len(r.get("retrieved_contexts") or [])==0 for r in rows))

def load_dataset(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return Dataset.from_list(rows)

def main():
    dset = load_dataset(INP)
    scheck = sanity_check("tests/eval_set_with_preds.jsonl")
    # Pass LangChain instances directly; Ragas will wrap internally
    # judge_llm = ChatOllama(
    #     model="llama3.2:3b", 
    #     temperature=0.0, 
    #     format="json",
    #     request_timeout=120,    # optional, helps with timeouts
    #     num_ctx=4096,           # optional, larger context
    #     keep_alive="5m",        # optional
    #     )
    
    judge_llm = ChatOllama(
        model="llama3.2:3b",         # keep your 3B model
        temperature=0.0,
        format="json",               # force JSON output
        top_p=0.1,                   # reduce sampling spread
        top_k=1,                     # make it extremely deterministic
        num_ctx=4096,                # avoid context overflow
        num_predict=256,             # keep outputs short (fits in JSON)
        request_timeout=180,         # give it a bit more time on Windows
        keep_alive="5m",
        system="Return ONLY a single valid JSON object. No prose, no preface, no examples."
        )

    emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={"normalize_embeddings": True},
        # needed for some HF models
        multi_process=False,
        show_progress=False,
    )
    
    metrics = [
    Faithfulness(),
    ResponseRelevancy(),
    ContextPrecision(),
    ContextRecall(),
    ]
    
    result = evaluate(
        dataset=dset,  # or just dset as the first positional arg
        metrics=metrics,
        llm=judge_llm,
        embeddings=emb,
        raise_exceptions=False,)
    
    df = result.to_pandas()
    print("Columns:", list(df.columns))

    wanted = {
        "faithfulness", "faithfulness_score",
        "response_relevancy", "answer_relevancy", "response_relevancy_score", "answer_relevancy_score",
        "context_precision", "context_precision_score",
        "context_recall", "context_recall_score",
    }

    metric_cols = sorted([c for c in df.columns if c in wanted])  # stable order
    if not metric_cols:
        # fallback if your build only has *_score columns with different names
        metric_cols = sorted([c for c in df.columns if c.endswith("_score")])

    print("=== RAGAS scores (mean) ===")
    for c in metric_cols:
        mean_val = pd.to_numeric(df[c], errors="coerce").mean()
        print(f"{c:>24s}: {mean_val:.3f}")

    df.to_csv("tests/ragas_scores.csv", index=False, encoding="utf-8-sig")
    print("✅ Detailed scores → tests/ragas_scores.csv")

    
    # df = result.to_pandas()
    # print("=== RAGAS scores (mean) ===")
    # for col in df.columns:
    #     if col.endswith("_score"):
    #         print(f"{col:>24s}: {df[col].mean():.3f}")

if __name__ == "__main__":
    main()