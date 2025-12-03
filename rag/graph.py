# app/rag/graph.py
from typing import List
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from rag.retriever import HybridRetriever
from rag.prompts import build_prompt
from rag.guardrails import (
    is_emergency, EMERGENCY_MSG,
    is_out_of_scope, OUT_OF_SCOPE_MSG,
    is_member_specific, MEMBER_SPECIFIC_MSG,
    needs_fallback_from_retrieval
)
from langchain_community.llms import Ollama
import re

class RAGState(BaseModel):
    # user question (validated/coerced to str)
    question: str
    # retrieved passages
    docs: list = []
    # final generated answer
    answer: str = ""
    # rendered citation lines ("- Title → URL")
    citations: list[str] = []

retriever = HybridRetriever()
llm = Ollama(model="llama3.2:3b", temperature=0.2, num_predict=256)  # small, RAM-safe

# ---------- helpers ----------
def _strip_model_sources(text: str) -> str:
    return re.split(r"\n\s*quellen\s*:\s*", text or "", flags=re.I)[0].strip()

# ---------- nodes (each must return dict) ----------
def route(state: RAGState):
    # No state change here; branching is handled by add_conditional_edges.
    return {}

def emergency_node(state: RAGState):
    return {"answer": EMERGENCY_MSG, "citations": []}

def out_of_scope_node(state: RAGState):
    return {"answer": OUT_OF_SCOPE_MSG, "citations": []}

def member_specific_node(state: RAGState):
    return {"answer": MEMBER_SPECIFIC_MSG, "citations": []}

def retrieve(state: RAGState):
    hits = retriever.retrieve(state.question, top_k=4)
    # fallback if retrieval found nothing
    if needs_fallback_from_retrieval(len(hits), min_needed=1):
        return {"answer": "Dafür habe ich in den bereitgestellten Quellen keinen Beleg.",
                "citations": [], "docs": []}
    return {"docs": hits}

def generate(state: RAGState):
    if not state.docs:
        # nothing to do (already set an answer in retrieve fallback)
        return {}
    contexts = [h["text"] for h in state.docs]
    prompt = build_prompt(state.question, contexts)
    raw = llm.invoke(prompt)
    cleaned = _strip_model_sources(raw)

    seen, cits = set(), []
    for h in state.docs:
        m = h["meta"] or {}
        title = m.get("title") or m.get("id") or "Quelle"
        url = m.get("url")
        if url and url not in seen:
            seen.add(url)
            cits.append(f"- {title} → {url}")

    answer = cleaned + ("\n\nQuellen:\n" + "\n".join(cits) if cits else "")
    return {"answer": answer, "citations": cits}

# ---------- routing function for conditional edges ----------
def _route_key(state: RAGState) -> str:
    q = (state.question or "").strip()
    if is_emergency(q):
        return "EMERGENCY"
    if is_out_of_scope(q):
        return "OOS"
    if is_member_specific(q):
        return "MEMBER"
    return "retrieve"

def build_graph():
    g = StateGraph(RAGState)

    # register nodes
    g.add_node("route", route)
    g.add_node("emergency", emergency_node)
    g.add_node("oos", out_of_scope_node)
    g.add_node("member", member_specific_node)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    # entry
    g.set_entry_point("route")

    # conditional branching out of route
    g.add_conditional_edges(
        "route",
        _route_key,
        {
            "EMERGENCY": "emergency",
            "OOS": "oos",
            "MEMBER": "member",
            "retrieve": "retrieve",
        },
    )

    # normal path
    g.add_edge("retrieve", "generate")

    # end nodes
    g.add_edge("emergency", END)
    g.add_edge("oos", END)
    g.add_edge("member", END)
    g.add_edge("generate", END)

    return g.compile()