# AOK Niedersachsen — Offline RAG Chatbot (Local)

Privacy-first medical FAQ assistant that runs fully offline.  
Hybrid retrieval (BM25 + dense bge-m3, optional bge-reranker), local LLM via Ollama, orchestration with LangGraph, and a Streamlit UI with citations. Includes an evaluation loop using RAGAS.

> ⚠️ Disclaimer: This demo is NOT medical advice. In emergencies call 112 (Germany) or your local emergency number.

---

## Features
- Offline end-to-end: ingest → clean → chunk → embed (bge-m3) → Chroma index.
- Hybrid retrieval: BM25 (sparse) + dense vectors; optional cross-encoder rerank (bge-reranker v2-m3).
- Guardrails & routing (LangGraph): emergency / out-of-scope / member-specific handoffs, retrieval-empty fallback.
- Local LLM: `llama3.2:3b` (default) via Ollama; switchable in UI.
- UI: Streamlit app with Top-K control, FAQ quick picks, and source citations.
- Evaluation: Build eval set from FAQ → collect answers → score with RAGAS.

---

## Project structure
aok-rag-bot/
├─ data/
│ ├─ raw/ # (optional) raw HTML/PDF cache
│ ├─ processed/ # chroma/ and chunks.jsonl live here
│ ├─ seed_sources.yaml # curated crawl targets (AOK/GKV etc.)
│ └─ faq_de.yaml # German FAQs (also used for eval seed)
├─ ingest/
│ ├─ ingest.py # fetch/clean/chunk/embed → Chroma
│ └─ cleaners.py # simple cleaners + char-window chunker
├─ rag/
│ ├─ retriever.py # BM25 + dense + optional rerank + fusion
│ ├─ prompts.py # German system & answer template
│ ├─ guardrails.py # regex routes & safe messages
│ └─ graph.py # LangGraph flow (route → retrieve → gen)
├─ ui/
│ └─ streamlit_app.py # local demo UI with citations
├─ tests/
│ ├─ build_evalset_from_faq.py # FAQ → eval_set.jsonl
│ ├─ collect_answers.py # run graph, save responses+contexts
│ └─ ragas_eval.py # RAGAS: faithfulness, relevancy, CP/CR
├─ requirements.txt
├─ aok_chatbot.yml # (optional) Conda env file
└─ README.md

Ingest: python -m ingest.ingest --manifest data/seed_sources.yaml --faq data/faq_de.yaml
Smoke Test: python scripts/smoke_retrieval.py; python -m scripts.smoke_hybrid
python -m streamlit run ui/streamlit_app.py

python -m scripts.smoke_graph
# Try:
# - "Notfall, starke Brustschmerzen – was tun?"
# - "Wie ändere ich meine Bankverbindung?"
# - "Schreib mir ein Python-Programm ..."

RAGAS - Future:
python -m tests.build_evalset_from_faq
python -m tests.collect_answers
python -m tests.ragas_eval

RAG Chatbot Pipeline:
                ┌──────────────────────┐
                │   Data Sources       │
                │  (AOK site, PDFs, FAQ│
                └─────────┬────────────┘
                          │
                ┌─────────▼────────────┐
                │     Ingestion        │
                │ ingest/ingest.py     │
                │ + cleaners.py        │
                └─────────┬────────────┘
                          │
                  ┌───────▼────────┐
                  │  Chunker       │
                  │ (split text)   │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │ Embedder       │
                  │ BAAI/bge-m3    │
                  └───────┬────────┘
                          │
          ┌───────────────▼─────────────────┐
          │   Vector DB (Dense Index)       │
          │   Chroma (data/processed/chroma)│
          └───────────────┬─────────────────┘
                          │
 ┌────────────────────────▼────────────────────────┐
 │ Hybrid Retriever (app/rag/retriever.py)         │
 │  • BM25Retriever (sparse, chunks.jsonl)         │
 │  • DenseRetriever (Chroma vectors)              │
 │  → EnsembleRetriever (fusion, weights)          │
 └────────────────────────┬────────────────────────┘
                          │
                  ┌───────▼────────┐
                  │   Reranker     │
                  │ bge-reranker   │
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │    LLM (Ollama)│
                  │ LLaMA-3.2 / Phi│
                  └───────┬────────┘
                          │
                  ┌───────▼────────┐
                  │ LangGraph      │
                  │ graph.py       │
                  │ (retrieval→gen)│
                  └───────┬────────┘
                          │
             ┌────────────▼────────────┐
             │  Answer + Citations     │
             │  Streamlit / FastAPI    │
             └─────────────────────────┘

Configuration Knobs
Embedding model: BAAI/bge-m3 (set in rag/retriever.py)
Reranker: BAAI/bge-reranker-v2-m3 (toggle in HybridRetriever(use_reranker=True))
Persistence: data/processed/chroma
Sparse corpus: data/processed/chunks.jsonl
Fusion weights: set in EnsembleRetriever (e.g., dense 0.55 / bm25 0.45)
LLM: llama3.2:3b (UI dropdown) via Ollama
Prompting: German templates in rag/prompts.py
Guardrails: regex routes in rag/guardrails.py (emergency / out-of-scope / member)
Evaluation: scripts in tests/ folder

Data (YAML FAQ, PDFs, AOK pages)
     │
     ▼
Ingestion (ingest.py + cleaners.py)
- fetch → clean → chunk
     │
     ▼
Embeddings: BAAI/bge-m3
     │
     ▼
Vector DB: Chroma (persisted)
     │
     ▼
Hybrid Retrieval (retriever.py)
- BM25 (chunks.jsonl) + Dense (Chroma)
- Fusion (weights) → Top-K
     │
     ▼
Reranker (bge-reranker v2-m3) [optional]
     │
     ▼
LangGraph (graph.py)
- route → retrieve → generate
- guardrails (emergency/oos/member)
     │
     ▼
LLM via Ollama (llama3.2:3b default)
     │
     ▼
UI: Streamlit (answer + citations)



System-design architecture:

                                   +-----------------------------+
                                   |        Datenquellen         |
                                   |  FAQ / PDF / YAML / Web     |
                                   +--------------+--------------+
                                                  |
                                                  v
                           OFFLINE (INGESTION & INDEX BUILD) PIPELINE
                                                  |
                           +----------------------+----------------------+
                           | Ingestion & Preprocessing Service           |
                           | Abruf, HTML/PDF-Parsing, Bereinigung,       |
                           | Chunking, Metadata                          |
                           +----------------------+----------------------+
                                                  |
                          +-----------------------+----------------------+
                          |                                              |
                          v                                              v
                +--------------------+                        +---------------------+
                | BM25 Korpus Index  |                        | Embedding Service   |
                | (Chunks)           |                        | (BGE-M3)            |
                +----------+---------+                        +----------+----------+
                           |                                           |
                           |                                           v
                           |                                +----------------------+
                           |                                | Vector DB (Chroma)   |
                           |                                +----------+-----------+
                           |                                           |
                           +-------------------------------+-----------+
                                                           |
                                        +------------------v------------------+
                                        | Document Store / Object Storage     |
                                        | (Rohdaten, Chunks, Versionierung)   |
                                        +------------------+------------------+
                                                           |
                                        +------------------v------------------+
                                        | Ingestion Scheduler / Orchestrator  |
                                        | (Cron, LangGraph Flows, etc.)       |
                                        +-------------------------------------+


                     ONLINE (QUERY, RETRIEVAL, GENERATION, OBSERVABILITY) PFAD


+-------------------+        +-------------------+       +------------------------+
|   Streamlit UI    +------->+   API Gateway     +------>+   AuthN / AuthZ        |
|  (Web Frontend)   |        | (Routing, Throttle)|      |  (User / Tenant)      |
+---------+---------+        +----------+--------+       +-----------+------------+
                                      |                                |
                                      |                                v
                                      |                     +----------------------+
                                      |                     |  Rate Limit / WAF    |
                                      |                     +----------+-----------+
                                      |                                |
                                      v                                |
                          +-----------+-------------------------------+
                          |         LangGraph Orchestrator           |
                          | Routing, Tools, Guardrails, State        |
                          +-----------+-------------------------------+
                                      |
                                      v
                         +------------+---------------------------+
                         |         RAG / Retrieval Layer          |
                         +------------+---------------------------+
                                      |
            +-------------------------+-----------------------------+
            |                                                       |
            v                                                       v
+------------------------+                           +---------------------------+
| BM25 Retrieval Service |                           | Vector Retrieval Service  |
| (über BM25 Korpus)     |                           | (Chroma, Dense, Filters)  |
+-----------+------------+                           +--------------+------------+
            \                                                      /
             \                                                    /
              v                                                  v
                         +----------------------------------+
                         |   Hybrid Retrieval + Reranker    |
                         |   (BM25 + Dense + BGE-Reranker)  |
                         +----------------+-----------------+
                                          |
                                          v
                               +----------+-----------+
                               |   LLM Gateway        |
                               |  (Ollama Llama 3.2   |
                               |   3B Instruct, etc.) |
                               +----------+-----------+
                                          |
                                          v
                         +----------------+----------------------+
                         |  Antwortgenerator / Output Composer   |
                         |  (Finale Antwort + Zitate + Metadata) |
                         +----------------+----------------------+
                                          |
                                          v
                                   +------+------+
                                   |  Streamlit  |
                                   |    UI       |
                                   +-------------+


            OBSERVABILITY, STORAGE & PLATFORM (SEITENKANAL / CROSS-CUTTING)


      +---------------------+       +---------------------+       +---------------------+
      | Request / Event Log |<------+  API Gateway       |       | Feedback Store      |
      +----------+----------+       +---------------------+       +----------+----------+
                 ^                                                          |
                 |                                                          v
      +----------+----------+       +---------------------+       +---------------------+
      | Metrics & Tracing   |<------+ LangGraph / RAG     |       | Eval / A/B Testing  |
      | (Latency, Fehler,   |       | LLM / Vector DB     |       | RAG/LLM Quality     |
      | Kosten, Hit-Rates)  |       +---------------------+       +---------------------+
      +----------+----------+
                 ^
                 |
      +----------+----------+
      | Monitoring & Alerts |
      | (Dashboards, Pager) |
      +---------------------+
```
