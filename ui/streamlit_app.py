# app/ui/streamlit_app.py
import os
import yaml
import streamlit as st
from datetime import datetime

# Your existing pipeline
from rag.graph import build_graph, RAGState

# -------------------- Page / Styles --------------------
st.set_page_config(page_title="AOK RAG Chatbot (Local)", layout="wide", page_icon="💚")
st.markdown("""
<style>
/* add safe space at very top so nothing gets clipped under the toolbar */
.stAppViewContainer { padding-top: 0 !important; }
.main .block-container { padding-top: 2.2rem !important; }

/* optional: a touch more room on small screens */
@media (max-width: 820px) {
  .main .block-container { padding-top: 2.8rem !important; }
}

/* your existing polish (keep these if you had them) */
.stButton>button {border-radius: 10px; padding: 0.5rem 1rem; font-weight: 600;}
.faq-btn button {width: 100%; text-align: left; border: 1px solid #eaeaea;}
.cite-box {background: #f8fbff; border: 1px solid #e6eef8; padding: .75rem 1rem; border-radius: 10px;}
.header-title {font-size: 2.0rem; font-weight: 800; margin: 0 0 .2rem 0;}
.header-sub {color: #475569; margin-bottom: .6rem;}
.small {font-size: 0.88rem; color: #64748b;}
.kpi {font-size: .85rem; color: #64748b;}
.code {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;}
a {text-decoration: none;}
</style>
""", unsafe_allow_html=True)
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# -------------------- Header --------------------
colA, colB = st.columns([5,2], vertical_alignment="center")
with colA:
    st.markdown('<div class="header-title">AOK Niedersachsen – RAG Chatbot (Local)</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Offline-Demo · Hybrid Retrieval (BM25 + Dense) · LLM: Ollama</div>', unsafe_allow_html=True)
with colB:
    st.markdown(f"""<div class="kpi">
    <b>Stand:</b> {datetime.now().strftime("%d.%m.%Y %H:%M")}<br/>
    <b>Index:</b> <span class="code">data/processed/chroma</span><br/>
    <b>FAQ:</b> <span class="code">data/faq_de.yaml</span>
    </div>""", unsafe_allow_html=True)

st.divider()

# -------------------- Sidebar Controls --------------------
with st.sidebar:
    st.header("Einstellungen")
    st.caption("Ollama muss lokal laufen und das Modell vorhanden sein.")
    model = st.text_input("Ollama-Modell", value="llama3.2:3b", help="z. B. llama3.2:3b oder phi3:3.8b-mini-instruct")
    top_k = st.slider("Top-K Dokumente", 2, 8, 4, 1)
    st.markdown("""
**Hinweis:** Wenn der Speicher knapp ist, schließen Sie andere Apps oder wählen Sie ein kleineres Modell
(z. B. `phi3:3.8b-mini-instruct`).
""")

# Build and cache the graph once
@st.cache_resource
def _get_graph():
    return build_graph()

graph = _get_graph()

# -------------------- FAQ Suggestions --------------------
faq_suggestions = []
faq_path = "data/faq_de.yaml"
if os.path.exists(faq_path):
    try:
        data = yaml.safe_load(open(faq_path, "r", encoding="utf-8"))
        faq_suggestions = [x.get("question") for x in data if isinstance(x, dict) and x.get("question")]
    except Exception:
        pass

# -------------------- How-to card --------------------
with st.expander("ℹ️ Wie benutze ich diese Demo?", expanded=True):
    st.markdown("""
- Geben Sie Ihre Frage links ein **oder** klicken Sie auf einen der FAQ-Buttons.
- Das System sucht kontextrelevante Textstellen (BM25 + Dense), generiert mit dem lokalen LLM eine Antwort
  und zeigt **Quellen** an (nur aus den geladenen Dokumenten).
- Bei medizinischen Notfällen wird eine **Sicherheitsmeldung** ausgegeben (112 / Clarimedis).
""")

# -------------------- Main Two Columns --------------------
left, right = st.columns([2,3], vertical_alignment="top")

with left:
    st.subheader("Frage stellen")
    q = st.text_area("Ihre Frage", placeholder="Beispiel: Was ist die elektronische Patientenakte (ePA)?", height=120)
    ask = st.button("Fragen", type="primary")

    if faq_suggestions:
        st.markdown("**Beispiele (aus FAQ):**")
        # Show up to 10 FAQ buttons in 2 columns
        fcol1, fcol2 = st.columns(2)
        for i, qx in enumerate(faq_suggestions[:10], start=1):
            target_col = fcol1 if i % 2 else fcol2
            with target_col:
                if st.button(f"#{i}  {qx}", key=f"faq_{i}", use_container_width=True, help="FAQ übernehmen", type="secondary"):
                    q = qx
                    ask = True

with right:
    st.subheader("Antwort")
    if ask and (q or "").strip():
        with st.spinner("Antworte…"):
            state = RAGState(question=q)
            result = graph.invoke(state)
        answer = (result.get("answer") or "").strip()
        citations = result.get("citations") or []

        # Main answer
        if answer:
            st.write(answer)

        # Citations accordion
        if citations:
            with st.expander("🔗 Quellen (aus den Dokumenten)"):
                st.markdown('<div class="cite-box">', unsafe_allow_html=True)
                for c in citations:
                    st.markdown(c)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Geben Sie links eine Frage ein oder wählen Sie einen FAQ-Vorschlag.")

# -------------------- Footer --------------------
st.divider()
st.markdown(
    '<span class="small">Demo: RAG-Pipeline (BM25 + Dense, optional Reranking) · '
    'LLM via <span class="code">Ollama</span> · Nur lokale Quellen, keine externen Anfragen.</span>',
    unsafe_allow_html=True,
)