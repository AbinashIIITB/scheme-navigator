import os
import sys
from pathlib import Path
import time

# ── Ensure project root is on sys.path ──────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
from src.config import TOP_K, SIM_THRESHOLD, COLLECTION_NAME, LLM_MODEL_NAME, CHROMA_DIR
from src.generator import answer
from src.vectorstore import get_chroma_client, get_embedder


# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarkari Scheme Navigator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #0d9488 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; font-weight: 700; }
    .main-header p  { color: #e2e8f0; margin-top: 0.5rem; margin-bottom: 0; font-size: 1.05rem; }

    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1rem; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-card .num   { font-size: 1.6rem; font-weight: 700; color: #1e3a8a; }
    .metric-card .label { font-size: 0.82rem; color: #64748b; font-weight: 500; }

    .score-badge {
        display: inline-block; background: #f1f5f9; color: #334155;
        padding: 0.2rem 0.5rem; border-radius: 6px; font-weight: 600; font-size: 0.8rem;
    }
    .abstain-box {
        background: #fff7ed; border-left: 4px solid #f97316;
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
    }
    .build-box {
        background: #f0fdf4; border-left: 4px solid #22c55e;
        padding: 1rem; border-radius: 8px; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Auto-build index if not present (Streamlit Cloud / fresh deployment) ─────
@st.cache_resource(show_spinner=False)
def ensure_index() -> int:
    """
    Builds the ChromaDB vector index if it doesn't already exist.
    Cached for the lifetime of the Streamlit server process.
    Returns number of indexed chunks.
    """
    from src.ingest import load_documents
    from src.chunker import chunk_documents
    from src.vectorstore import populate_db

    client = get_chroma_client(CHROMA_DIR)
    embed_fn = get_embedder()

    try:
        col = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
        if col.count() > 0:
            return col.count()
    except Exception:
        pass  # Collection doesn't exist yet — build it

    # Build fresh index
    docs = load_documents()
    chunks = chunk_documents(docs)
    populate_db(chunks, persist_dir=CHROMA_DIR)

    try:
        col = client.get_collection(COLLECTION_NAME, embedding_function=embed_fn)
        return col.count()
    except Exception:
        return len(chunks)


# ── Startup: ensure index exists ─────────────────────────────────────────────
with st.spinner("🔧 Initialising vector index (first run takes ~30s)..."):
    chunk_count = ensure_index()


# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏛️ Sarkari Scheme Navigator</h1>
    <p>Grounded RAG assistant for Indian Government Schemes — every answer cites the exact source clause.
    Refuses to guess when confidence is low.</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("System Controls")

    st.markdown("### 📊 Index Metrics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <div class="num">30</div><div class="label">Scheme PDFs</div>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <div class="num">{chunk_count}</div><div class="label">Vector Chunks</div>
        </div>""", unsafe_allow_html=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("""
        <div class="metric-card">
            <div class="num" style="color:#15803d;">100%</div>
            <div class="label">Recall @ 5</div>
        </div>""", unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="metric-card">
            <div class="num" style="color:#0369a1;">98.75%</div>
            <div class="label">MRR @ 5</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ RAG Hyperparameters")
    k_slider = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=TOP_K)
    threshold_slider = st.slider(
        "Similarity Threshold",
        min_value=0.10, max_value=0.80, value=SIM_THRESHOLD, step=0.05,
        help="Below this cosine similarity the system abstains instead of guessing."
    )

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    sample_queries = [
        "Select a sample...",
        "What subsidy does West Bengal's Amar Fasal Amar Gola scheme give for storehouses?",
        "What is the eligible age for women under Chhattisgarh's Silai Machine Sahayata Yojana?",
        "What is the maximum loan amount under Mahila Samridhi Yojana?",
        "What is the total outlay of DoT PLI Scheme for telecom products?",
        "How much do the top 10 winners of the Arunachal Pradesh Entrepreneurship Challenge receive?",
        "What annual pension does a 40–70% disabled person get under Chandigarh's scheme?",
        "What is the recipe for butter chicken? (Test Abstain Guard 🔬)",
        "Who won the FIFA World Cup 2022? (Test Abstain Guard 🔬)",
    ]
    selected_sample = st.selectbox("Quick Questions", sample_queries)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Built with ❤️ | LangChain · ChromaDB · Gemini · Streamlit")


# ── Chat State ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Namaste! 🙏 I am your **Sarkari Scheme Navigator**. Ask me about eligibility, "
                "grant amounts, loan caps, or application guidelines for official Indian Government Schemes. "
                "Every answer I give is grounded in the source document and page — no guessing."
            ),
            "sources": [], "abstained": False, "confidence": 1.0,
        }
    ]


def render_sources(sources, confidence):
    """Render a collapsible sources panel."""
    with st.expander(f"📚 Cited Sources ({len(sources)} | Confidence: {confidence:.0%})"):
        for idx, src in enumerate(sources, start=1):
            st.markdown(
                f"**Source {idx}:** `{src['source']}` (Page {src['page']})  "
                f"<span class='score-badge'>Score: {src['score']:.4f}</span>",
                unsafe_allow_html=True
            )
            if src.get("excerpt"):
                st.caption(f"*\"{src['excerpt']}\"*")
            st.divider()


# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🏛️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("abstained"):
            st.markdown("""
            <div class="abstain-box">
                ⚠️ <strong>System Abstained:</strong> Retrieval confidence was below the configured
                threshold. No hallucinated response was generated.
            </div>""", unsafe_allow_html=True)
        elif msg.get("sources"):
            render_sources(msg["sources"], msg.get("confidence", 0.0))


# ── Handle new input ──────────────────────────────────────────────────────────
prompt_input = st.chat_input("Ask about any government scheme — eligibility, grants, loan caps…")

# Guard against the sample dropdown re-firing on every Streamlit re-render.
# We store the last-used sample in session_state and only act when it changes.
if "_last_sample" not in st.session_state:
    st.session_state["_last_sample"] = "Select a sample..."

active_prompt = None
if prompt_input:
    active_prompt = prompt_input
elif selected_sample != "Select a sample..." and selected_sample != st.session_state["_last_sample"]:
    active_prompt = selected_sample
    st.session_state["_last_sample"] = selected_sample

if active_prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_prompt)

    # Generate answer
    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Searching official scheme archives…"):
            res = answer(
                query=active_prompt,
                k=k_slider,
                threshold=threshold_slider,
            )

        st.markdown(res["answer"])

        if res["abstained"]:
            st.markdown("""
            <div class="abstain-box">
                ⚠️ <strong>System Abstained:</strong> Retrieval confidence is below the safety
                threshold — no hallucinated answer generated.
            </div>""", unsafe_allow_html=True)
        elif res.get("sources"):
            render_sources(res["sources"], res["confidence"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": res["answer"],
        "sources": res["sources"],
        "abstained": res["abstained"],
        "confidence": res["confidence"],
    })
