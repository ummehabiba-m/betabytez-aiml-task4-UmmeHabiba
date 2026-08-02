import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

# ---- Config ----
DISTANCE_THRESHOLD = 1.5
REFUSAL_MESSAGE = "I cannot answer this from the provided documents."
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index", "faiss_index.index")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index", "chunks.pkl")

# ---- Load environment variables (Groq API key) ----
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it to your .env file (local) or Streamlit secrets (cloud).")
    st.stop()

client = Groq(api_key=groq_api_key)

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="◆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---- Custom styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #0B0E14;
    --surface: #141924;
    --surface-hover: #1A2030;
    --border: #232838;
    --gold: #D4A72C;
    --teal: #3FB8AF;
    --text: #E8EAED;
    --text-muted: #8890A0;
}

.stApp {
    background: var(--bg);
}

/* Hide Streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Main container width */
.block-container {
    max-width: 780px;
    padding-top: 2.5rem;
}

/* Header */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}
.app-header .mark {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    background: linear-gradient(135deg, var(--gold), #B8860B);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 18px;
    color: #0B0E14;
    flex-shrink: 0;
}
.app-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 26px !important;
    color: var(--text) !important;
    margin: 0 !important;
    line-height: 1.2;
}
.app-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    color: var(--text-muted);
    margin: 2px 0 28px 54px;
    line-height: 1.5;
}
.app-subtitle em {
    color: #A8B0C0;
    font-style: normal;
    border-bottom: 1px dotted #3A4256;
}

/* Chat message cards */
.msg-row {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    align-items: flex-start;
}
.msg-avatar {
    width: 30px;
    height: 30px;
    border-radius: 7px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    margin-top: 2px;
}
.msg-avatar.user {
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-muted);
}
.msg-avatar.assistant {
    background: rgba(212, 167, 44, 0.15);
    border: 1px solid rgba(212, 167, 44, 0.4);
    color: var(--gold);
}
.msg-avatar.refused {
    background: rgba(63, 184, 175, 0.1);
    border: 1px solid rgba(63, 184, 175, 0.3);
    color: var(--teal);
}
.msg-content {
    flex: 1;
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: var(--text);
    padding-top: 4px;
}
.msg-content.user-text {
    color: #C5CAD4;
}

/* Grounded badge */
.grounded-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 8px;
}
.grounded-tag.yes {
    background: rgba(212, 167, 44, 0.12);
    color: var(--gold);
    border: 1px solid rgba(212, 167, 44, 0.25);
}
.grounded-tag.no {
    background: rgba(63, 184, 175, 0.1);
    color: var(--teal);
    border: 1px solid rgba(63, 184, 175, 0.25);
}

/* Source cards */
.sources-wrap {
    margin-top: 12px;
    margin-left: 42px;
}
.source-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 2px solid var(--teal);
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
}
.source-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    color: var(--teal);
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
}
.source-meta .dist {
    color: var(--text-muted);
}
.source-snippet {
    font-family: 'Inter', sans-serif;
    font-size: 12.5px;
    color: var(--text-muted);
    line-height: 1.5;
}

/* Suggested questions */
.suggestion-chip {
    display: inline-block;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    padding: 7px 14px;
    border-radius: 20px;
    margin: 4px 6px 4px 0;
}

/* Chat input styling */
.stChatInput {
    border-color: var(--border) !important;
}
.stChatInput textarea {
    background: var(--surface) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Expander (sources) override */
.streamlit-expanderHeader {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    color: var(--teal) !important;
}

hr {
    border-color: var(--border) !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="app-header">
    <div class="mark">◆</div>
    <h1>Document Q&A</h1>
</div>
<div class="app-subtitle">
    Grounded in <em>Attention Is All You Need</em> &amp; <em>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks</em>
</div>
""", unsafe_allow_html=True)


# ---- Cache the heavy resources so they only load once per session ----
@st.cache_resource
def load_resources():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return embedding_model, index, chunks

embedding_model, index, chunks = load_resources()


# ---- Core RAG functions ----
def retrieve_chunks(query, k=4):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "text": chunks[idx].page_content,
            "source_paper": chunks[idx].metadata["source_paper"],
            "page": chunks[idx].metadata["page_label"],
            "distance": float(distances[0][rank])
        })
    return results


def generate_answer_with_grounding_check(query, k=4):
    retrieved = retrieve_chunks(query, k=k)
    best_distance = retrieved[0]["distance"]

    if best_distance > DISTANCE_THRESHOLD:
        return {"answer": REFUSAL_MESSAGE, "sources": retrieved, "grounded": False}

    context_block = "\n\n".join([
        f"[Source: {r['source_paper']}, Page {r['page']}]\n{r['text']}"
        for r in retrieved
    ])

    system_prompt = (
        "You are a document Q&A assistant. Answer the user's question using "
        "ONLY the context provided below. Do not use any outside knowledge, "
        "even if you know the answer from general training. "
        "If the answer is not contained in the provided context, respond "
        "exactly with: 'I cannot answer this from the provided documents.' "
        "Do not guess, speculate, or fill in gaps with information not present "
        "in the context."
    )
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    is_grounded = REFUSAL_MESSAGE.lower() not in answer.lower()

    return {"answer": answer, "sources": retrieved, "grounded": is_grounded}


def render_sources(sources):
    with st.expander("SOURCES · " + " / ".join(sorted(set(s["source_paper"].split()[0] for s in sources)))):
        for s in sources:
            st.markdown(f"""
<div class="source-card">
    <div class="source-meta">
        <span>{s['source_paper']} — p.{s['page']}</span>
        <span class="dist">d={s['distance']:.3f}</span>
    </div>
    <div class="source-snippet">{s['text'][:220].strip()}...</div>
</div>
""", unsafe_allow_html=True)


# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Empty state: suggested questions ----
if not st.session_state.messages:
    st.markdown("""
<div style="margin: 8px 0 24px 0;">
    <span class="suggestion-chip">What is self-attention?</span>
    <span class="suggestion-chip">How does RAG combine memory types?</span>
    <span class="suggestion-chip">What is multi-head attention?</span>
</div>
""", unsafe_allow_html=True)

# ---- Render history ----
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="msg-row">
    <div class="msg-avatar user">Q</div>
    <div class="msg-content user-text">{msg["content"]}</div>
</div>
""", unsafe_allow_html=True)
    else:
        avatar_class = "assistant" if msg.get("grounded", True) else "refused"
        tag_class = "yes" if msg.get("grounded", True) else "no"
        tag_text = "Grounded answer" if msg.get("grounded", True) else "Not in documents"
        st.markdown(f"""
<div class="msg-row">
    <div class="msg-avatar {avatar_class}">◆</div>
    <div class="msg-content">
        <span class="grounded-tag {tag_class}">{tag_text}</span><br/>
        {msg["content"]}
    </div>
</div>
""", unsafe_allow_html=True)
        if "sources" in msg:
            st.markdown('<div class="sources-wrap">', unsafe_allow_html=True)
            render_sources(msg["sources"])
            st.markdown('</div>', unsafe_allow_html=True)

# ---- Chat input ----
if user_question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.spinner("Searching documents..."):
        result = generate_answer_with_grounding_check(user_question)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "grounded": result["grounded"]
    })
    st.rerun()