import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ingest import load_or_build_index
from utils import summarize_textrank, highlight_hits

st.set_page_config(page_title="FlashFind", layout="wide")

# --- Model cache ---
@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # small, CPU-friendly

# --- UI header ---
st.title("FlashFind • AI search for your files")
st.caption("Drop PDFs / .txt / .md in ./data then click ‘Rebuild Index’. Runs fully local.")

# --- Sidebar controls ---
st.sidebar.header("Search settings")
min_conf = st.sidebar.slider("No-answer threshold", 0.0, 1.0, 0.35, 0.01,
                             help="If the best similarity is below this, show a ‘No good match’ message.")
top_k = st.sidebar.slider("Results to show", 3, 10, 5)

# --- Rebuild button & query box ---
c1, c2 = st.columns([3, 1])
with c1:
    query = st.text_input("Ask a question (about your indexed documents)", placeholder="e.g., What protections guard the Philosopher’s Stone?")
with c2:
    if st.button("Rebuild Index", use_container_width=True):
        st.session_state.idx = None

# --- Load or build index ---
model = get_model()
if "idx" not in st.session_state or st.session_state.idx is None:
    with st.spinner("Building or loading index..."):
        st.session_state.idx = load_or_build_index(model)

texts, embeds, meta = st.session_state.idx

# --- Corpus banner (what's actually indexed) ---
if meta:
    unique_sources = sorted({m.get("source", "") for m in meta if m.get("source")})[:12]
    more = "…" if len({m.get('source', '') for m in meta}) > 12 else ""
    st.info("**Indexed sources:** " + ", ".join(unique_sources) + more)

# --- Search flow ---
if query:
    # Embed query and compute similarity
    q_vec = model.encode([query], normalize_embeddings=True)
    sims = cosine_similarity(q_vec, embeds)[0]
    best = float(np.max(sims)) if len(sims) else 0.0

    # No-answer guardrail
    if best < min_conf:
        st.subheader("No good match found")
        st.write(
            "This question appears to be outside your indexed documents. "
            "Try adding relevant files to `./data` or rephrasing the query."
        )
        st.caption(f"Best similarity: {best:.3f}  •  Threshold: {min_conf:.2f}")
    else:
        # Retrieve top-k and summarize
        top_idx = np.argsort(-sims)[:top_k]
        joined = " ".join([texts[i] for i in top_idx])

        st.subheader("Answer")
        st.write(summarize_textrank(joined))

        st.subheader("Sources")
        for rank, i in enumerate(top_idx, start=1):
            src = meta[i].get("source", "unknown")
            pg = meta[i].get("page")
            label = f"{src} — page {pg}" if pg else src
            with st.expander(f"{rank}. {label}  •  score {sims[i]:.3f}"):
                st.markdown(highlight_hits(texts[i], query), unsafe_allow_html=True)

        st.caption(f"Best similarity: {best:.3f}  •  Threshold: {min_conf:.2f}")
