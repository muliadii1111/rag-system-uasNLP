"""
app.py
=======
Aplikasi utama Streamlit untuk RAG System.
Mengintegrasikan LangChain + LangGraph + LangSmith.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.rag_pipeline import (
    load_pdf,
    load_web,
    split_documents,
    create_vectorstore,
    save_vectorstore,
)
from src.graph_workflow import build_rag_graph, run_rag_graph

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System — UAS NLP",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Tema "Zine Lab Report"
# Putih cerah + pink/cyan/kuning jreng + border tebal hitam + hard shadow
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root{
    --paper: #FAFAF7;
    --ink: #15161B;
    --pink: #FF2E63;
    --pink-soft: #FFD9E4;
    --cyan: #00B8E6;
    --cyan-soft: #CFF3FF;
    --yellow: #FFD23F;
    --yellow-soft: #FFF3CC;
    --text-sub: #3A3C46;
}

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
.stApp { background-color: var(--paper); }
.main .block-container{ background: var(--paper); }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }

/* ---------- Hero ---------- */
.eyebrow-tag{
    display: inline-block;
    font-family:'JetBrains Mono', monospace; font-weight: 700; font-size: 0.72rem;
    letter-spacing: 0.08em; text-transform: uppercase;
    background: var(--yellow); color: var(--ink);
    border: 2.5px solid var(--ink); border-radius: 6px;
    padding: 5px 13px; transform: rotate(-2deg);
    box-shadow: 3px 3px 0 var(--ink);
    margin-bottom: 1.2rem;
}
.hero-title{
    font-family: 'Bebas Neue', sans-serif; font-weight: 400; font-size: 3.7rem;
    line-height: 0.96; color: var(--ink); text-transform: uppercase;
    letter-spacing: 0.01em; margin: 0 0 1.1rem 0;
}
.hl{ background: var(--pink); color: var(--paper); padding: 0 10px; display: inline-block; transform: rotate(-1deg); }
.hero-sub{ font-family:'Space Grotesk', sans-serif; font-size: 1.05rem; color: var(--text-sub); max-width: 540px; font-weight: 500; }

/* ---------- Flow divider ---------- */
.flow-divider{
    display: flex; align-items: center; gap: 12px; margin: 2.4rem 0 1.1rem;
    font-family:'JetBrains Mono', monospace; font-weight: 700; font-size: 0.74rem;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink);
}
.flow-divider::before, .flow-divider::after{ content:""; flex:1; height: 2.5px; background: var(--ink); }

/* ---------- Pipeline diagram (signature element) ---------- */
.pipeline-row{ display: flex; align-items: stretch; gap: 10px; overflow-x: auto; padding: 0.6rem 0.2rem 1rem; flex-wrap: wrap; }
.pl-node{
    flex: 1; min-width: 150px;
    border: 2.5px solid var(--ink); border-radius: 10px; padding: 1.15rem 1rem;
    box-shadow: 5px 5px 0 var(--ink);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.pl-node:hover{ transform: rotate(0deg) translate(-2px,-2px) !important; box-shadow: 7px 7px 0 var(--ink); }
.pl-node.c1{ background: var(--pink-soft); transform: rotate(-1.5deg); }
.pl-node.c2{ background: var(--cyan-soft); transform: rotate(1deg); }
.pl-node.c3{ background: var(--yellow-soft); transform: rotate(-1deg); }
.pl-node.c4{ background: var(--pink-soft); transform: rotate(1.5deg); }
.pl-num{ font-family:'JetBrains Mono', monospace; font-weight: 700; font-size: 0.7rem; color: var(--ink); opacity: 0.55; }
.pl-icon{ font-size: 1.7rem; margin: 0.35rem 0; }
.pl-label{ font-family:'Space Grotesk', sans-serif; font-weight: 700; font-size: 0.96rem; color: var(--ink); text-transform: uppercase; }
.pl-tag{ font-family:'JetBrains Mono', monospace; font-weight: 700; font-size: 0.66rem; margin-top: 5px; }
.pl-desc{ font-family:'Space Grotesk', sans-serif; font-size: 0.75rem; color: var(--text-sub); margin-top: 6px; font-weight: 500; line-height: 1.35; }
.pl-footer{
    margin-top: 0.4rem; font-family:'JetBrains Mono', monospace; font-weight: 600; font-size: 0.78rem;
    color: var(--ink); text-align: center; border-top: 2.5px dashed var(--ink); padding-top: 0.9rem;
}
.pl-footer b{ background: var(--cyan); padding: 1px 9px; border-radius: 4px; border: 2px solid var(--ink); }

/* ---------- Stat readout ---------- */
.stat-row{ display: flex; gap: 0.6rem; flex-wrap: wrap; margin: 0.4rem 0 1.8rem; }
.stat-chip{
    font-family:'JetBrains Mono', monospace; font-weight: 600; font-size: 0.74rem;
    background: var(--paper); border: 2px solid var(--ink); color: var(--ink);
    padding: 5px 13px; border-radius: 6px; box-shadow: 2px 2px 0 var(--ink);
}

/* ---------- Section title ---------- */
.section-title{ font-family:'Bebas Neue', sans-serif; font-size: 2.1rem; color: var(--ink); text-transform: uppercase; margin: 0.3rem 0 1rem; }

/* ---------- Empty state ---------- */
.empty-state{ border: 2.5px dashed var(--ink); border-radius: 14px; padding: 2.6rem 1.5rem; text-align: center; background: var(--yellow-soft); }
.empty-icon{ font-size: 2.2rem; margin-bottom: 0.6rem; }
.empty-title{ font-family:'Bebas Neue', sans-serif; font-size: 1.7rem; color: var(--ink); text-transform: uppercase; }
.empty-sub{ font-family:'Space Grotesk', sans-serif; font-size: 0.92rem; color: var(--text-sub); font-weight: 500; }

/* ---------- Chat ---------- */
.answer-card{
    background: var(--paper); border: 2.5px solid var(--ink); border-left: 7px solid var(--pink);
    border-radius: 4px 14px 14px 4px; padding: 1.3rem 1.5rem; color: var(--ink);
    font-family:'Space Grotesk', sans-serif; font-weight: 500; line-height: 1.7;
    box-shadow: 4px 4px 0 var(--ink);
}
.trace-wrap{ margin-top: 0.6rem; padding: 0.2rem; }
.trace-item{ display: flex; align-items: flex-start; gap: 10px; padding: 5px 0; }
.trace-dot{ width: 8px; height: 8px; border-radius: 2px; background: var(--cyan); border: 1.5px solid var(--ink); margin-top: 5px; flex-shrink: 0; }
.trace-text{ font-family:'JetBrains Mono', monospace; font-weight: 500; font-size: 0.78rem; color: var(--text-sub); }
.source-label{
    font-family:'JetBrains Mono', monospace; font-weight: 700; font-size: 0.7rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink); margin: 0.9rem 0 0.5rem;
}
.source-chip{
    display: inline-block; font-family:'JetBrains Mono', monospace; font-weight: 600; font-size: 0.74rem;
    background: var(--cyan-soft); color: var(--ink); border: 2px solid var(--ink);
    padding: 3px 11px; border-radius: 6px; margin: 3px 6px 3px 0;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{ background: var(--ink); border-right: 4px solid var(--pink); }
section[data-testid="stSidebar"] *{ color: var(--paper) !important; }
.side-label{
    font-family:'JetBrains Mono', monospace; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--yellow) !important; margin: 1.3rem 0 0.6rem;
}
section[data-testid="stSidebar"] hr{ border-color: #33343C; margin: 1rem 0; }

/* ---------- Widget restyle ---------- */
.stButton > button{
    background: var(--yellow); color: var(--ink) !important; border: 2.5px solid var(--ink);
    border-radius: 8px; font-weight: 700; font-family:'Space Grotesk', sans-serif;
    box-shadow: 4px 4px 0 var(--ink); transition: all 0.1s ease;
}
.stButton > button:hover{ background: var(--pink); color: var(--paper) !important; }
.stButton > button:active{ transform: translate(3px,3px); box-shadow: 0 0 0 var(--ink); }
.stChatInput textarea, .stTextInput input{
    background: var(--paper) !important; border: 2.5px solid var(--ink) !important;
    color: var(--ink) !important; border-radius: 8px !important; font-family:'Space Grotesk', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"]{ background: var(--paper) !important; border: 2.5px dashed var(--ink) !important; border-radius: 10px; }
[data-testid="stFileUploaderDropzone"] *{ color: var(--ink) !important; }
[data-testid="stFileUploaderDropzone"] button{
    background: var(--yellow) !important; color: var(--ink) !important;
    border: 2px solid var(--ink) !important; border-radius: 6px !important; font-weight: 700 !important;
}
[data-testid="stFileUploaderFile"]{ background: var(--paper) !important; border: 2px solid var(--ink) !important; border-radius: 8px !important; }
[data-testid="stFileUploaderFile"] *{ color: var(--ink) !important; }
[data-testid="stMetricValue"]{ font-family:'JetBrains Mono', monospace; }
[data-testid="stExpander"]{ background: var(--paper) !important; border: 2.5px solid var(--ink) !important; border-radius: 10px !important; box-shadow: 4px 4px 0 var(--ink); margin-bottom: 0.5rem; }
[data-testid="stExpander"] summary{ background: var(--paper) !important; color: var(--ink) !important; font-weight: 700 !important; border-radius: 8px !important; }
[data-testid="stExpander"] summary *{ color: var(--ink) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"]{ background: var(--paper) !important; color: var(--ink) !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] *{ color: var(--ink) !important; }
.stAlert{ border-radius: 8px; border: 2px solid var(--ink) !important; }
.stAlert *{ color: var(--ink) !important; }
[data-testid="stRadio"] label p{ color: var(--text-sub); font-weight: 600; }
section[data-testid="stSidebar"] [data-testid="stRadio"] label p{ color: var(--paper) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────────
for key, default in [
    ("vectorstore", None), ("rag_graph", None), ("chat_history", []),
    ("doc_loaded", False), ("doc_name", ""), ("chunk_count", 0), ("char_count", 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="side-label">⚙ Konfigurasi Kunci</div>', unsafe_allow_html=True)
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...",
                              value=os.getenv("GROQ_API_KEY", ""),
                              help="Daftar gratis di https://console.groq.com")
    langsmith_key = st.text_input("LangSmith API Key", type="password", placeholder="lsv2_...",
                                   value=os.getenv("LANGCHAIN_API_KEY", ""),
                                   help="Daftar gratis di https://smith.langchain.com")
    langsmith_project = st.text_input("LangSmith Project",
                                       value=os.getenv("LANGCHAIN_PROJECT", "rag-system-uas-nlp"))

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
    if langsmith_key:
        os.environ["LANGCHAIN_API_KEY"] = langsmith_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    if langsmith_project:
        os.environ["LANGCHAIN_PROJECT"] = langsmith_project

    st.markdown('<div class="side-label">📄 Sumber Dokumen</div>', unsafe_allow_html=True)
    source_type = st.radio("Pilih sumber:", ["Upload PDF", "URL Web"], horizontal=True, label_visibility="collapsed")

    if source_type == "Upload PDF":
        uploaded_file = st.file_uploader("Upload file PDF", type=["pdf"],
                                          help="PDF akan di-chunk dan di-embed ke vector store")
        process_btn = st.button("⚡ Proses Dokumen", use_container_width=True, type="primary")

        if process_btn and uploaded_file:
            if not groq_key:
                st.error("❌ Masukkan Groq API Key terlebih dahulu!")
            else:
                with st.spinner("⏳ Memproses dokumen..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.read())
                            tmp_path = tmp.name

                        docs = load_pdf(tmp_path)
                        chunks = split_documents(docs)
                        vectorstore = create_vectorstore(chunks)
                        rag_graph = build_rag_graph(vectorstore)

                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_graph = rag_graph
                        st.session_state.doc_loaded = True
                        st.session_state.doc_name = uploaded_file.name
                        st.session_state.chat_history = []
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.char_count = sum(len(c.page_content) for c in chunks)

                        os.unlink(tmp_path)
                        st.success(f"✅ {len(chunks)} chunk berhasil diproses!")
                        st.info(f"📄 {uploaded_file.name} siap ditanya!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    else:
        web_url = st.text_input("Masukkan URL", placeholder="https://example.com/artikel")
        process_web_btn = st.button("⚡ Proses URL", use_container_width=True, type="primary")

        if process_web_btn and web_url:
            if not groq_key:
                st.error("❌ Masukkan Groq API Key terlebih dahulu!")
            else:
                with st.spinner("⏳ Memuat halaman web..."):
                    try:
                        docs = load_web(web_url)
                        chunks = split_documents(docs)
                        vectorstore = create_vectorstore(chunks)
                        rag_graph = build_rag_graph(vectorstore)

                        st.session_state.vectorstore = vectorstore
                        st.session_state.rag_graph = rag_graph
                        st.session_state.doc_loaded = True
                        st.session_state.doc_name = web_url
                        st.session_state.chat_history = []
                        st.session_state.chunk_count = len(chunks)
                        st.session_state.char_count = sum(len(c.page_content) for c in chunks)

                        st.success(f"✅ {len(chunks)} chunk berhasil diproses!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    st.markdown('<div class="side-label">📊 Status Sistem</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LangSmith", "ON" if langsmith_key else "OFF")
    with col2:
        st.metric("Dokumen", "✅" if st.session_state.doc_loaded else "❌")

    if st.session_state.doc_loaded:
        name = st.session_state.doc_name
        st.caption(f"📁 `{name[:40]}...`" if len(name) > 40 else f"📁 `{name}`")

    if st.session_state.chat_history:
        if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption("UAS NLP · LangChain + LangGraph + LangSmith")


# ─────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────
st.markdown("""
<div class="eyebrow-tag">UAS NLP · RAG PIPELINE</div>
<p class="hero-title">Tanya apa saja<br>dari <span class="hl">dokumenmu.</span></p>
<p class="hero-sub">Retrieval-Augmented Generation yang membaca dokumenmu, menyaring bagian yang relevan, lalu menyusun jawaban — dibangun dengan LangChain, LangGraph, dan LangSmith.</p>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Pipeline Diagram — elemen signature
# ─────────────────────────────────────────────
st.markdown('<div class="flow-divider">Alur Pemrosesan</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-row">
    <div class="pl-node c1">
      <div class="pl-num">01</div>
      <div class="pl-icon">📄</div>
      <div class="pl-label">Muat &amp; Embed</div>
      <div class="pl-tag">LangChain</div>
      <div class="pl-desc">PDF/Web → chunks → vector FAISS</div>
    </div>
    <div class="pl-node c2">
      <div class="pl-num">02</div>
      <div class="pl-icon">🔎</div>
      <div class="pl-label">Retrieve</div>
      <div class="pl-tag">LangGraph</div>
      <div class="pl-desc">cari top-4 chunk paling mirip</div>
    </div>
    <div class="pl-node c3">
      <div class="pl-num">03</div>
      <div class="pl-icon">⚖️</div>
      <div class="pl-label">Grade</div>
      <div class="pl-tag">LangGraph</div>
      <div class="pl-desc">saring chunk yang relevan</div>
    </div>
    <div class="pl-node c4">
      <div class="pl-num">04</div>
      <div class="pl-icon">✨</div>
      <div class="pl-label">Generate</div>
      <div class="pl-tag">atau → Fallback</div>
      <div class="pl-desc">susun jawaban dari konteks</div>
    </div>
</div>
<div class="pl-footer"><b>LangSmith</b> &nbsp;melacak setiap langkah 01–04 secara otomatis ke dashboard</div>
""", unsafe_allow_html=True)

with st.expander("Detail tiap tahap pipeline"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🔗 LangChain**
- PDF / Web Loader
- RecursiveTextSplitter
- HuggingFace Embeddings
- FAISS Vector Store
- RAG Chain (LCEL)
- Groq LLM
        """)
    with col2:
        st.markdown("""
**🕸️ LangGraph**
- StateGraph definition
- Node: retrieve
- Node: grade_documents
- Node: generate
- Node: fallback
- Conditional edges
        """)
    with col3:
        st.markdown("""
**🔭 LangSmith**
- Auto tracing semua chain
- Monitor latency & token
- Debug step-by-step
- Logging runs
- Dashboard monitoring
- Evaluasi output
        """)

if st.session_state.doc_loaded:
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">{st.session_state.chunk_count} CHUNKS</div>
        <div class="stat-chip">{st.session_state.char_count:,} KARAKTER</div>
        <div class="stat-chip">MODEL: GPT-OSS-20B</div>
        <div class="stat-chip">EMBED: MINILM-L6-V2</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Chat Interface
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">💬 Chat dengan Dokumen</p>', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🗂️"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🔭"):
            st.markdown(f'<div class="answer-card">{msg["content"]}</div>', unsafe_allow_html=True)
            if "steps" in msg:
                with st.expander("🔍 Lihat proses LangGraph"):
                    trace_html = '<div class="trace-wrap">' + "".join(
                        f'<div class="trace-item"><div class="trace-dot"></div><div class="trace-text">{s}</div></div>'
                        for s in msg["steps"]
                    ) + '</div>'
                    st.markdown(trace_html, unsafe_allow_html=True)
            if "sources" in msg and msg["sources"]:
                st.markdown('<div class="source-label">Sumber</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="source-chip">{src}</span>' for src in msg["sources"])
                st.markdown(chips, unsafe_allow_html=True)

if not st.session_state.doc_loaded:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🗂️</div>
        <div class="empty-title">Belum Ada Dokumen</div>
        <div class="empty-sub">Upload PDF dari sidebar kiri untuk mulai mengobrol dengan isinya.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    question = st.chat_input("Tanyakan sesuatu tentang dokumen...")

    if question:
        with st.chat_message("user", avatar="🗂️"):
            st.write(question)

        with st.chat_message("assistant", avatar="🔭"):
            with st.spinner("🤔 Sedang berpikir..."):
                try:
                    result = run_rag_graph(st.session_state.rag_graph, question)

                    answer = result.get("generation", "Terjadi kesalahan dalam menghasilkan jawaban.")
                    steps = result.get("steps", [])
                    relevant_docs = result.get("relevant_documents", [])

                    sources = []
                    for doc in relevant_docs:
                        page = doc.metadata.get("page", None)
                        src = doc.metadata.get("source", "Dokumen")
                        sources.append(f"Hal. {page + 1}" if page is not None else str(src)[:30])
                    sources = list(set(sources))

                    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                    with st.expander("🔍 Lihat proses LangGraph"):
                        trace_html = '<div class="trace-wrap">' + "".join(
                            f'<div class="trace-item"><div class="trace-dot"></div><div class="trace-text">{s}</div></div>'
                            for s in steps
                        ) + '</div>'
                        st.markdown(trace_html, unsafe_allow_html=True)

                    if sources:
                        st.markdown('<div class="source-label">Sumber</div>', unsafe_allow_html=True)
                        chips = "".join(f'<span class="source-chip">{src}</span>' for src in sources)
                        st.markdown(chips, unsafe_allow_html=True)

                    st.session_state.chat_history.append({"role": "user", "content": question})
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer, "steps": steps, "sources": sources,
                    })

                    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
                        st.caption(f"✅ Run ini sudah di-trace ke LangSmith project: `{os.getenv('LANGCHAIN_PROJECT')}`")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Pastikan API key sudah diisi dengan benar.")