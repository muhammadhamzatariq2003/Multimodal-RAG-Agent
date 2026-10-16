import streamlit as st
import time
import hashlib
from src.core.config import settings
from src.services.vector_store import KnowledgeBaseManager
from src.services.rag_engine import EnterpriseRAGEngine
from src.processors.pdf_extractor import PDFProcessor
from src.storage.history_manager import get_session_history
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(
    page_title="Enterprise Multimodal RAG Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Enterprise Styling (Tailwind-inspired CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #334155;
    }
    
    .badge-live {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.4);
        display: inline-block;
        margin-bottom: 8px;
    }

    .citation-box {
        background: #f8fafc;
        border-left: 3px solid #3b82f6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 13px;
        color: #334155;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# State initialization
if "kb_manager" not in st.session_state:
    st.session_state.kb_manager = KnowledgeBaseManager()
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = EnterpriseRAGEngine(st.session_state.kb_manager)
if "session_id" not in st.session_state:
    st.session_state.session_id = "tenant_client_workspace"
if "history_store" not in st.session_state:
    st.session_state.history_store = get_session_history(st.session_state.session_id)

# Sidebar
with st.sidebar:
    st.markdown("### 🏢 Enterprise Document Hub")
    st.caption("AI-Powered Multimodal Ingestion Pipeline")
    
    # Document Ingestion Section
    uploaded_files = st.file_uploader(
        "Upload Client PDF Documents (Text + Visuals)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload complex PDFs containing tables, contracts, diagrams, or charts."
    )
    
    if uploaded_files:
        if st.button("🚀 Ingest & Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Extracting text, analyzing visual diagrams, and building vector index..."):
                for uf in uploaded_files:
                    file_bytes = uf.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    
                    # Extract text & diagrams
                    extractor = PDFProcessor(file_bytes, uf.name)
                    chunks = extractor.process()
                    
                    success, message = st.session_state.kb_manager.index_document(chunks, file_hash, uf.name)
                    if success:
                        st.success(f"✓ {message}")
                    else:
                        st.info(f"ℹ {message}")
    
    st.divider()
    
    # Knowledge Base Live Analytics
    stats = st.session_state.kb_manager.get_stats()
    st.markdown("#### 📊 Knowledge Base Status")
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 12px; color: #64748b;">TOTAL VECTOR CHUNKS</div>
        <div style="font-size: 20px; font-weight: 700; color: #0f172a;">{stats['total_chunks']}</div>
        <div style="font-size: 11px; color: #10b981; margin-top: 4px;">Indexed in ChromaDB</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Embedding Model:** `{stats['embedding_model']}`")
    st.markdown(f"**LLM Target:** `{settings.LLM_MODEL}`")
    
    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.history_store.clear()
        st.rerun()

# Main Header
st.markdown("""
<div class="main-header">
    <div class="badge-live">● PRODUCTION READY RAG</div>
    <h2 style="margin: 0; padding: 0; color: white;">Enterprise Multimodal Document Intelligence</h2>
    <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 14px;">
        High-precision contextual Q&A with OCR diagram captions, chunk-level citations, and persistent session memory.
    </p>
</div>
""", unsafe_allow_html=True)

# Display Messages from Persistent History
messages = st.session_state.history_store.messages
if not messages:
    st.info("💡 **Welcome to your Enterprise Document Assistant.** Upload your company contracts, reports, or technical manuals on the sidebar and ask any question!")
else:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="⚡"):
                st.markdown(msg.content)

# User Chat Input
user_input = st.chat_input("Ask a grounded question about your uploaded documents...")

if user_input:
    # Display user query
    with st.chat_message("user", avatar="👤"):
        st.write(user_input)
    
    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Searching multimodal vector store and synthesizing response..."):
            response = st.session_state.rag_engine.query(user_input, session_id=st.session_state.session_id)
            st.markdown(response["answer"])
            
            # Show interactive source citations accordion
            if response["citations"]:
                with st.expander("🔍 View Verified Source Citations & Evidence Chunks"):
                    for idx, c in enumerate(response["citations"]):
                        st.markdown(f"""
                        <div class="citation-box">
                            <strong>Passage #{idx+1}</strong> — Document: <code>{c['source']}</code> | Page: <strong>{c['page']}</strong> | Type: <em>{c['type'].capitalize()}</em><br/>
                            <span style="color: #475569;">"{c['preview']}"</span>
                        </div>
                        """, unsafe_allow_html=True)
    st.rerun()
