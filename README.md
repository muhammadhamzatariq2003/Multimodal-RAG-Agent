# 🚀 Enterprise Multimodal RAG Document Intelligence Platform

> **Production-grade Retrieval-Augmented Generation (RAG) system engineered for enterprise documents, contracts, reports, and visual diagrams with strict citation grounding and zero hallucination.**

[![Architecture](https://img.shields.io/badge/Architecture-Modular_Clean_Architecture-blue.svg)](#architecture)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![VectorDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![LLM Support](https://img.shields.io/badge/LLM-HuggingFace%20%7C%20OpenAI%20%7C%20Local-purple.svg)](#llm-providers)

---

## 💼 Why Upwork Clients Need This Solution (Industry Problems Solved)

When corporate and enterprise clients seek RAG solutions on Upwork, they repeatedly face 5 critical bottlenecks:

| # | Typical Client Pain Point | Our Enterprise Solution |
|---|---|---|
| **1** | **Hallucinations & Untruthful Answers** | **Strict Grounding Guardrails**: Prompt engineering ensures the model answers strictly from retrieved chunks, citing exact document name and page number. |
| **2** | **Blind to Charts & Diagrams** | **Multimodal Vision Pipeline**: Extracts embedded charts, images, and schematics via `PyMuPDF` + `VisionEncoderDecoder` (ViT + GPT-2) to index visual intelligence alongside text. |
| **3** | **Duplicate Document Uploads & Bloat** | **SHA-256 Deduplication Registry**: Real-time content hashing prevents double-indexing, saving vector database compute and indexing costs. |
| **4** | **Spaghetti Code & Prototype Mess** | **Modular Production Architecture**: Clean separation of `core`, `services`, `processors`, and `storage` layers ready for Docker and CI/CD. |
| **5** | **Lost Chat Sessions on Page Refresh** | **Thread-Safe Persistent History**: Structured JSON session management ensures clients never lose their audit trail. |

---

## 🏛️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   Enterprise UI (Streamlit)                 │
│  - Multi-file Ingestion   - Citation Inspector   - Analytics│
└──────────────────────────────┬──────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌──────────────────────┐              ┌──────────────────────┐
│  PDF Text Processor  │              │  Vision Image Engine │
│  (PyMuPDF Layout)    │              │  (ViT-GPT2 / OCR)    │
└───────────┬──────────┘              └───────────┬──────────┘
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │        Chunking & Deduplication Engine      │
        │    - RecursiveCharacterTextSplitter         │
        │    - SHA-256 Checksum Ingestion Filter      │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │      ChromaDB High-Performance Vector DB    │
        │      - Sentence-Transformers Embeddings     │
        └──────────────────────┬──────────────────────┘
                               ▼
        ┌─────────────────────────────────────────────┐
        │       Grounded RAG Orchestrator             │
        │       - Top-K Context Injection             │
        │       - Page-Level Source Traceability      │
        │       - Persistent Session Storage          │
        └─────────────────────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
Enterprise-Multimodal-RAG-Assistant/
├── .env.example                  # Environment configuration template
├── README.md                     # Executive & Technical documentation
├── requirements.txt              # Standardized enterprise dependencies
├── app.py                        # Streamlit Enterprise UI Dashboard
├── data/
│   ├── chat_history/             # Persistent JSON conversation sessions
│   ├── chroma_db/                # Chroma Vector Database persistent files
│   └── processed_hashes.txt      # SHA-256 deduplication registry
└── src/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── config.py             # Pydantic environment and app configuration
    │   └── prompts.py            # Guarded enterprise prompts & citations rules
    ├── processors/
    │   ├── __init__.py
    │   ├── pdf_extractor.py      # Dual text and image extraction pipeline
    │   └── image_captioner.py    # Transformer-based diagram/chart captioning
    ├── services/
    │   ├── __init__.py
    │   ├── vector_store.py       # ChromaDB lifecycle, indexing & deduplication
    │   └── rag_engine.py         # Grounded inference and citation assembler
    └── storage/
        ├── __init__.py
        └── history_manager.py    # Persistent chat history management
```

---

## ⚡ Quick Start Guide

### 1. Clone & Set Up Virtual Environment
```bash
git clone <repo-url>
cd Enterprise-Multimodal-RAG-Assistant
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your HuggingFace or OpenAI token in `.env`:
```ini
HUGGINGFACEHUB_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

### 4. Run the Enterprise Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 Key Client-Facing Capabilities

1. **Multi-Document Drag & Drop**: Upload unlimited PDF files in parallel.
2. **Interactive Citation Inspector**: Expand and inspect the raw text passages and page references backing every answer.
3. **Live Vector Index Metrics**: Real-time counter showing vector chunks and active embeddings.
4. **Persistent Multi-Session Memory**: Reset or resume discussions across multiple document sets.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
