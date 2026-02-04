# Local RAG Chatbot

End-to-end **local Retrieval-Augmented Generation (RAG) chatbot** for answering questions over PDF and Markdown documents using vector search and a locally hosted LLM.

This project focuses on **retrieval quality, hallucination control, and system reliability**, rather than prompt-only generation or hosted APIs.

---

## System Overview

The chatbot follows a standard RAG pipeline:

1. **Offline ingestion** of PDF and Markdown documents  
2. **Chunking with overlap** for semantic coverage  
3. **Embedding generation** using MiniLM  
4. **Vector similarity search** with FAISS  
5. **Deterministic local LLM inference** using Phi-3-mini (via llama-cpp)  
6. **Source-attributed responses** served through a FastAPI backend  
7. **Lightweight Streamlit UI** for interactive demos

All retrieval policies are enforced **server-side** to ensure consistent behavior.

---

## Key Features

- Local, offline-friendly RAG system (no external LLM APIs)
- Similarity-thresholded retrieval (top-k + minimum score)
- Deterministic generation (temperature = 0)
- Source attribution for grounded answers
- FastAPI inference service
- Streamlit chat UI for interactive exploration

---

## Tech Stack

- **Language:** Python  
- **Backend:** FastAPI  
- **Embeddings:** SentenceTransformers (MiniLM)  
- **Vector Store:** FAISS  
- **LLM:** Phi-3-mini (llama-cpp)  
- **UI:** Streamlit  

---


