# DocuMind

A document-grounded QA chatbot. Upload PDFs, DOCX, TXT, or CSV files and ask questions. Answers come only from your documents, with citations.

---

## Features

- Multi-format document support (PDF, DOCX, TXT, CSV)
- Hybrid retrieval — semantic search + lexical search
- Conversation-aware follow-up questions
- Grounded answers with source citations
- Multi-document support
- Streamlit interface

---

## Architecture

```
Upload Documents
        ↓
Chunk (RecursiveCharacterTextSplitter)
        ↓
BGE-small Embeddings → FAISS Index
BM25 Index
        ↓
Reciprocal Rank Fusion
        ↓
Query Reformulation (follow-ups only)
        ↓
Llama-3.3-70B via Groq
        ↓
Answer + Citations
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector search | FAISS (IndexFlatIP) |
| Lexical search | BM25 (rank_bm25) |
| Ranking fusion | Reciprocal Rank Fusion |
| Generation | Llama-3.3-70B via Groq |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| UI | Streamlit |

---

## Why Hybrid Retrieval?

Dense embeddings capture meaning but can miss exact terms and abbreviations. BM25 handles exact keyword matches. RRF combines both rankings, rewarding chunks that score well in both systems.

---

## Evaluation

Custom 40-question benchmark covering: fact, why, compare, multi-hop, cross-document, summary, topic-switch, refusal, and adversarial categories.

**Overall score: 8.3 / 10**

---

## Setup

```bash
pip install -r requirements.txt
```

Add your Groq API key to `.env`:

```
GROQ_API_KEY=your_key_here
```

```bash
streamlit run app.py
```

---

## Future Improvements

- Persistent vector store
- OCR support for scanned PDFs
- Cross-encoder reranking
- Retrieval metrics (Recall@k, MRR)
