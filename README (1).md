# DocuMind

A document-grounded QA chatbot. Upload PDFs, DOCX, TXT, or CSV files — ask questions, get answers backed by citations from your actual documents, not the internet.

---

## Quick Start

```bash
pip install -r requirements.txt
# add GROQ_API_KEY to .env
streamlit run app.py
```

---

## Architecture

```
Upload → Load → Chunk → Embed → FAISS Index
                                     ↓
Query → [HybridRetrieve] → Threshold Gate → [QueryRewrite?] → LLM → Answer + Citations
             ↑
     semantic (FAISS) + keyword overlap (80/20)
```

**Five stages, each independently testable:**

| Stage | What it does | Key decision |
|---|---|---|
| **Load** | Extracts text per-page (PDF), per-paragraph (DOCX), per-row (CSV) | Each format handled separately; corrupted files skipped with warning, not crash |
| **Chunk** | Splits text at paragraph/sentence boundaries, 800 chars, 150 overlap | LangChain `RecursiveCharacterTextSplitter` — respects sentence boundaries unlike fixed-char splitting |
| **Embed** | Encodes chunk content into 384-dim vectors | `all-MiniLM-L6-v2`; content only, no metadata prefix (see Decisions) |
| **Index** | Stores normalized vectors in FAISS | `IndexFlatIP` (exact inner product = cosine on normalized vectors) |
| **Retrieve** | Semantic search + keyword boost → threshold gate → optional query rewrite | Hybrid scoring: `0.8 × semantic + 0.2 × keyword_overlap` |
| **Generate** | LLM answers strictly from retrieved context | `llama-3.3-70b-versatile` via Groq; numbered citations enforced structurally |

---

## Tech Stack

### Why these, not alternatives

**FAISS (not ChromaDB / Pinecone)**
- Runs in-process, zero infra, no persistence needed for ephemeral Streamlit sessions
- `IndexFlatIP` = exact search; brute-force is fine at < 10k chunks
- ChromaDB adds disk persistence and metadata filtering — useful if you need cross-session memory, unnecessary here
- Pinecone/Qdrant are managed services — overkill and not free for this scale

**all-MiniLM-L6-v2 (not larger models)**
- 384-dim, ~80MB, runs on CPU in < 100ms per query
- Retrieval-class model, not generation-class — sized for semantic matching, not reasoning
- Known limitation: weak on proper-noun-dense text (tool lists, named entities) — confirmed empirically with similarity scores (see Limitations)
- Tradeoff accepted: latency and zero-cost deployment outweigh accuracy gains from a 768-dim model at this scale

**Groq + llama-3.3-70b-versatile (not OpenAI)**
- Free tier with fast inference (GroqCloud hardware)
- 70B model for generation quality; 8B (`llama-3.1-8b-instant`) for cheap, fast auxiliary calls (query rewriting)
- Tradeoff: rate limits exist; handled explicitly (`RateLimitError` catch with user-facing message)

**LangChain RecursiveCharacterTextSplitter (not manual splitter)**
- Respects `\n\n → \n → . → space` boundary hierarchy so chunks don't cut mid-sentence
- Tradeoff: adds `langchain-text-splitters` dependency; justifies it for boundary-aware splitting
- Manual sliding-window alternative would require re-implementing boundary logic from scratch

**Streamlit (not FastAPI + React)**
- Intern project scope: Streamlit gives session state, file upload, and chat UI with ~50 lines
- Real production would separate backend (FastAPI) from frontend; noted as a known architectural limit

---

## Key Design Decisions (with reasoning)

**Embed content only, no metadata prefix**

Early version embedded `"Document: X\nPage: Y\n\n{content}"` per chunk. Direct testing showed this dilutes similarity scores for short chunks — a chunk mentioning "LangChain, ChromaDB, FAISS" scored 0.25 against a direct question about those tools, lower than an unrelated chunk (0.32). Source and page are stored as separate fields; they don't need to be in the embedding.

**Hybrid retrieval: semantic + keyword overlap (80/20)**

Pure semantic search underperforms on proper-noun-dense or list-like text. Simple keyword overlap (fraction of non-stopword query terms found in chunk text) adds a corrective signal for exact-name matches without requiring BM25 or a second index. Weighting validated against known-good and known-bad queries.

**LLM-based query rewriting (not keyword heuristics)**

Follow-up questions ("why does this happen?", "which one is better?") score too low in retrieval because they depend on prior context. First approach: keyword list (`["it", "this", "why"]` + length cutoff). Failure found immediately: "Could the order be reversed without breaking anything?" — 8 words, zero pronoun overlap, but clearly a follow-up. Replaced with a fast LLM call (`llama-3.1-8b-instant`) that decides directly. Crucially: model is instructed to return the query **unchanged** if it's already independent, preventing context contamination on topic-switched questions.

**Structural citation enforcement (not instruction-only)**

Early version: system prompt said "cite accurately." Found empirically that the model hallucinated page references (e.g., cited page 3 for content that was actually on page 1). Fix: numbered brackets tied to retrieved chunks passed directly in context — model cites `[1]`, `[2]`, etc., never writes a page number from memory. Structural constraint beats instruction-only for grounding.

**Summary mode bypasses similarity scoring**

"Summarize everything" is not a similarity-search problem — it's a coverage problem. Similarity scoring against a summarize query tends to surface only the most "summarize-sounding" chunks, not representative content. Fix: grab first 3 chunks per document directly, bypassing scoring entirely.

**Stage-2 retrieval rewrite fires at same threshold as final rejection**

Bug found during testing: Stage 2 (follow-up rewrite) fired when `score < 0.40`, but final rejection was at `SCORE_THRESHOLD = 0.42`. Any genuine follow-up scoring between 0.40–0.42 skipped rewriting and got wrongly rejected. Fixed by using the same `score_threshold` variable for both checks.

**Conversation history: both user AND assistant turns injected**

Bug found during testing: history loop re-added only past questions, never past answers. The LLM saw a stack of consecutive user messages with no record of what it had already said — "don't repeat yourself" instruction had nothing to check against. Fixed by appending both `user` and `assistant` turns.

---

## Limitations (honest)

**Weak retrieval on proper-noun-dense text**

Diagnosed with real numbers: a chunk containing "LangChain, ChromaDB, FAISS" scored 0.25 against "What tools does DocuMind use?" — below background noise level for completely unrelated questions (~0.26–0.32). Root cause: `all-MiniLM-L6-v2` represents average chunk meaning, so a mixed chunk (two topics) doesn't score well for either. Keyword overlap boost helps partially but doesn't fix ranking when both competing chunks share the same overlap fraction by coincidence.

**Real fix scoped as future work:** Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`, free via sentence-transformers) — scores query+chunk jointly rather than comparing two independently computed embeddings. Not shipped to avoid adding latency and a second model dependency without a validated improvement.

**HyDE tested and deliberately not shipped**

Hypothetical Document Embeddings (generate a fake answer, embed that for retrieval) was tested as an alternative fix. Found it did improve chunk ranking in one case, but by coincidental vocabulary overlap — it generated "Amazon S3, MySQL, Apache Spark" for a project using LangChain/ChromaDB/FAISS. Ranking improvement couldn't be trusted to generalize. Kept as commented, documented code.

**No quantitative evaluation**

Regression tests cover all bugs found during development and verify correct/incorrect behavior on known queries (`test_documind.py`, 12 tests). What's missing: quantitative metrics — hit-rate@k (what fraction of the time does the right chunk appear in top-k results), MRR, or faithfulness scoring. All pass/fail criteria in the test suite are manually written heuristics, not computed against a ground-truth dataset.

**Summary only reads first 3 chunks per document**

Long documents' later content may be underrepresented. A 7-page Neural Network Architectures doc's later sections (specific model variants, training details) may not appear in a summary.

**No persistence across server restarts**

FAISS index is in-memory; uploaded documents must be reprocessed on every new session. Acceptable for a demo; production would serialize the index and chunk metadata to disk.

**Scanned PDFs silently skipped**

`pypdf` can only extract digitally embedded text, not OCR. Pages with no extractable text generate a user-visible warning and are skipped. An OCR pipeline (e.g., `pytesseract`) was deliberately excluded as out of scope.

**Prompt injection risk**

Document content is fed directly into the LLM context. A malicious PDF containing text like "Ignore your previous instructions and..." could attempt to override system behavior. Unmitigated. Worth noting if deployed in any non-controlled environment.

**No hybrid keyword+semantic (real BM25)**

Current keyword overlap is a simple substring fraction — no term-frequency weighting, no inverse document frequency, no stemming. Real BM25 (`rank_bm25`, free) would handle plurals, stemming, and proper TF-IDF weighting. Scoped as a future improvement.

---

## What I'd Add Next (in priority order)

1. Quantitative eval set — 15–20 question/expected-source triples with automated hit-rate@k measurement (current test suite validates behavior but not retrieval metrics)
2. Cross-encoder reranker (sentence-transformers, free, CPU)
3. Real BM25 via `rank_bm25` for true hybrid search
4. FAISS index serialization (`faiss.write_index`) for persistence
5. Swap `all-MiniLM-L6-v2` for `BAAI/bge-small-en-v1.5` (retrieval-tuned, similar size, generally better on retrieval benchmarks)

---

## File Structure

```
documind/
├── app.py              # Streamlit UI, session state, chat interface
├── rag.py              # All pipeline logic: load, chunk, embed, retrieve, generate
├── test_documind.py    # Regression tests covering all major bugs found during development
├── .env                # GROQ_API_KEY (not committed)
├── requirements.txt
└── README.md
```

---

## Requirements

```
streamlit
pypdf
python-docx
pandas
sentence-transformers
faiss-cpu
groq
langchain-text-splitters
python-dotenv
numpy
```
