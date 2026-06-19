import pypdf
import docx
import pandas as pd
import re
import os
from dotenv import load_dotenv
from groq import Groq, RateLimitError
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ---------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------

def load_pdf(file):
    """Extract text from a PDF, page by page.
    Pages with no extractable text (e.g. scanned images) are skipped
    with a warning rather than silently producing empty chunks.
    """
    reader = pypdf.PdfReader(file)
    documents, warnings = [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append({"content": text, "source": file.name, "page": page_num})
        else:
            warnings.append(
                f"Page {page_num} of {file.name} appears to be scanned "
                f"or empty — skipped (no extractable text)."
            )
    return documents, warnings


def load_docx(file):
    """DOCX has no real concept of 'pages', so the whole document is
    treated as one unit."""
    doc = docx.Document(file)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if not full_text.strip():
        return [], [f"{file.name} has no extractable text — skipped."]
    return [{"content": full_text, "source": file.name, "page": None}], []


def load_txt(file):
    content = file.read().decode("utf-8", errors="ignore")
    if not content.strip():
        return [], [f"{file.name} is empty — skipped."]
    return [{"content": content, "source": file.name, "page": None}], []


def load_csv(file):
    """Each row becomes its own document — a row is usually one
    discrete record, so this preserves precise retrieval per row."""
    df = pd.read_csv(file)
    if df.empty:
        return [], [f"{file.name} has no rows — skipped."]
    documents = []
    for idx, row in df.iterrows():
        row_text = ", ".join(f"{col}: {val}" for col, val in row.items())
        documents.append({"content": row_text, "source": file.name, "page": idx + 1})
    return documents, []


def load_documents(uploaded_files):
    """Master dispatcher. Each file is wrapped in try/except so one
    corrupted upload doesn't crash processing for the whole batch."""
    all_documents, all_warnings = [], []
    for file in uploaded_files:
        ext = file.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                docs, warns = load_pdf(file)
            elif ext == "docx":
                docs, warns = load_docx(file)
            elif ext == "txt":
                docs, warns = load_txt(file)
            elif ext == "csv":
                docs, warns = load_csv(file)
            else:
                docs, warns = [], [f"{file.name}: unsupported file type — skipped."]
        except Exception as e:
            docs, warns = [], [f"{file.name}: failed to process ({e}) — skipped."]
        all_documents.extend(docs)
        all_warnings.extend(warns)
    return all_documents, all_warnings


# ---------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def chunk_documents(documents):
    all_chunks = []
    for doc in documents:
        pieces = _splitter.split_text(doc["content"])
        for idx, piece in enumerate(pieces):
            # Skip chunks under 100 characters — likely TOC or headers
            if len(piece.strip()) < 100:
                continue
            all_chunks.append({
                "content": piece,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": idx
            })
    return all_chunks


# ---------------------------------------------------------------
# Embeddings + FAISS index
# ---------------------------------------------------------------

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index(chunks):
    """Embeds chunk content only (no metadata prefix — embedding
    'Document: X\\nPage: Y' alongside content diluted similarity
    scores for short chunks, confirmed via direct testing)."""
    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, chunks


# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------

SCORE_THRESHOLD = 0.38


def keyword_overlap_score(query, text):
    """Fraction of meaningful query words that appear literally in the
    chunk. Corrects cases where semantic similarity alone undervalues
    short, proper-noun-heavy chunks (e.g. a list of tool names)."""
    stopwords = {"what", "is", "the", "a", "an", "does", "do", "are",
                 "how", "why", "which", "this", "that", "of", "in", "on"}
    query_words = set(re.findall(r"\w+", query.lower())) - stopwords
    if not query_words:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for w in query_words if w in text_lower)
    return matches / len(query_words)


def retrieve(query, index, chunks, top_k=15):
    """Embeds the query, searches FAISS, and blends semantic similarity
    with a keyword-overlap boost (80/20 weighting)."""
    model = get_embedding_model()
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")
    faiss.normalize_L2(query_vector)

    scores, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        semantic_score = float(score)
        keyword_score = keyword_overlap_score(query, chunk["content"])
        combined_score = (0.8 * semantic_score) + (0.2 * keyword_score)

        results.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": combined_score,
            "semantic_score": semantic_score,
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    if results:
        best_score = results[0]["score"]
        results = [r for r in results if r["score"] >= max(0.35, 0.7 * best_score)]

    return results


# ---------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------

_groq_client = None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


# ---------------------------------------------------------------
# HyDE (explored, NOT used in the live pipeline)
#
# Tested as a fix for cases where query phrasing didn't match document
# vocabulary (e.g. "what tools does X use" vs a chunk listing tool
# names). It can improve ranking, but it does so by generating a
# hypothetical answer that may contain confidently fabricated facts
# (verified directly: it once invented "Amazon S3, MySQL, Apache
# Spark" for a project that actually uses LangChain/ChromaDB/FAISS).
# Since the ranking improvement depended on coincidental vocabulary
# overlap rather than factual relevance, it was judged too unreliable
# to ship. Kept here as documented, tested, deliberately-unused code.
# ---------------------------------------------------------------

# def hyde_expand_query(query):
#     client = get_groq_client()
#     prompt = (
#         f"Write a short, plausible-sounding paragraph (2-3 sentences) "
#         f"that could be the answer to this question, even if you're "
#         f"just guessing: {query}"
#     )
#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.3,
#         max_tokens=100
#     )
#     return response.choices[0].message.content.strip()


# def retrieve_with_hyde(query, index, chunks, top_k=15):
#     direct_results = retrieve(query, index, chunks, top_k=top_k)
#     hyde_query = hyde_expand_query(query)
#     hyde_results = retrieve(hyde_query, index, chunks, top_k=top_k)

#     merged = {}
#     for r in direct_results + hyde_results:
#         key = (r["source"], r["page"], r["content"][:50])
#         if key not in merged or r["score"] > merged[key]["score"]:
#             merged[key] = r

#     return sorted(merged.values(), key=lambda r: r["score"], reverse=True)


# ---------------------------------------------------------------
# Generation
# ---------------------------------------------------------------

SYSTEM_PROMPT = """
You are DocuMind, an academic document assistant.

Answer strictly using the provided context.

CRITICAL:
- Treat the provided context as the only source of truth.
- Never answer using general knowledge.
- If a fact is not explicitly supported by the provided context, do not mention it.
- Every factual statement in the answer must be grounded in the retrieved context.

Rules:
- If part of an answer is unsupported, omit that part instead of filling gaps with general knowledge.
- If the answer is absent from the context, say:
  "I couldn't find relevant information in your uploaded documents to answer this question."
- Use citations exactly as provided ([1], [2], etc). Never fabricate citations.
- For "why" questions, focus on motivation/purpose rather than implementation detail.
- Expand abbreviations if context makes their meaning clear.

Reasoning:
- Synthesize information across chunks and documents into one coherent answer.
- Avoid repeating the same information.

Summary and overview questions:
- Summarize the collection as a whole; identify major topics and each document's role.

Comparisons:
- Explain similarities and differences; state which approach is better when supported, and why.

Architectures, pipelines, and processes:
- Present stages in order; explain relationships between components.
- Prefer complete pipelines over isolated pieces (e.g. "CNN → RNN/LSTM → CTC", not "uses convolutional backbones").

Follow-up questions:
- Use prior conversation to resolve references like "it", "why", "which one".
- Do not repeat previous answers; provide only the new information requested.

Writing style:
- Concise but complete. Bullets when helpful. No disconnected facts.
"""


def build_prompt(query, retrieved_chunks, is_followup=False):
    """Each chunk is numbered and tagged with its source. The LLM is
    instructed to cite ONLY these bracket numbers — never to write out
    page numbers from memory — since unconstrained citation was found
    to hallucinate page references that were never actually retrieved.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_info = f", page {chunk['page']}" if chunk['page'] else ""
        context_blocks.append(f"[{i}] (Source: {chunk['source']}{page_info})\n{chunk['content']}")
    context_text = "\n\n".join(context_blocks)

    followup_note = (
        "\n\nNote: this is a follow-up question. Do NOT restate your previous "
        "answer's content. Reference it only briefly if needed, and focus "
        "entirely on the new information being asked for."
        if is_followup else ""
    )

    return f"""
        Context:
        {context_text}

        Question:
        {query}{followup_note}

        Instructions:
        - Answer ONLY using the context above.
        - If the answer is not fully supported by the context, respond exactly:
          "I couldn't find relevant information in your uploaded documents to answer this question."
        """


def rewrite_query_with_history(query, history):
    """Resolves pronoun-dependent follow-ups ('why does this happen')
    into standalone questions using the previous turn. Critically,
    if the new message is already independent — especially a topic
    switch — it's returned UNCHANGED. This is what prevents an
    unrelated new question from inheriting leftover context from a
    previous topic (found and fixed: a "capital of Malaysia" question
    was once contaminated by a prior unrelated OCR answer).
    """
    last_q = history[-1]["question"]
    rewrite_prompt = f"""
Given the previous question and a new user message, rewrite the new message into a fully standalone question ONLY if it depends on the previous question.

If the new message is already independent, return it unchanged.

Previous question:
{last_q}

New message:
{query}

Return ONLY the rewritten (or unchanged) question.
"""
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0,
        max_tokens=50
    )
    return response.choices[0].message.content.strip()


def generate_answer(query, index, chunks, history=None, top_k=10,
                     score_threshold=SCORE_THRESHOLD):
    """Full RAG generation step: retrieve, check confidence, build
    prompt with history, call Groq, return answer + sources.

    Uses adaptive two-stage retrieval:
    Stage 1 — retrieve with query alone.
    Stage 2 — if confidence is low and history exists, retry with
    expanded query (previous question + answer + current query).
    The expanded query is also passed to the LLM so it has context
    for ambiguous follow-ups like 'Why?' or 'Explain that.'
    Summary queries and short queries bypass the score threshold.
    """

    SUMMARY_WORDS = {
        "summary", "summarize", "summarise",
        "overview", "brief", "outline",
        "list", "topics", "covered",
        "everything", "all",
        "code", "formula", "equation", "implement"
    }

    expanded_query = None
    query_words = set(query.lower().split())
    is_summary = bool(query_words.intersection(SUMMARY_WORDS))

    # Short queries score low even when answer exists — bypass threshold
    is_short_query = len(query.split()) <= 4

    # Use more chunks for summary-type queries
    effective_top_k = 20 if is_summary else top_k

    # Stage 1: retrieve with query alone
    retrieved = retrieve(query, index, chunks, top_k=effective_top_k)

    # Stage 2: low confidence + history → retry with expanded query
    if history and (not retrieved or retrieved[0]["score"] < 0.40):
        expanded_query = (
            history[-1]["question"]
            + " "
            + history[-1]["answer"][:200]
            + " "
            + query
        )
        retrieved = retrieve(expanded_query, index, chunks, top_k=effective_top_k)

    # Summary and short queries bypass score threshold
    if not (is_summary or is_short_query):
        if not retrieved or retrieved[0]["score"] < score_threshold:
            return {
                "answer": "I couldn't find relevant information in your "
                           "uploaded documents to answer this question.",
                "sources": []
            }

    # Use expanded query for LLM prompt if stage 2 was triggered,
    # so the LLM sees a meaningful question instead of just "Why?"
    prompt_query = expanded_query if expanded_query else query
    user_prompt = build_prompt(prompt_query, retrieved)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        for exchange in history[-4:]:
            messages.append({"role": "user", "content": exchange["question"]})
            messages.append({"role": "assistant", "content": exchange["answer"]})

    messages.append({"role": "user", "content": user_prompt})

    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=1000 if is_summary else 600
        )

    except RateLimitError:
        return {
            "answer": "Groq rate limit reached. Please try again later.",
            "sources": []
        }

    except Exception as e:
        return {
            "answer": f"Error contacting the LLM: {e}",
            "sources": []
        }

    answer = response.choices[0].message.content
    sources = [
        {"source": r["source"], "page": r["page"], "score": r["score"]}
        for r in retrieved
    ]

    return {"answer": answer, "sources": sources}
