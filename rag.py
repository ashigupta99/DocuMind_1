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
from rank_bm25 import BM25Okapi

load_dotenv()


# ---------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------

def load_pdf(file):
    reader = pypdf.PdfReader(file)
    documents, warnings = [], []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text and text.strip():
            documents.append({"content": text, "source": file.name, "page": page_num})
        else:
            warnings.append(
                f"Page {page_num} of {file.name} appears to be scanned "
                f"or empty — skipped."
            )
    return documents, warnings


def load_docx(file):
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
    df = pd.read_csv(file)
    if df.empty:
        return [], [f"{file.name} has no rows — skipped."]
    documents = []
    for idx, row in df.iterrows():
        row_text = ", ".join(f"{col}: {val}" for col, val in row.items())
        documents.append({"content": row_text, "source": file.name, "page": idx + 1})
    return documents, []


def load_documents(uploaded_files):
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

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

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
            if len(piece.strip()) < 80:
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
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def build_index(chunks):
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
# BM25 index
# ---------------------------------------------------------------

def build_bm25_index(chunks):
    corpus = [
        re.findall(r"\w+", chunk["content"].lower())
        for chunk in chunks
    ]
    return BM25Okapi(corpus)


# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------

SCORE_THRESHOLD = 0.30

STOPWORDS = {
    "what", "is", "the", "a", "an", "does", "do", "are",
    "how", "why", "which", "this", "that", "of", "in", "on"
}


def retrieve(query, index, chunks, top_k=20):
    """Semantic retrieval using FAISS + BGE embeddings."""
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
        results.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(score),
        })

    results.sort(key=lambda r: r["score"], reverse=True)

    if results:
        best_score = results[0]["score"]
        score_floor = max(0.30, 0.65 * best_score)
        results = [r for r in results if r["score"] >= score_floor]

    return results


def retrieve_bm25(query, bm25_index, chunks, top_k=20):
    """Lexical retrieval using BM25."""
    query_tokens = [
        w for w in re.findall(r"\w+", query.lower())
        if w not in STOPWORDS
    ]

    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append({
            "content": chunk["content"],
            "source": chunk["source"],
            "page": chunk["page"],
            "score": float(scores[idx])
        })

    return results


def reciprocal_rank_fusion(dense_results, bm25_results, k=20):
    """Merge FAISS and BM25 rankings. Chunks appearing near the top
    in both lists get a higher combined score."""
    fused_scores = {}
    chunk_lookup = {}

    for rank, r in enumerate(dense_results):
        key = (r["source"], r["page"], r["content"])
        fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = r

    for rank, r in enumerate(bm25_results):
        key = (r["source"], r["page"], r["content"])
        fused_scores[key] = fused_scores.get(key, 0) + 1 / (k + rank + 1)
        chunk_lookup[key] = r

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_lookup[key] for key, _ in ranked]


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
# Generation
# ---------------------------------------------------------------

SYSTEM_PROMPT = """
You are a document-grounded assistant.

Answer using ONLY the provided context.

Guidelines:

1. If the answer is directly stated in the context, answer confidently.

2. If the answer requires combining multiple pieces of retrieved information, do so.

3. If the context provides partial information, answer with appropriate uncertainty rather than refusing.

4. Refuse ONLY when the provided context contains no relevant information.

5. Never invent facts, relationships, or assumptions that are not supported by the context.

6. Do not use outside knowledge.

7. When comparing concepts, explain the differences clearly using the retrieved information.

8. When answering "why" questions, provide the reason stated in the context. If the context only gives an observation and not an explanation, explicitly say so.

9. If multiple uploaded documents are relevant, synthesize information across them, but only if the connection itself is supported by the context.

10. Prefer giving a limited answer over saying "I couldn't find relevant information" when some evidence exists.

11. If the context does not contain enough information to answer, say:

"I couldn't find relevant information in the uploaded documents to answer this question."

Be concise and avoid repeating the context verbatim.
"""


def build_prompt(query, retrieved_chunks, is_followup=False):
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_info = f", page {chunk['page']}" if chunk['page'] else ""
        context_blocks.append(
            f"[{i}] (Source: {chunk['source']}{page_info})\n{chunk['content']}"
        )
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
        - Answer only from the context.
        - If some information is available, provide the best answer possible and mention any limitations.
        - Do not use outside knowledge.
        - Cite sources whenever possible.
        """


def rewrite_query_with_history(query, history):
    """Resolves pronoun-dependent follow-ups into standalone questions.
    If the new message is already independent, it is returned unchanged."""
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


def generate_answer(
        query,
        index,
        bm25_index,
        chunks,
        history=None,
        top_k=20,
        score_threshold=SCORE_THRESHOLD):
    """
    Full RAG pipeline: retrieve → check confidence → generate.

    Two-stage retrieval:
      Stage 1 — retrieve with query alone.
      Stage 2 — if confidence is low and history exists, retry with
                 expanded context (previous Q+A + current query).

    Summary and short queries bypass the score threshold.
    """

    SUMMARY_WORDS = {
        "summary", "summarize", "summarise", "overview", "brief",
        "outline", "list", "topics", "covered", "everything", "all",
        "code", "formula", "equation", "implement"
    }

    query_words = set(query.lower().split())
    is_summary = bool(query_words.intersection(SUMMARY_WORDS))
    is_short_query = len(query.split()) <= 4

    effective_top_k = 40 if is_summary else top_k
    expanded_query = None

    # Stage 1: retrieve with query alone
    dense_results = retrieve(query, index, chunks, top_k=effective_top_k)
    bm25_results = retrieve_bm25(query, bm25_index, chunks, top_k=effective_top_k)
    retrieved = reciprocal_rank_fusion(dense_results, bm25_results)[:10]

    # Stage 2: low confidence + history → retry with expanded query
    if history and (not retrieved or retrieved[0]["score"] < 0.35):
        expanded_query = (
            history[-1]["question"]
            + " "
            + history[-1]["answer"][:200]
            + " "
            + query
        )
        dense_results = retrieve(expanded_query, index, chunks, top_k=effective_top_k)
        bm25_results = retrieve_bm25(expanded_query, bm25_index, chunks, top_k=effective_top_k)
        retrieved = reciprocal_rank_fusion(dense_results, bm25_results)[:10]

    # Summary and short queries bypass score threshold
    if not (is_summary or is_short_query):
        if not retrieved or retrieved[0]["score"] < score_threshold:
            return {
                "answer": "I couldn't find relevant information in your "
                           "uploaded documents to answer this question.",
                "sources": []
            }

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