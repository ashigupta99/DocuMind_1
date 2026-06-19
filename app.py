import streamlit as st
from rag import load_documents, chunk_documents, build_index, generate_answer

st.set_page_config(page_title="DocuMind", page_icon="📚", layout="wide")

# Initialize session state once on true page load
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.qa_history = []

st.title("📚 DocuMind")
st.caption("Your documents. Your answers. No generic internet noise.")

# --- Sidebar: upload + process ---
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or CSV files",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True
    )

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            # Warn if reprocessing over existing session
            if st.session_state.index is not None:
                st.warning("⚠️ Reprocessing will clear your current chat history.")

            with st.spinner("Reading and indexing your documents..."):
                docs, warnings = load_documents(uploaded_files)

                if warnings:
                    with st.expander(f"⚠️ {len(warnings)} warning(s)"):
                        for w in warnings:
                            st.write(w)

                if not docs:
                    st.error("No usable text could be extracted from these files.")
                else:
                    chunks = chunk_documents(docs)
                    index, stored_chunks = build_index(chunks)
                    st.session_state.index = index
                    st.session_state.chunks = stored_chunks
                    st.session_state.qa_history = []
                    st.success(f"Indexed {len(chunks)} chunks from {len(docs)} pages/sections.")

    if st.session_state.index is not None:
        st.info(f"✅ {st.session_state.index.ntotal} chunks ready to search.")

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.qa_history = []
            st.rerun()

        # Show indexed documents
        if st.session_state.chunks is not None:
            sources = sorted(set(c["source"] for c in st.session_state.chunks))
            with st.expander("📂 Indexed documents"):
                for s in sources:
                    st.write(f"• {s}")

# --- Main chat interface ---
if st.session_state.index is None:
    st.info("Upload and process documents in the sidebar to get started.")
else:
    # Render past conversation on every rerun
    for exchange in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(exchange["question"])
        with st.chat_message("assistant"):
            st.write(exchange["answer"])
            if exchange["sources"]:
                with st.expander("📄 Sources"):
                    for i, s in enumerate(exchange["sources"], start=1):
                        page_info = f", page {s['page']}" if s['page'] else ""
                        if s["score"] is not None:
                            st.write(
                                f"[{i}] {s['source']}{page_info} "
                                f"(relevance: {s['score']:.2f})"
                            )
                        else:
                            st.write(
                                f"[{i}] {s['source']}{page_info}"
                            )

    query = st.chat_input("Ask a question about your documents...")
    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = generate_answer(
                    query,
                    st.session_state.index,
                    st.session_state.chunks,
                    history=st.session_state.qa_history
                )
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("📄 Sources"):
                    for i, s in enumerate(result["sources"], start=1):
                        page_info = f", page {s['page']}" if s['page'] else ""
                        if s["score"] is not None:
                            st.write(
                                f"[{i}] {s['source']}{page_info} "
                                f"(relevance: {s['score']:.2f})"
                            )
                        else:
                            st.write(
                                f"[{i}] {s['source']}{page_info}"
                            )

        st.session_state.qa_history.append({
            "question": query,
            "answer": result["answer"],
            "sources": result["sources"]
        })