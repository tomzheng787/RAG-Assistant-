"""Streamlit chat app for the RAG pipeline: add PDFs, index them, and ask questions with cited answers.

    streamlit run app.py

API keys come from the environment or a .env file (see .env.example).
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.chain import (
    RAGChain,
    RAGConfig,
    build_chain,
    build_embedder,
    build_llm,
    index_documents,
    list_pdfs,
    open_vector_store,
)
from rag.embedding import PROVIDERS
from rag.llm import CHAT_PROVIDERS
from rag.retriever import RetrievalResult
from rag.vector_store import VectorStore

load_dotenv()
DEFAULTS = RAGConfig()
SETUP_ERRORS = (ValueError, ImportError, FileNotFoundError)  # missing keys, model mismatch, missing packages

st.set_page_config(page_title="PDF RAG Assistant", page_icon=":page_facing_up:", layout="wide")


# Cached across reruns and sessions. Arguments starting with "_" are not hashed, so each cache is keyed by the
# settings tuple passed next to the config.
@st.cache_resource(show_spinner="Opening the vector store...")
def get_store(_config: RAGConfig, key: tuple) -> VectorStore:
    return open_vector_store(_config, build_embedder(_config))


@st.cache_resource(show_spinner=False)
def get_llm(_config: RAGConfig, key: tuple):
    return build_llm(_config)


@st.cache_resource(show_spinner="Preparing retrieval...")
def get_chain(_config: RAGConfig, _store: VectorStore, _llm, key: tuple) -> RAGChain:
    return build_chain(_config, _store, _llm)


def store_key(config: RAGConfig) -> tuple:
    return (
        config.store,
        config.embedding_provider,
        config.embedding_model,
        config.index_dir,
        config.milvus_uri,
        config.milvus_collection,
        config.milvus_token,
    )


def llm_key(config: RAGConfig) -> tuple:
    return config.llm_provider, config.llm_model


def chain_key(config: RAGConfig) -> tuple:
    return (
        store_key(config),
        llm_key(config),
        config.k,
        config.score_threshold,
        config.use_hybrid,
        config.dense_weight,
        config.use_hyde,
        config.use_decomposition,
    )


def settings() -> RAGConfig:
    """Render the settings in the sidebar and return them as a config."""
    config = RAGConfig()
    with st.sidebar.expander("Models"):
        config.embedding_provider = st.selectbox(
            "Embedding provider", PROVIDERS, index=PROVIDERS.index(DEFAULTS.embedding_provider)
        )
        config.embedding_model = st.text_input("Embedding model", placeholder="provider default") or None
        config.llm_provider = st.selectbox(
            "Chat model provider", CHAT_PROVIDERS, index=CHAT_PROVIDERS.index(DEFAULTS.llm_provider)
        )
        config.llm_model = st.text_input("Chat model", placeholder="provider default") or None
        st.caption("Changing the embedding model requires rebuilding the index.")

    with st.sidebar.expander("Vector store"):
        config.store = st.radio("Backend", ["faiss", "milvus"], format_func=str.upper, horizontal=True)
        if config.store == "milvus":
            config.milvus_uri = st.text_input(
                "Milvus URI",
                value=DEFAULTS.milvus_uri,
                help="A file path runs Milvus Lite locally; http://localhost:19530 connects to a Milvus server.",
            )
            config.milvus_collection = st.text_input("Collection", value=DEFAULTS.milvus_collection)
            config.milvus_token = os.environ.get("MILVUS_TOKEN")

    with st.sidebar.expander("Retrieval", expanded=True):
        config.k = st.slider("Passages per answer", 1, 10, DEFAULTS.k)
        config.score_threshold = st.slider(
            "Similarity threshold",
            0.0,
            1.0,
            DEFAULTS.score_threshold,
            0.05,
            help="Passages less similar to the question than this are dropped. Typical scores depend on the "
            "embedding model: lower it if answers miss facts that are in the documents.",
        )
        config.use_hybrid = st.toggle("Hybrid search (BM25 + embeddings)", value=DEFAULTS.use_hybrid)
        config.dense_weight = st.slider(
            "Embedding weight",
            0.0,
            1.0,
            DEFAULTS.dense_weight,
            0.1,
            disabled=not config.use_hybrid,
            help="Higher favors meaning, lower favors exact keywords such as names, codes and numbers.",
        )
        config.use_hyde = st.toggle("HyDE (hypothetical answer search)", help="One extra LLM call per query")
        config.use_decomposition = st.toggle("Query decomposition", help="Splits complex questions; extra LLM calls")
    return config


def documents_panel(config: RAGConfig) -> None:
    """Upload PDFs into the documents folder and index them."""
    st.subheader("Documents")
    folder = Path(config.documents_dir)
    for upload in st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True) or []:
        target = folder / Path(upload.name).name  # keep only the file name, never a path from the browser
        if not target.exists() or target.stat().st_size != upload.size:
            folder.mkdir(parents=True, exist_ok=True)
            target.write_bytes(upload.getbuffer())

    try:
        store = get_store(config, store_key(config))
    except SETUP_ERRORS as e:
        st.error(str(e))
        store = None
    status = st.empty()  # filled after the buttons, so the counts include what they just indexed
    left, right = st.columns(2)
    if left.button("Index new files", type="primary", disabled=store is None, width="stretch"):
        run_indexing(config, store, only_new=True)
    if right.button("Rebuild index", width="stretch", help="Re-index every file from scratch"):
        rebuild_index(config)
    chunks = f", {len(store)} chunks indexed" if store is not None else ""
    status.caption(f"{len(list_pdfs(folder))} PDF(s) in `{folder}/`{chunks}")


def run_indexing(config: RAGConfig, store: VectorStore, only_new: bool) -> None:
    bar = st.progress(0.0, text="Starting...")
    try:
        done = index_documents(
            config,
            store,
            only_new=only_new,
            progress=lambda i, total, path: bar.progress(i / total, text=f"{path.name} ({i + 1}/{total})"),
        )
    except Exception as e:  # surface API errors (quota, network, ...) in the UI instead of crashing the app
        bar.empty()
        st.error(f"Indexing failed: {e}")
        return
    bar.empty()
    st.success(f"Indexed {len(done)} file(s)." if done else "No new files to index.")


def rebuild_index(config: RAGConfig) -> None:
    # Release the cached store first: Milvus Lite allows only one open connection to its file
    try:
        close = getattr(get_store(config, store_key(config)), "close", None)
        if close:
            close()
    except SETUP_ERRORS:
        pass  # e.g. the old index was built with another model, which is exactly what a rebuild fixes
    get_store.clear()
    get_chain.clear()
    try:
        store = open_vector_store(config, build_embedder(config), rebuild=True)
    except SETUP_ERRORS as e:
        st.error(str(e))
        return
    run_indexing(config, store, only_new=False)
    if hasattr(store, "close"):
        store.close()  # get_store opens it again on the next run
    st.rerun()


def show_details(details: dict) -> None:
    """Sources, plus sub-questions and HyDE passages when those techniques ran."""
    if details["sub_questions"]:
        with st.expander("Sub-questions"):
            st.markdown("\n".join(f"- {q}" for q in details["sub_questions"]))
    if details["hyde"]:
        with st.expander("HyDE passages"):
            for query, passages in details["hyde"].items():
                st.markdown(f"**{query}**")
                for passage in passages:
                    st.text(passage)
    sources = details["sources"]
    with st.expander(f"Sources ({len(sources)})" if sources else "Sources (none passed the threshold)"):
        for i, source in enumerate(sources, 1):
            score = f"{details['score_name']} {source['score']:.3f}"
            st.markdown(f"**[{i}] {source['file']}**, page {source['page']}, {score}")
            st.text(source["text"])


def describe(result: RetrievalResult, config: RAGConfig) -> dict:
    """The parts of a retrieval result worth keeping in the chat history."""
    return {
        "sources": [
            {"file": doc.metadata.get("file_name"), "page": doc.metadata.get("page"), "score": score, "text": doc.text}
            for doc, score in result.documents
        ],
        "sub_questions": result.sub_questions,
        "hyde": result.hypothetical_documents,
        "score_name": "fusion score" if config.use_hybrid else "similarity",
    }


def chat(config: RAGConfig) -> None:
    st.title("PDF RAG Assistant")
    switches = [("hybrid", config.use_hybrid), ("HyDE", config.use_hyde), ("decomposition", config.use_decomposition)]
    techniques = [name for name, on in switches if on]
    st.caption(
        f"Embeddings: {config.embedding_provider} | Chat: {config.llm_provider} | Store: {config.store.upper()} | "
        f"Retrieval: {', '.join(techniques) or 'embeddings only'} | Threshold: {config.score_threshold:.2f}"
    )

    messages = st.session_state.setdefault("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "details" in message:
                show_details(message["details"])

    question = st.chat_input("Ask a question about your documents")
    if not question:
        if not messages:
            st.info("Add PDFs in the sidebar, click **Index new files**, then ask a question below.")
        return

    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            store = get_store(config, store_key(config))
            if len(store) == 0:
                st.warning("No documents are indexed yet: add PDFs in the sidebar and click **Index new files**.")
                return
            chain = get_chain(config, store, get_llm(config, llm_key(config)), chain_key(config))
            with st.spinner("Searching the documents..."):
                result = chain.retrieve(question)
            answer = st.write_stream(chain.stream_answer(question, result))
        except Exception as e:  # missing API keys, network or quota errors, ...
            st.error(f"{type(e).__name__}: {e}")
            return
        details = describe(result, config)
        show_details(details)
    messages.append({"role": "assistant", "content": answer, "details": details})


def main() -> None:
    documents_area = st.sidebar.container()  # placed above the settings, filled once the settings are known
    st.sidebar.divider()
    config = settings()
    with documents_area:
        documents_panel(config)
    if st.sidebar.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    chat(config)


main()
