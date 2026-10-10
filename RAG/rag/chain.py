"""RAG chain: retrieve chunks for a question, then have an LLM answer from them with numbered citations.

RAGConfig gathers every setting of the pipeline, and the build / index helpers below turn it into working objects;
both the command line (python -m rag) and the Streamlit app (app.py) are thin layers over this module.
"""

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from rag.chunker import RecursiveTextSplitter
from rag.document import Document
from rag.embedding import Embedder, create_embedder
from rag.llm import ChatModel, create_chat_model
from rag.pdf_loader import load_pdf
from rag.retriever import RetrievalResult, Retriever
from rag.vector_store import INDEX_FILE, FaissVectorStore, VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You answer questions about the user's documents, using only the passages you are given."

ANSWER_PROMPT = """Answer the question using only the numbered passages below. Each passage is labeled with its \
source file and page.

Rules:
- After each statement, cite the passages that support it by number in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, say that the documents do not cover it. Do not answer from your own \
knowledge and do not invent citations.
- Answer in the same language as the question.

Passages:
{context}

Question: {question}"""

# Stands in for the passages when nothing passed the similarity threshold; the LLM then says, in the question's
# language, that the documents don't cover it
NO_CONTEXT = "(No passage in the documents is relevant enough to this question.)"


@dataclass
class RAGConfig:
    """Every setting of the pipeline. The defaults suit a first run; relative paths are relative to the
    working directory."""

    documents_dir: str = "documents"
    # Models (None = the provider's default model)
    embedding_provider: str = "openai"
    embedding_model: str | None = None
    llm_provider: str = "openai"
    llm_model: str | None = None
    # Vector store: "faiss" (a local index saved in index_dir) or "milvus"
    store: str = "faiss"
    index_dir: str = "index"
    milvus_uri: str = "index/milvus.db"  # a file path runs Milvus Lite; http://localhost:19530 is a Milvus server
    milvus_token: str | None = None  # for Zilliz Cloud or a server with authentication
    milvus_collection: str = "rag_chunks"
    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 100
    # Retrieval
    k: int = 4  # passages given to the LLM
    # Minimum cosine similarity for a chunk to count as relevant. Deliberately strict, so weakly related chunks
    # don't reach the LLM; lower it (e.g. 0.3 for OpenAI text-embedding-3) if answers miss facts that are there
    score_threshold: float = 0.5
    use_hybrid: bool = True  # BM25 + embeddings; costs no extra API calls
    dense_weight: float = 0.5
    use_hyde: bool = False  # extra LLM calls per question
    use_decomposition: bool = False  # extra LLM calls per question


def build_embedder(config: RAGConfig) -> Embedder:
    return create_embedder(config.embedding_provider, config.embedding_model)


def build_llm(config: RAGConfig) -> ChatModel:
    return create_chat_model(config.llm_provider, config.llm_model)


def open_vector_store(config: RAGConfig, embedder: Embedder, rebuild: bool = False) -> VectorStore:
    """Open the configured vector store. rebuild=True starts empty, discarding the chunks indexed before (even ones
    embedded with another model, which a normal open refuses to mix with the current one)."""
    if config.store == "milvus":
        from rag.milvus_store import MilvusVectorStore  # pymilvus is only needed when Milvus is used

        return MilvusVectorStore(
            embedder, config.milvus_uri, config.milvus_collection, config.milvus_token, drop_existing=rebuild
        )
    if config.store != "faiss":
        raise ValueError(f"Unknown vector store {config.store!r}, choose 'faiss' or 'milvus'")
    if not rebuild and (Path(config.index_dir) / INDEX_FILE).exists():
        return FaissVectorStore.load(config.index_dir, embedder)
    return FaissVectorStore(embedder)


def list_pdfs(directory: str | Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def index_documents(
    config: RAGConfig,
    store: VectorStore,
    only_new: bool = True,
    progress: Callable[[int, int, Path], None] | None = None,
) -> list[Path]:
    """Load, split and embed the PDFs in config.documents_dir into store; returns the files that were indexed.

    With only_new, files already in the store (matched by path) are skipped, so after adding a few PDFs only those
    get embedded. Edited or deleted files are only picked up by rebuilding: open_vector_store(rebuild=True), then
    index_documents(only_new=False). progress(done, total, file) is called before each file.
    """
    files = list_pdfs(config.documents_dir)
    if only_new:
        indexed = {doc.metadata.get("source") for doc in store.all_documents()}
        files = [f for f in files if str(f.resolve()) not in indexed]
    splitter = RecursiveTextSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    done = []
    for i, path in enumerate(files):
        if progress:
            progress(i, len(files), path)
        try:
            pages = load_pdf(path)
        except Exception as e:
            logger.error("Failed to parse, skipped %s: %s", path.name, e)
            continue
        store.add_documents(splitter.split_documents(pages))  # file by file, so a failure keeps earlier files
        done.append(path)
    if isinstance(store, FaissVectorStore):
        store.save(config.index_dir)  # Milvus persists on insert; FAISS lives in memory until saved
    return done


def format_context(passages: list[tuple[Document, float]]) -> str:
    """Number the passages for the prompt, so the answer can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{i}] ({doc.metadata.get('file_name', 'unknown file')}, page {doc.metadata.get('page', '?')})\n{doc.text}"
        for i, (doc, _) in enumerate(passages, 1)
    )


@dataclass
class RAGAnswer:
    answer: str
    retrieval: RetrievalResult  # retrieval.documents are the cited passages, in order: [1] is the first


class RAGChain:
    """Question in, cited answer out: retrieve passages, put them in a numbered prompt, let the LLM answer."""

    def __init__(self, retriever: Retriever, llm: ChatModel | None, k: int = 4):
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def retrieve(self, question: str) -> RetrievalResult:
        return self.retriever.retrieve(question, k=self.k)

    def build_prompt(self, question: str, retrieval: RetrievalResult) -> str:
        context = format_context(retrieval.documents) or NO_CONTEXT
        return ANSWER_PROMPT.format(context=context, question=question)

    def answer(self, question: str) -> RAGAnswer:
        retrieval = self.retrieve(question)
        text = self.llm.generate(self.build_prompt(question, retrieval), system=SYSTEM_PROMPT)
        return RAGAnswer(answer=text, retrieval=retrieval)

    def stream_answer(self, question: str, retrieval: RetrievalResult) -> Iterator[str]:
        """Stream the answer for an already retrieved result, so a UI can show the sources while it streams."""
        return self.llm.stream(self.build_prompt(question, retrieval), system=SYSTEM_PROMPT)


def build_chain(config: RAGConfig, store: VectorStore, llm: ChatModel | None) -> RAGChain:
    """Assemble the retriever and chain from config. The same LLM writes answers and transforms queries; without
    one, the chain can still retrieve (as long as HyDE and decomposition are off) but not answer."""
    retriever = Retriever(
        store,
        llm,
        use_hyde=config.use_hyde,
        use_decomposition=config.use_decomposition,
        use_hybrid=config.use_hybrid,
        dense_weight=config.dense_weight,
        score_threshold=config.score_threshold,
    )
    return RAGChain(retriever, llm, k=config.k)
