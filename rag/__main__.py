"""Command line for the RAG pipeline.

    python -m rag ingest                 index new PDFs from the documents folder (--rebuild to start over)
    python -m rag search "question"      show the passages retrieval finds, without generating an answer
    python -m rag ask "question"         answer the question from the documents, with citations

Run `python -m rag <command> --help` for all options. API keys are read from the environment or a .env file.
"""

import argparse
import logging
import os
from dataclasses import fields

from dotenv import load_dotenv

from rag.chain import RAGConfig, build_chain, build_embedder, build_llm, index_documents, open_vector_store
from rag.embedding import PROVIDERS
from rag.llm import CHAT_PROVIDERS
from rag.retriever import RetrievalResult

DEFAULTS = RAGConfig()


def add_store_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("embedding and vector store")
    group.add_argument("--embedding-provider", default=DEFAULTS.embedding_provider, choices=PROVIDERS)
    group.add_argument("--embedding-model", help="defaults to the provider's default model")
    group.add_argument("--store", default=DEFAULTS.store, choices=["faiss", "milvus"], help="vector store backend")
    group.add_argument("--index-dir", default=DEFAULTS.index_dir, help="where the FAISS index is saved")
    group.add_argument(
        "--milvus-uri", default=DEFAULTS.milvus_uri, help="a file path runs Milvus Lite, a URL connects to a server"
    )
    group.add_argument("--milvus-collection", default=DEFAULTS.milvus_collection)


def add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("retrieval")
    group.add_argument("-k", type=int, default=DEFAULTS.k, help="number of passages to retrieve")
    group.add_argument(
        "--threshold",
        dest="score_threshold",
        type=float,
        default=DEFAULTS.score_threshold,
        help="minimum cosine similarity of a passage (0 keeps nearly everything)",
    )
    group.add_argument("--no-hybrid", dest="use_hybrid", action="store_false", help="embedding search only, no BM25")
    group.add_argument("--dense-weight", type=float, default=DEFAULTS.dense_weight, help="embedding share in hybrid")
    group.add_argument("--hyde", dest="use_hyde", action="store_true", help="use Hypothetical Document Embeddings")
    group.add_argument("--decompose", dest="use_decomposition", action="store_true", help="split into sub-questions")
    group = parser.add_argument_group("chat model")
    group.add_argument("--llm-provider", default=DEFAULTS.llm_provider, choices=CHAT_PROVIDERS)
    group.add_argument("--llm-model", help="defaults to the provider's default model")


def print_retrieval(result: RetrievalResult) -> None:
    if result.sub_questions:
        print("\nSub-questions:")
        for question in result.sub_questions:
            print(f"  - {question}")
    for query, passages in result.hypothetical_documents.items():
        print(f"\nHyDE passage for {query!r}:")
        for passage in passages:
            print(f"  {passage}")


def print_sources(result: RetrievalResult, full_text: bool) -> None:
    if not result.documents:
        print("\nNo passage passed the similarity threshold.")
        return
    print("\nSources:" if not full_text else "")
    for i, (doc, score) in enumerate(result.documents, 1):
        meta = doc.metadata
        print(f"  [{i}] {meta.get('file_name')}, page {meta.get('page')}  (score {score:.3f})")
        if full_text:
            print(f"{doc.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m rag", description="Ask questions about your PDF documents")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index the PDFs in the documents folder")
    ingest.add_argument("--documents-dir", default=DEFAULTS.documents_dir, help="folder with the PDFs")
    ingest.add_argument("--rebuild", action="store_true", help="discard the index and re-index every file")
    ingest.add_argument("--chunk-size", type=int, default=DEFAULTS.chunk_size, help="maximum chunk length")
    ingest.add_argument("--chunk-overlap", type=int, default=DEFAULTS.chunk_overlap, help="overlap between chunks")
    add_store_options(ingest)

    for name, help_text in [("search", "show the passages retrieved for a question"), ("ask", "answer a question")]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("question")
        add_store_options(command)
        add_retrieval_options(command)

    args = parser.parse_args()
    if args.command == "ingest" and args.chunk_overlap >= args.chunk_size:
        parser.error(f"--chunk-overlap ({args.chunk_overlap}) must be smaller than --chunk-size ({args.chunk_size})")
    load_dotenv()  # read API keys from a .env file in the working directory, if present
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    # Options not offered by the chosen command keep their RAGConfig defaults
    config = RAGConfig(**{f.name: getattr(args, f.name) for f in fields(RAGConfig) if hasattr(args, f.name)})
    config.milvus_token = os.environ.get("MILVUS_TOKEN")

    try:
        store = open_vector_store(config, build_embedder(config), rebuild=getattr(args, "rebuild", False))
        needs_llm = args.command == "ask" or config.use_hyde or config.use_decomposition
        llm = build_llm(config) if args.command != "ingest" and needs_llm else None
    except (ValueError, ImportError, FileNotFoundError) as e:
        parser.error(str(e))

    if args.command == "ingest":
        files = index_documents(
            config,
            store,
            only_new=not args.rebuild,
            progress=lambda done, total, path: print(f"[{done + 1}/{total}] {path.name}"),
        )
        if files:
            print(f"Indexed {len(files)} file(s); the store now holds {len(store)} chunks.")
        else:
            print(f"No new PDFs in {config.documents_dir}/ ({len(store)} chunks indexed; --rebuild re-indexes all).")
        return

    if len(store) == 0:
        parser.error("The vector store is empty; run `python -m rag ingest` first")
    chain = build_chain(config, store, llm)
    result = chain.retrieve(args.question)
    print_retrieval(result)
    if args.command == "search":
        print()
        print_sources(result, full_text=True)
        return
    print()
    for piece in chain.stream_answer(args.question, result):
        print(piece, end="", flush=True)
    print()
    print_sources(result, full_text=False)


if __name__ == "__main__":
    main()
