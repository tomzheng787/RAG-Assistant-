[README.md](https://github.com/user-attachments/files/33293475/README.md)
# PDF RAG Assistant

English 

Ask questions about your PDF documents and get answers with page-level citations. This is a retrieval-augmented generation (RAG) pipeline built from scratch, with no LangChain or LlamaIndex, so every step is plain, readable Python: PDF parsing, chunking, embeddings, FAISS or Milvus vector search, BM25 hybrid retrieval, HyDE, query decomposition, and cited answer generation. It comes with a Streamlit chat app and a command line.

## Features

- **PDF parsing** with PyMuPDF: text in reading order, paragraphs rebuilt from wrapped lines, and Chinese/English aware line joining
- **Recursive chunking** that breaks at paragraph, sentence and clause boundaries, with overlap between chunks
- **Embeddings from 8 providers**: OpenAI, Qwen (Alibaba DashScope), Zhipu, SiliconFlow, Gemini, Voyage AI, Ollama, or local sentence-transformers models
- **Two vector stores** behind one interface: **FAISS** (a local index file) or **Milvus** (Milvus Lite, a Milvus server, or Zilliz Cloud)
- **Retrieval techniques** you can combine:
  - **Hybrid search**: BM25 keyword search plus embedding search, fused with reciprocal rank fusion, so exact terms like names, codes and numbers are found as well as paraphrases
  - **HyDE** (Hypothetical Document Embeddings): search with an LLM-written hypothetical answer
  - **Query decomposition**: split complex questions into sub-questions and retrieve for each
  - **Similarity threshold**: drop weakly related passages before they reach the LLM
- **Cited answers**: the LLM answers only from the retrieved passages, cites them as [1], [2], ..., and says so when the documents don't cover a question; answers stream as they are generated
- **Chat models from 7 providers**: OpenAI, Claude, Qwen, DeepSeek, Zhipu, Gemini, Ollama

## How it works

```
Indexing   PDF ─► pages ─► chunks ─► embeddings ─► FAISS / Milvus
                                  └─────────────► BM25 index (built from the same chunks)

Answering  question ─► [query decomposition] ─► for each (sub-)question:
                          embedding search (optionally with HyDE) ─► similarity threshold ─┐
                          BM25 keyword search ─────────────────────────────────────────────┴─► RRF fusion
                       ─► top-k passages ─► LLM ─► answer with [1] [2] citations
```

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
cp .env.example .env          # then uncomment and fill in the API keys you use
```

Put your PDFs into `documents/`, then start the app:

```bash
streamlit run app.py
```

In the sidebar, click **Index new files**, then ask questions in the chat box. The defaults use OpenAI for both embeddings and answers; pick other providers under **Models**.

### Command line

```bash
python -m rag ingest                       # index new PDFs in documents/ (--rebuild to start over)
python -m rag ask "What does the report conclude?"
python -m rag search "error code E1234"    # show the retrieved passages only
```

Every command takes the same options as the app, e.g.:

```bash
python -m rag ingest --embedding-provider qwen --store milvus
python -m rag ask "..." --embedding-provider qwen --store milvus --llm-provider claude --hyde --decompose
```

Run `python -m rag <command> --help` for the full list.

## Configuration

### Model providers

Set the API key of each provider you use in `.env` (see `.env.example`).

| Provider | Embeddings (default model) | Chat (default model) | API key variable |
|---|---|---|---|
| `openai` | text-embedding-3-small | gpt-6-luna | `OPENAI_API_KEY` |
| `claude` | (no embedding API) | claude-opus-5-5 | `ANTHROPIC_API_KEY` |
| `qwen` | text-embedding-v4 | qwen-plus | `DASHSCOPE_API_KEY` |
| `deepseek` | (no embedding API) | deepseek-v4-flash | `DEEPSEEK_API_KEY` |
| `zhipu` | embedding-3 | glm-5.3 | `ZHIPUAI_API_KEY` |
| `siliconflow` | BAAI/bge-m3 | | `SILICONFLOW_API_KEY` |
| `gemini` | gemini-embedding-001 | gemini-3.6-flash | `GEMINI_API_KEY` |
| `voyage` | voyage-4 | | `VOYAGE_API_KEY` |
| `ollama` | bge-m3 (local) | qwen3 (local) | none |
| `local` | BAAI/bge-m3 via sentence-transformers | | none |

Any model of a provider can be used by entering its name (`--embedding-model`, `--llm-model`, or the text fields in the app). Other OpenAI-compatible services can be added in one line in `rag/embedding.py` / `rag/llm.py`.

> **Changing the embedding model means re-indexing.** Vectors from different models are not comparable, so the stores record the model they were built with and refuse to mix them; click **Rebuild index** (or run `python -m rag ingest --rebuild`).

### Vector store

| | FAISS (default) | Milvus |
|---|---|---|
| Setup | nothing | nothing for Milvus Lite; or run a [Milvus server](https://milvus.io/docs/install_standalone-docker.md) |
| Where data lives | `index/` | `index/milvus.db` (Lite), or the server |
| Select with | `--store faiss` | `--store milvus --milvus-uri <uri>` |

The Milvus URI picks the deployment: a file path such as `index/milvus.db` runs Milvus Lite inside the process, `http://localhost:19530` connects to a server, and a Zilliz Cloud endpoint works together with `MILVUS_TOKEN` in `.env`. Milvus Lite lets one process at a time open its file, so don't run the CLI against the same file while the app has it open.

### Retrieval options

| Option | App control | CLI flag | Default |
|---|---|---|---|
| Passages given to the LLM | Passages per answer | `-k` | 4 |
| Similarity threshold | Similarity threshold | `--threshold` | 0.5 |
| Hybrid BM25 + embedding search | Hybrid search | `--no-hybrid` to turn off | on |
| Share of embedding results in hybrid fusion | Embedding weight | `--dense-weight` | 0.5 |
| HyDE | HyDE | `--hyde` | off |
| Query decomposition | Query decomposition | `--decompose` | off |

**About the threshold**: it is a minimum cosine similarity between the question and a passage. It is set fairly strict so that weakly related passages don't reach the LLM, but typical scores depend on the embedding model: OpenAI's text-embedding-3 models score lower than BGE or Qwen embeddings. If answers say the documents don't cover something that is in them, lower it (for OpenAI embeddings, 0.3 is a good start). With hybrid search the threshold filters the embedding results; BM25 then adds keyword matches, unless nothing passed the threshold at all.

HyDE and query decomposition each add LLM calls per question, so they make answers slower and cost more; hybrid search costs no extra API calls.

## Project structure

```
├── app.py                 Streamlit chat app
├── documents/             your PDFs (not committed to git)
├── index/                 FAISS index / Milvus Lite database (created on first indexing, not committed)
├── rag/
│   ├── __main__.py        command line: python -m rag ingest | search | ask
│   ├── chain.py           RAG chain, pipeline config, indexing
│   ├── pdf_loader.py      PDF parsing
│   ├── chunker.py         recursive text splitting
│   ├── embedding.py       embedding providers
│   ├── llm.py             chat model providers, streaming
│   ├── vector_store.py    vector store interface + FAISS store
│   ├── milvus_store.py    Milvus store
│   ├── bm25.py            BM25 keyword index
│   ├── retriever.py       hybrid search, HyDE, query decomposition, threshold
│   └── document.py        the Document type passed between steps
├── requirements.txt
└── .env.example           API key template
```

Several modules can also be run on their own for debugging, e.g. `python -m rag.pdf_loader documents/` previews the extracted text and `python -m rag.chunker documents/` previews the chunks.

## Limitations

- Scanned PDFs (images of text) have no extractable text and are skipped with a warning; they need OCR first.
- Each page is chunked separately, so a paragraph spanning two pages ends up in two chunks.
- Tables are extracted as plain text, without their row/column structure.
- Questions are answered one at a time: follow-ups like "and its drawbacks?" don't see the previous question.

## Installation notes

- **macOS 13 or older**: recent `faiss-cpu` releases have no prebuilt wheel there, and pip then fails building it from source. Install it with `pip install --only-binary :all: faiss-cpu`.
- **NumPy 1.x environments** (e.g. older Anaconda): recent `pyarrow` (pulled in by Milvus Lite and Streamlit) requires NumPy 2. Either upgrade NumPy, or pin `pip install "pyarrow<26"`.
- **Local embedding models** (`local` provider) need `pip install sentence-transformers`, which installs PyTorch.
