"""Vector stores: keep chunk embeddings and retrieve the chunks most similar to a query.

VectorStore is the interface the retrieval code relies on; FaissVectorStore implements it with a local FAISS index
saved to disk, and rag.milvus_store.MilvusVectorStore implements it with the Milvus vector database.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np

from rag.document import Document
from rag.embedding import Embedder

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
DOCUMENTS_FILE = "documents.json"


class VectorStore(ABC):
    """Common interface of the vector store backends, so retrieval works the same with FAISS or Milvus."""

    embedder: Embedder

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored chunks."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Embed documents and add them to the store."""

    @abstractmethod
    def search_by_vector(self, vector: list[float], k: int = 4) -> list[tuple[Document, float]]:
        """Return up to k chunks most similar to vector, best first, each with its cosine similarity (-1 to 1)."""

    @abstractmethod
    def all_documents(self) -> list[Document]:
        """Every stored chunk, e.g. to build a BM25 index over the same chunks."""

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return up to k chunks most similar to query, best first, each with its cosine similarity (-1 to 1)."""
        if len(self) == 0:
            return []
        return self.search_by_vector(self.embedder.embed_query(query), k)


class FaissVectorStore(VectorStore):
    """Chunks and their vectors in a FAISS index, searched by cosine similarity.

    Vectors are L2-normalized and kept in an exact inner-product index (IndexFlatIP), so the inner product equals
    cosine similarity. Exact search stays fast up to roughly a million chunks; beyond that, switch to an
    approximate index such as IndexHNSWFlat or IndexIVFFlat.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: faiss.Index | None = None  # created on the first add, once the vector dimension is known
        self.documents: list[Document] = []  # documents[i] is the chunk stored at index position i

    def __len__(self) -> int:
        return len(self.documents)

    def add_documents(self, documents: list[Document]) -> None:
        """Embed documents and append them to the index."""
        if not documents:
            return
        vectors = self._to_matrix(self.embedder.embed_documents([doc.text for doc in documents]))
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents.extend(documents)
        logger.info("Indexed %d chunks (%d total)", len(documents), len(self.documents))

    def all_documents(self) -> list[Document]:
        return self.documents

    def search_by_vector(self, vector: list[float], k: int = 4) -> list[tuple[Document, float]]:
        if self.index is None:
            return []
        scores, ids = self.index.search(self._to_matrix([vector]), min(k, len(self.documents)))
        return [(self.documents[i], float(score)) for i, score in zip(ids[0], scores[0]) if i != -1]

    def save(self, directory: str | Path) -> None:
        """Write the FAISS index and the chunks (text + metadata) to directory. Saving an empty store removes
        a previously saved index, so an emptied store doesn't come back on the next load."""
        directory = Path(directory)
        if self.index is None:
            for name in (INDEX_FILE, DOCUMENTS_FILE):
                (directory / name).unlink(missing_ok=True)
            return
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        data = {
            "embedding_model": self.embedder.model,
            "dimension": self.index.d,
            "documents": [{"text": doc.text, "metadata": doc.metadata} for doc in self.documents],
        }
        (directory / DOCUMENTS_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d chunks to %s", len(self.documents), directory)

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "FaissVectorStore":
        """Load a store written by save(); embedder must use the same model the index was built with."""
        directory = Path(directory)
        if not (directory / INDEX_FILE).exists() or not (directory / DOCUMENTS_FILE).exists():
            raise FileNotFoundError(f"No vector store found in {directory}; build one first")
        data = json.loads((directory / DOCUMENTS_FILE).read_text(encoding="utf-8"))
        # Vectors from different models live in different spaces, so mixing them makes search results meaningless
        if data["embedding_model"] != embedder.model:
            raise ValueError(
                f"The index in {directory} was built with {data['embedding_model']!r}, but the embedder uses "
                f"{embedder.model!r}; switch back to that model or rebuild the index"
            )
        store = cls(embedder)
        store.index = faiss.read_index(str(directory / INDEX_FILE))
        store.documents = [Document(text=d["text"], metadata=d["metadata"]) for d in data["documents"]]
        if store.index.ntotal != len(store.documents):
            raise ValueError(
                f"Corrupted vector store in {directory}: "
                f"{store.index.ntotal} vectors but {len(store.documents)} documents"
            )
        return store

    def _to_matrix(self, vectors: list[list[float]]) -> np.ndarray:
        """Convert vectors to the normalized float32 matrix FAISS expects."""
        matrix = np.asarray(vectors, dtype=np.float32)
        if self.index is not None and matrix.shape[1] != self.index.d:
            raise ValueError(
                f"Embedding dimension {matrix.shape[1]} does not match the index ({self.index.d}); "
                "was the model or its dimensions setting changed?"
            )
        faiss.normalize_L2(matrix)  # in place; makes inner product equal to cosine similarity
        return matrix

