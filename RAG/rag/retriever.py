"""Retrieval on top of the FAISS vector store, with hybrid search and query transformations.

- Hybrid dense-sparse retrieval: embedding search (dense, matches meaning) and BM25 keyword search (sparse, matches
  exact terms like names, codes and numbers) run side by side, and their rankings are fused with reciprocal rank
  fusion, so a chunk either retriever finds can surface and one both agree on ranks highest.
- HyDE (Hypothetical Document Embeddings): an LLM first writes a passage that would answer the question, and the
  search uses that passage's embedding together with the question's. Questions and answers are phrased differently,
  so a hypothetical answer usually lands closer to the real answer passages in embedding space.
- Query decomposition: an LLM splits a complex question into simpler sub-questions. Each one (plus the original
  question) is retrieved separately and the result lists are merged by taking each query's best chunks in turn,
  so every part of the question gets supporting chunks.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from rag.bm25 import BM25Index
from rag.document import Document
from rag.llm import ChatModel, RefusalError
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Write a short passage (about 100-200 words) that directly answers the question below, written like \
an excerpt from a document or reference text that contains the answer. Use the same language as the question. If \
you are unsure of the facts, still write a plausible passage: it is only used for search, never shown to anyone. \
Output only the passage.

Question: {question}"""

DECOMPOSE_PROMPT = """Break the question below into at most {max_questions} simpler sub-questions that together \
cover everything needed to answer it. Each sub-question must be understandable on its own, without referring to the \
other sub-questions. If the question is already simple, return it unchanged as the only line. Use the same language \
as the question. Output one sub-question per line, with no numbering and nothing else.

Question: {question}"""

# List markers a model may add despite the instructions: "1. ", "2) ", "3" + CJK enumeration comma, "- ", "* ",
# bullet. A marker must be followed by whitespace, so a sub-question starting with "3.14" keeps its number
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]\s+|\d+\u3001\s*|[-*\u2022]\s+)")


def generate_hypothetical_documents(llm: ChatModel, question: str, n: int = 1) -> list[str]:
    """Have the LLM write n passages that would answer the question (the "H" in HyDE)."""
    return [llm.generate(HYDE_PROMPT.format(question=question)) for _ in range(n)]


def decompose_query(llm: ChatModel, question: str, max_questions: int = 4) -> list[str]:
    """Split a question into up to max_questions sub-questions; a simple question comes back as itself."""
    try:
        reply = llm.generate(DECOMPOSE_PROMPT.format(question=question, max_questions=max_questions))
    except RefusalError:
        logger.warning("The LLM declined to decompose the question; using it as is")
        return [question]
    sub_questions = [_LIST_MARKER_RE.sub("", line).strip() for line in reply.splitlines()]
    return [q for q in sub_questions if q][:max_questions] or [question]


def interleave_results(result_lists: list[list[tuple[Document, float]]], k: int = 4) -> list[tuple[Document, float]]:
    """Merge per-query result lists by taking each list's best remaining chunk in turn, skipping duplicates.

    Every query (e.g. each sub-question) gets its top chunks into the merged result, so the context covers all parts
    of the question. Fusing by score or by rank sum instead favors chunks that show up in many lists, which can push
    out a sub-question's single best chunk. A chunk found by several queries keeps its highest score.
    """
    best: dict[tuple, tuple[Document, float]] = {}
    for rank in range(max(map(len, result_lists), default=0)):
        for results in result_lists:
            if rank < len(results):
                doc, score = results[rank]
                key = doc.key
                if key not in best or score > best[key][1]:
                    best[key] = (doc, score)  # updating an existing key keeps its position
    return list(best.values())[:k]  # dicts keep insertion order, i.e. the order chunks were first taken


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[Document, float]]],
    weights: list[float] | None = None,
    k: int = 4,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse ranked result lists: a chunk scores the sum of weight / (rrf_k + rank) over the lists it appears in.

    Used to combine dense and BM25 results for the same query. Their raw scores live on unrelated scales (cosine
    similarity vs. unbounded BM25), so ranks are fused instead, and a chunk both retrievers rank highly comes out on
    top. rrf_k = 60 comes from the original RRF paper; it keeps the very top ranks from dominating.
    """
    weights = weights or [1.0] * len(result_lists)
    scores: dict[tuple, float] = {}
    documents: dict[tuple, Document] = {}
    for results, weight in zip(result_lists, weights):
        for rank, (doc, _) in enumerate(results, 1):
            key = doc.key
            documents[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
    best = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [(documents[key], scores[key]) for key in best]


@dataclass
class RetrievalResult:
    # (chunk, score) pairs. Scores are cosine similarities, or fusion scores with hybrid retrieval; with
    # decomposition the order comes from the round-robin merge rather than from the scores
    documents: list[tuple[Document, float]]
    sub_questions: list[str] = field(default_factory=list)  # empty unless query decomposition ran
    hypothetical_documents: dict[str, list[str]] = field(default_factory=dict)  # query -> HyDE passages


class Retriever:
    """Retrieve chunks for a question from a vector store; hybrid search, HyDE and decomposition can be combined.

    LLM calls per question: decomposition adds 1, HyDE adds hyde_documents per searched query (with both enabled,
    for the original question and each sub-question; those run in parallel). Hybrid search needs no LLM.

    score_threshold drops chunks whose cosine similarity to the query is below it. Typical similarities depend on
    the embedding model (OpenAI text-embedding-3 scores run lower than BGE or Qwen embeddings), so tune it per model.
    With hybrid search it filters the embedding results; BM25 then adds keyword matches, unless no chunk passed the
    threshold at all, in which case the question is taken to be unrelated to the documents and nothing is returned.
    """

    def __init__(
        self,
        store: VectorStore,
        llm: ChatModel | None = None,
        use_hyde: bool = False,
        use_decomposition: bool = False,
        use_hybrid: bool = False,
        dense_weight: float = 0.5,
        score_threshold: float | None = None,
        hyde_documents: int = 1,
        max_sub_questions: int = 4,
    ):
        if (use_hyde or use_decomposition) and llm is None:
            raise ValueError("HyDE and query decomposition need an llm (a ChatModel)")
        if not 0 <= dense_weight <= 1:
            raise ValueError(f"dense_weight must be between 0 and 1, got {dense_weight}")
        self.store = store
        self.llm = llm
        self.use_hyde = use_hyde
        self.use_decomposition = use_decomposition
        self.hyde_documents = hyde_documents
        self.max_sub_questions = max_sub_questions
        # Share of the fused score given to dense results; BM25 gets the rest. Raise it for paraphrased,
        # conceptual questions, lower it for exact-term lookups such as names, codes or IDs
        self.dense_weight = dense_weight
        self.score_threshold = score_threshold
        self.bm25 = BM25Index(store.all_documents()) if use_hybrid else None

    def retrieve(self, question: str, k: int = 4) -> RetrievalResult:
        if self.bm25 is not None and len(self.bm25) != len(self.store):
            self.bm25 = BM25Index(self.store.all_documents())  # the store changed since the index was built
        result = RetrievalResult(documents=[])
        if not self.use_decomposition:
            result.documents = self._search(question, k, result)
            return result

        result.sub_questions = decompose_query(self.llm, question, self.max_sub_questions)
        logger.info("Sub-questions: %s", result.sub_questions)
        # Keep the original question too, in case the decomposition misses something
        queries = [question, *(q for q in result.sub_questions if q != question)]
        with ThreadPoolExecutor(max_workers=len(queries)) as pool:
            result_lists = list(pool.map(lambda q: self._search(q, k, result), queries))
        result.documents = interleave_results(result_lists, k)
        return result

    def _search(self, query: str, k: int, result: RetrievalResult) -> list[tuple[Document, float]]:
        if self.bm25 is None:
            return self._dense_search(query, k, result)
        # Fuse deeper candidate lists than k, so chunks both retrievers rank moderately well can still surface
        depth = max(4 * k, 20)
        dense = self._dense_search(query, depth, result)
        if not dense:
            return []  # nothing is semantically close enough; keyword matches alone (often common words) aren't trusted
        sparse = self.bm25.search(query, depth)  # keywords come from the real query, never from a HyDE passage
        return reciprocal_rank_fusion([dense, sparse], [self.dense_weight, 1 - self.dense_weight], k)

    def _dense_search(self, query: str, k: int, result: RetrievalResult) -> list[tuple[Document, float]]:
        results = self._embedding_search(query, k, result)
        if self.score_threshold is None:
            return results
        return [(doc, score) for doc, score in results if score >= self.score_threshold]

    def _embedding_search(self, query: str, k: int, result: RetrievalResult) -> list[tuple[Document, float]]:
        if not self.use_hyde:
            return self.store.search(query, k)
        embedder = self.store.embedder
        try:
            passages = generate_hypothetical_documents(self.llm, query, self.hyde_documents)
        except RefusalError:
            logger.warning("The LLM declined to write a HyDE passage for %r; searching with the query alone", query)
            return self.store.search(query, k)
        result.hypothetical_documents[query] = passages

        # As in the HyDE paper, average the hypothetical passages' vectors together with the query's own vector.
        # Normalize first so that every vector carries equal weight regardless of the provider's vector lengths.
        vectors = np.asarray([*embedder.embed_documents(passages), embedder.embed_query(query)], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return self.store.search_by_vector(vectors.mean(axis=0).tolist(), k)

