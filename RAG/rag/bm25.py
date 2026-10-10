"""BM25 keyword search: the sparse half of hybrid retrieval.

Embeddings match meaning but can miss exact terms such as names, product codes or error numbers; BM25 scores chunks
by how often they contain the query's terms, weighting rare terms higher, so it catches exactly those cases.
"""

import math
import re
from array import array
from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np

from rag.document import Document

# Japanese kana and Chinese ideographs
_CJK_CHARS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
# A run of CJK characters, or a run of other letters / digits
_TOKEN_RE = re.compile(rf"[{_CJK_CHARS}]+|[^\W_{_CJK_CHARS}]+")
_CJK_RE = re.compile(rf"[{_CJK_CHARS}]")


def tokenize(text: str) -> list[str]:
    """Split text into BM25 terms: lowercase words for alphabetic text, characters plus character pairs for CJK.

    Chinese has no spaces between words, and overlapping pairs ("ab", "bc") approximate words without a dictionary,
    the same approach as Elasticsearch's CJK analyzer. Single characters are kept too, for one-character queries.
    For real word segmentation, pass a different tokenizer to BM25Index, e.g. jieba.lcut_for_search.
    """
    tokens = []
    for run in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.match(run):
            tokens.extend(run)
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        else:
            tokens.append(run)
    return tokens


class BM25Index:
    """An in-memory BM25 index over Documents.

    It is rebuilt from the vector store's documents instead of being saved separately: building takes about
    4 seconds per 10k chunks of 500 characters (well under a second for a few PDFs), and a query a few milliseconds.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        tokenizer: Callable[[str], list[str]] = tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = list(documents)
        self.tokenizer = tokenizer
        self.k1 = k1  # how quickly repeated occurrences of a term stop adding to the score
        self.b = b  # how strongly long chunks are penalized (0 = not at all, 1 = fully)

        # Collect one (term, chunk, frequency) posting per distinct term in each chunk, in compact arrays
        self.vocab: dict[str, int] = {}  # term -> term id
        term_ids, doc_ids, freqs, lengths = array("i"), array("i"), array("i"), []
        for i, doc in enumerate(self.documents):
            counts = Counter(tokenizer(doc.text))
            lengths.append(sum(counts.values()))
            new_terms = [term for term in counts if term not in self.vocab]
            self.vocab.update(zip(new_terms, range(len(self.vocab), len(self.vocab) + len(new_terms))))
            term_ids.extend(map(self.vocab.__getitem__, counts))
            doc_ids.extend(array("i", [i]) * len(counts))
            freqs.extend(counts.values())

        # Inverted index in compressed sparse row form: sort the postings by term once, so the chunks containing
        # term t are _doc_ids[_offsets[t]:_offsets[t + 1]], with matching frequencies in _freqs. The order within
        # one term doesn't matter for scoring, so the fast unstable sort is fine
        term_ids_np = np.frombuffer(term_ids, dtype=np.int32)
        order = np.argsort(term_ids_np)
        self._doc_ids = np.frombuffer(doc_ids, dtype=np.int32)[order]
        self._freqs = np.frombuffer(freqs, dtype=np.int32)[order].astype(np.float32)
        self._offsets = np.concatenate([[0], np.cumsum(np.bincount(term_ids_np, minlength=len(self.vocab)))])
        doc_lengths = np.asarray(lengths, dtype=np.float32)
        avg_length = max(float(doc_lengths.mean()), 1.0) if lengths else 1.0
        # Per-chunk part of the BM25 denominator, computed once instead of on every query
        self._length_norm = k1 * (1 - b + b * doc_lengths / avg_length)

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return up to k chunks sharing terms with query, best first, with their BM25 scores."""
        scores = np.zeros(len(self.documents), dtype=np.float32)
        n = len(self.documents)
        for term in set(self.tokenizer(query)):
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            start, end = self._offsets[term_id], self._offsets[term_id + 1]
            ids, freqs = self._doc_ids[start:end], self._freqs[start:end]
            # Rare terms carry more weight; this IDF variant (as in Lucene) never goes negative
            idf = math.log(1 + (n - len(ids) + 0.5) / (len(ids) + 0.5))
            scores[ids] += idf * freqs * (self.k1 + 1) / (freqs + self._length_norm[ids])
        top = np.argsort(-scores)[:k]
        return [(self.documents[i], float(scores[i])) for i in top if scores[i] > 0]
