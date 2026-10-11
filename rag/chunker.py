"""Chunking: split page-level Documents into overlapping chunks small enough to embed and retrieve precisely."""

import re
from collections.abc import Callable

from rag.document import Document

# Split boundaries from coarse to fine. Text is first split at the coarsest boundary; any piece still longer
# than chunk_size is split again at the next level, so chunks break at the most natural boundary available.
DEFAULT_SEPARATORS = [
    r"\n{2,}",  # paragraphs
    r"\n",  # lines
    # sentence ends: CJK full stop, full-width/ASCII ! and ?, plus any closing quotes or brackets;
    # an English period only counts before whitespace, so "3.14" stays whole
    r"[\u3002\uff01\uff1f!?]+[\u201d\u2019\"\uff09)]*|\.(?=\s)",
    r"[\uff1b;]",  # clauses: full-width/ASCII semicolons
    r"[\uff0c,\u3001]",  # phrases: full-width/ASCII commas, CJK enumeration comma
    r"\s+",  # words
]


def _split_after(text: str, pattern: str) -> list[str]:
    """Split text after every match of pattern, keeping the separator attached to the preceding piece."""
    pieces, start = [], 0
    for match in re.finditer(pattern, text):
        if match.end() > start:
            pieces.append(text[start : match.end()])
            start = match.end()
    if start < len(text):
        pieces.append(text[start:])
    return pieces


class RecursiveTextSplitter:
    """Recursively split text at natural boundaries, then pack the pieces into chunks of at most chunk_size.

    Lengths are measured with length_function: characters by default, which works for both Chinese and English;
    pass a tokenizer-based function to measure in tokens instead.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: list[str] | None = None,
        length_function: Callable[[str], int] = len,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = DEFAULT_SEPARATORS if separators is None else separators
        self.length_function = length_function

    def split_text(self, text: str) -> list[str]:
        chunks = (chunk.strip() for chunk in self._split(text, self.separators))
        return [chunk for chunk in chunks if chunk]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Split each Document; every chunk inherits its parent's metadata plus its position in the parent."""
        chunks = []
        for doc in documents:
            for i, text in enumerate(self.split_text(doc.text)):
                chunks.append(Document(text=text, metadata={**doc.metadata, "chunk_index": i}))
        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if self.length_function(text) <= self.chunk_size:
            return [text]
        # No boundary left to try: fall back to single characters, which _merge packs back up to chunk_size
        pieces = _split_after(text, separators[0]) if separators else list(text)

        chunks, fitting = [], []
        for piece in pieces:
            if self.length_function(piece) <= self.chunk_size:
                fitting.append(piece)
                continue
            if fitting:
                chunks.extend(self._merge(fitting))
                fitting = []
            chunks.extend(self._split(piece, separators[1:]))
        if fitting:
            chunks.extend(self._merge(fitting))
        return chunks

    def _merge(self, pieces: list[str]) -> list[str]:
        """Pack consecutive pieces into chunks; each new chunk starts with the tail of the previous one as overlap.

        The overlap is made of whole pieces (e.g. sentences) up to chunk_overlap long, so it never starts mid-sentence;
        a piece longer than chunk_overlap is not repeated, leaving those two neighboring chunks without overlap.
        """
        chunks, window, window_len = [], [], 0
        for piece in pieces:
            piece_len = self.length_function(piece)
            if window and window_len + piece_len > self.chunk_size:
                chunks.append("".join(window))
                # Drop pieces from the front until what remains fits the overlap budget and leaves room for this piece
                while window and (window_len > self.chunk_overlap or window_len + piece_len > self.chunk_size):
                    window_len -= self.length_function(window.pop(0))
            window.append(piece)
            window_len += piece_len
        if window:
            chunks.append("".join(window))
        return chunks


if __name__ == "__main__":
    import argparse
    import logging

    from rag.pdf_loader import load_pdfs

    parser = argparse.ArgumentParser(description="Load PDFs, split them into chunks and preview the result")
    parser.add_argument("path", help="a PDF file or a directory containing PDFs")
    parser.add_argument("--chunk-size", type=int, default=500, help="maximum chunk length in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="overlap between neighboring chunks")
    parser.add_argument("--show", type=int, default=5, help="number of chunks to print")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pages = load_pdfs(args.path)
    splitter = RecursiveTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(pages)

    lengths = [len(c.text) for c in chunks]
    print(f"\n{len(pages)} pages -> {len(chunks)} chunks", end="")
    if chunks:
        print(f", length min/avg/max = {min(lengths)}/{sum(lengths) // len(lengths)}/{max(lengths)}")
    print()
    for chunk in chunks[: args.show]:
        meta = chunk.metadata
        print(f"--- {meta['file_name']} page {meta['page']} chunk {meta['chunk_index']} ({len(chunk.text)} chars) ---")
        print(chunk.text)
        print()
