from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """The basic unit passed through the RAG pipeline: a piece of text plus where it came from."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple:
        """Identifies a chunk across searches and stores: same source, page, position and text means same chunk."""
        meta = self.metadata
        return meta.get("source"), meta.get("page"), meta.get("chunk_index"), self.text

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        return f"Document(text={preview!r}..., metadata={self.metadata})"
