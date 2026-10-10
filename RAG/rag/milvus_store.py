"""Milvus vector store: the same interface as the FAISS store, backed by the Milvus vector database.

The uri picks the deployment:
- a local file path such as "index/milvus.db" runs Milvus Lite inside this process, with nothing to install or start
  (one process at a time can open the file)
- "http://localhost:19530" connects to a Milvus server, e.g. one started with Docker
- a Zilliz Cloud endpoint, together with a token, connects to the managed service
"""

import logging
from pathlib import Path

from pymilvus import DataType, MilvusClient

from rag.document import Document
from rag.embedding import Embedder
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

MODEL_PROPERTY = "rag.embedding_model"  # collection property recording which model produced the vectors
INSERT_BATCH_SIZE = 1000  # rows per insert request, keeping requests well below Milvus' message size limit
MAX_QUERY_ROWS = 16384  # the most rows a single Milvus query may return


class MilvusVectorStore(VectorStore):
    """Chunks and their vectors in a Milvus collection, searched by cosine similarity."""

    def __init__(
        self,
        embedder: Embedder,
        uri: str = "index/milvus.db",
        collection: str = "rag_chunks",
        token: str | None = None,
        drop_existing: bool = False,
    ):
        self.embedder = embedder
        self.collection = collection
        if "://" not in uri:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)  # Milvus Lite creates the file, not its folder
        self.client = MilvusClient(uri=uri, token=token or "")
        if drop_existing and self.client.has_collection(collection):
            self.client.drop_collection(collection)
        if self.client.has_collection(collection):
            # Vectors from different models live in different spaces, so mixing them makes search results meaningless
            built_with = self.client.describe_collection(collection).get("properties", {}).get(MODEL_PROPERTY)
            if built_with and built_with != embedder.model:
                raise ValueError(
                    f"Milvus collection {collection!r} was built with {built_with!r}, but the embedder uses "
                    f"{embedder.model!r}; switch back to that model or rebuild the collection"
                )
            self.client.load_collection(collection)  # a collection must be loaded into memory before searches
        # The collection itself is created on the first add, once the vector dimension is known

    def __len__(self) -> int:
        if not self.client.has_collection(self.collection):
            return 0
        return self.client.query(self.collection, filter="", output_fields=["count(*)"])[0]["count(*)"]

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        vectors = self.embedder.embed_documents([doc.text for doc in documents])
        if not self.client.has_collection(self.collection):
            self._create_collection(dimension=len(vectors[0]))
        rows = [{"vector": vec, "text": doc.text, "metadata": doc.metadata} for doc, vec in zip(documents, vectors)]
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            self.client.insert(self.collection, rows[start : start + INSERT_BATCH_SIZE])
        logger.info("Indexed %d chunks in Milvus collection %r", len(documents), self.collection)

    def search_by_vector(self, vector: list[float], k: int = 4) -> list[tuple[Document, float]]:
        if not self.client.has_collection(self.collection):
            return []
        hits = self.client.search(
            self.collection,
            data=[vector],
            limit=k,
            output_fields=["text", "metadata"],
            search_params={"metric_type": "COSINE"},
        )
        # With the COSINE metric, Milvus reports the similarity itself as "distance": higher is closer
        return [(Document(h["entity"]["text"], h["entity"]["metadata"]), float(h["distance"])) for h in hits[0]]

    def all_documents(self) -> list[Document]:
        total = len(self)
        if total == 0:
            return []
        fields = ["text", "metadata"]
        if total <= MAX_QUERY_ROWS:
            rows = self.client.query(self.collection, filter="", limit=total, output_fields=fields)
        else:
            # Larger collections are read in batches. (On Milvus Lite, pymilvus logs a harmless warning here)
            iterator = self.client.query_iterator(self.collection, batch_size=1000, filter="", output_fields=fields)
            rows = []
            while batch := iterator.next():
                rows.extend(batch)
            iterator.close()
        return [Document(row["text"], row["metadata"]) for row in rows]

    def close(self) -> None:
        """Release the connection (and, for Milvus Lite, the lock on the database file)."""
        self.client.close()

    def _create_collection(self, dimension: int) -> None:
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)  # bytes; far above any chunk the splitter makes
        schema.add_field("metadata", DataType.JSON)
        index_params = self.client.prepare_index_params()
        # AUTOINDEX lets Milvus choose a suitable index; the cosine metric matches the FAISS store's scores
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        # Strong consistency makes freshly inserted chunks searchable right away
        self.client.create_collection(self.collection, schema=schema, consistency_level="Strong")
        # Don't block until the index is built: that wait relies on a call Milvus Lite doesn't implement (it logs a
        # scary error), and loading the collection waits for the index anyway
        self.client.create_index(self.collection, index_params, sync=False)
        self.client.load_collection(self.collection)
        self.client.alter_collection_properties(self.collection, properties={MODEL_PROPERTY: self.embedder.model})
