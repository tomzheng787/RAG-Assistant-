"""Embedding: turn text into vectors through different embedding model providers.

Most providers (OpenAI, Qwen / DashScope, Zhipu, SiliconFlow, Gemini, Ollama, ...) expose an OpenAI-compatible
/embeddings endpoint, so one client covers all of them. Voyage AI has its own API and local models run through
sentence-transformers; each of those gets a dedicated class. Use create_embedder() to pick one by name.
"""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


class Embedder(ABC):
    """Common interface for all embedding backends.

    Documents and queries go through separate methods because some models encode them differently
    (e.g. Voyage's input_type, or the query instruction some BGE / Qwen3 embedding models expect).
    """

    model: str
    batch_size: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for indexing, sending them in batches no larger than the provider accepts."""
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch, input_type="document"))
            logger.debug("Embedded %d/%d texts", start + len(batch), len(texts))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        return self._embed_batch([text], input_type="query")[0]

    @abstractmethod
    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Embed one batch; input_type is "document" or "query"."""


class OpenAICompatibleEmbedder(Embedder):
    """Any provider implementing OpenAI's /embeddings API, OpenAI itself included."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        # Without an explicit key the SDK falls back to OPENAI_API_KEY, which must never be sent to another provider
        if base_url and not api_key:
            raise ValueError(f"api_key is required when base_url is set ({base_url})")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        # The SDK retries rate limits (429) and server errors with exponential backoff
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        extra = {"dimensions": self.dimensions} if self.dimensions else {}
        # Ask for plain floats: the SDK's default base64 encoding is not supported by every compatible provider
        response = self.client.embeddings.create(model=self.model, input=texts, encoding_format="float", **extra)
        # Each result carries the position of its input; sort by it rather than trusting the response order
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


class VoyageEmbedder(Embedder):
    """Voyage AI's native embeddings API, which distinguishes documents from queries via input_type."""

    API_URL = "https://api.voyageai.com/v1/embeddings"
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        model: str = "voyage-4",
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 128,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError("Missing Voyage API key: set VOYAGE_API_KEY or pass api_key")
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=httpx.HTTPTransport(retries=max_retries),  # retries connection failures only
        )

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        payload = {"model": self.model, "input": texts, "input_type": input_type}
        if self.dimensions:
            payload["output_dimension"] = self.dimensions
        # Retry rate limits and server errors with exponential backoff: 1s, 2s, 4s, ...
        for attempt in range(self.max_retries + 1):
            response = self.client.post(self.API_URL, json=payload)
            if response.status_code not in self.RETRY_STATUS or attempt == self.max_retries:
                break
            delay = 2**attempt
            logger.warning("Voyage API returned %d, retrying in %ds", response.status_code, delay)
            time.sleep(delay)
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]


class SentenceTransformerEmbedder(Embedder):
    """Local open-source models (BGE, Qwen3-Embedding, ...) via sentence-transformers: no API key, works offline
    once the model is downloaded."""

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        dimensions: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
        query_prompt: str | None = None,
    ):
        # Optional dependency that pulls in PyTorch, so import it only when a local model is actually used
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embedding models need sentence-transformers: pip install sentence-transformers"
            ) from e

        self.model = model
        self.batch_size = batch_size
        # Some models expect an instruction in front of queries, e.g. bge-*-en-v1.5:
        # "Represent this sentence for searching relevant passages: "
        self.query_prompt = query_prompt
        # truncate_dim shortens vectors; only meaningful for models trained for it (Matryoshka embeddings)
        self._model = SentenceTransformer(model, device=device, truncate_dim=dimensions)

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        prompt = self.query_prompt if input_type == "query" else None
        vectors = self._model.encode(texts, batch_size=self.batch_size, prompt=prompt, normalize_embeddings=True)
        return vectors.tolist()


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str | None
    api_key_env: str | None  # None means the provider needs no key
    default_model: str
    batch_size: int  # max inputs per request the provider accepts


# OpenAI-compatible providers. Batch limits follow each provider's docs as of 2026-10; if a provider rejects
# a request as too large, pass a smaller batch_size to create_embedder().
OPENAI_COMPATIBLE_PROVIDERS = {
    "openai": ProviderSpec(None, "OPENAI_API_KEY", "text-embedding-3-small", 256),
    # Alibaba Cloud Model Studio (DashScope); use dashscope-intl.aliyuncs.com outside mainland China
    "qwen": ProviderSpec(
        "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "text-embedding-v4", 10
    ),
    "zhipu": ProviderSpec("https://open.bigmodel.cn/api/paas/v4/", "ZHIPUAI_API_KEY", "embedding-3", 10),
    # Hosts open-source models such as BAAI/bge-m3 and Qwen/Qwen3-Embedding-8B
    "siliconflow": ProviderSpec("https://api.siliconflow.cn/v1", "SILICONFLOW_API_KEY", "BAAI/bge-m3", 32),
    "gemini": ProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-embedding-001", 100
    ),
    # Local models served by Ollama, e.g. after `ollama pull bge-m3`
    "ollama": ProviderSpec("http://localhost:11434/v1", None, "bge-m3", 32),
}

PROVIDERS = [*OPENAI_COMPATIBLE_PROVIDERS, "voyage", "local"]


def create_embedder(provider: str, model: str | None = None, **kwargs) -> Embedder:
    """Create an embedder by provider name; see PROVIDERS for the options.

    API keys are read from each provider's environment variable unless api_key is passed. Extra keyword
    arguments (dimensions, batch_size, base_url, ...) go to the embedder class.
    """
    provider = provider.lower()
    if provider == "voyage":
        return VoyageEmbedder(model=model or "voyage-4", **kwargs)
    if provider == "local":
        return SentenceTransformerEmbedder(model=model or "BAAI/bge-m3", **kwargs)

    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown embedding provider {provider!r}, choose from: {', '.join(PROVIDERS)}")
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        # Keyless servers like Ollama still need a non-empty placeholder for the SDK
        api_key = os.environ.get(spec.api_key_env) if spec.api_key_env else "unused"
    if not api_key:
        raise ValueError(f"Missing API key for {provider}: set {spec.api_key_env} or pass api_key")
    kwargs.setdefault("base_url", spec.base_url)
    kwargs.setdefault("batch_size", spec.batch_size)
    return OpenAICompatibleEmbedder(model=model or spec.default_model, api_key=api_key, **kwargs)


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Embed a piece of text to check that a provider is set up correctly")
    parser.add_argument("text", nargs="?", default="What is retrieval-augmented generation?", help="text to embed")
    parser.add_argument("--provider", default="openai", choices=PROVIDERS, help="embedding provider")
    parser.add_argument("--model", help="model name (defaults to the provider's default model)")
    parser.add_argument("--dimensions", type=int, help="output vector size, for models that support it")
    args = parser.parse_args()

    load_dotenv()  # read API keys from a .env file in the working directory, if present
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        embedder = create_embedder(args.provider, args.model, dimensions=args.dimensions)
    except (ValueError, ImportError) as e:
        parser.error(str(e))
    vector = embedder.embed_query(args.text)
    print(f"{args.provider} / {embedder.model}: {len(vector)} dimensions")
    print("first values:", [round(v, 4) for v in vector[:8]])
