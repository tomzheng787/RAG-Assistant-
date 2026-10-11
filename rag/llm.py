"""Chat models: a small text-generation interface over different LLM providers, used for query transformations
(HyDE, query decomposition) and later for answer generation.

Most providers (OpenAI, Qwen / DashScope, DeepSeek, Zhipu, Gemini, Ollama, ...) expose OpenAI-compatible Chat
Completions, so one client covers all of them; Claude goes through Anthropic's official SDK. Use create_chat_model()
to pick one by name.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

# Open-source reasoning models (e.g. Qwen3 or DeepSeek-R1 served by Ollama) may put their reasoning inline
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Claude models that support server-side refusal fallback (fallbacks="default")
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


def _skip_leading_think(pieces: Iterator[str]) -> Iterator[str]:
    """Drop a leading <think>...</think> block from a streamed reply, passing everything after it through."""
    head = ""
    for piece in pieces:
        head += piece
        text = head.lstrip()
        if text.startswith("<think>"):
            end = text.find("</think>")
            if end == -1:
                continue  # still inside the reasoning block
            text = text[end + len("</think>") :].lstrip()
        elif "<think>".startswith(text):
            continue  # too short to tell yet, e.g. "<th"
        if text:
            yield text
            break
    else:
        # The stream ended while still undecided, e.g. a reply of just "<"; pass it on unless it was a think block
        if head.strip() and not head.lstrip().startswith("<think>"):
            yield head.strip()
        return
    yield from pieces


class RefusalError(RuntimeError):
    """The model declined to answer, e.g. because a safety filter blocked the request."""


class ChatModel(ABC):
    """Common interface for all chat backends: one prompt in, one text reply out."""

    model: str

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text reply to a single-turn prompt."""

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """Yield the reply in pieces as it is generated; backends without streaming yield it all at once."""
        yield self.generate(prompt, system)


class OpenAICompatibleChat(ChatModel):
    """Any provider implementing OpenAI's Chat Completions API, OpenAI itself included."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        **request_params: Any,
    ):
        # Without an explicit key the SDK falls back to OPENAI_API_KEY, which must never be sent to another provider
        if base_url and not api_key:
            raise ValueError(f"api_key is required when base_url is set ({base_url})")
        self.model = model
        # Extra Chat Completions parameters sent with every request, e.g. temperature or reasoning_effort
        self.request_params = request_params
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.model, messages=self._messages(prompt, system), **self.request_params
        )
        choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise RefusalError(f"{self.model} declined the request (content filter)")
        return _THINK_RE.sub("", choice.message.content or "").strip()

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model, messages=self._messages(prompt, system), stream=True, **self.request_params
        )
        yield from _skip_leading_think(self._stream_text(response))

    def _stream_text(self, response) -> Iterator[str]:
        for chunk in response:
            if not chunk.choices:
                continue  # e.g. a final chunk that only reports token usage
            choice = chunk.choices[0]
            if choice.finish_reason == "content_filter":
                raise RefusalError(f"{self.model} declined the request (content filter)")
            if choice.delta.content:
                yield choice.delta.content

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        return messages


class AnthropicChat(ChatModel):
    """Claude through Anthropic's official SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        api_key: str | None = None,
        effort: str = "low",
        max_tokens: int = 16000,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.model = model
        # Thinking depth: low / medium / high / xhigh / max. Low suits short tasks like rewriting a question
        self.effort = effort
        self.max_tokens = max_tokens
        # Without api_key the SDK finds credentials itself: ANTHROPIC_API_KEY, or a profile from `ant auth login`
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self.client.beta.messages.create(**self._request(prompt, system))
        if response.stop_reason == "refusal":
            raise RefusalError(f"{self.model} declined the request")
        # The reply may also hold thinking blocks; keep only the text
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        with self.client.beta.messages.stream(**self._request(prompt, system)) as stream:
            yield from stream.text_stream  # text only; thinking blocks are not included
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            raise RefusalError(f"{self.model} declined the request")

    def _request(self, prompt: str, system: str | None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        if self.model in _CLAUDE_FALLBACK_MODELS:
            # If a safety classifier declines, the API re-runs the request on a suitable fallback model in the same call
            request |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return request


@dataclass(frozen=True)
class ChatProviderSpec:
    base_url: str | None
    api_key_env: str | None  # None means the provider needs no key
    default_model: str
    # Parameters sent along with the default model; dropped when you pick another model, which may not accept them
    default_params: dict[str, Any] = field(default_factory=dict)


# OpenAI-compatible chat providers. Model names follow each provider's docs as of 2026-10.
OPENAI_COMPATIBLE_CHAT_PROVIDERS = {
    # gpt-6-luna is OpenAI's fast, low-cost model; low reasoning effort keeps short rewrites quick
    "openai": ChatProviderSpec(None, "OPENAI_API_KEY", "gpt-6-luna", {"reasoning_effort": "low"}),
    # Alibaba Cloud Model Studio (DashScope); use dashscope-intl.aliyuncs.com outside mainland China
    "qwen": ChatProviderSpec("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY", "qwen-plus"),
    "deepseek": ChatProviderSpec("https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-v4-flash"),
    # GLM-5.3 always thinks and defaults to maximum effort, which is slow for short rewrites
    "zhipu": ChatProviderSpec(
        "https://open.bigmodel.cn/api/paas/v4/", "ZHIPUAI_API_KEY", "glm-5.3", {"reasoning_effort": "low"}
    ),
    "gemini": ChatProviderSpec(
        "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "gemini-3.6-flash"
    ),
    # Local models served by Ollama, e.g. after `ollama pull qwen3`
    "ollama": ChatProviderSpec("http://localhost:11434/v1", None, "qwen3"),
}

CHAT_PROVIDERS = [*OPENAI_COMPATIBLE_CHAT_PROVIDERS, "claude"]


def create_chat_model(provider: str, model: str | None = None, **kwargs: Any) -> ChatModel:
    """Create a chat model by provider name; see CHAT_PROVIDERS for the options.

    API keys are read from each provider's environment variable unless api_key is passed. Extra keyword
    arguments go to the chat model class (for OpenAI-compatible providers, unknown ones become request parameters).
    """
    provider = provider.lower()
    if provider == "claude":
        return AnthropicChat(model=model or "claude-opus-5-5", **kwargs)

    spec = OPENAI_COMPATIBLE_CHAT_PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown chat provider {provider!r}, choose from: {', '.join(CHAT_PROVIDERS)}")
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        # Keyless servers like Ollama still need a non-empty placeholder for the SDK
        api_key = os.environ.get(spec.api_key_env) if spec.api_key_env else "unused"
    if not api_key:
        raise ValueError(f"Missing API key for {provider}: set {spec.api_key_env} or pass api_key")
    kwargs.setdefault("base_url", spec.base_url)
    params = {**spec.default_params, **kwargs} if model is None else kwargs
    return OpenAICompatibleChat(model=model or spec.default_model, api_key=api_key, **params)
