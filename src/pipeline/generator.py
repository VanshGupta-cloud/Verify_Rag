"""Modular LLM generation adapter for VerifyRAG.

Supports pluggable providers: Groq, OpenAI, Ollama, and deterministic MockGenerator
for offline testing and evaluation benchmarks.
"""

import abc
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.config import Settings, get_settings
from src.schemas import Chunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an accurate, fact-based AI research assistant.
Answer the user's query using ONLY the provided context chunks.
Ground every factual assertion directly in the provided evidence.
Do not invent or extrapolate facts beyond what is stated in the context."""


def format_context_prompt(query: str, context_chunks: List[Chunk]) -> str:
    """Format user query and retrieved context chunks into grounded prompt text."""
    if not context_chunks:
        context_str = "No reference context provided."
    else:
        formatted_chunks = [
            f"[Chunk {chunk.chunk_id}] (doc: {chunk.doc_id}):\n{chunk.text.strip()}"
            for chunk in context_chunks
        ]
        context_str = "\n\n".join(formatted_chunks)

    prompt = (
        f"Context Evidence:\n{context_str}\n\n"
        f"User Query: {query.strip()}\n\n"
        "Instructions: Formulate a clear, direct, and factual response grounded exclusively in the context above."
    )
    return prompt


class BaseGenerator(abc.ABC):
    """Abstract base interface for all LLM generator providers."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    @abc.abstractmethod
    def generate(
        self, query: str, context_chunks: List[Chunk], **kwargs: Any
    ) -> Tuple[str, float]:
        """Generate a draft response grounded on retrieved context chunks.

        Args:
            query: User's question or search query.
            context_chunks: List of retrieved Chunk objects to ground the generation.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Tuple containing:
                - generated_text: The string response.
                - latency_ms: Generation time in milliseconds.
        """
        pass


class MockGenerator(BaseGenerator):
    """Deterministic offline generator for unit tests and offline benchmarking."""

    def __init__(
        self,
        model_name: str = "mock-generator-v1",
        default_response: Optional[str] = None,
        latency_simulate_ms: float = 5.0,
    ):
        super().__init__(model_name=model_name, temperature=0.0)
        self.default_response = default_response
        self.latency_simulate_ms = latency_simulate_ms
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self, query: str, context_chunks: List[Chunk], **kwargs: Any
    ) -> Tuple[str, float]:
        start = time.perf_counter()

        if self.default_response is not None:
            response = self.default_response
        elif context_chunks:
            # Deterministically synthesize an answer from the first 2 chunks
            snippets = [c.text.strip().split(". ")[0] + "." for c in context_chunks[:2]]
            response = (
                f"Regarding {query.strip().rstrip('?')}, "
                + " ".join(snippets)
            )
        else:
            response = f"No context was provided to answer the question: '{query}'."

        if self.latency_simulate_ms > 0:
            time.sleep(self.latency_simulate_ms / 1000.0)

        latency_ms = (time.perf_counter() - start) * 1000.0
        self.call_history.append(
            {"query": query, "chunks_count": len(context_chunks), "response": response}
        )
        return response, round(latency_ms, 2)


class GroqGenerator(BaseGenerator):
    """High-speed LLM generation using Groq Cloud API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.api_key = api_key or get_settings().groq_api_key
        if not self.api_key:
            logger.warning(
                "GROQ_API_KEY is not configured. GroqGenerator will fail unless key is provided."
            )

    def generate(
        self, query: str, context_chunks: List[Chunk], **kwargs: Any
    ) -> Tuple[str, float]:
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required to generate responses with GroqGenerator. "
                "Set it in .env or pass api_key to GroqGenerator."
            )

        try:
            from groq import Groq  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'groq' package is not installed. Install via 'pip install groq'."
            ) from e

        client = Groq(api_key=self.api_key)
        user_prompt = format_context_prompt(query, context_chunks)

        start = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        latency_ms = (time.perf_counter() - start) * 1000.0
        text = response.choices[0].message.content or ""
        return text.strip(), round(latency_ms, 2)


class OpenAIGenerator(BaseGenerator):
    """LLM generation adapter using OpenAI models (e.g. gpt-4o-mini)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.api_key = api_key or get_settings().openai_api_key
        if not self.api_key:
            logger.warning(
                "OPENAI_API_KEY is not configured. OpenAIGenerator will fail unless key is provided."
            )

    def generate(
        self, query: str, context_chunks: List[Chunk], **kwargs: Any
    ) -> Tuple[str, float]:
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required to generate responses with OpenAIGenerator. "
                "Set it in .env or pass api_key to OpenAIGenerator."
            )

        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=self.api_key)
            user_prompt = format_context_prompt(query, context_chunks)

            start = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=kwargs.get("max_tokens", 1024),
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            text = response.choices[0].message.content or ""
            return text.strip(), round(latency_ms, 2)

        except ImportError:
            # Fallback to direct HTTP request using requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_context_prompt(query, context_chunks)},
                ],
                "temperature": self.temperature,
            }
            start = time.perf_counter()
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000.0
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text.strip(), round(latency_ms, 2)


class OllamaGenerator(BaseGenerator):
    """Local offline generation using Ollama REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3:8b",
        temperature: float = 0.1,
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        self.base_url = base_url.rstrip("/")

    def generate(
        self, query: str, context_chunks: List[Chunk], **kwargs: Any
    ) -> Tuple[str, float]:
        user_prompt = format_context_prompt(query, context_chunks)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        start = time.perf_counter()
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=kwargs.get("timeout", 120),
        )
        resp.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000.0
        data = resp.json()
        return data.get("response", "").strip(), round(latency_ms, 2)


def get_generator(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    settings: Optional[Settings] = None,
    **kwargs: Any,
) -> BaseGenerator:
    """Factory creating configured generator based on provider name or application settings.

    Args:
        provider: 'groq', 'openai', 'ollama', or 'mock'. Defaults to settings.llm_provider.
        model_name: Optional model override. Defaults to settings.llm_model.
        settings: Optional Settings instance. Defaults to get_settings().
        **kwargs: Extra arguments passed to generator constructor.

    Returns:
        Instance of BaseGenerator.
    """
    cfg = settings or get_settings()
    selected_provider = (provider or cfg.llm_provider).lower()
    selected_model = model_name or cfg.llm_model

    if selected_provider == "groq":
        return GroqGenerator(
            api_key=cfg.groq_api_key,
            model_name=selected_model,
            **kwargs,
        )
    elif selected_provider == "openai":
        return OpenAIGenerator(
            api_key=cfg.openai_api_key,
            model_name=selected_model if selected_model != "llama-3.3-70b-versatile" else "gpt-4o-mini",
            **kwargs,
        )
    elif selected_provider == "ollama":
        return OllamaGenerator(
            model_name=selected_model if selected_model != "llama-3.3-70b-versatile" else "llama3:8b",
            **kwargs,
        )
    elif selected_provider == "mock":
        return MockGenerator(
            model_name=selected_model or "mock-generator-v1",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{selected_provider}'. "
            "Supported options are: 'groq', 'openai', 'ollama', 'mock'."
        )
