"""Configuration and Settings for VerifyRAG.

Loads environment variables, default model paths, verification thresholds,
and hardware acceleration preferences.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_default_device() -> Literal["cuda", "cpu"]:
    """Detect whether CUDA is available on current system."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except (ImportError, Exception):
        pass
    return "cpu"


class Settings(BaseSettings):
    """Central configuration for VerifyRAG pipeline, API, and models."""

    # Project Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # LLM Settings & Keys
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    llm_provider: Literal["groq", "openai", "ollama"] = Field(
        default="groq", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="llama-3.3-70b-versatile", alias="LLM_MODEL")

    # Ingestion & Chunking Settings
    chunk_size: int = Field(default=500, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between consecutive chunks")
    top_k_chunks: int = Field(default=4, alias="TOP_K_CHUNKS")

    # Embedding Model
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME"
    )
    embedding_dim: int = Field(default=384, description="Embedding dimension for all-MiniLM-L6-v2")

    # NLI Cross-Encoder Model
    nli_model_name: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        alias="NLI_MODEL_NAME"
    )

    # Verification Decision Boundaries (from brain.md & PRD.md)
    entailment_threshold: float = Field(
        default=0.70,
        alias="ENTAILMENT_THRESHOLD",
        description="Minimum entailment probability to mark a claim as SUPPORTED"
    )
    contradiction_threshold: float = Field(
        default=0.60,
        alias="CONTRADICTION_THRESHOLD",
        description="Minimum contradiction probability to mark a claim as CONTRADICTED"
    )

    # Hardware & Acceleration
    device: Literal["cuda", "cpu"] = Field(
        default_factory=_detect_default_device,
        alias="DEVICE"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton cached provider for application settings."""
    return Settings()
