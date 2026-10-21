"""Canonical Pydantic v2 schemas for VerifyRAG.

Single source of truth defined in brain.md Section 4.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VerdictEnum(str, Enum):
    """NLI classification verdict for a claim against evidence chunks."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"


class Chunk(BaseModel):
    """Retrieved context chunk with similarity score and document provenance."""
    chunk_id: int
    text: str
    doc_id: str
    similarity_score: float = 0.0


class Claim(BaseModel):
    """Atomic factual claim decomposed from draft answer and evaluated via NLI."""
    claim_id: int
    text: str
    verdict: VerdictEnum = VerdictEnum.UNSUPPORTED
    best_chunk_id: Optional[int] = None
    confidence: float = 0.0
    scores: Dict[str, float] = Field(
        default_factory=lambda: {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    )


class LatencyBreakdown(BaseModel):
    """Millisecond telemetry for each pipeline execution phase."""
    retrieval_ms: float = 0.0
    draft_generation_ms: float = 0.0
    claim_split_ms: float = 0.0
    nli_verification_ms: float = 0.0
    remediation_ms: float = 0.0
    total_ms: float = 0.0


class VerificationResult(BaseModel):
    """Final output payload containing verified answer, claims, and telemetry."""
    query: str
    mode: str  # "baseline" | "filter" | "regeneration"
    draft_answer: str
    verified_answer: str
    claims: List[Claim] = Field(default_factory=list)
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    faithfulness_score: float = 0.0
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)


class ParsedDocument(BaseModel):
    """Standardized representation of an ingested document."""
    doc_id: str
    filename: str
    content: str
    num_pages: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
