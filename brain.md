# brain.md - VerifyRAG AI Context & Project Memory

> **Note for AI Assistants:**  
> This file is the **Single Source of Truth** for the `VerifyRAG` project. Read this file first before inspecting the codebase or generating new modules. It contains the architecture, design choices, data models, coding standards, and active roadmap.

---

## 1. Project Overview & Pitch

* **Project Title:** `VerifyRAG: Claim-Level Citation Verification for Reducing Hallucination in Retrieval-Augmented Generation`
* **Short Description:** A verifiable RAG system that fact-checks its own answers claim-by-claim against retrieved source chunks using Natural Language Inference (NLI) before presenting them to the user.
* **Target Outcome:** Lower the percentage of hallucinated/unsupported claims by $\ge 30\%$ compared to Plain RAG, with auditable citations and sub-2.5s verification latency.

---

## 2. Architectural Mental Model

```text
[User Query + Documents]
         │
         ├──► 1. Semantic Retrieval (SentenceTransformers + FAISS) ──► Top-k Chunks (E_1..E_k)
         │                                                                   │
         ├──► 2. Draft Generation (LLM: OpenAI / Groq / Ollama) ─────────────┼──► Draft Answer
         │                                                                   │          │
         │                                                                   │   3. Claim Splitter (spaCy)
         │                                                                   │          │
         │                                                                   ▼          ▼
         └──► 4. Batched NLI Verification (DeBERTa-v3 Cross-Encoder) ◄─── (Claims c_1..c_n)
                     │
                     ▼
              Verdict Matrix [Entailment / Contradiction / Neutral]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
    5a. Filtering          5b. Regeneration (Critique Prompting)
         │                       │
         └───────────┬───────────┘
                     ▼
         6. Audited Output UI + Telemetry HUD (Streamlit / FastAPI)
```

---

## 3. Technology Stack & Key Dependencies

| Component | Library / Model | Notes |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Strict type annotations throughout. |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight 384-dim dense embeddings. |
| **Vector DB** | `faiss-cpu>=1.8.0` | Fast in-memory L2 / Inner Product search. |
| **Generator LLM** | Modular Adapter (`OpenAI`, `Groq`, `Ollama`) | Default: Groq `llama-3.3-70b-versatile` or OpenAI `gpt-4o-mini`. |
| **Claim Splitter** | `spacy` (`en_core_web_sm`) + Rule Regex | Breaks complex sentences into atomic testable assertions. |
| **NLI Cross-Encoder**| `cross-encoder/nli-deberta-v3-base` | 3 classes: Contradiction (0), Entailment (1), Neutral (2). |
| **Backend API** | `fastapi`, `uvicorn`, `pydantic>=2.0` | Clean REST contracts with async endpoints. |
| **Frontend UI** | `streamlit>=1.35.0` | Clean, interactive research UI with citation highlights. |

---

## 4. Canonical Data Structures (Pydantic v2)

When writing or modifying code in `src/`, always align with these schemas:

```python
from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"

class Chunk(BaseModel):
    chunk_id: int
    text: str
    doc_id: str
    similarity_score: float

class Claim(BaseModel):
    claim_id: int
    text: str
    verdict: VerdictEnum
    best_chunk_id: Optional[int] = None
    confidence: float
    scores: Dict[str, float] = Field(
        default_factory=lambda: {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    )

class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    draft_generation_ms: float
    claim_split_ms: float
    nli_verification_ms: float
    remediation_ms: float
    total_ms: float

class VerificationResult(BaseModel):
    query: str
    mode: str  # "baseline" | "filter" | "regeneration"
    draft_answer: str
    verified_answer: str
    claims: List[Claim]
    retrieved_chunks: List[Chunk]
    faithfulness_score: float
    latency: LatencyBreakdown
```

---

## 5. Coding Standards & AI Guardrails

1. **No Fake / Mock Verification:**  
   Always use real DeBERTa NLI inference in the pipeline. If CUDA is unavailable, run on CPU with vectorized batches (do not run $N \times K$ loops sequentially).
2. **Model Caching:**  
   Load the SentenceTransformer embedding model and DeBERTa cross-encoder as singletons or application state (`@lru_cache` or FastAPI lifespan) to avoid re-initializing weights on every request.
3. **Pydantic Validation:**  
   All request payloads and pipeline responses must strictly validate through Pydantic v2 schemas.
4. **Environment Variables:**  
   Never hardcode API keys or secret tokens. Access them through `src/config.py` using `pydantic-settings` or `python-dotenv`.
5. **Decoupled Remediation:**  
   Keep `remediator.py` completely independent from UI rendering. The remediator accepts claims + chunks and returns clean text; presentation logic belongs in `frontend/`.

---

## 6. Project Roadmap & Implementation Tasks

- [x] **Milestone 0: Project Specifications & Blueprint**
  - [x] `PRD.md` (Product Requirements Document)
  - [x] `TRD.md` (Technical Requirements Document)
  - [x] `UI_UX_DESIGN_BRIEF.md` (Design tokens & layout)
  - [x] `APP_FLOW.md` (User journeys & edge cases)
  - [x] `brain.md` (AI memory & conventions)
- [x] **Milestone 1: Ingestion & Vector Retrieval**
  - [x] Implement `src/ingestion/parser.py` (PDF & TXT parser).
  - [x] Implement `src/ingestion/chunker.py` (Recursive text splitter).
  - [x] Implement `src/ingestion/vector_store.py` (FAISS index wrapper).
- [ ] **Milestone 2: LLM Generation & Claim Decomposition**
  - [ ] Implement `src/pipeline/generator.py` (Pluggable LLM interface).
  - [ ] Implement `src/pipeline/claim_splitter.py` (spaCy atomic splitter).
- [ ] **Milestone 3: NLI Verification & Remediation Engine**
  - [ ] Implement `src/pipeline/verifier.py` (Batched DeBERTa cross-encoder).
  - [ ] Implement `src/pipeline/remediator.py` (Filter and Regeneration modes).
  - [ ] Implement `src/pipeline/orchestrator.py` (Full end-to-end pipeline).
- [ ] **Milestone 4: Interactive Frontend & API**
  - [ ] Implement `src/api/app.py` (FastAPI endpoints).
  - [ ] Implement `frontend/app.py` (Streamlit 3-panel UI with citation badges).
- [ ] **Milestone 5: Benchmark Suite & Presentation Preparation**
  - [ ] Create benchmark dataset with 15 trick questions.
  - [ ] Automated evaluation script comparing Plain RAG vs. Filter vs. Regenerate.
  - [ ] Generate comparative tables and publication graphs.

---

## 7. Quick Commands

```bash
# Setup virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run FastAPI Backend
uvicorn src.api.app:app --reload --port 8000

# Run Streamlit Interactive Demo
streamlit run frontend/app.py

# Run Automated Benchmark Suite
python -m src.evaluation.benchmark
```
