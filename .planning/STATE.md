# Project State (VerifyRAG)

## Current Status
- **Active Milestone:** Milestone 2 (Completed) -> Transitioning to Milestone 3
- **Completed Milestones:**
  - Milestone 1 (Ingestion & Vector Retrieval Layer)
  - Milestone 2 (Generation & Claim Decomposition)
- **Next Phase:** Milestone 3 (NLI Verification & Remediation Engine)

## Key Deliverables Implemented

### Milestone 1: Ingestion & Vector Retrieval
- `src/schemas.py`: Canonical Pydantic v2 schemas (`VerdictEnum`, `Chunk`, `Claim`, `LatencyBreakdown`, `VerificationResult`, `ParsedDocument`).
- `src/config.py`: Centralized settings with Pydantic v2 BaseSettings, thresholds, model names, and CUDA auto-detection.
- `src/ingestion/parser.py`: Multi-format parser for PDF, TXT, MD with encoding fallbacks and whitespace normalization.
- `src/ingestion/chunker.py`: Recursive character text splitter with 500-char size and 50-char overlap.
- `src/ingestion/vector_store.py`: In-memory dense vector store supporting FAISS IndexFlatIP (cosine similarity) and vectorized NumPy fallback with singleton embedding caching.
- `src/ingestion/demo_ingest.py`: End-to-end CLI demonstration of parsing, chunking, indexing, and semantic retrieval.
- `tests/test_milestone_1.py`: 11/11 unit and integration tests passing.

### Milestone 2: Generation & Claim Decomposition
- `src/pipeline/generator.py`: Pluggable generator adapter supporting Groq (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), Ollama (`llama3:8b`), and deterministic `MockGenerator` with prompt-grounding template and latency telemetry.
- `src/pipeline/claim_splitter.py`: Atomic proposition decomposition engine with punctuation masking for abbreviations/decimals, markdown list normalization, clause splitting on conjunctions, and conversational noise stripping.
- `src/pipeline/demo_pipeline_m2.py`: Interactive CLI demonstrating full flow from evidence retrieval to draft answer generation and atomic claim table.
- `tests/test_milestone_2.py`: 14/14 unit and integration tests passing (25/25 total suite passing).

## Progress Log
- **2026-10-21:** Milestone 0 blueprints loaded and GSD planning directory initialized.
- **2026-10-21:** Milestone 1 planned (`01-01-PLAN.md`) and executed across all 4 waves (11/11 tests passing).
- **2026-10-21:** Remote GitHub repository linked and rebased cleanly with origin/main.
- **2026-10-21:** Milestone 2 planned (`02-01-PLAN.md`) and executed across all 3 waves (25/25 tests passing).
