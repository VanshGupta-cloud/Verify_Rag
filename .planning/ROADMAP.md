# Roadmap (VerifyRAG)

## Phase 0: Project Specifications & Blueprint (Completed)
- [x] PRD, TRD, UI/UX Design Brief, APP Flow, brain.md established in root.

## Phase 1: Ingestion & Vector Retrieval Layer (Completed)
- [x] Plan 01-01: Milestone 1 Ingestion Pipeline
  - [x] Task 1: `src/config.py` - Pydantic settings, model paths, thresholds, device resolver.
  - [x] Task 2: `src/ingestion/parser.py` - Multi-format extractor for PDF, TXT, MD.
  - [x] Task 3: `src/ingestion/chunker.py` - Recursive character chunker (500 chars / 50 overlap).
  - [x] Task 4: `src/ingestion/vector_store.py` - FAISS dense index with SentenceTransformers & canonical `Chunk` return.
  - [x] Task 5: Integration test verifying parser -> chunker -> vector_store -> top-k retrieval (11/11 tests passing).

## Phase 2: Generation & Claim Decomposition (Completed)
- [x] Plan 02-01: Milestone 2 Generation & Claim Decomposition
  - [x] Task 1: Modular LLM generation adapter (`src/pipeline/generator.py`).
  - [x] Task 2: Atomic claim decomposition engine (`src/pipeline/claim_splitter.py`).
  - [x] Task 3: Interactive CLI demo (`src/pipeline/demo_pipeline_m2.py`).
  - [x] Task 4: Unit & integration test suite (`tests/test_milestone_2.py` - 14/14 passing).

## Phase 3: NLI Verification & Remediation Engine (Next)
- [ ] Batched DeBERTa cross-encoder verifier (`src/pipeline/verifier.py`).
- [ ] Remediation algorithms (`src/pipeline/remediator.py`).
- [ ] Pipeline orchestrator (`src/pipeline/orchestrator.py`).

## Phase 4: Interactive Frontend & API
- [ ] FastAPI backend (`src/api/app.py` & schemas).
- [ ] Streamlit 3-panel UI (`frontend/app.py`).

## Phase 5: Benchmark Suite & Presentation Preparation
- [ ] Curated evaluation corpus (`data/evaluation/`).
- [ ] Baseline comparison evaluation script (`src/evaluation/benchmark.py`).
