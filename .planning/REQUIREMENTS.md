# Requirements & Scope (VerifyRAG)

## Milestone 1: Ingestion & Vector Retrieval Layer
- [ ] **REQ-M1-01 (Configuration):** Provide centralized Pydantic v2 Settings in `src/config.py` loading thresholds (`tau_entail=0.70`, `tau_contra=0.60`, `top_k=4`), API keys, device selection (`cuda` vs `cpu`), and model identifiers.
- [ ] **REQ-M1-02 (Document Parsing):** Support `.pdf`, `.txt`, and `.md` ingestion via `src/ingestion/parser.py` with multi-page extraction, UTF-8 fallback, whitespace normalization, and deterministic `doc_id`.
- [ ] **REQ-M1-03 (Recursive Chunking):** Implement `src/ingestion/chunker.py` with default `chunk_size=500` characters and `chunk_overlap=50` characters, respecting paragraph and sentence boundaries.
- [ ] **REQ-M1-04 (FAISS Vector Store):** Implement `src/ingestion/vector_store.py` with `IndexFlatIP` (cosine similarity on normalized vectors), singleton `SentenceTransformer` embeddings, top-$k$ search returning canonical `Chunk` models, and pure NumPy fallback for maximum runtime portability.

## Milestone 2: Generation & Claim Decomposition
- [ ] **REQ-M2-01 (LLM Adapter):** Unified interface for Groq (`llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), and Ollama local.
- [ ] **REQ-M2-02 (Claim Splitter):** spaCy-based atomic proposition and sentence decomposition in `src/pipeline/claim_splitter.py`.

## Milestone 3: NLI Verification & Remediation Engine
- [ ] **REQ-M3-01 (Batched NLI Cross-Encoder):** DeBERTa-v3 cross-encoder evaluation over $N \times K$ candidate pairs with vectorization in `src/pipeline/verifier.py`.
- [ ] **REQ-M3-02 (Remediation Logic):** Filtering strategy (flag/strip) and Critique-guided LLM regeneration in `src/pipeline/remediator.py`.
- [ ] **REQ-M3-03 (Pipeline Orchestrator):** Unified end-to-end flow producing canonical `VerificationResult` with latency profiling.

## Milestone 4: Interactive Frontend & API
- [ ] **REQ-M4-01 (FastAPI Endpoints):** REST endpoints for document upload, query verification, and benchmark evaluation.
- [ ] **REQ-M4-02 (Streamlit UI):** 3-panel UI with interactive citation badges, bi-directional highlighting, telemetry HUD, and threshold sliders.

## Milestone 5: Benchmark Suite & Evaluation
- [ ] **REQ-M5-01 (Benchmark Dataset):** 15 curated trick questions designed to induce hallucinations.
- [ ] **REQ-M5-02 (Automated Benchmark Runner):** Tri-mode evaluation comparing Plain RAG vs. Filter vs. Regenerate across Faithfulness, Citation Precision, and Latency.
