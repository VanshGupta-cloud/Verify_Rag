# VerifyRAG

**Claim-Level Citation Verification for Reducing Hallucination in Retrieval-Augmented Generation**

## Executive Summary
Standard RAG pipelines assume generator LLMs remain faithful to retrieved context. In practice, LLMs hallucinate plausible, fluent, yet unsupported assertions. **VerifyRAG** introduces an automated claim-level verification and remediation layer between generation and presentation:
1. Decomposes draft responses into atomic, testable claims.
2. Performs Natural Language Inference (NLI) with a cross-encoder model (`cross-encoder/nli-deberta-v3-base`) against retrieved chunks.
3. Classifies claims into **SUPPORTED (Entailment)**, **UNSUPPORTED (Neutral)**, or **CONTRADICTED**.
4. Applies active remediation:
   - **Mode 1 (Filtering):** Strips or flags unsupported/contradictory claims.
   - **Mode 2 (Regeneration):** Dynamically re-prompts the LLM with a targeted critique containing only validated premise chunks.
5. Emits an audited response with interactive claim-level citation badges and quantifiable faithfulness metrics in Streamlit.

## Objectives & Success Criteria
- **Hallucination Reduction:** $\ge 30\%$ reduction in unsupported claims compared to Plain RAG.
- **Citation Precision:** $\ge 90\%$ precision on claim-to-chunk attribution links.
- **Latency Budget:** Total verification + repair overhead $< 2.5\text{s}$ with batched GPU inference.
- **Explainability:** $100\%$ of displayed assertions link back to specific source chunk IDs and NLI confidence scores.

## Architecture Stack
- **Language & Runtime:** Python 3.10+, Windows 11 64-bit, PyTorch CUDA FP16
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector Index:** FAISS in-memory index (`IndexFlatIP` with normalized vectors)
- **Generator LLM:** Modular adapter (Groq `llama-3.3-70b-versatile`, OpenAI `gpt-4o-mini`, Ollama local)
- **Claim Splitter:** `spacy` (`en_core_web_sm`) + rule-based proposition decomposition
- **NLI Cross-Encoder:** `cross-encoder/nli-deberta-v3-base`
- **Backend API:** FastAPI with Pydantic v2 schemas
- **Frontend UI:** Streamlit interactive 3-panel UI with citation pills and latency HUD
