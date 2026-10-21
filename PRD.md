# Product Requirements Document (PRD)

## Project Name: VerifyRAG
**Full Academic Title:** VerifyRAG: Claim-Level Citation Verification for Reducing Hallucination in Retrieval-Augmented Generation  
**Domain:** Generative AI, Natural Language Processing, Information Retrieval, Deep Learning  
**Target Milestone:** B.Tech CSE 2nd Year Project / Mini Project  

---

## 1. Executive Summary & Problem Statement

### 1.1 The Problem
Retrieval-Augmented Generation (RAG) is the industry standard for grounding Large Language Models (LLMs) in external proprietary or factual knowledge. However, **standard RAG operates on an open-loop trust assumption**:
1. The retrieval engine fetches relevant context passages.
2. The generator LLM produces an answer.
3. **The Silent Failure:** The generator LLM frequently introduces subtle hallucinations, extralinguistic assumptions, or misattributed facts that appear fully fluent and confident.
4. Existing systems display answers without verifying whether every specific assertion in the generated text is faithfully grounded in the retrieved passages.

### 1.2 The Solution (VerifyRAG)
**VerifyRAG** introduces an automated **claim-level verification and remediation layer** between generation and presentation:
* Decomposes the draft response into atomic, testable claims.
* Runs Natural Language Inference (NLI) using a cross-encoder model to classify each claim against retrieved evidence into **Entailment**, **Neutral**, or **Contradiction**.
* Applies active remediation:
  * **Mode 1 (Filtering):** Removes or visually flags unsupported/contradictory claims.
  * **Mode 2 (Regeneration):** Dynamically re-prompts the LLM with a targeted critique containing only validated premise chunks.
* Emits a final grounded response with interactive claim-level citation badges and quantifiable faithfulness metrics.

---

## 2. Product Goals & Core Objectives

| Objective | Target Metric |
| :--- | :--- |
| **Hallucination Reduction** | Achieve $\ge 30\%$ reduction in unsupported claims compared to Plain RAG baseline. |
| **Citation Precision** | Achieve $\ge 90\%$ precision on claim-to-chunk attribution links. |
| **Latency Budget** | Total verification + repair overhead $< 2.5\text{s}$ on GPU or $< 5\text{s}$ on modern CPU. |
| **Explainability** | $100\%$ of displayed assertions link back to specific source chunk IDs and NLI confidence scores. |
| **Evaluation Framework** | Provide automated comparative benchmarks (Plain RAG vs. VerifyRAG Filtering vs. VerifyRAG Regeneration). |

---

## 3. Target Audience & Stakeholders

1. **B.Tech Evaluation Committee & Project Guide:** Require scientific rigor, quantifiable baselines, mathematical grounding, ablation studies, and clear architectural modularity.
2. **Domain Researchers & Enterprise Knowledge Workers:** Need trustworthy question-answering over dense technical, medical, legal, or financial corpora where hallucination carries high penalties.
3. **Developers / Engineers:** Need an auditable drop-in pipeline for verifiable enterprise RAG workflows.

---

## 4. Feature Specifications

### 4.1 Document Ingestion & Retrieval (RAG Foundation)
* **Document Formats:** PDF, TXT, Markdown.
* **Chunking Strategy:** Recursive character chunking (Chunk size: 500 tokens, Overlap: 50 tokens).
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`.
* **Vector Store:** FAISS (in-memory index) or ChromaDB (persistent vector store).
* **Retrieval Mode:** Top-$k$ semantic similarity search ($k=3 \text{ to } 5$).

### 4.2 Generation Engine
* **Generator LLM:** Modular adapter supporting OpenAI API (GPT-4o-mini), Local LLM via Ollama (Llama-3-8B / Mistral-7B), or Groq API for low-latency testing.
* **Draft Generation:** Produces a comprehensive response based on the top-$k$ retrieved chunks.

### 4.3 Claim Decomposition Engine
* **Decomposition Strategy:** Splits complex compound sentences into atomic propositional claims.
* **Granularity:** Sentence-level or atomic-proposition level (avoiding conjunction-tangled statements so each fact is independently verifiable).

### 4.4 Claim-Level NLI Verification Engine
* **Verification Model:** Cross-encoder NLI model (e.g., `cross-encoder/nli-deberta-v3-base`).
* **Inference Pipeline:**
  * For each claim $c_i$ and each retrieved chunk $e_j$:
    $$\text{Score}(c_i, e_j) = \text{Softmax}(\text{DeBERTa}([e_j; c_i]))$$
  * Classes: `Entailment`, `Neutral`, `Contradiction`.
* **Verdict Decision Boundary:**
  * **Supported (Entailed):** $\max_j P(\text{Entailment} \mid e_j, c_i) \ge \tau_{\text{entail}}$ (default: $0.70$).
  * **Contradicted:** $\max_j P(\text{Contradiction} \mid e_j, c_i) \ge \tau_{\text{contra}}$ (default: $0.60$).
  * **Neutral / Unsupported:** Fails entailment threshold without explicit contradiction.

### 4.5 Remediation Strategies (Ablation / Comparison)
* **Strategy A: Plain RAG (Baseline):** Draft response output directly without post-processing.
* **Strategy B: Filtering:** Automatically strips unsupported/contradicted claims while preserving syntactic flow, or presents them with warning markers.
* **Strategy C: Targeted Regeneration:** Feeds back the failed claims and only the strongly entailed evidence chunks to the generator LLM with a strict grounding constraint prompt.

### 4.6 Interactive Citation & Trust UI
* **Claim Attribution Badges:** Green (Entailed), Yellow (Neutral/Unverified), Red (Contradiction).
* **Chunk Highlighting:** Hovering or clicking a claim highlights the exact source passage that proved or disproved it.
* **Telemetry Display:** Side-by-side latency breakdown (Retrieval ms, Generation ms, Claim Split ms, NLI Verification ms, Remediation ms).

---

## 5. User Personas & User Stories

### Persona 1: Academic Evaluator / Professor
* *"As an evaluator, I want to see a side-by-side benchmark comparing baseline RAG against VerifyRAG across standard metrics (Faithfulness, Citation Recall, Latency) so I can verify the project's scientific validity."*

### Persona 2: Research Student / End-User
* *"As an analyst reviewing research papers, I want to click any claim in the AI's summary and immediately view the exact sentence in the source paper that proves it, so I never act on false information."*

---

## 6. Success Metrics & Evaluation Protocol

1. **Faithfulness Score ($F$):**
   $$F = \frac{\text{Number of Entailed Claims}}{\text{Total Generated Claims}} \times 100$$
2. **Citation Precision ($P_{\text{cite}}$):** Percentage of cited chunks that genuinely entail the associated claim.
3. **Citation Recall ($R_{\text{cite}}$):** Percentage of ground-truth statements from source context retained in the final response.
4. **Latency Overhead ($\Delta T$):**
   $$\Delta T = T_{\text{VerifyRAG}} - T_{\text{PlainRAG}}$$
   Target: $\Delta T \le 2.0\text{s}$ with batched DeBERTa inference.

---

## 7. Out of Scope (For Initial B.Tech Prototype)
* Multimodal retrieval (images, audio, video).
* Real-time web crawler integrations (corpus is fixed to user-uploaded or benchmark documents).
* Multi-hop external graph reasoning beyond vector context.
