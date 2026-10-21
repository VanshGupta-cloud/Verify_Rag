# SYNOPSIS
ON
### “VerifyRAG: Claim-Level Citation Verification for Reducing Hallucination in Retrieval-Augmented Generation”

**Submitted in Partial Fulfillment of the requirements for the Award of the Degree of**  
**Bachelor of Technology**  
**In**  
**Computer Science and Engineering**  

by  
**(Project Id: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;)**  
**Vansh Gupta (2501640101006)**  

Under the supervision of  
**Mohd Shuaib Zahid**  
(Assistant Professor)  

---

**Department of Computer Science and Engineering**  
**Pranveer Singh Institute of Technology**  
Kanpur - Agra - Delhi National Highway - 19, Bhauti - Kanpur - 209305  

---

## 1. Introduction

With the rapid emergence of Generative Artificial Intelligence (GenAI) and Large Language Models (LLMs), natural language interfaces have become foundational tools for automated question answering and data synthesis. Despite their fluency, LLMs frequently suffer from *hallucination*—the generation of factually incorrect, unverifiable, or contradictory assertions presented with undue confidence. Retrieval-Augmented Generation (RAG) was introduced to alleviate this vulnerability by grounding generation in external, domain-specific text corpora via dense vector retrieval. 

However, conventional RAG systems operate under an unverified, open-loop assumption: context chunks are retrieved, the LLM generates a synthesized draft, and the resulting response is directly presented to the user without validating whether the LLM faithfully adhered to the retrieved premises. In high-stakes applications—such as legal research, biomedical inquiry, and academic study—even minor extrinsic hallucinations can lead to catastrophic misinformation.

**VerifyRAG** proposes an automated, claim-level verification and remediation framework that fact-checks generated answers against retrieved source passages prior to presentation. The system decomposes complex generated responses into atomic, independently testable propositions and scores each claim using a cross-encoder Natural Language Inference (NLI) model. Based on semantic entailment probabilities, unsupported or contradicted statements are either actively filtered or systematically regenerated through critique-guided prompt synthesis.

### Key Technical Terms:
* **Retrieval-Augmented Generation (RAG):** A hybrid architecture combining non-parametric dense retrieval (vector database) with parametric generation (LLMs).
* **Hallucination:** Extrinsic or intrinsic fabrications where generated text deviates from ground-truth source evidence.
* **Natural Language Inference (NLI):** The task of determining whether a hypothesis is logically supported (*Entailment*), refuted (*Contradiction*), or undetermined (*Neutral*) given a premise text.
* **Cross-Encoder Architecture:** A deep transformer model that processes premise and hypothesis jointly via full self-attention, offering superior classification precision over bi-encoder similarity.
* **Atomic Claim Decomposition:** Syntactic and semantic deconstruction of compound sentences into single-assertion factual statements.

---

## 2. Project Objective

The primary objective of **VerifyRAG** is to develop an auditable, verifiable, and hallucination-resistant retrieval-augmented generation pipeline. The explicit project goals are:

1. **Develop an End-to-End Grounded Retrieval Pipeline:** Implement an efficient document ingestion, recursive text chunking, and dense vector indexing pipeline using Sentence-Transformers and FAISS.
2. **Implement Propositional Claim Decomposition:** Build an automated claim-splitting engine using linguistic sentence tokenization and syntactic parsing to isolate compound statements into atomic assertions.
3. **Integrate Cross-Encoder NLI Verification:** Incorporate a high-performance cross-encoder model (`cross-encoder/nli-deberta-v3-base`) to compute pairwise entailment probabilities between extracted claims and retrieved context chunks.
4. **Deploy Dual Remediation Strategies:** 
   * *Filtering Strategy:* Dynamically suppress or visually flag claims classified as neutral or contradictory.
   * *Regeneration Strategy:* Re-synthesize the response by feeding verified premises and critique constraints back into the generator LLM.
5. **Quantifiable Hallucination Reduction:** Achieve a measurable reduction ($\ge 30\%$) in unsupported claims compared to the standard Plain RAG baseline while sustaining an acceptable latency overhead ($< 2.5\text{ seconds}$).
6. **Deliver an Interactive Demonstration Interface:** Provide an interactive, explainable Streamlit dashboard featuring claim-level citation badges, bi-directional evidence highlighting, and real-time pipeline latency telemetry.

---

## 3. Feasibility Study

### 3.1 Technical Feasibility
The technical infrastructure required for VerifyRAG is thoroughly viable on contemporary deep learning workstations. Dense retrieval is accelerated via Facebook AI Similarity Search (FAISS), enabling sub-50ms vector lookups. Cross-encoder verification utilizes the pre-trained DeBERTa-v3 architecture, which provides state-of-the-art inference on benchmark fact-checking datasets (MNLI, FEVER). By leveraging an NVIDIA GeForce RTX 3050 GPU (4 GB VRAM) with half-precision floating-point (FP16) tensor operations and batched inference, the $N \times K$ cross-attention scoring executes with low computational overhead. Generator LLM adapters (via Groq API or quantized local Ollama instances) ensure reliable, high-throughput text synthesis.

### 3.2 Operational Feasibility
From an operational perspective, the system demands no specialized technical training from the end-user. The application provides an intuitive web interface where users upload standard documents (`.pdf`, `.txt`) and submit queries in plain English. Responses are accompanied by color-coded verification badges and interactive citation pills, allowing researchers, students, and professionals to inspect source citations with single-click precision.

### 3.3 Economic Feasibility
VerifyRAG relies entirely on open-source frameworks, open-access pre-trained weights from Hugging Face, and local vector indexing libraries. The project eliminates costly proprietary cloud infrastructure by running inference locally on consumer-grade hardware. External LLM queries are optimized through lightweight model APIs or fully offline quantized open-weights models (e.g., Llama-3-8B), incurring negligible development and operational expenditures.

### 3.4 Schedule Feasibility
The project development lifecycle is structured across five sequential milestones:
* *Month 1:* Literature review, requirement finalization, document parsing, and FAISS vector index integration.
* *Month 2:* LLM draft generation, atomic claim extraction module, and DeBERTa cross-encoder verification pipeline.
* *Month 3:* Remediation algorithms (filtering and critique-guided regeneration) and Streamlit user interface development.
* *Month 4:* Benchmark evaluation, baseline comparative analysis (Plain RAG vs. VerifyRAG), report drafting, and final presentation.  
This timeline fits comfortably within the academic semester schedule.

### 3.5 Legal Feasibility
All third-party libraries, including PyTorch, Hugging Face Transformers, FAISS, and Streamlit, are distributed under permissive open-source licenses (MIT and Apache 2.0). The underlying datasets and benchmark documents comprise public-domain research literature, adhering strictly to copyright, fair use, and academic integrity policies.

---

## 4. Methodology / Planning of Work

The engineering methodology of VerifyRAG is designed as a closed-loop verification architecture divided into four coordinated subsystems:

### 4.1 Subsystem 1: Document Indexing & Dense Semantic Retrieval
Unstructured knowledge documents (PDF, TXT, Markdown) are parsed and partitioned using recursive character boundary chunking with a segment length of $500$ characters and a $50$-character sliding overlap. Each text chunk $e_j$ is projected into a $384$-dimensional continuous latent space using the `sentence-transformers/all-MiniLM-L6-v2` embedding model. The vectors are indexed using Facebook AI Similarity Search (FAISS) with inner-product cosine similarity. When a user submits query $q$, the vectorized query $\mathbf{v}_q$ performs a top-$k$ nearest-neighbor search to retrieve the most semantically relevant evidence passages $E = \{e_1, e_2, \dots, e_k\}$ (default $k=4$) within sub-50ms latency.

### 4.2 Subsystem 2: Contextual Generation & Propositional Decomposition
The generator LLM (Groq Llama-3.3-70B or OpenAI GPT-4o-mini) receives an augmented prompt containing query $q$ and evidence passages $E$ to synthesize a preliminary draft answer $D$. Because standard LLM generation is prone to hallucination, $D$ is not shown directly to the user. Instead, $D$ is routed to the Claim Decomposition Engine, which uses spaCy's dependency parser and rule-based clausal segmentation to decompose compound sentences into a discrete set of atomic, independently testable propositions $C = \{c_1, c_2, \dots, c_n\}$. Each claim encapsulates a single factual assertion.

### 4.3 Subsystem 3: Batched Cross-Encoder NLI Verification Engine
To fact-check claims against retrieved context with maximum semantic precision, VerifyRAG utilizes `cross-encoder/nli-deberta-v3-base` rather than bi-encoder cosine distance. A Cartesian candidate pairing layer constructs all candidate combinations $\mathcal{P} = \{(c_i, e_j) \mid 1 \le i \le n, 1 \le j \le k\}$. For an answer of $5$ claims and $4$ retrieved chunks, the resulting $20$ pairs are vectorized into a single batch (batch size $\le 32$) and evaluated concurrently on the GPU using PyTorch FP16 tensor operations.

The cross-encoder performs bidirectional joint cross-attention over sequence $[\text{CLS}] \, e_j \, [\text{SEP}] \, c_i \, [\text{SEP}]$, generating logit vector $\mathbf{z}_{ij} \in \mathbb{R}^3$. A Softmax transformation yields class probabilities for Contradiction, Neutral, and Entailment:
$$\mathbf{p}_{ij} = \text{Softmax}(\mathbf{z}_{ij}) = \big[ P(\text{Contradiction}), P(\text{Neutral}), P(\text{Entailment}) \big]$$
For each claim $c_i$, max-pooling over all evidence chunks determines the strongest supporting chunk $j^*$ and peak entailment probability:
$$\hat{E}_i = \max_{j \in [1, k]} P(\text{Entailment} \mid e_j, c_i), \quad j^* = \arg\max_{j \in [1, k]} P(\text{Entailment} \mid e_j, c_i)$$
A decision engine classifies the claim as:
* **SUPPORTED:** if $\hat{E}_i \ge \tau_{\text{entail}}$ (default: $0.70$), attributing citation to chunk $j^*$.
* **CONTRADICTED:** else if $\max_j P(\text{Contradiction} \mid e_j, c_i) \ge \tau_{\text{contra}}$ (default: $0.60$).
* **UNSUPPORTED (Neutral):** otherwise.

### 4.4 Subsystem 4: Dual Remediation & Attributed Presentation
Once claims are classified, VerifyRAG executes one of two remediation pathways before displaying the final response on the Streamlit dashboard:
* **Remediation Strategy A (Active Filtering):** Unsupported and neutral claims are systematically purged from the response text, while contradictory statements are either redacted or accompanied by inline warning callouts. The remaining supported claims are re-concatenated with grammatical transitions.
* **Remediation Strategy B (Critique-Guided Regeneration):** The system isolates strictly verified chunks $E_{\text{verified}}$ and feeds a structured critique constraint back to the generator LLM: *"Rewrite the answer using ONLY the verified facts from the provided sources; do not assume or introduce unverified details."* The LLM re-synthesizes a concise, 100% grounded response.

The final output renders on the web interface with interactive citation pills (e.g., `[Chunk 2] (✓)`), bi-directional hover highlighting connecting text to source chunks, and an automated Telemetry HUD tracking Faithfulness Score, Hallucination Suppression Rate, and component latency breakdown.

---

## 5. Tools / Technology Used

### 5.1 Minimum Hardware Requirements
* **Processor (CPU):** Intel Core i5 (11th Gen / 12th Gen equivalent, Multi-core $\ge 2.5\text{ GHz}$)
* **System Memory (RAM):** 16 GB DDR4 / DDR5
* **Graphics Processing Unit (GPU):** NVIDIA GeForce RTX 3050 (4 GB GDDR6 VRAM, CUDA Compute capability 8.6)
* **Storage (HDD/SSD):** 512 GB NVMe M.2 Solid State Drive
* **Peripheral / Connectivity:** Stable Broadband Internet connectivity (for model repository weights and API transactions)

### 5.2 Minimum Software Requirements
* **Operating System:** Microsoft Windows 11 (64-bit) / Ubuntu Linux 22.04 LTS
* **Programming Language:** Python 3.10.x / 3.11.x
* **Deep Learning Framework:** PyTorch 2.2.0 with CUDA 12.1 acceleration
* **NLP & Model Hub:** Hugging Face `transformers` (v4.40+), `sentence-transformers` (v2.7+)
* **NLI Inference Model:** `cross-encoder/nli-deberta-v3-base`
* **Vector Search Library:** `faiss-cpu` / `faiss-gpu` (v1.8.0+)
* **Text Processing Tools:** `spacy` (v3.7+, `en_core_web_sm`), `pypdf` (v4.2+)
* **Web Presentation & Dashboard:** `streamlit` (v1.35+)
* **Development Environment (IDE):** Antigravity IDE / Visual Studio Code

---

## 6. References [IEEE Format]

1. P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.
2. P. He, X. Liu, J. Gao, and W. Chen, "DeBERTa: Decoding-enhanced BERT with Disentangled Attention," in *Proc. 9th Int. Conf. Learn. Represent. (ICLR)*, Virtual Event, 2021.
3. J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, "FEVER: A Large-scale Dataset for Fact Extraction and VERification," in *Proc. 2018 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT)*, New Orleans, LA, pp. 809–819, 2018.
4. S. Min, K. Krishna, X. Lyu, M. Lewis, W. Yih, P. Kohli, S. Iyyer, L. Zettlemoyer, and H. Hajishirzi, "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," in *Proc. 2023 Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, Singapore, pp. 12076–12100, 2023.
5. Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang, "Retrieval-Augmented Generation for Large Language Models: A Survey," *arXiv preprint arXiv:2312.10997*, 2023.
6. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," in *Thirty-fifth Conf. Neural Inf. Process. Syst. Datasets and Benchmarks Track*, 2021.
