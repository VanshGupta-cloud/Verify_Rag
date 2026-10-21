import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import os

src_template = r"C:\Users\Vansh Gupta\.gemini\antigravity\brain\16ac2cf2-2f4f-4b71-b2ee-c3d1e4e82e54\.user_uploaded\media_1792417391621_bd0e5c64.docx"
out_docx = r"C:\Users\Vansh Gupta\.gemini\antigravity\scratch\VerifyRAG\VerifyRAG_Synopsis.docx"
out_pdf = r"C:\Users\Vansh Gupta\.gemini\antigravity\scratch\VerifyRAG\VerifyRAG_Synopsis.pdf"

doc = docx.Document(src_template)

# --- 1. UPDATE COVER PAGE ---
# Paragraph 0: 'SYNOPSIS'
# Paragraph 1: 'ON'
# Paragraph 2: Project Title
p2 = doc.paragraphs[2]
p2.text = "“VerifyRAG: Claim-Level Citation Verification for Reducing Hallucination in Retrieval-Augmented Generation”"
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p2.runs:
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True

# Paragraph 14: Student Name (Roll No)
p14 = doc.paragraphs[14]
p14.text = "Vansh Gupta (2501640101006)"
p14.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p14.runs:
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

# Paragraph 17: Supervisor Name
p17 = doc.paragraphs[17]
p17.text = "Mohd Shuaib Zahid"
p17.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p17.runs:
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True

# Paragraph 18: Designation
p18 = doc.paragraphs[18]
p18.text = "(Assistant Professor)"
p18.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in p18.runs:
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

# Ensure all cover page text has Times New Roman
for i in range(24):
    p = doc.paragraphs[i]
    for r in p.runs:
        r.font.name = "Times New Roman"

# Ensure page break after cover page (at paragraph 24)
p24 = doc.paragraphs[24]
p24.text = ""

# --- 2. REMOVE PLACEHOLDER TEXT (PARAGRAPHS 25 ONWARDS) ---
paras_to_remove = doc.paragraphs[25:]
for p in paras_to_remove:
    p._p.getparent().remove(p._p)

# Helper function to add page break
def add_page_break():
    p = doc.add_paragraph()
    r = p.add_run()
    r.add_break(docx.enum.text.WD_BREAK.PAGE)

# Add page break right after cover page
add_page_break()

# Helper formatting functions matching PSIT Guidelines
def add_sec_heading(title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(14)
    r.font.bold = True
    return p

def add_sub_heading(title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.font.bold = True
    return p

def add_body_p(text, indent=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(6)
    if indent:
        p.paragraph_format.left_indent = Inches(0.25)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    return p

def add_bullet_item(prefix, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.space_after = Pt(4)
    r_bold = p.add_run(prefix + " ")
    r_bold.font.name = "Times New Roman"
    r_bold.font.size = Pt(12)
    r_bold.font.bold = True
    r_text = p.add_run(text)
    r_text.font.name = "Times New Roman"
    r_text.font.size = Pt(12)
    return p

def add_diagram_block(ascii_art):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.1)
    r = p.add_run(ascii_art)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    return p

# --- 3. SECTION 1: INTRODUCTION ---
add_sec_heading("1. Introduction")
add_body_p(
    "With the rapid emergence of Generative Artificial Intelligence (GenAI) and Large Language Models (LLMs), natural language interfaces have become foundational tools for automated question answering, data synthesis, and technical reasoning. Despite their remarkable fluency, LLMs frequently suffer from hallucination—the generation of factually unsupported, extralinguistic, or contradictory assertions presented with undue statistical confidence. Retrieval-Augmented Generation (RAG) was introduced to alleviate this vulnerability by grounding text generation in external, domain-specific corpora via dense vector retrieval."
)
add_body_p(
    "However, conventional RAG systems operate under an unverified, open-loop assumption: context passages are retrieved, the LLM produces an answer, and the synthesized response is presented directly to the user without validating whether every constituent statement is faithfully grounded in the retrieved context. In high-stakes environments—such as legal analysis, medical inquiry, financial audits, and scientific research—even minor hallucinations can yield catastrophic misinformation."
)
add_body_p(
    "VerifyRAG proposes an automated, claim-level verification and remediation framework that fact-checks generated answers against retrieved source passages prior to presentation. The system decomposes complex responses into atomic propositions, scores each claim using a cross-encoder Natural Language Inference (NLI) model, and remediates unsupported claims via active filtering or critique-guided regeneration."
)

add_sub_heading("Special Technical Terms:")
add_bullet_item("• Retrieval-Augmented Generation (RAG):", "A hybrid AI architecture combining non-parametric dense retrieval (vector database) with parametric generation (LLM).")
add_bullet_item("• Hallucination:", "Extrinsic or intrinsic fabrications where generated text diverges from factual ground-truth source evidence.")
add_bullet_item("• Natural Language Inference (NLI):", "The classification task of determining whether a hypothesis is logically Entailed, Contradicted, or Neutral given a premise text.")
add_bullet_item("• Cross-Encoder Architecture:", "A deep transformer architecture that processes premise and hypothesis jointly via full cross-attention, providing superior verification precision over bi-encoder similarity.")
add_bullet_item("• Atomic Claim Decomposition:", "Syntactic deconstruction of compound statements into discrete, independently testable factual propositions.")

# --- 4. SECTION 2: PROJECT OBJECTIVE ---
add_sec_heading("2. Project Objective")
add_body_p(
    "The primary objective of VerifyRAG is to construct a verifiable, transparent, and hallucination-resistant retrieval-augmented generation framework. The specific project objectives are:"
)
add_bullet_item("1.", "Develop an end-to-end grounded retrieval pipeline using recursive text chunking and dense vector indexing with FAISS.")
add_bullet_item("2.", "Implement propositional claim decomposition to split compound generated sentences into atomic, testable assertions using linguistic parsers.")
add_bullet_item("3.", "Integrate a cross-encoder NLI model (cross-encoder/nli-deberta-v3-base) to classify each claim against retrieved evidence chunks into Entailment, Neutral, or Contradiction.")
add_bullet_item("4.", "Implement dual post-verification remediation strategies: (a) Filtering Mode to purge or flag unverified claims, and (b) Regeneration Mode to re-synthesize responses using strictly verified premises.")
add_bullet_item("5.", "Achieve a quantifiable reduction (≥ 30%) in unsupported claims compared to baseline Plain RAG, while sustaining an acceptable verification latency overhead (< 2.5 seconds).")
add_bullet_item("6.", "Deliver an interactive, explainable Streamlit demonstration dashboard featuring claim-level citation badges, bi-directional evidence highlighting, and real-time latency telemetry.")

# --- 5. SECTION 3: FEASIBILITY STUDY ---
add_sec_heading("3. Feasibility Study")
add_body_p(
    "A comprehensive feasibility study was conducted to analyze the technical, operational, economic, schedule, and legal viability of the proposed VerifyRAG system:"
)

add_sub_heading("3.1 Technical Feasibility")
add_body_p(
    "The project is technically feasible using current hardware and open-source deep learning frameworks. Dense retrieval is performed using Facebook AI Similarity Search (FAISS), which completes top-k vector searches in under 50ms. Cross-encoder verification is powered by the pre-trained DeBERTa-v3 architecture, renowned for state-of-the-art accuracy on NLI and fact-checking benchmarks (MNLI, FEVER). By leveraging an NVIDIA GeForce RTX 3050 GPU (4 GB VRAM) with CUDA acceleration, half-precision (FP16) tensor operations, and batched matrix inference, the cross-attention verification of all (claim, chunk) pairs executes with low latency. High-throughput LLM adapters (via Groq API or quantized local Ollama instances) ensure reliable generation."
)

add_sub_heading("3.2 Operational Feasibility")
add_body_p(
    "VerifyRAG is designed with high operational usability. The end-user does not require specialized machine learning knowledge to interact with the system. Users upload standard documents (PDF, TXT) through a clean web interface and type queries in plain language. Every generated answer is augmented with color-coded verification badges and interactive citation pills. Single-click inspection allows users and evaluators to immediately examine the exact source passage that substantiated each claim."
)

add_sub_heading("3.3 Economic Feasibility")
add_body_p(
    "The project exhibits complete economic feasibility as it relies strictly on open-source software and open-access model weights available via Hugging Face. Development, indexing, and verification run locally on existing consumer hardware without requiring paid enterprise cloud clusters. External LLM querying utilizes free and ultra-low-cost API tiers (such as Groq or OpenAI gpt-4o-mini) or fully free offline local models, resulting in negligible financial expenditure."
)

add_sub_heading("3.4 Schedule Feasibility")
add_body_p(
    "The development lifecycle is divided into structured milestones matching the academic semester schedule: Month 1 focuses on document parsing, chunking, and FAISS vector indexing; Month 2 covers LLM draft generation, atomic claim splitting, and DeBERTa cross-encoder integration; Month 3 implements filtering, regeneration algorithms, and the Streamlit interface; Month 4 is dedicated to benchmark evaluation, comparative baseline analysis, and project documentation."
)

add_sub_heading("3.5 Legal Feasibility")
add_body_p(
    "VerifyRAG complies fully with legal and licensing standards. All underlying libraries (PyTorch, Hugging Face Transformers, FAISS, Streamlit) are licensed under permissive open-source licenses (MIT and Apache 2.0). All benchmark evaluation datasets and test documents utilize public-domain academic publications, honoring copyright laws, fair-use doctrines, and institutional academic ethics."
)

# --- 6. SECTION 4: METHODOLOGY / PLANNING OF WORK ---
add_sec_heading("4. Methodology/ Planning of work")
add_body_p(
    "The engineering methodology of VerifyRAG is designed as a closed-loop verification architecture divided into four coordinated subsystems. The dataflow transitions from raw corpus indexing to atomic claim decomposition, batched cross-encoder inference, and dual-strategy remediation, as shown in Figure 1."
)

# Insert the high-resolution architecture diagram
img_path = r"C:\Users\Vansh Gupta\.gemini\antigravity\scratch\VerifyRAG\architecture_diagram.png"
if os.path.exists(img_path):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(8)
    p_img.paragraph_format.space_after = Pt(4)
    run_img = p_img.add_run()
    run_img.add_picture(img_path, width=Inches(6.4))
    
    # Caption
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    r_cap_lbl = p_cap.add_run("Figure 1: ")
    r_cap_lbl.font.name = "Times New Roman"
    r_cap_lbl.font.size = Pt(10.5)
    r_cap_lbl.font.bold = True
    r_cap_txt = p_cap.add_run("End-to-end system architecture of VerifyRAG illustrating document indexing, generation, batched cross-encoder NLI verification, and dual remediation strategies.")
    r_cap_txt.font.name = "Times New Roman"
    r_cap_txt.font.size = Pt(10.5)
    r_cap_txt.font.italic = True

add_sub_heading("4.1 Subsystem 1: Document Indexing & Dense Semantic Retrieval")
add_body_p(
    "Unstructured knowledge documents (PDF, TXT, Markdown) are parsed and partitioned using recursive character boundary chunking with a segment length of 500 characters and a 50-character sliding overlap. Each text chunk e_j is projected into a 384-dimensional continuous latent space using the sentence-transformers/all-MiniLM-L6-v2 embedding model. The vectors are indexed using Facebook AI Similarity Search (FAISS) with inner-product cosine similarity. When a user submits query q, the vectorized query v_q performs a top-k nearest-neighbor search to retrieve the most semantically relevant evidence passages E = {e_1, e_2, ..., e_k} (default k=4) within sub-50ms latency."
)

add_sub_heading("4.2 Subsystem 2: Contextual Generation & Propositional Decomposition")
add_body_p(
    "The generator LLM (Groq Llama-3.3-70B or OpenAI GPT-4o-mini) receives an augmented prompt containing query q and evidence passages E to synthesize a preliminary draft answer D. Because standard LLM generation is prone to hallucination, D is not shown directly to the user. Instead, D is routed to the Claim Decomposition Engine, which uses spaCy's dependency parser and rule-based clausal segmentation to decompose compound sentences into a discrete set of atomic, independently testable propositions C = {c_1, c_2, ..., c_n}. Each claim encapsulates a single factual assertion."
)

add_sub_heading("4.3 Subsystem 3: Batched Cross-Encoder NLI Verification Engine")
add_body_p(
    "To fact-check claims against retrieved context with maximum semantic precision, VerifyRAG utilizes cross-encoder/nli-deberta-v3-base rather than bi-encoder cosine distance. A Cartesian candidate pairing layer constructs all candidate combinations P = {(c_i, e_j) | 1 <= i <= n, 1 <= j <= k}. For an answer of 5 claims and 4 retrieved chunks, the resulting 20 pairs are vectorized into a single batch (batch size <= 32) and evaluated concurrently on the GPU using PyTorch FP16 tensor operations."
)
add_body_p(
    "The cross-encoder performs bidirectional joint cross-attention over sequence [CLS] e_j [SEP] c_i [SEP], generating logit vector z_ij in R^3. A Softmax transformation yields class probabilities for Contradiction, Neutral, and Entailment. For each claim c_i, max-pooling over all evidence chunks determines the strongest supporting chunk j* and peak entailment probability: E_i = max_j P(Entailment | e_j, c_i). A decision engine classifies the claim as SUPPORTED if E_i >= 0.70, CONTRADICTED if max_j P(Contradiction) >= 0.60, or UNSUPPORTED (Neutral) otherwise."
)

add_sub_heading("4.4 Subsystem 4: Dual Remediation & Attributed Presentation")
add_body_p(
    "Once claims are classified, VerifyRAG executes one of two remediation pathways before displaying the final response on the Streamlit dashboard:"
)
add_bullet_item("• Remediation Strategy A (Active Filtering):", "Unsupported and neutral claims are systematically purged from the response text, while contradictory statements are either redacted or accompanied by inline warning callouts. The remaining supported claims are re-concatenated with grammatical transitions.")
add_bullet_item("• Remediation Strategy B (Critique-Guided Regeneration):", "The system isolates strictly verified chunks E_verified and feeds a structured critique constraint back to the generator LLM: 'Rewrite the answer using ONLY the verified facts from the provided sources; do not assume or introduce unverified details.' The LLM re-synthesizes a concise, 100% grounded response.")
add_body_p(
    "The final output renders on the web interface with interactive citation pills (e.g., [Chunk 2] (✓)), bi-directional hover highlighting connecting text to source chunks, and an automated Telemetry HUD tracking Faithfulness Score, Hallucination Suppression Rate, and component latency breakdown."
)

# --- 7. SECTION 5: TOOLS / TECHNOLOGY USED ---
add_sec_heading("5. Tools/Technology Used:")

add_sub_heading("5.1 Minimum Hardware Requirements")
add_body_p("Hardware required for the development and execution of the project:")
add_bullet_item("CPU:", "Intel Core i5 (11th / 12th Generation equivalent, Multi-core ≥ 2.5 GHz)")
add_bullet_item("RAM:", "16 GB DDR4 / DDR5")
add_bullet_item("GPU:", "NVIDIA GeForce RTX 3050 (4 GB GDDR6 VRAM, CUDA Compute 8.6)")
add_bullet_item("HDD/SSD:", "512 GB NVMe Solid State Drive")
add_bullet_item("Others (if any):", "Broadband Internet connection for model weight caching and API services")

add_sub_heading("5.2 Minimum Software Requirements")
add_body_p("Software and toolchain required for the development of the project:")
add_bullet_item("OS:", "Microsoft Windows 11 (64-bit) / Ubuntu Linux 22.04 LTS")
add_bullet_item("Programming Language:", "Python 3.10 / 3.11")
add_bullet_item("Deep Learning Framework:", "PyTorch 2.2.0 with CUDA 12.1 acceleration")
add_bullet_item("NLP & Transformers Hub:", "Hugging Face Transformers (v4.40+), Sentence-Transformers (v2.7+)")
add_bullet_item("NLI Verification Model:", "cross-encoder/nli-deberta-v3-base")
add_bullet_item("Vector Database:", "FAISS-CPU / FAISS-GPU (v1.8.0+)")
add_bullet_item("Text Parsing Tools:", "spaCy (v3.7+, en_core_web_sm), PyPDF (v4.2+)")
add_bullet_item("Web Presentation / Demo:", "Streamlit (v1.35+)")
add_bullet_item("Development IDE:", "Antigravity IDE / Visual Studio Code")

# --- 8. SECTION 6: REFERENCES [IEEE FORMAT] ---
add_sec_heading("6. References: [IEEE format]:")
add_body_p("The following study material and research publications were referenced for the development of this project:")

refs = [
    "[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 9459–9474, 2020.",
    "[2] P. He, X. Liu, J. Gao, and W. Chen, \"DeBERTa: Decoding-enhanced BERT with Disentangled Attention,\" in Proc. 9th Int. Conf. Learn. Represent. (ICLR), Virtual Event, 2021.",
    "[3] J. Thorne, A. Vlachos, C. Christodoulopoulos, and A. Mittal, \"FEVER: A Large-scale Dataset for Fact Extraction and VERification,\" in Proc. 2018 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol. (NAACL-HLT), New Orleans, LA, pp. 809–819, 2018.",
    "[4] S. Min, K. Krishna, X. Lyu, M. Lewis, W. Yih, P. Kohli, S. Iyyer, L. Zettlemoyer, and H. Hajishirzi, \"FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation,\" in Proc. 2023 Conf. Empirical Methods Natural Lang. Process. (EMNLP), Singapore, pp. 12076–12100, 2023.",
    "[5] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang, \"Retrieval-Augmented Generation for Large Language Models: A Survey,\" arXiv preprint arXiv:2312.10997, 2023.",
    "[6] N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, \"BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models,\" in Thirty-fifth Conf. Neural Inf. Process. Syst. Datasets and Benchmarks Track, 2021."
]

for ref in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(ref)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)

# Save edited DOCX
doc.save(out_docx)
print(f"Successfully generated DOCX at: {out_docx}")
