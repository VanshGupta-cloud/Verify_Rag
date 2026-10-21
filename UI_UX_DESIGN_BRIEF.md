# UI/UX Design Brief

## Project: VerifyRAG
**Aesthetic Style:** Academic Precision meets Modern AI SaaS  
**Visual Tone:** Trustworthy, Data-Dense, Analytical, High-Contrast  
**Inspiration:** Perplexity AI, Elicit, Consensus, Linear  

---

## 1. Design Philosophy

In high-stakes technical and scientific question-answering, **trust is visual**. The UI must not just provide an answer; it must visually prove *why* the answer is credible by exposing the connection between individual claims, NLI inference scores, and underlying source documents.

### Key UX Principles:
1. **Zero Black-Box Responses:** Every assertion has an explicit citation pill displaying its verification status.
2. **Bi-Directional Highlighting:** Hovering a claim highlights its supporting document chunk; hovering a chunk highlights every claim derived from it.
3. **Transparent Latency:** Users see real-time pipeline telemetry (Retrieval $\rightarrow$ Draft $\rightarrow$ NLI $\rightarrow$ Remediation) so they understand the value of verification time.
4. **Immediate Baseline Comparison:** Evaluators can toggle between "Plain RAG" (unfiltered) and "VerifyRAG" (verified) to immediately witness hallucination suppression in real time.

---

## 2. Color System & Semantic Tokens

### 2.1 Base Theme (Dark / Slate Theme Recommended)
* **Background (Canvas):** `#090D16` (Deep Midnight Slate)
* **Card Surface:** `#111827` (Charcoal Slate)
* **Elevated Surface (Modals/Popovers):** `#1F2937` (Cool Gray)
* **Borders / Dividers:** `#374151` (Subtle boundary line)
* **Text Primary:** `#F9FAFB` (High-contrast pure white)
* **Text Muted:** `#9CA3AF` (Medium silver for metadata and secondary labels)

### 2.2 Semantic Verification Palette (Critical for Claim Badging)

| Status | Color Name | Hex Code | Background Tint | Meaning in UI |
| :--- | :--- | :--- | :--- | :--- |
| **Supported / Entailed** | Emerald Green | `#10B981` | `#064E3B` (20% opacity) | Factually backed by retrieved document chunk. |
| **Contradicted** | Crimson Rose | `#EF4444` | `#7F1D1D` (25% opacity) | Directly conflicts with retrieved source passage. |
| **Unsupported / Neutral** | Warning Amber | `#F59E0B` | `#78350F` (20% opacity) | Plausible or hallucinated; not verifiable from context. |
| **Citation Reference** | Cyan Accent | `#06B6D4` | `#164E63` (20% opacity) | Interactive citation button linking to source chunk. |

---

## 3. Typography Hierarchy

* **Primary Font Family (UI & Body):** `Inter`, `-apple-system`, `sans-serif`
  * Clean, highly readable at dense sizes (12px to 15px).
* **Monospace Font Family (Citations, Latency, Confidence Scores):** `JetBrains Mono`, `Fira Code`, `monospace`
  * Conveys algorithmic rigor and mathematical precision.

### Type Scale:
* **H1 / Page Title:** 24px / Semi-Bold / `letter-spacing: -0.02em`
* **H2 / Section Headers:** 18px / Medium
* **Body Text:** 14px / Regular / `line-height: 1.6`
* **Badges & Citations:** 11px / Medium / Monospace / Uppercase
* **Telemetry & Metrics:** 12px / Regular / Monospace

---

## 4. Layout Architecture & Screen Zones

```text
+--------------------------------------------------------------------------------------------------+
|  VerifyRAG [Live]     | Model: DeBERTa-v3 + Llama-3-70B | Docs Indexed: 42 chunks | [Academic Mode] |
+--------------------------------------------------------------------------------------------------+
|  LEFT SIDEBAR            |  CENTER: QUERY & ANSWER COMPARISON          |  RIGHT: SOURCE INSPECTOR   |
|                          |                                             |                            |
|  [ Upload PDF / TXT ]    |  [ Search Bar: "Explain surface codes..." ] |  Selected Source:          |
|                          |  [ Mode: [Plain] [Filter] [Regenerate] ]    |  Chunk #2 (Score: 0.89)    |
|  Configuration:          |                                             |  ------------------------- |
|  - Top-k Chunks: [ 4 ]   |  Generated Answer:                          |  "Surface codes detect     |
|  - Tau_Entail:   [0.70]  |  Surface codes detect syndrome errors       |   syndrome errors using    |
|  - Tau_Contra:   [0.60]  |  using X and Z stabilizers [Chunk 2] (✓).   |   X and Z stabilizers      |
|                          |  They guarantee zero physical qubit         |   arranged in a 2D         |
|  Benchmark Presets:      |  overhead. (✗ Contradiction removed)        |   lattice..."              |
|  - Hallucination Trap 1  |                                             |                            |
|  - Medical QA Test       |  ------------------------------------------ |  NLI Confidence Matrix:    |
|  - CS Theory Corpus      |  Telemetry: Latency: 1.84s | Faithfulness: 92% |  Entail: 94% | Contra: 1%  |
+--------------------------------------------------------------------------------------------------+
```

---

## 5. Key Interactive Components

### 5.1 Interactive Claim Card
Each decomposed claim renders as an expandable micro-card:
* **Left Badge:** Green checkmark (Entailed), Red cross (Contradiction), or Yellow dash (Neutral).
* **Center:** Atomic claim sentence.
* **Right Metrics:** Horizontal mini stacked bar representing NLI class distribution:
  `[ 94% Entailment | 4% Neutral | 2% Contradiction ]`.
* **Click Action:** Opens the exact chunk in the right-hand Source Inspector, auto-scrolling to the matching phrase.

### 5.2 Real-Time Pipeline Progress Bar
When a query executes, replace indeterminate spinners with a 5-step active progress tracker:
1. `[✓] Semantic Retrieval (Top-4 Chunks found in 42ms)`
2. `[✓] LLM Draft Generation (420 tokens generated in 810ms)`
3. `[✓] Atomic Claim Decomposition (5 independent claims extracted)`
4. `[⟳] Batched DeBERTa NLI Verification (Scoring 20 premise-hypothesis pairs...)`
5. `[ ] Targeted Regeneration / Remediation`

### 5.3 Side-by-Side Diff View (The "Evaluator View")
Allows toggling a split screen:
* **Left Pane (Plain RAG):** Shows the raw answer with hidden hallucinations highlighted in faint red.
* **Right Pane (VerifyRAG):** Shows the remediated answer with citations and zero hallucinations.
* This is the centerpiece visualization for professor and committee demonstrations.

---

## 6. Accessibility & Responsiveness
* Minimum contrast ratio of **4.5:1** for all text against dark surfaces (WCAG AA compliant).
* All interactive badges and buttons have `:focus-visible` rings with `#06B6D4` outline.
* Graceful collapse on screens under 1200px width into a stacked 2-column or tabbed interface.
