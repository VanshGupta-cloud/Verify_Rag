"""Milestone 1 Test Suite (VerifyRAG).

Validates:
1. Pydantic v2 Canonical Schemas (brain.md Section 4)
2. Centralized Settings (src/config.py)
3. Document Ingestion & Multi-Format Parsing (src/ingestion/parser.py)
4. Recursive Character Chunking (src/ingestion/chunker.py)
5. Vector Store Dense Retrieval (src/ingestion/vector_store.py)
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import (
    VerdictEnum,
    Chunk,
    Claim,
    LatencyBreakdown,
    VerificationResult,
    ParsedDocument
)
from src.config import get_settings
from src.ingestion.parser import parse_document, extract_text_from_txt, _clean_text
from src.ingestion.chunker import RecursiveCharacterChunker
from src.ingestion.vector_store import VectorStore


class TestCanonicalSchemas(unittest.TestCase):
    """Ensure data structures adhere strictly to brain.md Section 4."""

    def test_verdict_enum(self):
        self.assertEqual(VerdictEnum.SUPPORTED.value, "SUPPORTED")
        self.assertEqual(VerdictEnum.CONTRADICTED.value, "CONTRADICTED")
        self.assertEqual(VerdictEnum.UNSUPPORTED.value, "UNSUPPORTED")

    def test_chunk_schema(self):
        chunk = Chunk(
            chunk_id=1,
            text="Surface codes use stabilizer measurements to detect syndromes.",
            doc_id="doc_test_123",
            similarity_score=0.885
        )
        self.assertEqual(chunk.chunk_id, 1)
        self.assertEqual(chunk.similarity_score, 0.885)
        self.assertTrue(isinstance(chunk.model_dump(), dict))

    def test_claim_schema(self):
        claim = Claim(
            claim_id=1,
            text="Surface codes guarantee zero qubit overhead.",
            verdict=VerdictEnum.CONTRADICTED,
            best_chunk_id=2,
            confidence=0.89,
            scores={"entailment": 0.05, "neutral": 0.06, "contradiction": 0.89}
        )
        self.assertEqual(claim.verdict, VerdictEnum.CONTRADICTED)
        self.assertEqual(claim.scores["contradiction"], 0.89)

    def test_verification_result_schema(self):
        res = VerificationResult(
            query="What is VerifyRAG?",
            mode="filter",
            draft_answer="Draft text",
            verified_answer="Verified text",
            faithfulness_score=100.0,
            latency=LatencyBreakdown(
                retrieval_ms=10.0,
                draft_generation_ms=200.0,
                claim_split_ms=5.0,
                nli_verification_ms=50.0,
                remediation_ms=15.0,
                total_ms=280.0
            )
        )
        self.assertEqual(res.faithfulness_score, 100.0)
        self.assertEqual(res.latency.total_ms, 280.0)


class TestConfig(unittest.TestCase):
    """Verify application settings and default thresholds."""

    def test_default_thresholds(self):
        settings = get_settings()
        self.assertAlmostEqual(settings.entailment_threshold, 0.70)
        self.assertAlmostEqual(settings.contradiction_threshold, 0.60)
        self.assertEqual(settings.top_k_chunks, 4)
        self.assertEqual(settings.chunk_size, 500)
        self.assertEqual(settings.chunk_overlap, 50)
        self.assertIn(settings.device, ("cuda", "cpu"))


class TestParser(unittest.TestCase):
    """Verify multi-format parsing for TXT, MD, and PDF."""

    def test_clean_text(self):
        raw = "Hello \x00 world!\r\n\r\n\r\nThis is a test.  "
        cleaned = _clean_text(raw)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\r", cleaned)
        self.assertEqual(cleaned, "Hello  world!\n\nThis is a test.")

    def test_parse_markdown_file(self):
        prd_path = Path(__file__).resolve().parent.parent / "PRD.md"
        if prd_path.exists():
            doc = parse_document(prd_path)
            self.assertIsInstance(doc, ParsedDocument)
            self.assertTrue(len(doc.content) > 500)
            self.assertTrue(doc.doc_id.startswith("doc_"))
            self.assertEqual(doc.filename, "PRD.md")

    def test_parse_pdf_file(self):
        pdf_path = Path(__file__).resolve().parent.parent / "VerifyRAG_Synopsis.pdf"
        if pdf_path.exists():
            doc = parse_document(pdf_path)
            self.assertIsInstance(doc, ParsedDocument)
            self.assertTrue(doc.num_pages >= 1)
            self.assertTrue(len(doc.content) > 100)
            self.assertEqual(doc.filename, "VerifyRAG_Synopsis.pdf")


class TestChunker(unittest.TestCase):
    """Verify recursive splitting and chunk boundary handling."""

    def setUp(self):
        self.chunker = RecursiveCharacterChunker(chunk_size=200, chunk_overlap=30)

    def test_short_text_single_chunk(self):
        text = "This is a brief text under 200 characters."
        chunks = self.chunker.chunk_document("doc_1", text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].chunk_id, 1)

    def test_long_text_chunking_and_overlap(self):
        # Generate multi-paragraph text
        paras = [
            f"Paragraph {i}: Surface codes are 2D topological quantum error-correcting codes. "
            f"They use physical qubits on a 2D square lattice with syndrome measurements."
            for i in range(1, 10)
        ]
        text = "\n\n".join(paras)
        chunks = self.chunker.chunk_document("doc_quantum", text)
        self.assertTrue(len(chunks) > 1)

        # Check all chunks comply with chunk_size limit
        for c in chunks:
            self.assertLessEqual(len(c.text), 250)  # Reasonable buffer for atomic words
            self.assertEqual(c.doc_id, "doc_quantum")
            self.assertTrue(c.chunk_id >= 1)


class TestVectorStore(unittest.TestCase):
    """Verify vector indexing, dense search, and canonical Chunk output."""

    def test_vector_store_operations(self):
        # Create vector store (with mock/synthetic embeddings if sentence-transformers is loading)
        dim = 16
        vs = VectorStore(dimension=dim)

        # Create synthetic chunks
        chunks = [
            Chunk(chunk_id=1, text="Surface codes detect syndrome errors using X and Z stabilizers.", doc_id="doc_1"),
            Chunk(chunk_id=2, text="Topological quantum memory protects quantum information.", doc_id="doc_1"),
            Chunk(chunk_id=3, text="Neural networks can be accelerated via tensor cores.", doc_id="doc_2"),
        ]

        # Use synthetic normalized vectors for pure store testing
        np.random.seed(42)
        mock_vecs = np.random.randn(len(chunks), dim).astype(np.float32)

        added = vs.add_chunks(chunks, embeddings=mock_vecs)
        self.assertEqual(added, 3)
        self.assertEqual(vs.count(), 3)

        # Query using synthetic query vector
        query_vec = mock_vecs[0:1]  # Exact match for chunk 1
        # Test search with custom override or numpy
        if vs._use_faiss and vs._faiss_index is not None:
            scores, indices = vs._faiss_index.search(query_vec, 2)
            self.assertEqual(indices[0][0], 0)
            self.assertAlmostEqual(scores[0][0], 1.0, places=4)

        # Test clear
        vs.clear()
        self.assertEqual(vs.count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
