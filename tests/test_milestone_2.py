"""Unit and integration test suite for VerifyRAG Milestone 2.

Covers:
1. LLM Generator adapters (Mock, formatting, factory resolution).
2. Atomic Claim Decomposition (abbreviations, decimals, conjunctions, noise filtering).
3. End-to-end integration: retrieved chunks -> generated draft -> atomic claims validation.
"""

import pytest
from src.schemas import Chunk, Claim, VerdictEnum
from src.config import Settings
from src.pipeline.generator import (
    BaseGenerator,
    MockGenerator,
    GroqGenerator,
    OpenAIGenerator,
    OllamaGenerator,
    get_generator,
    format_context_prompt,
)
from src.pipeline.claim_splitter import ClaimSplitter


class TestGenerators:
    """Tests for LLM generation adapters."""

    def test_mock_generator_default_synthesis(self):
        chunks = [
            Chunk(chunk_id=1, text="Photosynthesis converts light into chemical energy.", doc_id="doc_bio"),
            Chunk(chunk_id=2, text="Chloroplasts contain chlorophyll pigments.", doc_id="doc_bio"),
        ]
        generator = MockGenerator(latency_simulate_ms=2.0)
        query = "How does photosynthesis work?"
        response, latency_ms = generator.generate(query, chunks)

        assert "Regarding How does photosynthesis work" in response
        assert "Photosynthesis converts light into chemical energy." in response
        assert latency_ms > 0
        assert len(generator.call_history) == 1

    def test_mock_generator_custom_response(self):
        generator = MockGenerator(default_response="Quantum entanglement is non-local.")
        response, latency_ms = generator.generate("Explain entanglement", [])
        assert response == "Quantum entanglement is non-local."
        assert latency_ms >= 0

    def test_format_context_prompt(self):
        chunks = [
            Chunk(chunk_id=1, text="Alpha text.", doc_id="d1"),
            Chunk(chunk_id=2, text="Beta text.", doc_id="d2"),
        ]
        prompt = format_context_prompt("What is alpha and beta?", chunks)
        assert "[Chunk 1] (doc: d1)" in prompt
        assert "[Chunk 2] (doc: d2)" in prompt
        assert "Alpha text." in prompt
        assert "Beta text." in prompt
        assert "User Query: What is alpha and beta?" in prompt

    def test_format_context_prompt_empty_chunks(self):
        prompt = format_context_prompt("Hello?", [])
        assert "No reference context provided." in prompt
        assert "User Query: Hello?" in prompt

    def test_generator_factory_resolution(self):
        mock_gen = get_generator("mock")
        assert isinstance(mock_gen, MockGenerator)

        groq_gen = get_generator("groq")
        assert isinstance(groq_gen, GroqGenerator)

        openai_gen = get_generator("openai")
        assert isinstance(openai_gen, OpenAIGenerator)

        ollama_gen = get_generator("ollama")
        assert isinstance(ollama_gen, OllamaGenerator)

    def test_generator_factory_invalid_provider(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_generator("unknown_provider")


class TestClaimSplitter:
    """Tests for atomic claim decomposition engine."""

    @pytest.fixture
    def splitter(self):
        return ClaimSplitter(min_claim_words=3, split_compound_sentences=True)

    def test_empty_and_whitespace_input(self, splitter):
        claims, latency = splitter.split_claims("   \n\t  ")
        assert claims == []
        assert latency >= 0

    def test_simple_sentences(self, splitter):
        text = "The Pacific Ocean is the largest ocean. It covers more than 30% of Earth's surface."
        claims, latency = splitter.split_claims(text)
        assert len(claims) == 2
        assert claims[0].claim_id == 1
        assert claims[0].text == "The Pacific Ocean is the largest ocean."
        assert claims[1].claim_id == 2
        assert "30% of Earth's surface" in claims[1].text
        assert claims[0].verdict == VerdictEnum.UNSUPPORTED
        assert latency >= 0

    def test_compound_sentence_decomposition(self, splitter):
        text = (
            "Surface codes detect quantum errors, and stabilizer measurements identify syndromes; "
            "however, physical qubit overhead remains high."
        )
        claims, _ = splitter.split_claims(text)
        assert len(claims) >= 3
        claim_texts = [c.text for c in claims]
        assert any("Surface codes detect quantum errors" in t for t in claim_texts)
        assert any("Stabilizer measurements identify syndromes" in t for t in claim_texts)
        assert any("Physical qubit overhead remains high" in t for t in claim_texts)

    def test_abbreviation_and_decimal_protection(self, splitter):
        text = "Dr. Smith noted that the temperature reached 98.6 degrees in the U.S. during July."
        claims, _ = splitter.split_claims(text)
        # Should NOT split inside "Dr.", "98.6", or "U.S."
        assert len(claims) == 1
        assert "Dr. Smith" in claims[0].text
        assert "98.6 degrees" in claims[0].text
        assert "U.S." in claims[0].text

    def test_preamble_and_noise_filtering(self, splitter):
        text = (
            "Based on the provided context, Python was released in 1991.\n"
            "Here is the answer: It supports object-oriented programming.\n"
            "I hope this helps!"
        )
        claims, _ = splitter.split_claims(text)
        claim_texts = [c.text for c in claims]
        # Preamble should be stripped
        assert any("Python was released in 1991" in t for t in claim_texts)
        assert not any("based on the provided context" in t.lower() for t in claim_texts)
        # Conversational closer should be filtered out
        assert not any("hope this helps" in t.lower() for t in claim_texts)

    def test_markdown_and_numbered_lists(self, splitter):
        text = """
        Key architectural components:
        - 1. Ingestion parser extracts multi-format documents.
        * 2. Recursive text chunker splits content into slices.
        - 3. Dense vector store indexes normalized embeddings.
        """
        claims, _ = splitter.split_claims(text)
        assert len(claims) == 3
        for c in claims:
            # Bullet markers must be stripped
            assert not c.text.startswith("-")
            assert not c.text.startswith("*")
            assert not c.text.startswith("1.")

    def test_claim_schema_conformance(self, splitter):
        text = "Transformer models utilize self-attention mechanisms."
        claims, _ = splitter.split_claims(text)
        assert len(claims) == 1
        claim = claims[0]
        assert isinstance(claim, Claim)
        assert claim.verdict == VerdictEnum.UNSUPPORTED
        assert claim.best_chunk_id is None
        assert claim.scores["entailment"] == 0.0
        assert claim.scores["contradiction"] == 0.0


class TestMilestone2PipelineIntegration:
    """Integration test: chunk retrieval -> draft generation -> atomic claim decomposition."""

    def test_pipeline_generation_and_claim_splitting(self):
        # 1. Simulate retrieved chunks from Milestone 1
        chunks = [
            Chunk(
                chunk_id=1,
                text="Albert Einstein published the general theory of relativity in 1915.",
                doc_id="physics_doc",
                similarity_score=0.91,
            ),
            Chunk(
                chunk_id=2,
                text="The theory describes gravity as a geometric property of spacetime.",
                doc_id="physics_doc",
                similarity_score=0.88,
            ),
        ]

        # 2. Generate draft using MockGenerator
        generator = MockGenerator(latency_simulate_ms=1.0)
        query = "When was general relativity published and what does it describe?"
        draft_text, gen_latency = generator.generate(query, chunks)

        assert len(draft_text) > 0
        assert gen_latency > 0

        # 3. Decompose draft into atomic claims
        splitter = ClaimSplitter(min_claim_words=3)
        claims, split_latency = splitter.split_claims(draft_text)

        assert len(claims) >= 1
        assert split_latency >= 0

        # Verify all claims conform to canonical Pydantic schema
        for c in claims:
            assert isinstance(c, Claim)
            assert c.claim_id > 0
            assert len(c.text.split()) >= 3
