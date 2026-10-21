"""Atomic claim decomposition engine for VerifyRAG.

Decomposes raw generated draft text into discrete, verifiable factual claims
using sentence boundary analysis, clause splitting, and conversational noise filtering.
Supports optional spaCy integration with resilient zero-dependency rule-based fallback.
"""

import logging
import re
import time
from typing import List, Optional, Tuple

from src.schemas import Claim, VerdictEnum

logger = logging.getLogger(__name__)

# Common conversational preambles/disclaimers and discourse markers to strip
CONVERSATIONAL_PREFIXES = [
    r"^based on the (provided|given) context,?\s*",
    r"^according to the (provided|given) (context|documents|text),?\s*",
    r"^(here is|here's) (the answer|a summary|what the text says):?\s*",
    r"^sure,?\s*(here is|here's)?\s*",
    r"^certainly,?\s*",
    r"^in summary,?\s*",
    r"^to summarize,?\s*",
    r"^in conclusion,?\s*",
    r"^overall,?\s*",
    # Discourse transitional adverbs when at beginning of claim
    r"^(however|furthermore|moreover|additionally|therefore|in addition|meanwhile),?\s*",
    # Leading conjunctions left over from clause splits
    r"^(and|or|but|yet)\s+",
]

# Patterns for abbreviations that should NOT trigger a sentence split
ABBREVIATIONS = (
    r"\be\.g\.", r"\bi\.e\.", r"\bu\.s\.", r"\bu\.k\.", r"\bdr\.", r"\bmr\.",
    r"\bmrs\.", r"\bms\.", r"\bprof\.", r"\bfig\.", r"\bal\.", r"\betc\.",
    r"\bvs\.", r"\binc\.", r"\bltd\.", r"\bcorp\.", r"\bno\.", r"\bapprox\.",
    r"\bdept\.", r"\best\.", r"\bmin\.", r"\bmax\."
)


class ClaimSplitter:
    """Decomposes generated draft answers into atomic, verifiable Claim objects."""

    def __init__(
        self,
        min_claim_words: int = 3,
        split_compound_sentences: bool = True,
        use_spacy: bool = True,
    ):
        """Initialize ClaimSplitter.

        Args:
            min_claim_words: Minimum word count for a fragment to qualify as an atomic claim.
            split_compound_sentences: Whether to split independent clauses joined by conjunctions.
            use_spacy: Attempt to load spaCy if available in the environment.
        """
        self.min_claim_words = min_claim_words
        self.split_compound_sentences = split_compound_sentences
        self.nlp = None

        if use_spacy:
            try:
                import spacy  # type: ignore
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = None
            except ImportError:
                self.nlp = None

    def split_claims(self, text: str) -> Tuple[List[Claim], float]:
        """Extract atomic claims from input text and measure execution latency.

        Args:
            text: Raw draft answer string.

        Returns:
            Tuple of:
                - List of canonical Claim objects with sequential IDs.
                - Execution latency in milliseconds (claim_split_ms).
        """
        start_time = time.perf_counter()
        raw_text = text.strip()
        if not raw_text:
            return [], round((time.perf_counter() - start_time) * 1000.0, 2)

        # Step 1: Normalize lines and list formatting
        normalized_paragraphs = self._normalize_structure(raw_text)

        # Step 2: Sentence segmentation
        raw_sentences: List[str] = []
        for para in normalized_paragraphs:
            if self.nlp is not None:
                doc = self.nlp(para)
                raw_sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
            else:
                raw_sentences.extend(self._rule_based_sentence_split(para))

        # Step 3: Clause decomposition and noise filtering
        candidate_propositions: List[str] = []
        for sentence in raw_sentences:
            cleaned = self._clean_preambles(sentence)
            if not cleaned:
                continue

            if self.split_compound_sentences:
                clauses = self._decompose_compound_sentence(cleaned)
                for clause in clauses:
                    cleaned_clause = self._clean_claim_text(clause)
                    if self._is_valid_claim(cleaned_clause):
                        candidate_propositions.append(cleaned_clause)
            else:
                cleaned_sentence = self._clean_claim_text(cleaned)
                if self._is_valid_claim(cleaned_sentence):
                    candidate_propositions.append(cleaned_sentence)

        # Step 4: Deduplication while preserving order
        seen = set()
        unique_claims: List[str] = []
        for prop in candidate_propositions:
            lower = prop.lower()
            if lower not in seen:
                seen.add(lower)
                unique_claims.append(prop)

        # Step 5: Construct canonical Claim objects
        claims: List[Claim] = [
            Claim(
                claim_id=idx + 1,
                text=prop,
                verdict=VerdictEnum.UNSUPPORTED,
                best_chunk_id=None,
                confidence=0.0,
                scores={"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0},
            )
            for idx, prop in enumerate(unique_claims)
        ]

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return claims, round(latency_ms, 2)

    def _normalize_structure(self, text: str) -> List[str]:
        """Convert markdown bullet points, numbered lists, and line breaks into standard text."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Skip lines that are purely section headers or colon introductions
            if line_str.rstrip().endswith(":"):
                continue

            # Strip markdown list markers: "-", "*", "1.", "2)", "[1]"
            line_str = re.sub(r"^(\s*[-*•]\s+|\s*\d+[\.\)]\s+|\[\d+\]\s*)", "", line_str).strip()
            # Also handle combined list items like "- 1. text"
            line_str = re.sub(r"^(\s*[-*•]\s+|\s*\d+[\.\)]\s+|\[\d+\]\s*)", "", line_str).strip()

            if line_str and not line_str.endswith(":"):
                cleaned_lines.append(line_str)
        return cleaned_lines

    def _clean_preambles(self, sentence: str) -> str:
        """Strip conversational assistant noise and filler greetings."""
        text = sentence.strip()
        for pattern in CONVERSATIONAL_PREFIXES:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return text

    def _clean_claim_text(self, text: str) -> str:
        """Standardize capitalization, transitional markers, and punctuation for an atomic claim."""
        cleaned = text.strip()
        if not cleaned:
            return ""

        # Strip discourse transitions like 'however,', 'moreover,'
        for pattern in CONVERSATIONAL_PREFIXES:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Remove trailing commas, colons, semicolons, dashes
        cleaned = re.sub(r"[,;:\-\s]+$", "", cleaned)

        # Ensure first character is capitalized
        if len(cleaned) > 1:
            cleaned = cleaned[0].upper() + cleaned[1:]
        elif len(cleaned) == 1:
            cleaned = cleaned.upper()

        # Ensure sentence ends with period if not ending in question or exclamation
        if cleaned and cleaned[-1] not in {".", "?", "!"}:
            cleaned += "."

        return cleaned

    def _is_valid_claim(self, text: str) -> bool:
        """Check if candidate text constitutes an informative factual proposition."""
        if not text:
            return False

        words = text.split()
        if len(words) < self.min_claim_words:
            return False

        # Reject pure questions, colon titles, or conversational closers
        lower = text.lower()
        if text.endswith("?"):
            return False
        if lower.startswith(("i hope this helps", "let me know if", "feel free to")):
            return False

        return True

    def _decompose_compound_sentence(self, sentence: str) -> List[str]:
        """Split compound sentences with coordinating conjunctions into independent clauses."""
        if ";" in sentence:
            parts = [p.strip() for p in sentence.split(";") if p.strip()]
            sub_clauses: List[str] = []
            for part in parts:
                sub_clauses.extend(self._decompose_conjunctions(part))
            return sub_clauses

        return self._decompose_conjunctions(sentence)

    def _decompose_conjunctions(self, clause: str) -> List[str]:
        """Split a clause on coordinating conjunctions (e.g. ', and ', ', but ', ', while ')."""
        pattern = r",\s+(?:and|but|while|whereas|however)\s+"
        parts = re.split(pattern, clause, flags=re.IGNORECASE)

        if len(parts) <= 1:
            return [clause]

        valid_parts = []
        for part in parts:
            part = part.strip()
            if len(part.split()) >= self.min_claim_words:
                valid_parts.append(part)

        return valid_parts if len(valid_parts) > 1 else [clause]

    def _rule_based_sentence_split(self, text: str) -> List[str]:
        """High-precision sentence segmenter handling abbreviations and decimals."""
        if not text:
            return []

        masked_text = text

        # Protect decimals like 98.6 or 3.14
        masked_text = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL__\2", masked_text)

        # Protect abbreviations preserving original casing
        for abbr_pattern in ABBREVIATIONS:
            masked_text = re.sub(
                abbr_pattern,
                lambda m: m.group(0).replace(".", "__DOT__"),
                masked_text,
                flags=re.IGNORECASE,
            )

        # Split on sentence boundaries (. ! ?)
        raw_splits = re.split(r"(?<=[.!?])\s+", masked_text)

        restored_sentences = []
        for item in raw_splits:
            restored = item.replace("__DECIMAL__", ".").replace("__DOT__", ".")
            restored = restored.strip()
            if restored:
                restored_sentences.append(restored)

        return restored_sentences
