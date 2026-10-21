"""Recursive Character Text Splitter for VerifyRAG.

Splits documents into overlapping chunks honoring semantic boundaries
(paragraphs, sentences, words) and returns canonical Chunk objects.
"""

from typing import List, Optional

from src.schemas import Chunk


class RecursiveCharacterChunker:
    """Recursively splits text using a hierarchy of separators to satisfy chunk_size.

    Default settings from brain.md & TRD.md:
      - chunk_size: 500 characters
      - chunk_overlap: 50 characters
      - separators: ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or list(self.DEFAULT_SEPARATORS)

    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator, preserving separator context where possible."""
        if not separator:
            return list(text)

        splits = text.split(separator)
        result = []
        for i, piece in enumerate(splits):
            if not piece and i > 0 and i < len(splits) - 1:
                continue
            # Retain punctuation on sentence boundaries
            if separator == ". " and i < len(splits) - 1:
                result.append(piece + ".")
            elif separator in ("\n\n", "\n") and i < len(splits) - 1:
                result.append(piece + separator)
            else:
                result.append(piece)
        return [s for s in result if s]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first available separator in the list."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # Fallback: slice by characters
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        sep = separators[0]
        remaining_seps = separators[1:]

        # Find first separator present in the text
        active_sep = None
        for candidate in separators:
            if candidate == "" or candidate in text:
                active_sep = candidate
                # Remaining separators are those after this candidate
                idx = separators.index(candidate)
                remaining_seps = separators[idx + 1 :]
                break

        if active_sep is None or active_sep == "":
            # Character-level slicing fallback
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        splits = self._split_text_with_separator(text, active_sep)

        # Merge splits into windows <= chunk_size with overlap
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in splits:
            piece_len = len(piece)

            if piece_len > self.chunk_size:
                # Flush existing buffer
                if current_chunk:
                    merged = "".join(current_chunk).strip()
                    if merged:
                        chunks.append(merged)
                    current_chunk = []
                    current_len = 0

                # Recurse on the oversized piece using finer separators
                sub_chunks = self._recursive_split(piece, remaining_seps)
                chunks.extend(sub_chunks)
                continue

            # Check if adding piece exceeds chunk_size
            if current_len + piece_len > self.chunk_size:
                if current_chunk:
                    merged = "".join(current_chunk).strip()
                    if merged:
                        chunks.append(merged)

                    # Build overlap from end of current_chunk
                    overlap_buffer: List[str] = []
                    overlap_len = 0
                    for prev_piece in reversed(current_chunk):
                        if overlap_len + len(prev_piece) <= self.chunk_overlap:
                            overlap_buffer.insert(0, prev_piece)
                            overlap_len += len(prev_piece)
                        else:
                            break
                    current_chunk = overlap_buffer
                    current_len = overlap_len

            current_chunk.append(piece)
            current_len += piece_len

        if current_chunk:
            merged = "".join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Split a raw text string into a list of cleaned text chunks."""
        if not text or not text.strip():
            return []
        raw_chunks = self._recursive_split(text.strip(), self.separators)
        # Deduplicate and filter empty
        return [c.strip() for c in raw_chunks if c.strip()]

    def chunk_document(
        self,
        doc_id: str,
        text: str,
        start_chunk_id: int = 1
    ) -> List[Chunk]:
        """Split document text into canonical Chunk schema objects."""
        text_slices = self.chunk_text(text)
        chunks: List[Chunk] = []

        for offset, slice_text in enumerate(text_slices):
            chunk_obj = Chunk(
                chunk_id=start_chunk_id + offset,
                text=slice_text,
                doc_id=doc_id,
                similarity_score=0.0
            )
            chunks.append(chunk_obj)

        return chunks
