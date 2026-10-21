"""FAISS Vector Store wrapper for VerifyRAG.

Manages dense vector storage and top-k retrieval using sentence-transformers.
Guarantees canonical Chunk return format and includes a vectorized NumPy fallback.
"""

from functools import lru_cache
import logging
from typing import Any, List, Optional, Tuple
import numpy as np

from src.config import get_settings
from src.schemas import Chunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model(model_name: Optional[str] = None, device: Optional[str] = None) -> Any:
    """Singleton cached loader for the SentenceTransformer embedding model.

    Guarantees weights are loaded once in memory (brain.md Coding Standard #2).
    """
    settings = get_settings()
    name = model_name or settings.embedding_model_name
    dev = device or settings.device

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info(f"Loading embedding model '{name}' on device '{dev}'...")
        model = SentenceTransformer(name, device=dev)
        return model
    except ImportError:
        logger.warning(
            "sentence_transformers is not installed. Dense embedding generation "
            "will be unavailable until dependencies are installed."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model '{name}': {e}")
        return None


class VectorStore:
    """In-memory dense vector store supporting FAISS IndexFlatIP (cosine similarity)

    with automatic fallback to vectorized NumPy operations.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        settings = get_settings()
        self.dimension = dimension or settings.embedding_dim
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.device

        self.chunks: List[Chunk] = []
        self._faiss_index: Any = None
        self._numpy_embeddings: Optional[np.ndarray] = None
        self._use_faiss = self._init_faiss()

    def _init_faiss(self) -> bool:
        """Attempt to initialize FAISS IndexFlatIP."""
        try:
            import faiss  # type: ignore
            # IndexFlatIP with L2-normalized vectors calculates cosine similarity
            self._faiss_index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"FAISS IndexFlatIP initialized (dim={self.dimension}).")
            return True
        except ImportError:
            logger.info("faiss is not installed; using native NumPy cosine vector index.")
            self._faiss_index = None
            return False
        except Exception as e:
            logger.warning(f"Error initializing FAISS ({e}); falling back to NumPy.")
            self._faiss_index = None
            return False

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize vectors so inner product equals cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return (vectors / norms).astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of text strings into normalized dense embeddings."""
        model = get_embedding_model(self.model_name, self.device)
        if model is None:
            raise RuntimeError(
                "Embedding model is not available. Please ensure sentence-transformers "
                "is installed and reachable."
            )
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[np.ndarray] = None) -> int:
        """Add chunks and their corresponding embeddings into the vector store.

        If embeddings are not provided, they will be computed automatically.
        """
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        if embeddings is None:
            dense_vectors = self.encode(texts)
        else:
            dense_vectors = self._normalize(np.asarray(embeddings, dtype=np.float32))

        if self._use_faiss and self._faiss_index is not None:
            self._faiss_index.add(dense_vectors)
        else:
            if self._numpy_embeddings is None:
                self._numpy_embeddings = dense_vectors
            else:
                self._numpy_embeddings = np.vstack([self._numpy_embeddings, dense_vectors])

        self.chunks.extend(chunks)
        return len(chunks)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Chunk]:
        """Perform semantic similarity search for query and return top-k Chunk objects.

        Each returned Chunk has its similarity_score field populated with the cosine score.
        """
        if not self.chunks:
            return []

        settings = get_settings()
        k = top_k or settings.top_k_chunks
        k = min(k, len(self.chunks))

        query_vec = self.encode([query])  # Shape: (1, dim)

        if self._use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vec, k)
            top_scores = scores[0]
            top_indices = indices[0]
        else:
            # NumPy Cosine Search
            assert self._numpy_embeddings is not None
            sims = np.dot(self._numpy_embeddings, query_vec.T).squeeze(axis=1)
            # Argsort descending
            top_indices = np.argsort(sims)[::-1][:k]
            top_scores = sims[top_indices]

        results: List[Chunk] = []
        for score, idx in zip(top_scores, top_indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            orig_chunk = self.chunks[idx]
            # Emit a copy with the similarity_score populated
            result_chunk = Chunk(
                chunk_id=orig_chunk.chunk_id,
                text=orig_chunk.text,
                doc_id=orig_chunk.doc_id,
                similarity_score=float(score)
            )
            results.append(result_chunk)

        return results

    def clear(self) -> None:
        """Reset the vector store and clear all indexed chunks."""
        self.chunks = []
        self._numpy_embeddings = None
        self._init_faiss()

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self.chunks)

    def get_all_chunks(self) -> List[Chunk]:
        """Return all indexed chunks."""
        return list(self.chunks)
