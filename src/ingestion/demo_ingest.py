"""Demonstration and CLI runner for Milestone 1 Ingestion & Vector Retrieval.

Usage:
    python -m src.ingestion.demo_ingest [file_path] [query]
"""

import sys
import time
from pathlib import Path

from src.config import get_settings
from src.ingestion.parser import parse_document
from src.ingestion.chunker import RecursiveCharacterChunker
from src.ingestion.vector_store import VectorStore


def run_demo(file_path: str = "PRD.md", query: str = "How does VerifyRAG reduce hallucination?"):
    print("=" * 70)
    print(" VerifyRAG — Milestone 1: Ingestion & Retrieval Pipeline Demo")
    print("=" * 70)

    settings = get_settings()
    print(f"Device: {settings.device}")
    print(f"Embedding Model: {settings.embedding_model_name}")
    print(f"Chunk Size: {settings.chunk_size} chars | Overlap: {settings.chunk_overlap} chars")
    print(f"Top-K Retrieval: {settings.top_k_chunks}")
    print("-" * 70)

    target_path = Path(file_path)
    if not target_path.exists():
        print(f"Target file '{file_path}' not found, falling back to 'PRD.md'...")
        target_path = Path("PRD.md")

    print(f"Step 1: Parsing document: {target_path} ...")
    t0 = time.perf_counter()
    doc = parse_document(target_path)
    t_parse = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Parsed doc_id='{doc.doc_id}', filename='{doc.filename}', "
          f"length={len(doc.content)} chars, pages={doc.num_pages} ({t_parse:.2f} ms)")

    print("\nStep 2: Chunking document ...")
    t0 = time.perf_counter()
    chunker = RecursiveCharacterChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = chunker.chunk_document(doc_id=doc.doc_id, text=doc.content)
    t_chunk = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Created {len(chunks)} chunks ({t_chunk:.2f} ms)")

    print("\nStep 3: Indexing chunks into Vector Store ...")
    t0 = time.perf_counter()
    vector_store = VectorStore()
    indexed_count = vector_store.add_chunks(chunks)
    t_index = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Indexed {indexed_count} chunks into in-memory vector store ({t_index:.2f} ms)")

    print(f"\nStep 4: Executing Semantic Similarity Search ...")
    print(f"  Query: \"{query}\"")
    t0 = time.perf_counter()
    retrieved = vector_store.search(query, top_k=settings.top_k_chunks)
    t_search = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Retrieved {len(retrieved)} chunks ({t_search:.2f} ms)\n")

    print("-" * 70)
    print(" Top Retrieved Chunks:")
    print("-" * 70)
    for rank, chunk in enumerate(retrieved, start=1):
        print(f"[{rank}] Chunk ID: #{chunk.chunk_id} | Similarity: {chunk.similarity_score:.4f} | Doc: {chunk.doc_id}")
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 140:
            preview = preview[:137] + "..."
        print(f"    Excerpt: {preview}\n")

    print("=" * 70)
    print(" Milestone 1 Ingestion & Vector Retrieval Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "PRD.md"
    q = sys.argv[2] if len(sys.argv) > 2 else "How does VerifyRAG reduce hallucination?"
    run_demo(fp, q)
