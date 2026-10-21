"""Interactive demonstration CLI for VerifyRAG Milestone 2.

Runs an end-to-end flow:
1. Ingests a sample document and indexes chunks into the vector store.
2. Queries the vector store to retrieve top-k context chunks.
3. Generates a draft answer using the pluggable generator adapter.
4. Decomposes the draft response into atomic, verifiable Claim objects.
5. Displays formatted claim propositions and latency telemetry.
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.ingestion.parser import parse_document
from src.ingestion.chunker import RecursiveCharacterChunker
from src.ingestion.vector_store import VectorStore
from src.pipeline.generator import MockGenerator, get_generator
from src.pipeline.claim_splitter import ClaimSplitter

console = Console()

SAMPLE_DOCUMENT = """
# Advances in Quantum Computing

Quantum error correction protects quantum information from decoherence and noise.
The surface code is currently the leading candidate for fault-tolerant quantum computing;
it arranges physical qubits on a 2D square lattice and performs syndrome measurements.
Stabilizer operations detect bit-flip and phase-flip errors using parity checks,
and physical error rates below 1% are required to achieve fault tolerance.
In 2019, Google demonstrated quantum supremacy with a 53-qubit Sycamore processor.
However, practical quantum computers will require millions of physical qubits to encode
hundreds of logical qubits.
"""


def run_demo() -> None:
    console.print(Panel.fit("[bold cyan]VerifyRAG - Milestone 2 Demonstration[/bold cyan]\n"
                            "[dim]LLM Generation & Atomic Claim Decomposition[/dim]"))

    # 1. Chunk & Index
    console.print("\n[bold yellow]Step 1: Chunking & Indexing Evidence...[/bold yellow]")
    chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=30)
    chunks = chunker.chunk_document(doc_id="doc_qc_001", text=SAMPLE_DOCUMENT)
    vector_store = VectorStore()
    vector_store.add_chunks(chunks)
    console.print(f"Indexed [green]{len(chunks)}[/green] chunks into vector store.")

    # 2. Retrieve
    query = "What is the surface code and what threshold is required for fault tolerance?"
    console.print(f"\n[bold yellow]Step 2: Semantic Retrieval for Query:[/bold yellow] '{query}'")
    retrieved = vector_store.search(query=query, top_k=2)
    for i, c in enumerate(retrieved, start=1):
        console.print(f"  [cyan][Chunk {c.chunk_id}][/cyan] (score: {c.similarity_score:.4f}): {c.text[:90]}...")

    # 3. Generate Draft Response
    console.print("\n[bold yellow]Step 3: Generating Draft Answer with Pluggable Adapter...[/bold yellow]")
    generator = MockGenerator(latency_simulate_ms=25.0)
    draft_answer, gen_latency = generator.generate(query=query, context_chunks=retrieved)
    console.print(Panel(draft_answer, title="[bold green]Generated Draft Response[/bold green]", border_style="green"))
    console.print(f"[dim]Generation Latency: {gen_latency:.1f} ms[/dim]")

    # 4. Decompose into Atomic Claims
    console.print("\n[bold yellow]Step 4: Decomposing Answer into Atomic Claims...[/bold yellow]")
    splitter = ClaimSplitter(min_claim_words=3, split_compound_sentences=True)
    claims, split_latency = splitter.split_claims(draft_answer)

    table = Table(title="Extracted Atomic Claims", show_header=True, header_style="bold magenta")
    table.add_column("Claim ID", justify="center", width=10)
    table.add_column("Atomic Proposition", justify="left")
    table.add_column("Initial Status", justify="center", width=16)

    for claim in claims:
        table.add_row(
            str(claim.claim_id),
            claim.text,
            f"[yellow]{claim.verdict.value}[/yellow]",
        )

    console.print(table)
    console.print(f"[dim]Claim Decomposition Latency: {split_latency:.1f} ms[/dim]")
    console.print("\n[bold green][OK] Milestone 2 Generation & Claim Decomposition Operational![/bold green]\n")


if __name__ == "__main__":
    run_demo()
