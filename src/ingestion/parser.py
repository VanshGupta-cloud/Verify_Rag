"""Document parser for VerifyRAG.

Extracts text and metadata from PDF, TXT, and Markdown files.
Supports both file paths and in-memory bytes (for Streamlit and FastAPI uploads).
"""

import hashlib
import io
import re
from pathlib import Path
from typing import BinaryIO, Optional, Union

from src.schemas import ParsedDocument


def _generate_doc_id(content: str, filename: str) -> str:
    """Generate a deterministic document identifier based on filename and content."""
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8", errors="ignore"))
    hasher.update(content[:1024].encode("utf-8", errors="ignore"))
    return f"doc_{hasher.hexdigest()[:12]}"


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace and strip null/unprintable control characters."""
    if not raw_text:
        return ""
    # Strip null bytes and non-printable control characters except \n and \t
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_text)
    # Normalize carriage returns
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple consecutive blank lines into double newline
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Strip leading/trailing whitespaces per line while preserving indentation
    lines = [line.strip() for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def extract_text_from_txt(
    source: Union[str, Path, bytes, BinaryIO],
    encoding: str = "utf-8"
) -> str:
    """Extract and decode plain text from string, Path, bytes, or file-like buffer."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        raw_bytes = path.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    elif hasattr(source, "read"):
        raw_bytes = source.read()
    else:
        raise ValueError(f"Unsupported source type: {type(source)}")

    # Attempt UTF-8 decoding first, then fall back to Latin-1
    for enc in (encoding, "utf-8", "latin-1", "cp1252"):
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    return raw_bytes.decode("utf-8", errors="replace")


def extract_text_from_pdf(source: Union[str, Path, bytes, BinaryIO]) -> tuple[str, int]:
    """Extract concatenated text and page count from a PDF file or bytes buffer."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required for PDF parsing. Install with `pip install pypdf`."
        ) from e

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        stream: BinaryIO = open(path, "rb")
        should_close = True
    elif isinstance(source, bytes):
        stream = io.BytesIO(source)
        should_close = False
    elif hasattr(source, "read"):
        stream = source
        should_close = False
    else:
        raise ValueError(f"Unsupported source type for PDF: {type(source)}")

    try:
        reader = PdfReader(stream)
        num_pages = len(reader.pages)
        page_texts: list[str] = []

        for idx, page in enumerate(reader.pages):
            page_content = page.extract_text() or ""
            if page_content.strip():
                page_texts.append(page_content)

        full_text = "\n\n".join(page_texts)
        return full_text, num_pages
    finally:
        if should_close:
            stream.close()


def parse_document(
    source: Union[str, Path, bytes, BinaryIO],
    filename: Optional[str] = None,
    doc_id: Optional[str] = None
) -> ParsedDocument:
    """Parse a document into a canonical ParsedDocument object.

    Handles .pdf, .txt, .md, .rst formats automatically.
    """
    inferred_filename = "document.txt"
    if isinstance(source, (str, Path)):
        path = Path(source)
        inferred_filename = path.name
    elif filename:
        inferred_filename = filename

    resolved_filename = filename or inferred_filename
    suffix = Path(resolved_filename).suffix.lower()

    if suffix == ".pdf":
        raw_text, num_pages = extract_text_from_pdf(source)
    elif suffix in (".txt", ".md", ".markdown", ".rst", ".log", ""):
        raw_text = extract_text_from_txt(source)
        num_pages = 1
    else:
        # Fallback to text parsing for unknown text-based formats
        raw_text = extract_text_from_txt(source)
        num_pages = 1

    cleaned_content = _clean_text(raw_text)
    if not cleaned_content:
        raise ValueError(f"No extractable text found in '{resolved_filename}'.")

    final_doc_id = doc_id or _generate_doc_id(cleaned_content, resolved_filename)

    return ParsedDocument(
        doc_id=final_doc_id,
        filename=resolved_filename,
        content=cleaned_content,
        num_pages=num_pages,
        metadata={
            "char_count": len(cleaned_content),
            "file_type": suffix or ".txt",
        }
    )
