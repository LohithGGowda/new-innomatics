"""
PDF Loader
----------
Loads one or more PDF files using PDFPlumber and returns raw page-level text.
Each page is wrapped in a lightweight Document dataclass so downstream modules
stay decoupled from the PDF library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """Represents a single page extracted from a PDF."""

    source: str          # absolute path to the PDF file
    page_number: int     # 1-based page index
    text: str            # raw extracted text
    metadata: dict = field(default_factory=dict)


def load_pdf(pdf_path: str | Path) -> List[RawDocument]:
    """
    Extract text from every page of a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of RawDocument objects, one per page.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the PDF contains no extractable text.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Loading PDF: %s", path)
    documents: List[RawDocument] = []

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        logger.debug("Total pages: %d", total_pages)

        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                logger.warning("Page %d/%d has no extractable text — skipping.", i, total_pages)
                continue

            documents.append(
                RawDocument(
                    source=str(path),
                    page_number=i,
                    text=text,
                    metadata={
                        "source": str(path),
                        "page": i,
                        "total_pages": total_pages,
                    },
                )
            )

    if not documents:
        raise ValueError(f"No extractable text found in PDF: {path}")

    logger.info("Loaded %d pages from %s", len(documents), path.name)
    return documents


def load_pdfs(pdf_paths: List[str | Path]) -> List[RawDocument]:
    """
    Load multiple PDFs and return a combined list of RawDocuments.

    Args:
        pdf_paths: List of paths to PDF files.

    Returns:
        Combined list of RawDocument objects from all PDFs.
    """
    all_docs: List[RawDocument] = []
    for p in pdf_paths:
        try:
            all_docs.extend(load_pdf(p))
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Skipping %s — %s", p, exc)
    return all_docs
