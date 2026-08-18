"""
ingest.py — Load and extract text from government scheme PDF documents.

Uses pdfplumber for reliable text extraction that handles tables,
multi-column layouts, and mixed content better than basic PDF parsers.
"""
from pathlib import Path
from typing import List, Dict, Any, Union

import pdfplumber

from src.config import RAW_DIR


def load_documents(raw_dir: Union[str, Path] = RAW_DIR) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page from every PDF in the given directory.

    Args:
        raw_dir: Folder containing the PDF files (defaults to data/raw/).

    Returns:
        List of page dicts, one per non-empty page:
            {
                'source': PDF filename  (e.g. "mahila_samridhi.pdf"),
                'page':   Page number starting from 1,
                'text':   Extracted text for that page,
            }

    Raises:
        FileNotFoundError: If raw_dir does not exist.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {raw_path}")

    pdf_files = sorted(raw_path.glob("*.pdf"))
    documents: List[Dict[str, Any]] = []

    print(f"Loading documents from {raw_path} ({len(pdf_files)} PDF files found)...")

    for pdf_file in pdf_files:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    extracted_text = page.extract_text()
                    # Skip pages with no extractable text (e.g. scanned images)
                    if extracted_text and extracted_text.strip():
                        documents.append({
                            "source": pdf_file.name,
                            "page": page_idx,
                            "text": extracted_text.strip(),
                        })
        except Exception as e:
            print(f"  Warning: could not read {pdf_file.name} — {e}")

    print(f"Loaded {len(documents)} pages across {len(pdf_files)} PDF documents.")
    return documents