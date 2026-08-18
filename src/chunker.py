"""
chunker.py — Split extracted PDF pages into overlapping text chunks.

Each chunk keeps track of which document and page it came from,
so we can cite the source when answering questions.
"""
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    pages: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split page-level documents into smaller, overlapping text chunks.

    Why overlap? A small overlap between consecutive chunks ensures that
    sentences near a boundary are not cut off mid-thought, improving
    retrieval quality.

    Args:
        pages:      List of page dicts — each must have keys
                    'source' (filename), 'page' (int), and 'text' (str).
        chunk_size: Maximum length of each chunk in characters.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        List of chunk dicts:
            {
                'id':     Unique string ID  (e.g. "myscheme.pdf__p2__c0"),
                'source': PDF filename,
                'page':   Page number (int),
                'text':   Chunk text (str),
            }
    """
    # RecursiveCharacterTextSplitter tries each separator in order,
    # choosing the one that keeps chunks closest to chunk_size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []

    for page_doc in pages:
        source = page_doc["source"]
        page_num = page_doc["page"]
        raw_text = page_doc["text"]

        splits = splitter.split_text(raw_text)

        for chunk_idx, chunk_text in enumerate(splits):
            if chunk_text.strip():
                chunks.append({
                    "id": f"{source}__p{page_num}__c{chunk_idx}",
                    "source": source,
                    "page": page_num,
                    "text": chunk_text.strip(),
                })

    return chunks
