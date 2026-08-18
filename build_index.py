"""
build_index.py — One-command index builder.

Run this once after cloning the repo (or whenever you add new PDFs):

    python build_index.py

It reads every PDF from data/raw/, splits the text into chunks, generates
vector embeddings, and stores them in ChromaDB at data/chroma_db/.
Subsequent runs of the app load from this pre-built index instantly.
"""
import sys
import time
from pathlib import Path

# Ensure src/ is importable when this script is run from any directory
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingest import load_documents
from src.chunker import chunk_documents
from src.vectorstore import populate_db
from src.config import RAW_DIR, CHROMA_DIR


def main():
    print("=" * 60)
    print("  Sarkari Scheme Navigator — Vector Index Builder")
    print("=" * 60)

    start = time.time()

    print(f"\n1. Ingesting PDFs from:  {RAW_DIR}")
    docs = load_documents(raw_dir=RAW_DIR)
    print(f"   → {len(docs)} pages loaded.")

    print(f"\n2. Chunking documents (chunk_size=500, overlap=50)...")
    chunks = chunk_documents(docs)
    print(f"   → {len(chunks)} text chunks created.")

    print(f"\n3. Generating embeddings & building ChromaDB at {CHROMA_DIR}...")
    populate_db(chunks, persist_dir=CHROMA_DIR)

    elapsed = time.time() - start
    print(f"\n✓  Index build complete in {elapsed:.1f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
